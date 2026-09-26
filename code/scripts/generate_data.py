"""Generate expert demonstrations: random arena + instruction -> expert rollout.

Episode i uses seed `base_seed + i` for both the task sampler and the sensor
noise, so the dataset is exactly reproducible from (walker.npz, base_seed, n).
Failed expert episodes (fall / collision / wrong final pose) are discarded and
counted in the manifest. Parallelized over CPU processes; one line is printed
per episode (seed, family, outcome, simulated vs. wall-clock seconds). If a
worker process dies (e.g. killed or crashed in the GL driver), the run stops
with an error instead of waiting forever; rerunning skips saved episodes.

    python scripts/generate_data.py --walker $G1NAV_DATA/walker/walker.npz \
        --out $G1NAV_DATA/episodes/expert --n 600 --base_seed 0
"""

import argparse
import faulthandler
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import rollout, tasks  # noqa: E402
from g1nav.walker import WalkerPolicy  # noqa: E402

_WALKER = None


def _init(walker_path):
  global _WALKER
  _WALKER = WalkerPolicy(walker_path)


def _work(job):
  seed, out_dir = job
  t0 = time.time()
  # Watchdog: if one episode runs for over 3 min of wall time, print where this
  # worker is (stack trace to stderr) every 3 min, to tell "slow" from "stuck".
  print(f"      start seed={seed}", flush=True)
  faulthandler.dump_traceback_later(180, repeat=True)
  try:
    return _run(seed, out_dir, t0)
  finally:
    faulthandler.cancel_dump_traceback_later()


def _run(seed, out_dir, t0):
  path = Path(out_dir) / f"ep_{seed:07d}.npz"
  if path.exists():
    meta = rollout.load_episode(path)["meta"]
    return seed, meta["task"]["family"], True, meta["steps"], "cached", time.time() - t0
  rng = np.random.default_rng(seed)
  task = tasks.sample_task(rng, split="train")
  ep = rollout.run_episode(task, _WALKER, seed=seed, progress_every=500)
  ok = ep["meta"]["result"]["success"]
  if ok:
    rollout.save_episode(path, ep)
  return (seed, task.family, ok, ep["meta"]["steps"], ep["meta"]["result"]["reason"],
          time.time() - t0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--walker", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=600, help="number of expert attempts")
  ap.add_argument("--base_seed", type=int, default=0)
  ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
  args = ap.parse_args()
  print(f"{args.workers} worker processes, MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')}",
        flush=True)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  jobs = [(args.base_seed + i, str(out)) for i in range(args.n)]
  t0 = time.time()
  stats = {}
  # ProcessPoolExecutor (unlike multiprocessing.Pool) raises BrokenProcessPool
  # when a worker dies instead of hanging forever.
  with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                           initializer=_init, initargs=(args.walker,)) as pool:
    futures = [pool.submit(_work, job) for job in jobs]
    for k, fut in enumerate(as_completed(futures), 1):
      seed, fam, ok, steps, reason, wall = fut.result()
      s = stats.setdefault(fam, {"ok": 0, "fail": 0, "steps": 0, "reasons": {}})
      s["ok" if ok else "fail"] += 1
      s["steps"] += steps if ok else 0
      if not ok:
        s["reasons"][reason] = s["reasons"].get(reason, 0) + 1
      elapsed = time.time() - t0
      print(f"{k:4d}/{len(jobs)} seed={seed:<7d} {fam:9s} {reason:17s} "
            f"sim={steps * 0.02:5.1f}s wall={wall:6.1f}s | elapsed {elapsed / 60:5.1f} min", flush=True)

  walker_md5 = hashlib.md5(Path(args.walker).read_bytes()).hexdigest()
  manifest = dict(generator="scripts/generate_data.py", base_seed=args.base_seed, n=args.n,
                  walker_md5=walker_md5, template_split="train", stats=stats,
                  total_ok=sum(s["ok"] for s in stats.values()),
                  total_hours=sum(s["steps"] for s in stats.values()) * 0.02 / 3600,
                  wall_min=(time.time() - t0) / 60)
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
