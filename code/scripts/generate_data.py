"""Generate expert demonstrations: random arena + instruction -> expert rollout.

Episode i uses seed `base_seed + i` for both the task sampler and the sensor
noise, so the dataset is exactly reproducible from (walker.npz, base_seed, n).
Failed expert episodes (fall / collision / wrong final pose) are discarded and
counted in the manifest. Parallelized over CPU processes.

    python scripts/generate_data.py --walker $G1NAV_DATA/walker/walker.npz \
        --out $G1NAV_DATA/episodes/expert --n 600 --base_seed 0
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
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
  path = Path(out_dir) / f"ep_{seed:07d}.npz"
  if path.exists():
    meta = rollout.load_episode(path)["meta"]
    return seed, meta["task"]["family"], True, meta["steps"], "cached"
  rng = np.random.default_rng(seed)
  task = tasks.sample_task(rng, split="train")
  ep = rollout.run_episode(task, _WALKER, seed=seed)
  ok = ep["meta"]["result"]["success"]
  if ok:
    rollout.save_episode(path, ep)
  return seed, task.family, ok, ep["meta"]["steps"], ep["meta"]["result"]["reason"]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--walker", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=600, help="number of expert attempts")
  ap.add_argument("--base_seed", type=int, default=0)
  ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  jobs = [(args.base_seed + i, str(out)) for i in range(args.n)]
  t0 = time.time()
  stats = {}
  with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.walker,)) as pool:
    for k, (seed, fam, ok, steps, reason) in enumerate(pool.imap_unordered(_work, jobs), 1):
      s = stats.setdefault(fam, {"ok": 0, "fail": 0, "steps": 0, "reasons": {}})
      s["ok" if ok else "fail"] += 1
      s["steps"] += steps if ok else 0
      if not ok:
        s["reasons"][reason] = s["reasons"].get(reason, 0) + 1
      if k % 20 == 0 or k == len(jobs):
        rate = k / (time.time() - t0)
        print(f"{k}/{len(jobs)} episodes ({rate:.2f}/s)", flush=True)

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
