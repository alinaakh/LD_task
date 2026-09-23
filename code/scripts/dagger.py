"""DAgger data collection: the student drives, the expert labels every step.

Behaviour cloning alone suffers from compounding errors, which for a humanoid
means falling. Here the student acts in closed loop; the privileged expert
(navigator + walker) labels each visited state with its joint targets, and
takes over for 1 s whenever the torso tilts past ~25 deg so the data also
contains recoveries. Episodes run for the full time limit, so the student
also learns to stop and stand at the goal. Features are extracted in-process.

    python scripts/dagger.py --student $G1NAV_DATA/vla/bc/student.pt \
        --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/episodes/dagger1 \
        --n 200 --base_seed 1000000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import rollout, tasks  # noqa: E402
from g1nav.runtime import StudentRunner  # noqa: E402
from g1nav.walker import WalkerPolicy  # noqa: E402
from extract_features import default_subset_path, extract_dir  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--student", required=True)
  ap.add_argument("--walker", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=200)
  ap.add_argument("--base_seed", type=int, default=1_000_000)
  ap.add_argument("--subset", default=None)
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  walker = WalkerPolicy(args.walker)
  subset = Path(args.subset) if args.subset else default_subset_path()
  runner = StudentRunner.from_files(args.student, subset)
  stats = {}
  t0 = time.time()
  for i in range(args.n):
    seed = args.base_seed + i
    path = out / f"ep_{seed:07d}.npz"
    if path.exists():
      continue
    task = tasks.sample_task(np.random.default_rng(seed), split="train")
    ep = rollout.run_episode(task, walker, seed=seed, student=runner, dagger=True)
    rollout.save_episode(path, ep)
    s = stats.setdefault(task.family, {"n": 0, "success": 0, "expert_frac": 0.0})
    s["n"] += 1
    s["success"] += int(ep["meta"]["result"]["success"])
    s["expert_frac"] += float(ep["expert_drove"].mean())
    if (i + 1) % 10 == 0:
      print(f"{i + 1}/{args.n} episodes, {(time.time() - t0) / 60:.1f} min", flush=True)
  for s in stats.values():
    s["expert_frac"] /= max(1, s["n"])
  (out / "manifest.json").write_text(json.dumps(dict(
      generator="scripts/dagger.py", student=args.student, base_seed=args.base_seed, n=args.n,
      stats=stats), indent=2))
  print(json.dumps(stats, indent=2))
  extract_dir(out, runner.vision, runner.text, runner.device)


if __name__ == "__main__":
  main()
