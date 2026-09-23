"""End-to-end plumbing test with tiny random stand-ins for the GR00T encoders.

Exercises: expert episodes -> feature extraction -> student training ->
closed-loop student rollout (DAgger mode) -> video. Numbers are meaningless;
it only checks that shapes, indexing, and file formats agree across stages.
Real GR00T weights are checked separately (names/shapes vs the checkpoint).

    python tests/test_pipeline_smoke.py --walker path/to/walker.npz --tmp /tmp/g1nav_smoke
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))
from g1nav import groot, rollout, tasks  # noqa: E402
from g1nav.model import StudentVLA  # noqa: E402
from g1nav.runtime import StudentRunner, pick_device  # noqa: E402
from g1nav.walker import WalkerPolicy  # noqa: E402
from extract_features import extract_dir  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--walker", required=True)
  ap.add_argument("--tmp", required=True)
  args = ap.parse_args()
  tmp = Path(args.tmp)
  ep_dir = tmp / "episodes"
  ep_dir.mkdir(parents=True, exist_ok=True)
  walker = WalkerPolicy(args.walker)

  for seed in range(6):
    task = tasks.sample_task(np.random.default_rng(seed))
    ep = rollout.run_episode(task, walker, seed=seed, time_limit=1.5)
    rollout.save_episode(ep_dir / f"ep_{seed:07d}.npz", ep)
  print("episodes ok")

  device = pick_device()
  torch.manual_seed(0)
  vision = groot.EagleVision(depth=2).to(device).eval()
  text = groot.EagleText(num_layers=1).to(device).eval()
  extract_dir(ep_dir, vision, text, device, batch=8)

  ckpt = tmp / "vla"
  subprocess.run([sys.executable, str(CODE / "scripts" / "train_vla.py"), "--data", str(ep_dir),
                  "--out", str(ckpt), "--steps", "30", "--batch", "32", "--eval_every", "15",
                  "--warmup", "5"], check=True)

  runner = StudentRunner(StudentVLA.load(ckpt / "student.pt"), vision, text, device, torch.float32)
  task = tasks.demo_task("pass_yellow_cube_turn_right", np.random.default_rng(0))
  ep = rollout.run_episode(task, walker, seed=0, student=runner, dagger=True, time_limit=2.0,
                           video_path=tmp / "student.mp4")
  print("student episode:", ep["meta"]["result"], "steps", ep["meta"]["steps"],
        "expert share", float(ep["expert_drove"].mean()))
  print("latency:", runner.timing_summary())


if __name__ == "__main__":
  main()
