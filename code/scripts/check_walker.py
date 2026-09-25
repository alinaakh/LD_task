"""Sim-to-sim check of the exported walker in plain MuJoCo (C engine, our arena).

The walker is trained in MJX; data generation and evaluation run in C MuJoCo.
This script drives it with random velocity-command schedules and then with
the full expert on sampled tasks, and reports falls and tracking error.
Run it before generating data: if the expert fails here, fix the walker first.

    python scripts/check_walker.py --walker $G1NAV_DATA/walker/walker.npz --video out.mp4
    python scripts/check_walker.py --walker ... --trials 0 --video_dir videos/   # one video per task
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import robot, rollout, tasks  # noqa: E402
from g1nav.arena import Scene  # noqa: E402
from g1nav.sim import G1NavSim  # noqa: E402
from g1nav.walker import WalkerPolicy  # noqa: E402


def command_tracking(walker, n_trials, seed):
  rng = np.random.default_rng(seed)
  falls, errs = 0, []
  for trial in range(n_trials):
    sim = G1NavSim(Scene(objects=[], robot_xy=(0.0, 0.0), robot_yaw=0.0), seed=trial)
    cmd = np.zeros(3)
    for t in range(int(20 / robot.CTRL_DT)):
      if t % 150 == 0:  # new command every 3 s, in the navigator's operating range
        cmd = np.array([rng.uniform(0, 0.6), 0.0, rng.uniform(-0.8, 0.8)])
        if rng.random() < 0.2:
          cmd[:] = 0
      sim.step(walker(sim.walker_obs(cmd)))
      if sim.fallen():
        falls += 1
        break
      if t % 150 > 50:  # skip the transient after a command change; noise-free readings
        lin = sim._read("local_linvel_pelvis")
        ang = sim.data.qvel[5]  # free-joint angular velocity is in the pelvis frame
        errs.append([abs(lin[0] - cmd[0]), abs(ang - cmd[2])])
    sim.close()
  errs = np.asarray(errs).reshape(-1, 2)
  if not len(errs):
    return dict(trials=n_trials, falls=falls)
  return dict(trials=n_trials, falls=falls, vx_mae=float(errs[:, 0].mean()),
              wz_mae=float(errs[:, 1].mean()))


def expert_tasks(walker, n, seed, video, start=1, video_dir=None):
  rng = np.random.default_rng(seed)
  by_family = {}
  for i in range(n):
    task = tasks.sample_task(rng, family=tasks.FAMILIES[i % len(tasks.FAMILIES)])
    if i + 1 < start:  # replay the sampler so task numbering matches earlier runs
      continue
    path = video if (video and i + 1 == start) else None
    if video_dir:
      path = str(Path(video_dir) / f"task_{i + 1:02d}_{task.family}.mp4")
    ep = rollout.run_episode(task, walker, seed=seed + i, video_path=path)
    by_family.setdefault(task.family, []).append(ep["meta"]["result"]["success"])
    print(f"{i + 1:3d} {task.family:10s} {ep['meta']['result']['reason']:17s} {task.instruction!r}")
  return {f: float(np.mean(v)) for f, v in by_family.items()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--walker", required=True)
  ap.add_argument("--trials", type=int, default=10)
  ap.add_argument("--tasks", type=int, default=20)
  ap.add_argument("--video", default=None)
  ap.add_argument("--seed", type=int, default=12345)
  ap.add_argument("--start", type=int, default=1, help="first task number to run (1-based)")
  ap.add_argument("--video_dir", default=None, help="save one video per task in this folder")
  args = ap.parse_args()
  walker = WalkerPolicy(args.walker)
  if args.trials > 0:
    tracking = command_tracking(walker, args.trials, args.seed)
    print("command tracking:", json.dumps(tracking))
  if args.video_dir:
    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
  success = expert_tasks(walker, args.tasks, args.seed, args.video, args.start, args.video_dir)
  print("expert success by family:", json.dumps(success))


if __name__ == "__main__":
  main()
