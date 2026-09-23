"""Sanity-check the expert navigator and success checks with an ideal unicycle.

This bypasses the walker entirely (test only; never used for data or eval):
if the expert cannot solve its own tasks with perfect velocity tracking, the
planner or the success check is wrong.

    python tests/test_expert_kinematic.py
"""

import collections
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import tasks  # noqa: E402

DT = 0.02


def rollout(task):
  nav = tasks.ExpertNavigator(task, DT)
  x, y = task.scene.robot_xy
  yaw = task.scene.robot_yaw
  traj, min_clear = [], 1e9
  for step in range(int(task.time_limit / DT)):
    vx, vy, wz = nav.command(x, y, yaw, step * DT)
    x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * DT
    y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * DT
    yaw = tasks.wrap(yaw + wz * DT)
    traj.append((x, y, yaw))
    for o in task.scene.objects:
      min_clear = min(min_clear, math.hypot(x - o.x, y - o.y) - o.radius)
    if nav.done and step * DT - nav.done_time > tasks.HOLD_AFTER_DONE:
      break
  traj = np.array(traj)
  # The G1's legs extend ~0.25 m from the pelvis; closer than that is a collision.
  res = tasks.evaluate(task, traj, fell=False, collided=min_clear < 0.25)
  return res, nav.done_time, min_clear


def main(n=400):
  rng = np.random.default_rng(0)
  stats = collections.defaultdict(list)
  for split in ("train", "novel"):
    for _ in range(n // 2):
      task = tasks.sample_task(rng, split=split)
      res, t_done, clear = rollout(task)
      stats[task.family].append((res["success"], t_done or np.nan, clear))
      if not res["success"]:
        print("FAIL", task.family, repr(task.instruction), res["reason"],
              f"done={t_done} clear={clear:.2f}")
  for fam, v in stats.items():
    v = np.array(v, dtype=float)
    print(f"{fam:10s} n={len(v):3d} success={v[:, 0].mean():.3f} "
          f"median_time={np.nanmedian(v[:, 1]):5.1f}s max_time={np.nanmax(v[:, 1]):5.1f}s "
          f"min_clearance={v[:, 2].min():.2f}m")
  rng = np.random.default_rng(1)
  for _ in range(8):
    print("  e.g.", repr(tasks.sample_task(rng).instruction))


if __name__ == "__main__":
  main()
