"""Closed-loop evaluation in MuJoCo: success rates, failure reasons, real-time budget, videos.

Test scenes use seeds disjoint from training/DAgger seeds. Two instruction
splits: "train" templates (new scenes, familiar phrasings) and "novel"
templates (phrasings never seen in training). The policy runs for the full
time limit and is judged on where it ends up.

    python scripts/evaluate.py --student $G1NAV_DATA/vla/dagger2/student.pt \
        --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/eval --n 20 --videos 3
    python scripts/evaluate.py --expert ...   # same protocol with the privileged expert
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import robot, rollout, tasks  # noqa: E402
from g1nav.walker import WalkerPolicy  # noqa: E402

TEST_SEED = 50_000_000


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--student", default=None)
  ap.add_argument("--expert", action="store_true", help="evaluate the privileged expert instead")
  ap.add_argument("--walker", required=True)
  ap.add_argument("--subset", default=None)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=20, help="episodes per family per split")
  ap.add_argument("--splits", nargs="+", default=["train", "novel"])
  ap.add_argument("--videos", type=int, default=3, help="videos per split (plus the 2 demos)")
  args = ap.parse_args()
  assert args.expert or args.student, "--student or --expert"

  out = Path(args.out)
  (out / "videos").mkdir(parents=True, exist_ok=True)
  walker = WalkerPolicy(args.walker)
  runner = None
  if not args.expert:
    from extract_features import default_subset_path
    from g1nav.runtime import StudentRunner
    runner = StudentRunner.from_files(args.student, Path(args.subset or default_subset_path()))

  records = []

  def run(task, seed, video=None, demo=False):
    t0 = time.perf_counter()
    ep = rollout.run_episode(task, walker, seed=seed, student=runner, video_path=video,
                             stop_when_done=False)
    wall = time.perf_counter() - t0
    m = ep["meta"]
    rec = dict(family=task.family, split=task.template_split, instruction=task.instruction,
               seed=seed, success=m["result"]["success"], reason=m["result"]["reason"],
               sim_time=m["steps"] * robot.CTRL_DT, wall_time=wall, video=video,
               latency_ms=m.get("latency_ms"), demo=demo)
    records.append(rec)
    print(f"[{task.template_split:5s}] {task.family:9s} {'OK ' if rec['success'] else 'FAIL'} "
          f"{rec['reason']:17s} {task.instruction!r}", flush=True)
    return rec

  # The two instructions from the task statement, recorded on video.
  for j, kind in enumerate(["follow_red_ball", "pass_yellow_cube_turn_right"]):
    seed = TEST_SEED - 1 - j
    run(tasks.demo_task(kind, np.random.default_rng(seed)), seed,
        str(out / "videos" / f"demo_{kind}.mp4"), demo=True)

  for s_i, split in enumerate(args.splits):
    n_vid = 0
    for f_i, fam in enumerate(tasks.FAMILIES):
      for i in range(args.n):
        seed = TEST_SEED + 100_000 * s_i + 1000 * f_i + i
        task = tasks.sample_task(np.random.default_rng(seed), split=split, family=fam)
        video = None
        if n_vid < args.videos and i == 0 and fam in ("goto", "pass_turn", "sequence"):
          video = str(out / "videos" / f"{split}_{fam}_{seed}.mp4")
          n_vid += 1
        run(task, seed, video)

  summary = collections.defaultdict(dict)
  for split in args.splits:
    rs = [r for r in records if r["split"] == split and not r["demo"]]
    for fam in tasks.FAMILIES:
      fr = [r for r in rs if r["family"] == fam]
      summary[split][fam] = dict(n=len(fr), success=float(np.mean([r["success"] for r in fr])),
                                 reasons=dict(collections.Counter(r["reason"] for r in fr)))
    summary[split]["all"] = dict(n=len(rs), success=float(np.mean([r["success"] for r in rs])))
  no_video = [r for r in records if r["video"] is None]
  result = dict(policy="expert" if args.expert else args.student, summary=summary,
                realtime=dict(
                    sim_seconds=sum(r["sim_time"] for r in no_video),
                    wall_seconds=sum(r["wall_time"] for r in no_video),
                    realtime_factor=sum(r["sim_time"] for r in no_video) /
                    max(1e-9, sum(r["wall_time"] for r in no_video))),
                episodes=records)
  if runner is not None:
    result["component_latency"] = runner.timing_summary()
  (out / "results.json").write_text(json.dumps(result, indent=2))
  print(json.dumps(dict(summary=summary, realtime=result["realtime"],
                        component_latency=result.get("component_latency")), indent=2))


if __name__ == "__main__":
  main()
