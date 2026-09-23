"""Assemble g1nav_<name>.zip with the layout the task asks for.

    python scripts/package_submission.py --name jane_doe --report /path/to/report.pdf \
        --student $G1NAV_DATA/vla/dagger2/student.pt --eval $G1NAV_DATA/eval
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import paths  # noqa: E402

REPO = paths.CODE_ROOT.parent


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--name", required=True)
  ap.add_argument("--report", required=True, help="your report.pdf")
  ap.add_argument("--student", required=True, help="final student checkpoint")
  ap.add_argument("--eval", required=True, help="evaluate.py output dir (results.json, videos/)")
  ap.add_argument("--data", default=str(paths.DATA_ROOT))
  ap.add_argument("--out_dir", default=str(REPO.parent))
  ap.add_argument("--include_episodes", action="store_true",
                  help="also copy raw episode files (large); default ships manifests only")
  args = ap.parse_args()

  data = Path(args.data)
  stage = Path(args.out_dir) / f"g1nav_{args.name}"
  if stage.exists():
    shutil.rmtree(stage)
  stage.mkdir(parents=True)

  shutil.copy(args.report, stage / "report.pdf")
  shutil.copy(REPO / "README.md", stage / "README.md")
  shutil.copytree(paths.CODE_ROOT, stage / "code",
                  ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs", ".ipynb_checkpoints",
                                                "MUJOCO_LOG.TXT"))

  ck = stage / "checkpoint"
  ck.mkdir()
  shutil.copy(args.student, ck / "student.pt")
  shutil.copy(data / "walker" / "walker.npz", ck / "walker.npz")
  for p in sorted((data / "vla").glob("*/student.pt")):
    if p.resolve() != Path(args.student).resolve():
      dst = ck / "intermediate" / p.parent.name
      dst.mkdir(parents=True, exist_ok=True)
      shutil.copy(p, dst / "student.pt")  # needed to regenerate DAgger data
  for f in ("progress.jsonl",):
    if (data / "walker" / f).exists():
      shutil.copy(data / "walker" / f, ck / f"walker_{f}")
  (ck / "GROOT_WEIGHTS.md").write_text(
      "The frozen GR00T N1.6 tensors (vision tower, mlp1 projector, 16-layer LLM; ~3.1 GB)\n"
      "are not duplicated here. `code/scripts/extract_features.py` downloads exactly those\n"
      "tensors from nvidia/GR00T-N1.6-3B (revision recorded in the safetensors metadata)\n"
      "via `g1nav.groot.download_subset`.\n")

  ds = stage / "dataset"
  ds.mkdir()
  for d in sorted((data / "episodes").iterdir()):
    if not d.is_dir():
      continue
    (ds / d.name).mkdir()
    if (d / "manifest.json").exists():
      shutil.copy(d / "manifest.json", ds / d.name / "manifest.json")
    if args.include_episodes:
      for f in d.glob("ep_*.npz"):
        shutil.copy(f, ds / d.name / f.name)
  shutil.copy(REPO / "dataset_REGENERATE.md", ds / "REGENERATE.md")

  vids = stage / "videos"
  vids.mkdir()
  for v in sorted((Path(args.eval) / "videos").glob("*.mp4")):
    shutil.copy(v, vids / v.name)
  if (Path(args.eval) / "results.json").exists():
    shutil.copy(Path(args.eval) / "results.json", ck / "eval_results.json")

  archive = shutil.make_archive(str(stage), "zip", stage.parent, stage.name)
  n_vid = len(list(vids.glob("*.mp4")))
  print(json.dumps(dict(zip=archive, videos=n_vid,
                        size_mb=round(Path(archive).stat().st_size / 1e6, 1)), indent=2))
  if n_vid < 3:
    print("WARNING: the task asks for at least three videos")


if __name__ == "__main__":
  main()
