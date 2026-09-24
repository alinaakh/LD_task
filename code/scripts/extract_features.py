"""Precompute frozen GR00T features for every episode (vision per frame, text per instruction).

Downloads the reused GR00T tensors on first use (about 3.1 GB).

    python scripts/extract_features.py --dirs $G1NAV_DATA/episodes/expert
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import groot, paths, rollout  # noqa: E402
from g1nav.dataset import feats_path, text_feats_path  # noqa: E402
from g1nav.runtime import pick_device  # noqa: E402


def default_subset_path() -> Path:
  return paths.DATA_ROOT / "groot" / "groot_n16_subset.safetensors"


@torch.no_grad()
def extract_dir(ep_dir: Path, vision, text, device, batch: int = 64):
  eps = sorted(ep_dir.glob("ep_*.npz"))
  tpath = text_feats_path(ep_dir)
  cache = torch.load(tpath) if tpath.exists() else {}
  instrs = sorted({rollout.load_episode(p)["meta"]["task"]["instruction"] for p in eps} - set(cache))
  for i in range(0, len(instrs), 32):
    chunk = instrs[i:i + 32]
    for s, f in zip(chunk, text(chunk)):
      cache[s] = f.to("cpu", torch.float16)
  torch.save(cache, tpath)

  todo = [p for p in eps if not feats_path(p).exists()]
  t0, frames = time.time(), 0
  for j, p in enumerate(todo):
    rgb = np.stack(rollout.load_episode(p, with_images=True)["rgb"])
    qs, scales = [], []
    for i in range(0, len(rgb), batch):
      q, s = groot.quantize_features(vision(torch.from_numpy(rgb[i:i + batch]).to(device)))
      qs.append(q.cpu())
      scales.append(s.cpu())
    np.savez(feats_path(p), q=torch.cat(qs).numpy(), scale=torch.cat(scales).numpy())
    frames += len(rgb)
    if (j + 1) % 25 == 0 or j + 1 == len(todo):
      print(f"  {ep_dir.name}: {j + 1}/{len(todo)} episodes, {frames / (time.time() - t0):.0f} frames/s",
            flush=True)
  print(f"{ep_dir}: {len(eps)} episodes, {len(instrs)} new instructions, {len(todo)} newly encoded")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dirs", nargs="+", required=True)
  ap.add_argument("--subset", default=None, help="path of the reused GR00T tensors")
  ap.add_argument("--batch", type=int, default=64)
  args = ap.parse_args()
  subset = Path(args.subset) if args.subset else default_subset_path()
  groot.download_subset(subset)
  device = pick_device()
  dtype = torch.float16 if device.type != "cpu" else torch.float32
  vision, text = groot.load_encoders(subset, device, dtype)
  for d in args.dirs:
    extract_dir(Path(d), vision, text, device, args.batch)


if __name__ == "__main__":
  main()
