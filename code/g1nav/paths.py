"""Filesystem locations. Override the data root with the G1NAV_DATA env var."""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
G1_ASSETS = CODE_ROOT / "assets" / "g1"
SCENE_XML = G1_ASSETS / "scene_mjx_feetonly_flat_terrain.xml"

DATA_ROOT = Path(os.environ.get("G1NAV_DATA", CODE_ROOT.parent / "runs"))


def data_dir(*parts: str) -> Path:
  p = DATA_ROOT.joinpath(*parts)
  p.mkdir(parents=True, exist_ok=True)
  return p
