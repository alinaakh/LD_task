"""Download the Unitree G1 model used by MuJoCo Playground's G1 joystick task.

The XMLs come from mujoco_playground and the meshes from mujoco_menagerie, both
pinned to fixed commits so the robot is byte-identical across machines. Mesh
paths are rewritten to a local `meshes/` directory.

    python scripts/setup_assets.py
"""

import argparse
import re
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import paths  # noqa: E402

PLAYGROUND_SHA = "4057c147714b6ac09b395377f1a1724bbeacc4d3"
MENAGERIE_SHA = "1b86ece576591213e2b666ebf59508454200ca97"
PLAYGROUND_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_playground/"
    f"{PLAYGROUND_SHA}/mujoco_playground/_src/locomotion/g1/xmls/"
)
MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{MENAGERIE_SHA}/unitree_g1/assets/"
)
XML_FILES = ["g1_mjx_feetonly.xml", "scene_mjx_feetonly_flat_terrain.xml", "sensor.xml"]
MESH_PREFIX = "../../../../../mujoco_menagerie/unitree_g1/assets/"


def _ssl_context():
  try:  # python.org builds on macOS ship without root certificates.
    import certifi
    return ssl.create_default_context(cafile=certifi.where())
  except ImportError:
    return None


def fetch(url: str) -> bytes:
  with urllib.request.urlopen(url, context=_ssl_context()) as r:
    return r.read()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()

  out = paths.G1_ASSETS
  (out / "meshes").mkdir(parents=True, exist_ok=True)
  for name in XML_FILES:
    dst = out / name
    if dst.exists() and not args.force:
      continue
    text = fetch(PLAYGROUND_URL + name).decode()
    text = text.replace(MESH_PREFIX, "meshes/")
    dst.write_text(text)
    print("xml  ", name)

  robot_xml = (out / "g1_mjx_feetonly.xml").read_text()
  meshes = sorted(set(re.findall(r'file="meshes/([^"]+)"', robot_xml)))
  for name in meshes:
    dst = out / "meshes" / name
    if dst.exists() and not args.force:
      continue
    dst.write_bytes(fetch(MENAGERIE_URL + name))
    print("mesh ", name)
  print(f"G1 assets ready in {out} ({len(meshes)} meshes)")


if __name__ == "__main__":
  main()
