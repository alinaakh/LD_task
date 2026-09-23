"""Side-by-side episode videos: egocentric RGB + depth | third-person view."""

from __future__ import annotations

import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from g1nav import arena

_TURBO = np.array([  # coarse turbo colormap anchors
    [48, 18, 59], [70, 107, 227], [40, 187, 236], [49, 242, 153],
    [162, 252, 60], [237, 208, 58], [251, 128, 34], [208, 47, 5], [122, 4, 3]], dtype=np.float32)


def colorize_depth(depth: np.ndarray, dmax: float = arena.DEPTH_MAX) -> np.ndarray:
  x = np.clip(depth / dmax, 0, 1) * (len(_TURBO) - 1)
  i = np.floor(x).astype(int).clip(0, len(_TURBO) - 2)
  f = (x - i)[..., None]
  return ((1 - f) * _TURBO[i] + f * _TURBO[i + 1]).astype(np.uint8)


def _font(size):
  try:
    return ImageFont.load_default(size=size)
  except TypeError:  # Pillow < 10.1
    return ImageFont.load_default()


def compose(ego_rgb, ego_depth, third, instruction: str, status: str) -> np.ndarray:
  """1320x760 frame: [ego RGB over ego depth | third-person], caption bar on top."""
  side = 360
  ego = Image.fromarray(ego_rgb).resize((side, side), Image.BILINEAR)
  dep = Image.fromarray(colorize_depth(ego_depth)).resize((side, side), Image.NEAREST)
  tp = Image.fromarray(third).resize((960, 720), Image.BILINEAR)
  canvas = Image.new("RGB", (side + 960, 720 + 40), (20, 20, 20))
  canvas.paste(ego, (0, 40))
  canvas.paste(dep, (0, 40 + side))
  canvas.paste(tp, (side, 40))
  d = ImageDraw.Draw(canvas)
  d.text((10, 10), f'"{instruction}"', fill=(255, 255, 255), font=_font(18))
  d.text((side + 10, 50), status, fill=(255, 255, 0), font=_font(16))
  for y, label in ((44, "ego RGB"), (44 + side, "ego depth")):
    d.rectangle((2, y - 2, 90, y + 18), fill=(0, 0, 0))
    d.text((6, y), label, fill=(255, 255, 255), font=_font(14))
  return np.asarray(canvas)


class VideoWriter:
  """Streams RGB frames to an H.264 mp4 through the ffmpeg binary."""

  def __init__(self, path, width=1320, height=760, fps=25):
    self.proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)

  def add(self, frame: np.ndarray):
    self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

  def close(self):
    self.proc.stdin.close()
    self.proc.wait()
