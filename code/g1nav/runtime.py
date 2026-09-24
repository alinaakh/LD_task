"""Real-time policy runtime: GR00T encoders + student VLA inside the control loop.

Per episode : instruction -> frozen LLM (once).
Per 10 Hz   : new RGBD frame -> frozen vision tower (1 image) + slow transformer.
Per 50 Hz   : fast MLP -> 15 joint targets.
Component latencies are measured with device synchronization.
"""

from __future__ import annotations

import collections
import time

import numpy as np
import torch

from g1nav import groot
from g1nav.model import StudentVLA
from g1nav.rollout import downsample_depth


def pick_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def _sync(device):
  if device.type == "cuda":
    torch.cuda.synchronize()
  elif device.type == "mps":
    torch.mps.synchronize()


class StudentRunner:

  def __init__(self, student: StudentVLA, vision, text, device, dtype=torch.float16):
    self.student = student.to(device).eval()
    self.vision = vision
    self.text = text
    self.device = device
    self.dtype = dtype
    self.cfg = student.cfg
    self.timings = collections.defaultdict(list)

  @classmethod
  def from_files(cls, student_ckpt, groot_subset, device=None):
    device = device or pick_device()
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    vision, text = groot.load_encoders(groot_subset, device, dtype)
    runner = cls(StudentVLA.load(student_ckpt), vision, text, device, dtype)
    runner.warmup()
    return runner

  def warmup(self, steps: int = 20):
    """Run dummy inputs once so kernel compilation / tokenizer loading are not timed."""
    from g1nav import arena
    self.reset("warm up")
    rgb = np.zeros((arena.EGO_RES, arena.EGO_RES, 3), np.uint8)
    depth = np.ones((arena.EGO_RES, arena.EGO_RES), np.float32)
    for t in range(steps):
      self.act(np.zeros(self.cfg.proprio_dim, np.float32), (rgb, depth) if t % 5 == 0 else None)
    self.timings.clear()

  def _timed(self, name, fn):
    _sync(self.device)
    t0 = time.perf_counter()
    out = fn()
    _sync(self.device)
    self.timings[name].append(time.perf_counter() - t0)
    return out

  @torch.no_grad()
  def reset(self, instruction: str):
    feats = self._timed("text_once", lambda: self.text([instruction])[0])
    self.text_feat = feats[None]
    self.text_mask = torch.ones(1, feats.shape[0], dtype=torch.bool, device=self.device)
    self.vis_hist = []
    self.depth_hist = []
    self.prop_hist = collections.deque(maxlen=self.cfg.prop_hist)
    self.z = None

  @torch.no_grad()
  def act(self, proprio: np.ndarray, frame=None) -> np.ndarray:
    p = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
    if not self.prop_hist:
      self.prop_hist.extend([p] * self.cfg.prop_hist)
    self.prop_hist.append(p)
    if frame is not None:
      rgb, depth = frame
      img = torch.as_tensor(rgb, device=self.device)[None]
      feat = self._timed("vision", lambda: self._encode(img))
      self.vis_hist.append(feat)
      self.depth_hist.append(torch.as_tensor(downsample_depth(depth).astype(np.float32),
                                             device=self.device))
      self.z, self.cmd = self._timed("slow", lambda: self._slow(p))
    if self.z is None:
      raise RuntimeError("the first act() call of an episode must include a camera frame")
    hist = torch.stack(list(self.prop_hist))[None]
    a = self._timed("fast", lambda: self.student.fast_forward(self.z, self.cmd, hist))
    return a[0].float().cpu().numpy()

  def _encode(self, img):
    # Same int8 round trip as the training features (see groot.quantize_features).
    return groot.dequantize_features(*groot.quantize_features(self.vision(img)[0]))

  def _slow(self, prop_now):
    k = len(self.vis_hist) - 1
    vis = torch.stack([self.vis_hist[max(0, k - o)] for o in self.cfg.vis_offsets])[None]
    valid = torch.tensor([[k - o >= 0 for o in self.cfg.vis_offsets]], device=self.device)
    depth = torch.stack([self.depth_hist[max(0, k - o)] for o in self.cfg.depth_offsets])[None]
    return self.student.slow_forward(self.text_feat, self.text_mask, vis, valid, depth,
                                     prop_now[None])

  def timing_summary(self) -> dict:
    return {k: dict(mean_ms=1e3 * float(np.mean(v)), p99_ms=1e3 * float(np.percentile(v, 99)),
                    n=len(v)) for k, v in self.timings.items()}
