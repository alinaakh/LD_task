"""Episode feature bank for student training.

Everything (frozen GR00T vision features, depth, proprioception, labels) is
concatenated into flat tensors, on the GPU when it fits, and batches are
gathered with index arithmetic that mirrors StudentRunner exactly:
  - a control step t sees the latest camera frame k = t // VISION_EVERY
  - vision history offsets are clamped at the episode start and masked
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch

from g1nav import groot, rollout
from g1nav.model import StudentConfig


def feats_path(ep_path: Path) -> Path:
  return ep_path.with_suffix(".feats.npy")


def text_feats_path(ep_dir: Path) -> Path:
  return ep_dir / "text_feats.pt"


def is_val(ep_path: Path, val_frac: float) -> bool:
  return (zlib.crc32(ep_path.name.encode()) % 1000) < val_frac * 1000


class FeatureBank:

  def __init__(self, dirs, cfg: StudentConfig, device, val_frac=0.05, max_gpu_gb=10.0):
    self.cfg = cfg
    self.device = device
    paths = [p for d in dirs for p in sorted(Path(d).glob("ep_*.npz")) if feats_path(p).exists()]
    if not paths:
      raise FileNotFoundError(f"no episodes with features in {dirs}")
    metas, n_t, n_k = [], 0, 0
    for p in paths:
      ep = rollout.load_episode(p)
      metas.append((p, ep, n_t, n_k))
      n_t += len(ep["action"])
      n_k += len(ep["vis_t"])
    feat_gb = n_k * cfg.vis_tokens * cfg.vis_dim * 2 / 1e9
    store = device if (device.type != "cpu" and feat_gb < max_gpu_gb) else torch.device("cpu")
    print(f"{len(paths)} episodes, {n_t * 0.02 / 3600:.2f} h, {n_k} frames "
          f"({feat_gb:.1f} GB features on {store})")

    self.vis = torch.empty(n_k, cfg.vis_tokens, cfg.vis_dim, dtype=torch.float16, device=store)
    self.depth = torch.empty(n_k, rollout.DEPTH_RES, rollout.DEPTH_RES, dtype=torch.float16,
                             device=device)
    self.prop = torch.empty(n_t, cfg.proprio_dim, device=device)
    self.action = torch.empty(n_t, cfg.action_dim, device=device)
    self.command = torch.empty(n_t, 3, device=device)
    step_ep, step_local = np.empty(n_t, np.int64), np.empty(n_t, np.int64)
    ep_t0, ep_k0, ep_text, is_v = [], [], [], []
    texts, text_ids, text_cache = [], {}, {}
    for e, (p, ep, t0, k0) in enumerate(metas):
      T, K = len(ep["action"]), len(ep["vis_t"])
      f = np.load(feats_path(p))
      assert f.shape[0] == K, (p, f.shape, K)
      self.vis[k0:k0 + K] = torch.from_numpy(f).to(store)
      self.depth[k0:k0 + K] = torch.from_numpy(ep["depth"]).to(device)
      self.prop[t0:t0 + T] = torch.from_numpy(ep["proprio"]).to(device)
      self.action[t0:t0 + T] = torch.from_numpy(ep["action"]).to(device)
      self.command[t0:t0 + T] = torch.from_numpy(ep["command"]).to(device)
      step_ep[t0:t0 + T] = e
      step_local[t0:t0 + T] = np.arange(T)
      instr = ep["meta"]["task"]["instruction"]
      if instr not in text_ids:
        if p.parent not in text_cache:
          text_cache[p.parent] = torch.load(text_feats_path(p.parent))
        text_ids[instr] = len(texts)
        texts.append(text_cache[p.parent][instr])
      ep_t0.append(t0)
      ep_k0.append(k0)
      ep_text.append(text_ids[instr])
      is_v.append(is_val(p, val_frac))

    L = cfg.max_text
    self.text = torch.zeros(len(texts), L, cfg.text_dim, dtype=torch.float16, device=device)
    self.text_mask = torch.zeros(len(texts), L, dtype=torch.bool, device=device)
    for i, t in enumerate(texts):
      n = min(L, t.shape[0])
      self.text[i, :n] = t[:n].to(device)
      self.text_mask[i, :n] = True

    as_t = lambda a: torch.as_tensor(np.asarray(a), device=device)  # noqa: E731
    self.step_ep, self.step_local = as_t(step_ep), as_t(step_local)
    self.ep_t0, self.ep_k0, self.ep_text = as_t(ep_t0), as_t(ep_k0), as_t(ep_text)
    val_ep = as_t(is_v)
    self.split_steps = {"train": torch.nonzero(~val_ep[self.step_ep]).squeeze(1),
                        "val": torch.nonzero(val_ep[self.step_ep]).squeeze(1)}
    self.vis_offsets = as_t(cfg.vis_offsets)
    self.depth_offsets = as_t(cfg.depth_offsets)
    self.hist = as_t(np.arange(cfg.prop_hist - 1, -1, -1))
    self.num_episodes = len(paths)
    self.num_val_episodes = int(sum(is_v))

  def prop_stats(self):
    x = self.prop[self.split_steps["train"]]
    return x.mean(0).cpu(), x.std(0).cpu()

  def sample(self, batch_size, split="train", generator=None, frame_dropout=0.0):
    idx = self.split_steps[split]
    gi = idx[torch.randint(len(idx), (batch_size,), device=self.device, generator=generator)]
    e, tl = self.step_ep[gi], self.step_local[gi]
    k = tl // rollout.VISION_EVERY
    rel = k[:, None] - self.vis_offsets
    valid = rel >= 0
    if frame_dropout > 0:
      drop = torch.rand(valid.shape, device=self.device, generator=generator) < frame_dropout
      drop[:, 0] = False
      valid = valid & ~drop
    vis_idx = self.ep_k0[e, None] + rel.clamp_min(0)
    dep_idx = self.ep_k0[e, None] + (k[:, None] - self.depth_offsets).clamp_min(0)
    prop_idx = self.ep_t0[e, None] + (tl[:, None] - self.hist).clamp_min(0)
    ti = self.ep_text[e]
    return dict(
        vis=self.vis[vis_idx.to(self.vis.device)].to(self.device, non_blocking=True),
        vis_valid=valid,
        depth=self.depth[dep_idx],
        prop_hist=self.prop[prop_idx],
        text=self.text[ti],
        text_mask=self.text_mask[ti],
        action=self.action[gi],
        command=self.command[gi],
    )
