"""The student VLA (trained from scratch on top of frozen GR00T features).

Two rates, like GR00T's System-2 / System-1 split but much smaller:

  slow (10 Hz, on each new camera frame)
    tokens = instruction (frozen GR00T LLM states, projected)
           + GR00T vision tokens for 8 frames spanning the last 3.2 s
           + depth CNN tokens for 2 frames
           + current proprioception / odometry
           + a learned [CTRL] query
    -> 4-layer transformer -> latent z (256) and an auxiliary velocity-command head

  fast (50 Hz, every control step)
    [z, predicted command, proprioception history (4 steps)] -> MLP -> 15 joint
    targets (normalized action in [-1, 1]; target = default + 0.5 * action)

The command head is supervised with the expert's velocity command; it is an
auxiliary signal only. The policy's output is always the joint-target vector.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from g1nav import groot, robot
from g1nav.sim import PROPRIO_DIM

CMD_SCALE = (0.6, 0.3, 0.8)


@dataclasses.dataclass
class StudentConfig:
  d_model: int = 256
  heads: int = 8
  slow_layers: int = 4
  ff: int = 1024
  dropout: float = 0.1
  vis_offsets: tuple = (0, 1, 2, 4, 8, 16, 24, 32)  # in vision steps (0.1 s)
  depth_offsets: tuple = (0, 2)
  prop_hist: int = 4                                 # control steps (0.02 s)
  fast_hidden: int = 512
  text_dim: int = groot.LLM_DIM
  vis_dim: int = groot.LLM_DIM
  vis_tokens: int = groot.NUM_VIS_TOKENS
  max_text: int = groot.MAX_TEXT_TOKENS
  proprio_dim: int = PROPRIO_DIM
  action_dim: int = robot.NUM_LOWER


class DepthEncoder(nn.Module):
  """64x64 metric depth -> 4x4 grid of tokens."""

  def __init__(self, d):
    super().__init__()
    chans = [1, 32, 64, 128, d]
    layers = []
    for i in range(4):
      layers += [nn.Conv2d(chans[i], chans[i + 1], 3 if i else 5, stride=2, padding=1 if i else 2),
                 nn.GroupNorm(8, chans[i + 1]), nn.GELU()]
    self.net = nn.Sequential(*layers)

  def forward(self, depth):  # (N, 64, 64) metres
    x = self.net((depth.float() / 3.0 - 1.0).unsqueeze(1))
    return x.flatten(2).transpose(1, 2)  # (N, 16, d)


def _mlp(i, h, o, n=2):
  layers, d = [], i
  for _ in range(n):
    layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
    d = h
  return nn.Sequential(*layers, nn.Linear(d, o))


class StudentVLA(nn.Module):

  def __init__(self, cfg: StudentConfig = StudentConfig()):
    super().__init__()
    self.cfg = cfg
    d = cfg.d_model
    self.text_proj = nn.Sequential(nn.LayerNorm(cfg.text_dim), nn.Linear(cfg.text_dim, d))
    self.vis_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
    self.depth_enc = DepthEncoder(d)
    self.prop_proj = _mlp(cfg.proprio_dim, d, d, 1)
    self.type_emb = nn.Parameter(torch.zeros(4, d))
    self.vis_pos = nn.Parameter(torch.zeros(cfg.vis_tokens, d))
    self.vis_age = nn.Parameter(torch.zeros(len(cfg.vis_offsets), d))
    self.depth_pos = nn.Parameter(torch.zeros(16, d))
    self.depth_age = nn.Parameter(torch.zeros(len(cfg.depth_offsets), d))
    self.text_pos = nn.Parameter(torch.zeros(cfg.max_text, d))
    self.ctrl = nn.Parameter(torch.zeros(1, 1, d))
    for p in (self.type_emb, self.vis_pos, self.vis_age, self.depth_pos, self.depth_age,
              self.text_pos, self.ctrl):
      nn.init.normal_(p, std=0.02)
    layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    self.slow = nn.TransformerEncoder(layer, cfg.slow_layers, enable_nested_tensor=False)
    self.z_norm = nn.LayerNorm(d)
    self.cmd_head = _mlp(d, 256, 3, 1)
    self.fast = _mlp(d + 3 + cfg.prop_hist * cfg.proprio_dim, cfg.fast_hidden, cfg.action_dim, 3)
    self.register_buffer("prop_mean", torch.zeros(cfg.proprio_dim))
    self.register_buffer("prop_std", torch.ones(cfg.proprio_dim))
    self.register_buffer("cmd_scale", torch.tensor(CMD_SCALE))

  def set_normalization(self, mean, std):
    self.prop_mean.copy_(torch.as_tensor(mean))
    self.prop_std.copy_(torch.as_tensor(std).clamp_min(1e-3))

  def norm_prop(self, p):
    return (p - self.prop_mean) / self.prop_std

  def slow_forward(self, text, text_mask, vis, vis_valid, depth, prop_now):
    """text (B,L,2048), text_mask (B,L) True=real token, vis (B,V,16,2048),
    vis_valid (B,V) bool, depth (B,D,64,64), prop_now (B,P) -> z (B,d), cmd (B,3)."""
    b, v = vis.shape[:2]
    n_depth = depth.shape[1]
    t_tok = self.text_proj(text.float()) + self.text_pos[: text.shape[1]] + self.type_emb[0]
    v_tok = self.vis_proj(vis.float()) + self.vis_pos + self.vis_age[:, None] + self.type_emb[1]
    d_tok = self.depth_enc(depth.flatten(0, 1)).view(b, n_depth, 16, -1)
    d_tok = d_tok + self.depth_pos + self.depth_age[:, None] + self.type_emb[2]
    p_tok = self.prop_proj(self.norm_prop(prop_now))[:, None] + self.type_emb[3]
    tokens = torch.cat([self.ctrl.expand(b, -1, -1), t_tok, v_tok.flatten(1, 2),
                        d_tok.flatten(1, 2), p_tok], dim=1)
    ones = torch.ones(b, 1, dtype=torch.bool, device=vis.device)
    keep = torch.cat([ones, text_mask, vis_valid.repeat_interleave(self.cfg.vis_tokens, 1),
                      torch.ones(b, n_depth * 16 + 1, dtype=torch.bool, device=vis.device)], 1)
    h = self.slow(tokens, src_key_padding_mask=~keep)
    z = self.z_norm(h[:, 0])
    return z, self.cmd_head(z)

  def fast_forward(self, z, cmd_norm, prop_hist):
    """prop_hist (B, H, P) oldest..newest -> normalized action (B, 15)."""
    x = torch.cat([z, cmd_norm, self.norm_prop(prop_hist).flatten(1)], dim=1)
    return torch.tanh(self.fast(x))

  def forward(self, batch):
    z, cmd = self.slow_forward(batch["text"], batch["text_mask"], batch["vis"],
                               batch["vis_valid"], batch["depth"], batch["prop_hist"][:, -1])
    return self.fast_forward(z, cmd, batch["prop_hist"]), cmd

  def loss(self, batch, cmd_weight=0.5):
    act, cmd = self(batch)
    l_act = F.mse_loss(act, batch["action"])
    l_cmd = F.mse_loss(cmd, batch["command"] / self.cmd_scale)
    return l_act + cmd_weight * l_cmd, dict(action_mse=l_act.detach(), cmd_mse=l_cmd.detach())

  def save(self, path, extra=None):
    torch.save(dict(state_dict=self.state_dict(), config=dataclasses.asdict(self.cfg),
                    extra=extra or {}), path)

  @staticmethod
  def load(path, map_location="cpu") -> "StudentVLA":
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    cfg = StudentConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()})
    m = StudentVLA(cfg)
    m.load_state_dict(ck["state_dict"])
    return m
