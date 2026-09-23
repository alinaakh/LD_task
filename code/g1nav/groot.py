"""Reused GR00T N1.6 parameters: the Eagle vision tower, its projector, and the LLM.

What we take from nvidia/GR00T-N1.6-3B (all frozen):
  backbone.model.vision_model.vision_model.*  SigLIP-style ViT, 27 layers, 1152-d,
                                              14 px patches on a 224 px image (256 tokens)
  backbone.model.mlp1.*                       2x2 pixel-shuffle projector 4608 -> 2048
  backbone.model.language_model.model.*       Qwen3-style LLM truncated to 16 layers
                                              (GR00T's select_layer=16), 2048-d
What we do not take: the 1.1B-parameter DiT action head, its state/action
encoders (embodiment-specific, 50-step flow-matching chunks; too slow for a
50 Hz reactive loop on a T4), and lm_head.

Only the needed tensors are downloaded, via HTTP range requests into the
checkpoint's safetensors shards (about 3.1 GB instead of 6.6 GB).
"""

from __future__ import annotations

import json
import math
import ssl
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "nvidia/GR00T-N1.6-3B"
VISION_PREFIX = "backbone.model.vision_model.vision_model."
MLP1_PREFIX = "backbone.model.mlp1."
LLM_PREFIX = "backbone.model.language_model.model."
KEEP = (VISION_PREFIX + "embeddings.", VISION_PREFIX + "encoder.",
        VISION_PREFIX + "post_layernorm.", MLP1_PREFIX, LLM_PREFIX)
TOKENIZER_REPO = "Qwen/Qwen3-1.7B"  # tokenizer files only (no weights); same BPE vocab
MAX_TEXT_TOKENS = 32

IMAGE_SIZE = 224
PATCH = 14
GRID = IMAGE_SIZE // PATCH  # 16
VIS_DIM = 1152
LLM_DIM = 2048
POOLED_GRID = 4             # 8x8 projector tokens -> 4x4 after 2x2 average pooling
NUM_VIS_TOKENS = POOLED_GRID * POOLED_GRID


# ----------------------------------------------------------------------------- download

def _ctx():
  try:
    import certifi
    return ssl.create_default_context(cafile=certifi.where())
  except ImportError:
    return None


def _get(url, start=None, end=None) -> bytes:
  req = urllib.request.Request(url)
  if start is not None:
    req.add_header("Range", f"bytes={start}-{end}")
  with urllib.request.urlopen(req, context=_ctx()) as r:
    return r.read()


def download_subset(out_path: str | Path, revision: str = "main") -> Path:
  """Fetch only the reused tensors into a single local safetensors file."""
  from safetensors.torch import save_file

  out_path = Path(out_path)
  if out_path.exists():
    return out_path
  info = json.loads(_get(f"https://huggingface.co/api/models/{REPO}/revision/{revision}"))
  sha = info["sha"]
  base = f"https://huggingface.co/{REPO}/resolve/{sha}/"
  index = json.loads(_get(base + "model.safetensors.index.json"))
  shards = sorted({f for k, f in index["weight_map"].items() if k.startswith(KEEP)})
  tensors = {}
  for shard in shards:
    url = base + shard
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    header = json.loads(_get(url, 8, 8 + n - 1))
    header.pop("__metadata__", None)
    wanted = sorted(((k, v) for k, v in header.items() if k.startswith(KEEP)),
                    key=lambda kv: kv[1]["data_offsets"][0])
    # Coalesce neighbouring tensors into ~256 MB range requests.
    i = 0
    while i < len(wanted):
      j, lo = i, wanted[i][1]["data_offsets"][0]
      hi = wanted[i][1]["data_offsets"][1]
      while j + 1 < len(wanted) and wanted[j + 1][1]["data_offsets"][0] - hi < (1 << 20) and \
          wanted[j + 1][1]["data_offsets"][1] - lo < (256 << 20):
        j += 1
        hi = wanted[j][1]["data_offsets"][1]
      blob = _get(url, 8 + n + lo, 8 + n + hi - 1)
      for k, v in wanted[i:j + 1]:
        s, e = v["data_offsets"]
        assert v["dtype"] == "BF16", (k, v["dtype"])
        raw = bytearray(blob[s - lo:e - lo])
        tensors[k] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(v["shape"]).clone()
      print(f"  {shard}: {j + 1}/{len(wanted)} tensors", flush=True)
      i = j + 1
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_file(tensors, str(out_path), metadata={"repo": REPO, "revision": sha})
  total = sum(t.numel() for t in tensors.values())
  print(f"saved {len(tensors)} tensors ({total / 1e6:.0f}M params) from {REPO}@{sha[:10]} -> {out_path}")
  return out_path


def load_subset(path, prefix: str) -> dict[str, torch.Tensor]:
  from safetensors import safe_open
  out = {}
  with safe_open(str(path), framework="pt") as f:
    for k in f.keys():
      if k.startswith(prefix):
        out[k[len(prefix):]] = f.get_tensor(k)
  return out


# ----------------------------------------------------------------------------- vision

class _Attention(nn.Module):

  def __init__(self, dim, heads):
    super().__init__()
    self.heads = heads
    self.q_proj = nn.Linear(dim, dim)
    self.k_proj = nn.Linear(dim, dim)
    self.v_proj = nn.Linear(dim, dim)
    self.out_proj = nn.Linear(dim, dim)

  def forward(self, x):
    b, n, d = x.shape
    q, k, v = (p(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
               for p in (self.q_proj, self.k_proj, self.v_proj))
    y = F.scaled_dot_product_attention(q, k, v)
    return self.out_proj(y.transpose(1, 2).reshape(b, n, d))


class _MLP(nn.Module):

  def __init__(self, dim, hidden):
    super().__init__()
    self.fc1 = nn.Linear(dim, hidden)
    self.fc2 = nn.Linear(hidden, dim)

  def forward(self, x):
    return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _Block(nn.Module):

  def __init__(self, dim, heads, hidden):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
    self.self_attn = _Attention(dim, heads)
    self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
    self.mlp = _MLP(dim, hidden)

  def forward(self, x):
    x = x + self.self_attn(self.layer_norm1(x))
    return x + self.mlp(self.layer_norm2(x))


class _Embeddings(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.patch_embedding = nn.Linear(3 * PATCH * PATCH, dim)
    self.position_embedding = nn.Embedding(GRID * GRID, dim)


class _Encoder(nn.Module):

  def __init__(self, dim, depth, heads, hidden):
    super().__init__()
    self.layers = nn.ModuleList(_Block(dim, heads, hidden) for _ in range(depth))


class EagleVision(nn.Module):
  """GR00T's vision path up to the LLM input: ViT -> pixel shuffle -> mlp1,
  followed by our 2x2 average pooling to 16 tokens of 2048-d."""

  def __init__(self, depth=27, dim=VIS_DIM, heads=16, hidden=4304):
    super().__init__()
    self.embeddings = _Embeddings(dim)
    self.encoder = _Encoder(dim, depth, heads, hidden)
    self.post_layernorm = nn.LayerNorm(dim, eps=1e-6)
    self.mlp1 = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, LLM_DIM), nn.GELU(),
                              nn.Linear(LLM_DIM, LLM_DIM))

  @classmethod
  def from_subset(cls, path) -> "EagleVision":
    m = cls()
    sd = load_subset(path, VISION_PREFIX)
    sd.update({f"mlp1.{k}": v for k, v in load_subset(path, MLP1_PREFIX).items()})
    m.load_state_dict(sd, strict=True)
    return m

  @staticmethod
  def patchify(images: torch.Tensor) -> torch.Tensor:
    """(B, 224, 224, 3) uint8 -> (B, 256, 588) in [-1, 1].

    Patches are flattened (row, col, channel): verified on the checkpoint's
    patch_embedding, whose weights correlate strongly at index lags 3 and 42
    (x and y neighbours in that layout) and not at lags 14 / 196.
    """
    x = images.float() / 127.5 - 1.0
    b = x.shape[0]
    x = x.view(b, GRID, PATCH, GRID, PATCH, 3).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, GRID * GRID, PATCH * PATCH * 3)

  @staticmethod
  def pixel_shuffle(x, scale=0.5):
    """InternVL/Eagle pixel shuffle on a (B, W, H, C) grid."""
    n, w, h, c = x.shape
    x = x.reshape(n, w, int(h * scale), int(c / scale)).permute(0, 2, 1, 3)
    x = x.reshape(n, int(h * scale), int(w * scale), int(c / (scale * scale)))
    return x.permute(0, 2, 1, 3)

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    """(B, 224, 224, 3) uint8 -> (B, 16, 2048)."""
    dtype = self.embeddings.patch_embedding.weight.dtype
    x = self.embeddings.patch_embedding(self.patchify(images).to(dtype))
    x = x + self.embeddings.position_embedding.weight
    for blk in self.encoder.layers:
      x = blk(x)
    x = self.post_layernorm(x)
    b = x.shape[0]
    x = self.pixel_shuffle(x.reshape(b, GRID, GRID, -1))          # (B, 8, 8, 4608)
    x = self.mlp1(x)                                              # (B, 8, 8, 2048)
    x = F.avg_pool2d(x.permute(0, 3, 1, 2), 2)                    # (B, 2048, 4, 4)
    return x.flatten(2).transpose(1, 2)                           # (B, 16, 2048)


# ----------------------------------------------------------------------------- language

def normalize_instruction(text: str) -> str:
  return " ".join(text.lower().strip().rstrip(".!?").split())


class EagleText(nn.Module):
  """GR00T's 16-layer language model, used once per episode to embed the instruction."""

  def __init__(self, num_layers=16):
    super().__init__()
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(vocab_size=151680, hidden_size=LLM_DIM, intermediate_size=6144,
                      num_hidden_layers=num_layers, num_attention_heads=16,
                      num_key_value_heads=8, head_dim=128, rope_theta=1_000_000.0,
                      rms_norm_eps=1e-6, max_position_embeddings=40960,
                      tie_word_embeddings=False, attention_bias=False)
    self.model = Qwen3Model(cfg)
    self.tokenizer = None

  @classmethod
  def from_subset(cls, path) -> "EagleText":
    m = cls()
    m.model.load_state_dict(load_subset(path, LLM_PREFIX), strict=True)
    return m

  def _tok(self):
    if self.tokenizer is None:
      from transformers import AutoTokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    return self.tokenizer

  @torch.no_grad()
  def forward(self, texts: list[str]) -> list[torch.Tensor]:
    """Returns one (L_i, 2048) tensor of final-layer token states per text."""
    tok = self._tok()
    enc = tok([normalize_instruction(t) for t in texts], return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_TEXT_TOKENS)
    dev = next(self.parameters()).device
    out = self.model(input_ids=enc.input_ids.to(dev),
                     attention_mask=enc.attention_mask.to(dev)).last_hidden_state
    lens = enc.attention_mask.sum(1).tolist()
    # Tokenizer pads on the left for Qwen: keep the real tokens.
    if tok.padding_side == "left":
      return [out[i, out.shape[1] - n:] for i, n in enumerate(lens)]
    return [out[i, :n] for i, n in enumerate(lens)]


def load_encoders(subset_path, device, dtype=torch.float16):
  vision = EagleVision.from_subset(subset_path).to(device, dtype).eval().requires_grad_(False)
  text = EagleText.from_subset(subset_path).to(device, dtype).eval().requires_grad_(False)
  return vision, text
