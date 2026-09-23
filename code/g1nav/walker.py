"""Numpy runtime for the PPO walker trained in MuJoCo Playground.

The Brax policy is `tanh(MLP(normalize(obs)))` with swish activations. We export
the normalizer statistics and dense layers to an .npz (see
scripts/train_walker.py, which also checks this runtime against Brax's own
inference function) so the expert runs anywhere without JAX.
"""

from pathlib import Path

import numpy as np


def _swish(x):
  return x / (1.0 + np.exp(-x))


class WalkerPolicy:
  """Joystick walker: (walker obs, velocity command) -> 15 normalized actions."""

  def __init__(self, npz_path: str | Path):
    npz_path = Path(npz_path)
    if not npz_path.exists():
      ckpts = sorted(p.name for p in npz_path.parent.glob("checkpoints/run_*/*"))
      raise FileNotFoundError(
          f"{npz_path} does not exist: the walker has not been trained/exported yet.\n"
          f"  checkpoints found: {ckpts[-3:] if ckpts else 'none'}\n"
          "  Run stage 4 (scripts/train_walker.py) until it prints 'numpy runtime vs Brax',\n"
          "  or export the latest checkpoint with: train_walker.py --out <walker dir> --export_only")
    z = np.load(npz_path)
    self.mean = z["obs_mean"].astype(np.float64)
    self.std = z["obs_std"].astype(np.float64)
    n = int(z["num_layers"])
    self.layers = [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64))
                   for i in range(n)]
    self.action_size = self.layers[-1][0].shape[1] // 2  # loc and scale halves

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.mean) / self.std
    for i, (w, b) in enumerate(self.layers):
      x = x @ w + b
      if i < len(self.layers) - 1:
        x = _swish(x)
    return np.tanh(x[..., : self.action_size]).astype(np.float32)

  @staticmethod
  def random(path: str | Path, obs_dim: int = 103, act_dim: int = 15, seed: int = 0):
    """Write an untrained walker with the right shapes, for plumbing tests only."""
    rng = np.random.default_rng(seed)
    sizes = [obs_dim, 512, 256, 128, 2 * act_dim]
    arrays = {"obs_mean": np.zeros(obs_dim), "obs_std": np.ones(obs_dim),
              "num_layers": len(sizes) - 1}
    for i in range(len(sizes) - 1):
      arrays[f"W{i}"] = rng.normal(0, 0.05, (sizes[i], sizes[i + 1]))
      arrays[f"b{i}"] = np.zeros(sizes[i + 1])
    np.savez(path, **arrays)
    return WalkerPolicy(path)
