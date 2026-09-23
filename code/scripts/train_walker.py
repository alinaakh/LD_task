"""Train the G1 lower-body walker with PPO in MuJoCo Playground (MJX, GPU).

This is our own RL policy (trained from scratch here, no pretrained walker):
Playground's G1 Joystick task with the action space restricted to the 15
lower-body joints; arms hold their default pose. It serves as the low-level
half of the *expert* that labels demonstrations; the student VLA never calls it.

Checkpoints go to <out>/checkpoints/run_<k>/<step>; rerunning the script
resumes from the newest checkpoint and trains only the remaining steps, so a
Colab disconnect costs at most one checkpoint interval.

    python scripts/train_walker.py --out $G1NAV_DATA/walker            # train / resume
    python scripts/train_walker.py --out $G1NAV_DATA/walker --export_only
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground.config import locomotion_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav import paths, robot  # noqa: E402
from g1nav.walker import WalkerPolicy  # noqa: E402

ENV_NAME = "G1JoystickFlatTerrain"


class LowerBodyJoystick(g1_joystick.Joystick):
  """G1 Joystick where the policy commands only legs + waist (15 joints)."""

  @property
  def action_size(self) -> int:
    return robot.NUM_LOWER

  def step(self, state, action):
    full = jp.zeros(self.mjx_model.nu).at[: robot.NUM_LOWER].set(action)
    return super().step(state, full)


def _link_menagerie():
  """Point Playground at the G1 meshes fetched by setup_assets.py (same pinned
  menagerie commit) instead of cloning the whole menagerie repository."""
  from mujoco_playground._src import mjx_env
  dst = Path(mjx_env.MENAGERIE_PATH) / "unitree_g1" / "assets"
  if not dst.exists():
    src = paths.G1_ASSETS / "meshes"
    assert src.exists(), "run scripts/setup_assets.py first"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)


def make_env(impl: str):
  _link_menagerie()
  cfg = g1_joystick.default_config()
  return LowerBodyJoystick(task="flat_terrain", config=cfg, config_overrides={"impl": impl})


def ppo_config(impl, num_timesteps, num_envs, smoke=False):
  p = locomotion_params.brax_ppo_config(ENV_NAME, impl)
  p.num_timesteps = num_timesteps
  if num_envs:
    p.num_envs = num_envs
  if smoke:  # tiny CPU run that only exercises the code path
    p.update(num_envs=8, batch_size=8, num_minibatches=2, unroll_length=5,
             episode_length=50, num_eval_envs=4, num_resets_per_eval=0)
  return p


def _all_checkpoints(ckpt_root: Path):
  out = []
  for run in sorted(ckpt_root.glob("run_*")):
    steps = sorted((int(d.name), d) for d in run.iterdir() if d.is_dir() and d.name.isdigit())
    if steps:
      out.append((run, steps))
  return out


def train(args):
  out = Path(args.out)
  ckpt_root = out / "checkpoints"
  ckpt_root.mkdir(parents=True, exist_ok=True)

  runs = _all_checkpoints(ckpt_root)
  done_steps = sum(steps[-1][0] for _, steps in runs)
  restore = runs[-1][1][-1][1] if runs else None
  remaining = args.num_timesteps - done_steps
  print(f"devices={jax.devices()} done={done_steps:,} remaining={remaining:,} restore={restore}")
  if remaining <= 0:
    return

  env = make_env(args.impl)
  eval_env = make_env(args.impl)
  p = ppo_config(args.impl, remaining, args.num_envs, args.smoke)
  # Keep the checkpoint interval roughly constant (~10M steps) across restarts.
  p.num_evals = max(2, int(remaining // 10_000_000) + 1)
  params = dict(p)
  net_cfg = params.pop("network_factory")
  num_eval_envs = params.pop("num_eval_envs", 128)
  run_dir = ckpt_root / f"run_{len(runs):02d}"
  run_dir.mkdir()

  log = open(out / "progress.jsonl", "a")
  t0 = time.time()

  def progress(step, metrics):
    rec = {"step": done_steps + int(step), "time_min": (time.time() - t0) / 60}
    keys = {"eval/episode_reward": "reward", "eval/avg_episode_length": "ep_len",
            "eval/episode_reward/tracking_lin_vel": "track_lin",
            "eval/episode_reward/tracking_ang_vel": "track_ang",
            "eval/episode_reward/termination": "termination"}
    rec.update({short: round(float(metrics[k]), 3) for k, short in keys.items() if k in metrics})
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.flush()

  ppo.train(
      environment=env,
      eval_env=eval_env,
      **params,
      network_factory=functools.partial(ppo_networks.make_ppo_networks, **net_cfg),
      randomization_fn=registry.get_domain_randomizer(ENV_NAME),
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      seed=args.seed + len(runs),
      progress_fn=progress,
      save_checkpoint_path=str(run_dir),
      restore_checkpoint_path=str(restore) if restore else None,
  )
  print(f"training finished in {(time.time() - t0) / 60:.1f} min")


def export(args):
  """Checkpoint -> walker.npz, then check the numpy runtime against Brax."""
  from brax.training import checkpoint

  out = Path(args.out)
  runs = _all_checkpoints(out / "checkpoints")
  assert runs, "no checkpoints found"
  path = runs[-1][1][-1][1]
  params = checkpoint.load(str(path))
  normalizer, policy = params[0], params[1]
  mean, std = normalizer.mean, normalizer.std
  if isinstance(mean, dict):
    mean, std = mean["state"], std["state"]
  layers = policy["params"]
  names = sorted(layers, key=lambda k: int(k.split("_")[-1]))
  arrays = {"obs_mean": np.asarray(mean), "obs_std": np.asarray(std), "num_layers": len(names)}
  for i, k in enumerate(names):
    arrays[f"W{i}"] = np.asarray(layers[k]["kernel"])
    arrays[f"b{i}"] = np.asarray(layers[k]["bias"])
  npz = out / "walker.npz"
  np.savez(npz, **arrays)
  print(f"exported {path} -> {npz}")

  env = make_env(args.impl)
  p = ppo_config(args.impl, 0, None)
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=__import__(
          "brax.training.acme.running_statistics", fromlist=["normalize"]).normalize,
      **p.network_factory)
  infer = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
  ours = WalkerPolicy(npz)
  state = jax.jit(env.reset)(jax.random.PRNGKey(0))
  step = jax.jit(env.step)
  err = 0.0
  for _ in range(50):
    a_ref, _ = infer(state.obs, jax.random.PRNGKey(0))
    a_np = ours(np.asarray(state.obs["state"]))
    err = max(err, float(np.abs(np.asarray(a_ref) - a_np).max()))
    state = step(state, a_ref)
  print(f"numpy runtime vs Brax: max |diff| = {err:.2e}")
  assert err < 1e-3, "exported walker does not match Brax"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--num_timesteps", type=int, default=200_000_000)
  ap.add_argument("--num_envs", type=int, default=None)
  ap.add_argument("--impl", default="jax", choices=["jax", "warp"])
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--export_only", action="store_true")
  ap.add_argument("--smoke", action="store_true", help="tiny CPU run to test the pipeline")
  args = ap.parse_args()
  if not args.export_only:
    train(args)
  export(args)


if __name__ == "__main__":
  main()
