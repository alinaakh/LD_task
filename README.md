# G1Nav: a small real-time VLA for Unitree G1 navigation in MuJoCo

A student VLA maps **egocentric RGBD frames, onboard sensor readings, their histories, and a free-form English instruction** to **15 lower-body joint targets** (legs + waist) of a Unitree G1 at 50 Hz. The robot moves only through MuJoCo physics (position actuators tracking those targets). The student reuses frozen **GR00T N1.6** parameters (vision tower, projector, LLM) and trains a small head (~5.6M parameters) on top.

## Layout

```
README.md                  this file
report.pdf                 approach, results, failure modes
checkpoint/
  student.pt               final student VLA (trained parameters only)
  walker.npz               RL walker used by the expert (data labelling only)
  intermediate/<stage>/    earlier students (needed to regenerate DAgger data)
  eval_results.json        per-episode evaluation results
  GROOT_WEIGHTS.md         how the frozen GR00T tensors are fetched
dataset/
  <split>/manifest.json    seeds, counts, walker hash for every data split
  REGENERATE.md            exact commands to regenerate the data
code/
  g1nav/                   library (arena, sim, tasks/expert, GR00T encoders, student, runtime)
  scripts/                 one script per pipeline stage (below)
  tests/                   expert sanity test, end-to-end plumbing test
  notebooks/G1Nav_Colab.ipynb   the full pipeline on a free Colab T4
videos/                    egocentric RGB + depth next to a third-person view, one mp4 per episode
```

## Requirements

| | Used for this submission | Minimum |
|---|---|---|
| GPU | NVIDIA T4 16 GB (Google Colab free tier) | any CUDA GPU with ≥ 12 GB for walker PPO and feature extraction; ≥ 6 GB for evaluation only |
| System RAM | 12.7 GB (Colab) | 12 GB |
| Disk | ~9 GB persistent (episodes ~3.5 GB + int8 vision features ~5.5 GB) + 3.1 GB GR00T subset (local, re-downloadable) | same |
| Software | Python 3, `code/requirements.txt`; walker training in a separate env with `code/requirements-walker.txt` | |

Evaluation alone (given `checkpoint/`) needs only `requirements.txt` and the GR00T download (3.1 GB).

## Reproduce

Set a data directory, then run from `code/` (`$G1NAV_DATA` defaults to `../runs`):

```bash
export G1NAV_DATA=/path/to/runs MUJOCO_GL=egl
pip install -r requirements.txt
python scripts/setup_assets.py                       # G1 model: pinned Playground + Menagerie commits
python tests/test_expert_kinematic.py                # expert sanity check

# 1. Walker (PPO, MuJoCo Playground / MJX). Separate env to isolate JAX's CUDA libs.
uv venv jaxenv && uv pip install --python jaxenv/bin/python -r requirements-walker.txt
jaxenv/bin/python scripts/train_walker.py --out $G1NAV_DATA/walker --num_timesteps 200000000
python scripts/check_walker.py --walker $G1NAV_DATA/walker/walker.npz

# 2. Expert demonstrations (deterministic: episode i uses seed base_seed + i)
python scripts/generate_data.py --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/episodes/expert --n 600 --base_seed 0

# 3. Frozen GR00T features (downloads only the reused tensors)
python scripts/extract_features.py --dirs $G1NAV_DATA/episodes/expert

# 4. Behaviour cloning, then two DAgger rounds
python scripts/train_vla.py --data $G1NAV_DATA/episodes/expert --out $G1NAV_DATA/vla/bc --steps 30000
python scripts/dagger.py --student $G1NAV_DATA/vla/bc/student.pt --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/episodes/dagger1 --n 200 --base_seed 1000000
python scripts/train_vla.py --data $G1NAV_DATA/episodes/expert $G1NAV_DATA/episodes/dagger1 --init $G1NAV_DATA/vla/bc/student.pt --out $G1NAV_DATA/vla/dagger1 --steps 15000
python scripts/dagger.py --student $G1NAV_DATA/vla/dagger1/student.pt --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/episodes/dagger2 --n 200 --base_seed 2000000
python scripts/train_vla.py --data $G1NAV_DATA/episodes/expert $G1NAV_DATA/episodes/dagger1 $G1NAV_DATA/episodes/dagger2 --init $G1NAV_DATA/vla/dagger1/student.pt --out $G1NAV_DATA/vla/dagger2 --steps 15000

# 5. Closed-loop evaluation, videos, real-time measurements
python scripts/evaluate.py --student $G1NAV_DATA/vla/dagger2/student.pt --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/eval --n 20 --videos 3
python scripts/evaluate.py --expert --walker $G1NAV_DATA/walker/walker.npz --out $G1NAV_DATA/eval_expert --n 20 --videos 0
```

To evaluate the shipped checkpoint directly: `python scripts/evaluate.py --student ../checkpoint/student.pt --walker ../checkpoint/walker.npz --out eval`. The walker is still loaded there because the expert labels each step for logging. The student never calls it.

The same sequence is in `code/notebooks/G1Nav_Colab.ipynb`, with every stage resumable across Colab disconnects.

## How it fits together

- **Arena** (`g1nav/arena.py`): 9 × 9 m walled room, 3–6 static objects (6 colours × {ball, cube, cylinder, cone}), random floor, wall and light colours. Robot = Playground's G1 feet-only-collision model. Added contact pairs make the legs and hands collide with objects and walls.
- **Sensors** (`g1nav/sim.py`): 224 × 224 RGBD head camera at 10 Hz; pelvis IMU (gyro, gravity), velocimeter, 29 joint encoders, previous joint targets, gait clock, and dead-reckoning odometry integrated from gyro + velocimeter. Noise follows the walker's training noise.
- **Expert** (`g1nav/tasks.py`, `g1nav/walker.py`): a privileged A* navigator turns the instruction's world-frame plan into velocity commands. A PPO walker trained from scratch in MuJoCo Playground turns those into joint targets. Used only to produce labels.
- **Student** (`g1nav/model.py`, `g1nav/runtime.py`): frozen GR00T vision tower + `mlp1` projector (16 tokens per frame), frozen GR00T 16-layer LLM (instruction, once per episode), a depth CNN, a 4-layer transformer at 10 Hz, and an MLP at 50 Hz that outputs the joint targets.
- **Instructions**: 5 families (go to / follow, pass-and-turn, two-object sequence, turn, walk forward N m) with many paraphrases. Evaluation also uses held-out phrasings.

## Third-party assets

G1 model XMLs from `google-deepmind/mujoco_playground` and meshes from `google-deepmind/mujoco_menagerie` (Apache-2.0), fetched at pinned commits by `setup_assets.py`. GR00T N1.6 parameters from `nvidia/GR00T-N1.6-3B` (NVIDIA license terms on the model card). Tokenizer files (no parameters) from `Qwen/Qwen3-1.7B`, whose vocabulary GR00T's LLM uses.
