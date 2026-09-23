"""Unitree G1 constants shared by the walker, the simulator, and the student.

Joint order follows the 29 position actuators in g1_mjx_feetonly.xml (which is
also the qpos[7:] order): left leg (6), right leg (6), waist (3), left arm (7),
right arm (7). "Lower body" here means legs + waist = the first 15 joints;
the arms are held at their default pose by fixed PD targets.
"""

import numpy as np

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
NUM_JOINTS = 29
LOWER_BODY = np.arange(15)
NUM_LOWER = len(LOWER_BODY)

# "knees_bent" keyframe of the Playground scene; the walker's nominal pose.
DEFAULT_POSE = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0.073,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0,
])
INIT_BASE_HEIGHT = 0.755

CTRL_DT = 0.02   # 50 Hz policy
SIM_DT = 0.002   # 500 Hz physics
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT))
ACTION_SCALE = 0.5  # joint target = DEFAULT_POSE + ACTION_SCALE * action
GAIT_FREQ = 1.4     # Hz, within the walker's training range U(1.25, 1.5)

# Sensor noise used during walker training (Playground defaults, level 1.0).
NOISE_SCALES = dict(joint_pos=0.03, joint_vel=1.5, gravity=0.05, linvel=0.1, gyro=0.2)

WALKER_OBS_DIM = 3 + 3 + 3 + 3 + 29 + 29 + 29 + 4  # = 103


def action_to_targets(action_lower: np.ndarray) -> np.ndarray:
  """Normalized lower-body action in [-1, 1] -> 15 absolute joint targets (rad)."""
  return DEFAULT_POSE[LOWER_BODY] + ACTION_SCALE * action_lower


def targets_to_action(targets_lower: np.ndarray) -> np.ndarray:
  return (targets_lower - DEFAULT_POSE[LOWER_BODY]) / ACTION_SCALE


def walker_obs(linvel, gyro, gravity, command, qpos_j, qvel_j, last_act29, phase):
  """Observation layout of Playground's G1 Joystick 'state' key."""
  return np.concatenate([
      linvel, gyro, gravity, command,
      qpos_j - DEFAULT_POSE, qvel_j, last_act29,
      np.cos(phase), np.sin(phase),
  ]).astype(np.float32)
