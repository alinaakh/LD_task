"""MuJoCo simulation of one G1Nav episode: 500 Hz physics, 50 Hz control.

The robot moves only through physics: each control step writes 15 lower-body
PD targets (arms held at default) and integrates 10 physics substeps.

Onboard "sensors" = what the student may use: pelvis IMU (gyro, gravity
direction), pelvis velocimeter, joint encoders, the previous joint target, a
gait clock, and dead-reckoning odometry integrated from gyro + velocimeter.
The privileged pose (`pose()`) is used only by the expert and for scoring.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from g1nav import arena, robot
from g1nav.arena import Scene

PROPRIO_DIM = 3 + 3 + 3 + 29 + 29 + 15 + 4 + 4  # = 90


class G1NavSim:

  def __init__(self, scene: Scene, seed: int = 0, noise: float = 1.0):
    self.scene = scene
    self.model = arena.build_model(scene)
    self.data = mujoco.MjData(self.model)
    self.noise = noise
    self.rng = np.random.default_rng(seed)
    m = self.model
    self._imu_site = m.site("imu_in_pelvis").id
    self._sens = {n: (m.sensor_adr[m.sensor(n).id], m.sensor_dim[m.sensor(n).id])
                  for n in ["gyro_pelvis", "local_linvel_pelvis", "upvector_torso"]}
    self._env_geoms = {m.geom(f"obj{i}").id for i in range(len(scene.objects))}
    self._env_geoms |= {m.geom(f"wall{i}").id for i in range(4)}
    self._ego = None
    self._ego_depth = None
    self._third = None
    self.reset()

  # ------------------------------------------------------------------ state

  def reset(self):
    m, d = self.model, self.data
    mujoco.mj_resetDataKeyframe(m, d, m.key("knees_bent").id)
    x, y = self.scene.robot_xy
    yaw = self.scene.robot_yaw
    d.qpos[0:3] = [x, y, robot.INIT_BASE_HEIGHT]
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.ctrl[:] = robot.DEFAULT_POSE
    mujoco.mj_forward(m, d)
    self.t = 0
    self.phase = np.array([0.0, np.pi])
    self.phase_dt = 2 * np.pi * robot.CTRL_DT * robot.GAIT_FREQ
    self.last_act = np.zeros(robot.NUM_JOINTS)
    self.odom = np.zeros(3)  # x, y, yaw in the start frame
    self.collided = False
    self._sensors = None

  def _read(self, name):
    adr, dim = self._sens[name]
    return self.data.sensordata[adr:adr + dim].copy()

  def _noisy(self, x, key):
    s = robot.NOISE_SCALES[key] * self.noise
    return x + (2 * self.rng.random(x.shape) - 1) * s

  def sensors(self) -> dict:
    """Noisy onboard readings for the current control step (cached until step())."""
    if self._sensors is None:
      d = self.data
      gravity = d.site_xmat[self._imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
      self._sensors = dict(
          gyro=self._noisy(self._read("gyro_pelvis"), "gyro"),
          linvel=self._noisy(self._read("local_linvel_pelvis"), "linvel"),
          gravity=self._noisy(gravity, "gravity"),
          qpos=self._noisy(d.qpos[7:].copy(), "joint_pos"),
          qvel=self._noisy(d.qvel[6:].copy(), "joint_vel"),
          last_act=self.last_act.copy(),
          phase=self.phase.copy(),
          odom=self.odom.copy(),
      )
    return self._sensors

  def walker_obs(self, command: np.ndarray) -> np.ndarray:
    s = self.sensors()
    return robot.walker_obs(s["linvel"], s["gyro"], s["gravity"], command,
                            s["qpos"], s["qvel"], s["last_act"], s["phase"])

  def proprio(self) -> np.ndarray:
    """The student's per-step sensor vector (PROPRIO_DIM)."""
    s = self.sensors()
    o = s["odom"]
    return np.concatenate([
        s["gyro"], s["gravity"], s["linvel"],
        s["qpos"] - robot.DEFAULT_POSE, s["qvel"], s["last_act"][:robot.NUM_LOWER],
        np.cos(s["phase"]), np.sin(s["phase"]),
        [o[0], o[1], math.cos(o[2]), math.sin(o[2])],
    ]).astype(np.float32)

  # ------------------------------------------------------------------ stepping

  def step(self, action_lower: np.ndarray):
    """Apply 15 normalized lower-body actions for one 50 Hz control step."""
    a = np.clip(np.asarray(action_lower, dtype=np.float64), -1.0, 1.0)
    s = self.sensors()
    d = self.data
    d.ctrl[:robot.NUM_LOWER] = robot.action_to_targets(a)
    d.ctrl[robot.NUM_LOWER:] = robot.DEFAULT_POSE[robot.NUM_LOWER:]
    for _ in range(robot.N_SUBSTEPS):
      mujoco.mj_step(self.model, d)
    if not self.collided:
      self.collided = self._env_contact()

    dt = robot.CTRL_DT
    yaw = self.odom[2] + s["gyro"][2] * dt
    c, sn = math.cos(yaw), math.sin(yaw)
    vx, vy = s["linvel"][0], s["linvel"][1]
    self.odom = np.array([self.odom[0] + (c * vx - sn * vy) * dt,
                          self.odom[1] + (sn * vx + c * vy) * dt, yaw])
    self.last_act = np.zeros(robot.NUM_JOINTS)
    self.last_act[:robot.NUM_LOWER] = a
    self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
    self.t += 1
    self._sensors = None

  def _env_contact(self) -> bool:
    d = self.data
    for i in range(d.ncon):
      c = d.contact[i]
      if c.dist < 0 and (c.geom1 in self._env_geoms or c.geom2 in self._env_geoms):
        return True
    return False

  # ------------------------------------------------------------------ privileged

  def pose(self) -> np.ndarray:
    q = self.data.qpos
    w, x, y, z = q[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([q[0], q[1], yaw])

  def fallen(self) -> bool:
    return bool(self.data.qpos[2] < 0.45 or self._read("upvector_torso")[2] < 0.6)

  # ------------------------------------------------------------------ rendering

  def render_ego(self) -> tuple[np.ndarray, np.ndarray]:
    """Egocentric RGB (224x224x3 uint8) and depth (224x224 float32, metres, clipped)."""
    if self._ego is None:
      self._ego = mujoco.Renderer(self.model, arena.EGO_RES, arena.EGO_RES)
      self._ego_depth = mujoco.Renderer(self.model, arena.EGO_RES, arena.EGO_RES)
      self._ego_depth.enable_depth_rendering()
    self._ego.update_scene(self.data, camera=arena.EGO_CAM)
    rgb = self._ego.render()
    self._ego_depth.update_scene(self.data, camera=arena.EGO_CAM)
    depth = np.clip(self._ego_depth.render(), 0.0, arena.DEPTH_MAX).astype(np.float32)
    return rgb, depth

  def render_third(self, width: int = 640, height: int = 480) -> np.ndarray:
    if self._third is None or self._third.width != width:
      self._third = mujoco.Renderer(self.model, height, width)
    self._third.update_scene(self.data, camera=arena.THIRD_CAM)
    return self._third.render()

  def close(self):
    for r in (self._ego, self._ego_depth, self._third):
      if r is not None:
        r.close()
