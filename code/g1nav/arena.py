"""Arena construction: the Playground G1 plus static colored objects, walls, cameras.

Objects are static geoms (no joints). The G1 XML disables generic collisions
and lists contact pairs explicitly, so we add pairs between the robot's
collision geoms and every object and wall; walking into an object is physical,
and it is also reported as a failure.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np

from g1nav import paths

COLORS = {
    "red": (0.85, 0.08, 0.08),
    "yellow": (0.95, 0.82, 0.08),
    "blue": (0.10, 0.25, 0.90),
    "green": (0.10, 0.65, 0.20),
    "orange": (1.00, 0.45, 0.05),
    "purple": (0.55, 0.15, 0.75),
}

# Collision footprint radius (m) used by the planner and the success checks.
SHAPES = {
    "ball": dict(radius=0.18),
    "cube": dict(radius=0.24),
    "cylinder": dict(radius=0.15),
    "cone": dict(radius=0.20),
}

ARENA_HALF = 4.5            # walls at +-4.5 m
ROBOT_COLLISION_GEOMS = [
    "left_foot", "right_foot", "left_shin", "right_shin",
    "left_thigh", "right_thigh", "left_hand_collision", "right_hand_collision",
]

EGO_CAM = "head_rgbd"
THIRD_CAM = "overview"
EGO_RES = 224               # SigLIP input size; rendered directly at this size
EGO_FOVY = 60.0
EGO_PITCH_DEG = 30.0        # camera tilted down
DEPTH_MAX = 6.0


@dataclasses.dataclass
class Obj:
  color: str
  shape: str
  x: float
  y: float
  yaw: float = 0.0

  @property
  def radius(self) -> float:
    return SHAPES[self.shape]["radius"]

  @property
  def xy(self) -> np.ndarray:
    return np.array([self.x, self.y])

  @property
  def name(self) -> str:
    return f"{self.color} {self.shape}"


@dataclasses.dataclass
class Scene:
  objects: list[Obj]
  robot_xy: tuple[float, float]
  robot_yaw: float
  floor_rgb: tuple[float, float, float] = (0.55, 0.55, 0.55)
  wall_rgb: tuple[float, float, float] = (0.80, 0.78, 0.72)
  light: float = 0.8

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)

  @staticmethod
  def from_dict(d: dict) -> "Scene":
    d = dict(d)
    d["objects"] = [Obj(**o) for o in d["objects"]]
    return Scene(**d)


def _yaw_quat(yaw: float) -> list[float]:
  return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat).ravel())
  return q


def _ego_camera_quat(pitch_deg: float) -> np.ndarray:
  """Camera looking along body +x, tilted down by `pitch_deg` (MuJoCo cams look along -z)."""
  p = math.radians(pitch_deg)
  forward = np.array([math.cos(p), 0.0, -math.sin(p)])
  up = np.array([math.sin(p), 0.0, math.cos(p)])
  right = np.cross(forward, up)
  return _mat_to_quat(np.stack([right, up, -forward], axis=1))


def _add_cone_mesh(spec: mujoco.MjSpec, name: str, radius: float, height: float, n: int = 24):
  ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
  base = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], axis=1)
  verts = np.vstack([base, [[0.0, 0.0, height]]])
  mesh = spec.add_mesh()
  mesh.name = name
  mesh.uservert = verts.ravel().tolist()


def build_model(scene: Scene) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(paths.SCENE_XML))

  tex = spec.texture("groundplane")
  f = np.array(scene.floor_rgb)
  tex.rgb1 = f.tolist()
  tex.rgb2 = (f * 0.85).tolist()
  tex.markrgb = (f * 0.6).tolist()

  world = spec.worldbody
  light = world.add_light()
  light.name = "sun"
  light.pos = [0, 0, 6]
  light.dir = [0, 0, -1]
  light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  light.diffuse = [scene.light] * 3
  light.castshadow = False

  _add_cone_mesh(spec, "cone_mesh", SHAPES["cone"]["radius"], 0.5)

  env_geoms = []
  for i, o in enumerate(scene.objects):
    g = world.add_geom()
    g.name = f"obj{i}"
    g.rgba = [*COLORS[o.color], 1.0]
    g.contype = 0
    g.conaffinity = 0
    g.condim = 3
    g.quat = _yaw_quat(o.yaw)
    if o.shape == "ball":
      g.type = mujoco.mjtGeom.mjGEOM_SPHERE
      g.size = [0.18, 0, 0]
      g.pos = [o.x, o.y, 0.18]
    elif o.shape == "cube":
      g.type = mujoco.mjtGeom.mjGEOM_BOX
      g.size = [0.17, 0.17, 0.17]
      g.pos = [o.x, o.y, 0.17]
    elif o.shape == "cylinder":
      g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
      g.size = [0.15, 0.25, 0]
      g.pos = [o.x, o.y, 0.25]
    elif o.shape == "cone":
      g.type = mujoco.mjtGeom.mjGEOM_MESH
      g.meshname = "cone_mesh"
      g.pos = [o.x, o.y, 0.0]
    else:
      raise ValueError(o.shape)
    env_geoms.append(g.name)

  h = ARENA_HALF
  for i, (x, y, sx, sy) in enumerate([(h, 0, 0.05, h), (-h, 0, 0.05, h),
                                      (0, h, h, 0.05), (0, -h, h, 0.05)]):
    g = world.add_geom()
    g.name = f"wall{i}"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [sx, sy, 0.4]
    g.pos = [x, y, 0.4]
    g.rgba = [*scene.wall_rgb, 1.0]
    g.contype = 0
    g.conaffinity = 0
    env_geoms.append(g.name)

  for env_geom in env_geoms:
    for robot_geom in ROBOT_COLLISION_GEOMS:
      p = spec.add_pair()
      p.name = f"{robot_geom}__{env_geom}"
      p.geomname1 = robot_geom
      p.geomname2 = env_geom
      p.condim = 3

  # Egocentric RGBD camera in the G1 head (the real robot carries a D435 there).
  cam = spec.body("torso_link").add_camera()
  cam.name = EGO_CAM
  cam.pos = [0.06, 0.0, 0.43]
  cam.quat = _ego_camera_quat(EGO_PITCH_DEG).tolist()
  cam.fovy = EGO_FOVY

  # Third-person camera: fixed high in a corner, always aimed at the robot.
  cam = world.add_camera()
  cam.name = THIRD_CAM
  cam.pos = [-h + 0.3, -h + 0.3, 4.2]
  cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
  cam.targetbody = "pelvis"
  cam.fovy = 50.0

  model = spec.compile()
  model.vis.global_.offwidth = 1280
  model.vis.global_.offheight = 960
  return model
