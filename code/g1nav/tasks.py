"""Instruction families, scene sampling, the privileged expert navigator, and success checks.

Every task is compiled into world-frame *segments* (go to object, go to point,
turn to yaw). The expert executes them closed-loop from the true robot pose,
so it can also label states visited by the student (DAgger).

Families
  goto       "follow the red ball", "walk to the blue cylinder", ...
  pass_turn  "go straight and turn right after passing the yellow cube"
  sequence   "go to the red ball, then go to the green cone"
  turn       "turn left", "turn around"
  forward    "walk forward two meters"
Each family has training templates and held-out templates (split="novel")
used only for evaluation of instruction generalization.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from g1nav import arena
from g1nav.arena import ARENA_HALF, COLORS, SHAPES, Obj, Scene

FAMILIES = ["goto", "pass_turn", "sequence", "turn", "forward"]
FAMILY_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.15]
TIME_LIMIT = {"goto": 35.0, "pass_turn": 30.0, "sequence": 50.0, "turn": 12.0, "forward": 20.0}

SHAPE_WORDS = {
    "ball": ["ball", "sphere"],
    "cube": ["cube", "box", "block"],
    "cylinder": ["cylinder", "pillar"],
    "cone": ["cone"],
}

TEMPLATES = {
    "goto": {
        "train": ["go to {o}", "walk to {o}", "head over to {o}", "approach {o}",
                  "follow {o}", "move towards {o}", "walk up to {o}", "navigate to {o}",
                  "please go to {o}", "go stand next to {o}", "find {o} and walk to it",
                  "can you walk over to {o}"],
        "novel": ["make your way to {o}", "get close to {o}", "go find {o}"],
    },
    "pass_turn": {
        "train": ["go straight and turn {d} after passing {o}",
                  "walk straight ahead and turn {d} once you pass {o}",
                  "keep going straight until you pass {o}, then turn {d}",
                  "go past {o} and then turn {d}",
                  "move forward past {o} and take a {d}"],
        "novel": ["head straight, and after you have passed {o}, go {d}",
                  "continue forward beyond {o}, then make a {d} turn"],
    },
    "sequence": {
        "train": ["go to {o1}, then go to {o2}", "first visit {o1} and then {o2}",
                  "walk to {o1} and after that to {o2}", "go to {o1} first, then {o2}"],
        "novel": ["stop by {o1} on your way to {o2}", "visit {o1}, followed by {o2}"],
    },
    "turn": {
        "train": ["turn {d}", "turn to your {d}", "rotate {d}", "face {d}",
                  "turn 90 degrees {d}", "make a {d} turn"],
        "novel": ["rotate ninety degrees to the {d}", "pivot {d}"],
    },
    "turn_around": {
        "train": ["turn around", "do a u-turn", "face the other way", "turn 180 degrees"],
        "novel": ["look behind you", "spin around to face backwards"],
    },
    "forward": {
        "train": ["walk forward {n}", "go straight for {n}", "move ahead {n}",
                  "walk {n} straight ahead", "go forward about {n}"],
        "novel": ["advance {n}", "proceed straight ahead for {n}"],
    },
}
NUM_WORDS = {1: ["one meter", "1 meter"], 2: ["two meters", "2 meters"], 3: ["three meters", "3 meters"]}

# Navigation constants (m, rad, s).
STOP_DIST = 0.55        # stop this far from an object's footprint
CLEARANCE = 0.45        # robot half-width + margin, added to obstacle radius
FOV_HALF = math.radians(26.0)
VMAX = 0.6
WZ_MAX = 0.8
HOLD_AFTER_DONE = 3.0   # expert keeps "standing" (stepping in place) at the end
REACH_MARGIN = 0.85     # success radius beyond the object footprint


def wrap(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


def rot(v, ang):
  c, s = math.cos(ang), math.sin(ang)
  return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclasses.dataclass
class Task:
  family: str
  instruction: str
  scene: Scene
  segments: list[dict]
  goal: dict                 # data for the success check
  template_split: str = "train"

  @property
  def time_limit(self) -> float:
    return TIME_LIMIT[self.family]

  def to_dict(self) -> dict:
    return dict(family=self.family, instruction=self.instruction, scene=self.scene.to_dict(),
                segments=self.segments, goal=self.goal, template_split=self.template_split)

  @staticmethod
  def from_dict(d: dict) -> "Task":
    d = dict(d)
    d["scene"] = Scene.from_dict(d["scene"])
    return Task(**d)


# ----------------------------------------------------------------------------- scenes

def _sample_objects(rng, n, avoid_xy, min_sep=1.0, half=ARENA_HALF - 1.0):
  combos = [(c, s) for c in COLORS for s in SHAPES]
  picks = rng.choice(len(combos), size=n, replace=False)
  objs = []
  for idx in picks:
    for _ in range(200):
      xy = rng.uniform(-half, half, 2)
      if all(np.linalg.norm(xy - np.asarray(p)) > min_sep for p in avoid_xy) and \
         all(np.linalg.norm(xy - o.xy) > min_sep for o in objs):
        c, s = combos[idx]
        objs.append(Obj(c, s, float(xy[0]), float(xy[1]), float(rng.uniform(0, np.pi))))
        break
  return objs


def _visuals(rng):
  g = rng.uniform(0.35, 0.7)
  tint = rng.uniform(-0.05, 0.05, 3)
  return dict(floor_rgb=tuple(float(v) for v in np.clip(g + tint, 0, 1)),
              wall_rgb=tuple(float(v) for v in rng.uniform(0.6, 0.9, 3)),
              light=float(rng.uniform(0.6, 1.0)))


def _refer(obj: Obj, objects: list[Obj], rng) -> str:
  opts = [f"{obj.color} {w}" for w in SHAPE_WORDS[obj.shape]]
  if sum(o.color == obj.color for o in objects) == 1:
    opts += [f"{obj.color} object", f"{obj.color} one"]
  if sum(o.shape == obj.shape for o in objects) == 1:
    opts += SHAPE_WORDS[obj.shape]
  return "the " + str(rng.choice(opts))


def _seg_dist(p, a, b):
  ab = b - a
  t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
  return float(np.linalg.norm(p - (a + t * ab)))


def _path_clear(a, b, objects, skip=(), margin=CLEARANCE + 0.15):
  return all(_seg_dist(o.xy, a, b) > o.radius + margin
             for i, o in enumerate(objects) if i not in skip)


def _inside(p, margin=0.8):
  return bool(np.all(np.abs(p) <= ARENA_HALF - margin))


def sample_task(rng: np.random.Generator, split: str = "train", family: str | None = None) -> Task:
  """Rejection-sample a scene + instruction whose expert path is feasible."""
  family = family or str(rng.choice(FAMILIES, p=FAMILY_WEIGHTS))
  for _ in range(1000):
    task = _try_sample(rng, family, split)
    if task is not None:
      return task
  raise RuntimeError(f"could not sample a {family} task")


def _try_sample(rng, family, split) -> Task | None:
  start = rng.uniform(-(ARENA_HALF - 1.5), ARENA_HALF - 1.5, 2)
  yaw0 = float(rng.uniform(-np.pi, np.pi))
  h = np.array([math.cos(yaw0), math.sin(yaw0)])
  objects = _sample_objects(rng, int(rng.integers(3, 7)), [start])
  if len(objects) < 3:
    return None
  tmpl = lambda fam: str(rng.choice(TEMPLATES[fam][split]))  # noqa: E731
  segments, goal = [], {}

  if family == "goto":
    i = int(rng.integers(len(objects)))
    text = tmpl("goto").format(o=_refer(objects[i], objects, rng))
    segments = [dict(type="goto_obj", obj=i)]
    goal = dict(obj=i)

  elif family == "sequence":
    i, j = (int(v) for v in rng.choice(len(objects), 2, replace=False))
    text = tmpl("sequence").format(o1=_refer(objects[i], objects, rng),
                                   o2=_refer(objects[j], objects, rng))
    segments = [dict(type="goto_obj", obj=i), dict(type="goto_obj", obj=j)]
    goal = dict(obj1=i, obj2=j)

  elif family == "pass_turn":
    along = rng.uniform(1.4, 2.6)
    lateral = rng.choice([-1, 1]) * rng.uniform(0.75, 1.1)
    side = np.array([-h[1], h[0]])
    obj_xy = start + along * h + lateral * side
    combos = [(c, s) for c in COLORS for s in SHAPES
              if not any(o.color == c and o.shape == s for o in objects)]
    c, s = combos[int(rng.integers(len(combos)))]
    target = Obj(c, s, float(obj_xy[0]), float(obj_xy[1]), float(rng.uniform(0, np.pi)))
    objects = [o for o in objects if np.linalg.norm(o.xy - obj_xy) > 1.0]
    objects.insert(int(rng.integers(len(objects) + 1)), target)
    i = objects.index(target)
    turn_left = bool(rng.random() < 0.5)
    pass_pt = start + (along + 0.8) * h
    goal_pt = pass_pt + 1.5 * rot(h, np.pi / 2 if turn_left else -np.pi / 2)
    if not (_inside(target.xy, 0.6) and _inside(pass_pt) and _inside(goal_pt)):
      return None
    if not (_path_clear(start, pass_pt, objects, skip=(i,)) and
            _path_clear(pass_pt, goal_pt, objects)):
      return None
    d = "left" if turn_left else "right"
    text = tmpl("pass_turn").format(o=_refer(target, objects, rng), d=d)
    turn_yaw = wrap(yaw0 + (np.pi / 2 if turn_left else -np.pi / 2))
    segments = [dict(type="goto_point", xy=pass_pt.tolist()),
                dict(type="turn_to", yaw=float(turn_yaw)),
                dict(type="goto_point", xy=goal_pt.tolist())]
    goal = dict(point=goal_pt.tolist(), obj=i)

  elif family == "turn":
    kind = rng.choice(["left", "right", "around"], p=[0.4, 0.4, 0.2])
    if kind == "around":
      text, angle = tmpl("turn_around"), np.pi
    else:
      text = tmpl("turn").format(d=kind)
      angle = np.pi / 2 if kind == "left" else -np.pi / 2
    yaw_goal = float(wrap(yaw0 + angle))
    # "turn around" is always executed counter-clockwise; plain turn_to would
    # pick a side at random because +-pi are equally short.
    segments = ([dict(type="turn_to", yaw=float(wrap(yaw0 + np.pi / 2)))] if kind == "around" else [])
    segments.append(dict(type="turn_to", yaw=yaw_goal))
    goal = dict(yaw=yaw_goal, xy=start.tolist())

  elif family == "forward":
    n = int(rng.integers(1, 4))
    goal_pt = start + n * h
    if not (_inside(goal_pt) and _path_clear(start, goal_pt, objects)):
      return None
    text = tmpl("forward").format(n=str(rng.choice(NUM_WORDS[n])))
    segments = [dict(type="goto_point", xy=goal_pt.tolist(), keep_heading=True)]
    goal = dict(point=goal_pt.tolist())
  else:
    raise ValueError(family)

  scene = Scene(objects=objects, robot_xy=(float(start[0]), float(start[1])), robot_yaw=yaw0,
                **_visuals(rng))
  return Task(family, text, scene, segments, goal, split)


def demo_task(kind: str, rng: np.random.Generator) -> Task:
  """The two example instructions from the task statement, in random scenes."""
  family, color, shape, text = {
      "follow_red_ball": ("goto", "red", "ball", "follow the red ball"),
      "pass_yellow_cube_turn_right": ("pass_turn", "yellow", "cube",
                                      "go straight and turn right after passing the yellow cube"),
  }[kind]
  while True:
    task = _try_sample(rng, family, "train")
    if task is None or (family == "pass_turn" and "right" not in task.instruction):
      continue
    objs = task.scene.objects
    i = task.goal["obj"]
    if any(o.color == color and o.shape == shape for j, o in enumerate(objs) if j != i):
      continue
    objs[i] = dataclasses.replace(objs[i], color=color, shape=shape)
    task.instruction = text
    return task


# ----------------------------------------------------------------------------- expert

class GridPlanner:
  """8-connected A* on a 10 cm grid. Cells inside an obstacle's footprint plus
  0.3 m are blocked; cells within footprint + CLEARANCE + 0.2 m are penalized,
  so paths keep their distance but can still thread moderate gaps."""

  RES = 0.1

  def __init__(self, objects: list[Obj]):
    self.objects = objects
    self.n = int(round(2 * ARENA_HALF / self.RES))
    c = (np.arange(self.n) + 0.5) * self.RES - ARENA_HALF
    self.xs, self.ys = np.meshgrid(c, c, indexing="ij")
    self._cache = {}

  def _costmap(self, skip):
    key = tuple(sorted(skip))
    if key not in self._cache:
      cost = np.ones((self.n, self.n))
      wall = ARENA_HALF - np.maximum(np.abs(self.xs), np.abs(self.ys))
      cost[wall < 0.5] = np.inf
      for i, o in enumerate(self.objects):
        if i in skip:
          continue
        d = np.hypot(self.xs - o.x, self.ys - o.y) - o.radius
        cost += 8.0 * np.clip((CLEARANCE + 0.2 - d) / (CLEARANCE + 0.2), 0, 1)
        cost[d < 0.3] = np.inf
      self._cache[key] = cost
    return self._cache[key]

  def _cell(self, p):
    return tuple(np.clip(((np.asarray(p) + ARENA_HALF) / self.RES).astype(int), 0, self.n - 1))

  def plan(self, start, goal, skip=()) -> list[np.ndarray]:
    import heapq
    cost = self._costmap(skip)
    s, g = self._cell(start), self._cell(goal)
    if not np.isfinite(cost[g]):
      return [np.asarray(goal, dtype=float)]
    moves = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    dist = {s: 0.0}
    parent = {s: None}
    heap = [(0.0, s)]
    while heap:
      _, u = heapq.heappop(heap)
      if u == g:
        break
      du = dist[u]
      for dx, dy, step in moves:
        v = (u[0] + dx, u[1] + dy)
        if not (0 <= v[0] < self.n and 0 <= v[1] < self.n):
          continue
        cv = cost[v]
        if not np.isfinite(cv) and np.isfinite(cost[u]):
          continue  # never enter blocked cells; leaving them is allowed
        nd = du + step * min(cv, 20.0)
        if nd < dist.get(v, 1e18):
          dist[v] = nd
          parent[v] = u
          heapq.heappush(heap, (nd + math.hypot(g[0] - v[0], g[1] - v[1]), v))
    if g not in parent:
      return [np.asarray(goal, dtype=float)]
    cells, u = [], g
    while u is not None:
      cells.append(u)
      u = parent[u]
    pts = [np.array([self.xs[c], self.ys[c]]) for c in reversed(cells)]
    pts[-1] = np.asarray(goal, dtype=float)
    return pts


class ExpertNavigator:
  """Privileged closed-loop navigator: robot pose -> (vx, vy, wz) command."""

  def __init__(self, task: Task, dt: float):
    self.task = task
    self.objects = task.scene.objects
    self.dt = dt
    self.seg = 0
    self.seen = False
    self.cmd = np.zeros(3)
    self.done_time = None
    self._planner = GridPlanner(self.objects)
    self._plan, self._plan_key, self._plan_age, self._plan_goal = None, None, 0, np.zeros(2)

  @property
  def done(self) -> bool:
    return self.seg >= len(self.task.segments)

  def command(self, x: float, y: float, yaw: float, t: float) -> np.ndarray:
    p = np.array([x, y])
    target = np.zeros(3)
    while not self.done:
      seg = self.task.segments[self.seg]
      target, finished = self._segment(seg, p, yaw)
      if not finished:
        break
      self.seg += 1
      self.seen = False
      target = np.zeros(3)
    if self.done and self.done_time is None:
      self.done_time = t
    # Acceleration limits keep the walker in its comfortable regime.
    lim = np.array([1.5, 1.0, 3.0]) * self.dt
    self.cmd = self.cmd + np.clip(target - self.cmd, -lim, lim)
    return self.cmd.copy()

  def _segment(self, seg, p, yaw):
    kind = seg["type"]
    if kind == "turn_to":
      err = wrap(seg["yaw"] - yaw)
      if abs(err) < 0.08:
        return np.zeros(3), True
      return np.array([0.0, 0.0, np.clip(2.0 * err, -WZ_MAX, WZ_MAX)]), False

    if kind == "goto_point":
      goal = np.asarray(seg["xy"])
      dist = np.linalg.norm(goal - p)
      if dist < 0.2:
        return np.zeros(3), True
      if seg.get("keep_heading"):
        return self._track(p, yaw, goal, dist, skip=()), False
      return self._track(p, yaw, self._waypoint(p, goal, skip=()), dist, skip=()), False

    if kind == "goto_obj":
      obj = self.objects[seg["obj"]]
      to_obj = obj.xy - p
      dist = np.linalg.norm(to_obj)
      bearing = wrap(math.atan2(to_obj[1], to_obj[0]) - yaw)
      if abs(bearing) < FOV_HALF:
        self.seen = True
      if dist <= obj.radius + STOP_DIST:
        if abs(bearing) > 0.2:  # finish by facing the object
          return np.array([0.0, 0.0, np.clip(2.0 * bearing, -WZ_MAX, WZ_MAX)]), False
        return np.zeros(3), True
      if not self.seen:  # search: always rotate counter-clockwise until in view
        return np.array([0.0, 0.0, 0.7]), False
      stop_pt = self._approach_point(obj, seg["obj"], -to_obj / dist)
      wp = self._waypoint(p, stop_pt, skip=(seg["obj"],))
      return self._track(p, yaw, wp, dist - obj.radius - STOP_DIST, skip=(seg["obj"],)), False
    raise ValueError(kind)

  def _approach_point(self, obj, idx, direction):
    """Stop point next to `obj`, on the robot's side if free, else rotated around it."""
    cost = self._planner._costmap((idx,))
    base = math.atan2(direction[1], direction[0])
    for k in range(19):
      ang = base + (1 if k % 2 else -1) * ((k + 1) // 2) * math.radians(20)
      q = obj.xy + (obj.radius + STOP_DIST - 0.05) * np.array([math.cos(ang), math.sin(ang)])
      if cost[self._planner._cell(q)] < 3.0:
        return q
    return obj.xy + (obj.radius + STOP_DIST - 0.05) * direction

  def _waypoint(self, p, goal, skip):
    """Lookahead point on an A* path around the inflated obstacles (replanned at 2 Hz)."""
    key = (self.seg, tuple(skip))
    if (self._plan_key != key or self._plan_age >= 25 or
        np.linalg.norm(self._plan_goal - goal) > 0.3):
      self._plan = self._planner.plan(p, goal, skip)
      self._plan_key, self._plan_age, self._plan_goal = key, 0, goal.copy()
    self._plan_age += 1
    path = self._plan
    # Drop points already passed, then look 0.5 m ahead.
    while len(path) > 1 and np.linalg.norm(path[0] - p) < 0.3:
      path = path[1:]
    self._plan = path
    for q in path:
      if np.linalg.norm(q - p) >= 0.5:
        return q
    return path[-1]

  def _track(self, p, yaw, wp, remaining, skip):
    d = wp - p
    err = wrap(math.atan2(d[1], d[0]) - yaw)
    wz = np.clip(1.8 * err, -WZ_MAX, WZ_MAX)
    vx = VMAX * max(0.0, math.cos(err)) ** 2 if abs(err) < 1.0 else 0.0
    vx *= np.clip(remaining / 0.6, 0.35, 1.0)
    return np.array([vx, 0.0, wz])


# ----------------------------------------------------------------------------- success

def evaluate(task: Task, traj: np.ndarray, fell: bool, collided: bool) -> dict:
  """traj: (T, 3) array of privileged (x, y, yaw). Judged on the final pose."""
  if fell:
    return dict(success=False, reason="fell")
  if collided:
    return dict(success=False, reason="collision")
  x, y, yaw = traj[-1]
  p = np.array([x, y])
  g, objs = task.goal, task.scene.objects

  def near(i, pts):
    return np.linalg.norm(pts - objs[i].xy, axis=-1) <= objs[i].radius + REACH_MARGIN

  if task.family == "goto":
    ok = bool(near(g["obj"], p))
  elif task.family == "sequence":
    ok = bool(near(g["obj1"], traj[:, :2]).any() and near(g["obj2"], p))
  elif task.family in ("pass_turn", "forward"):
    ok = bool(np.linalg.norm(p - np.asarray(g["point"])) <= (0.7 if task.family == "pass_turn" else 0.5))
  elif task.family == "turn":
    ok = bool(abs(wrap(yaw - g["yaw"])) <= math.radians(30) and
              np.linalg.norm(p - np.asarray(g["xy"])) <= 0.6)
  else:
    raise ValueError(task.family)
  return dict(success=ok, reason="ok" if ok else "wrong_final_pose")
