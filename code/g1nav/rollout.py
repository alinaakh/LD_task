"""Run one episode with the expert or the student, recording everything needed
for training (expert labels at every step) and for scoring.

Timing: control at 50 Hz; the egocentric camera is sampled every
VISION_EVERY control steps (10 Hz), which is also the student's vision rate.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
from PIL import Image

from g1nav import robot, tasks
from g1nav.sim import G1NavSim
from g1nav.tasks import Task
from g1nav.video import VideoWriter, compose

VISION_EVERY = 5      # 50 Hz / 5 = 10 Hz
DEPTH_RES = 64
VIDEO_EVERY = 2       # 25 fps videos
TAKEOVER_TILT = 0.9   # expert takes over (DAgger) when gravity_z > -0.9 (~25 deg tilt)
TAKEOVER_STEPS = 50


def downsample_depth(depth: np.ndarray) -> np.ndarray:
  img = Image.fromarray(depth.astype(np.float32), mode="F")
  return np.asarray(img.resize((DEPTH_RES, DEPTH_RES), Image.BOX), dtype=np.float16)


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
  buf = io.BytesIO()
  Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
  return buf.getvalue()


def decode_jpeg(b: bytes) -> np.ndarray:
  return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def run_episode(task: Task, walker, seed: int, student=None, *, dagger: bool = False,
                video_path=None, time_limit: float | None = None, noise: float = 1.0,
                stop_when_done: bool | None = None, progress_every: int = 0) -> dict:
  """Returns a dict of arrays + metadata (see save_episode for the on-disk format).

  student=None -> the expert (navigator + walker) drives.
  student given -> the student drives; with dagger=True the expert takes over
  briefly whenever the robot tilts dangerously (recovery demonstrations).
  Expert labels (`action`, `command`) are recorded in all cases.
  """
  sim = G1NavSim(task.scene, seed=seed, noise=noise)
  nav = tasks.ExpertNavigator(task, robot.CTRL_DT)
  limit = time_limit or task.time_limit
  n_steps = int(round(limit / robot.CTRL_DT))
  if stop_when_done is None:
    stop_when_done = student is None
  if student is not None:
    student.reset(task.instruction)
  video = VideoWriter(video_path) if video_path else None

  rec = {k: [] for k in ("proprio", "action", "action_exec", "command", "pose", "expert_drove")}
  vis_t, depth, jpegs = [], [], []
  latency = []
  takeover = 0
  fell = False
  t_start = time.perf_counter()
  for t in range(n_steps):
    if progress_every and t and t % progress_every == 0:
      print(f"      seed={seed} step {t}/{n_steps} ({t * robot.CTRL_DT:.0f}s sim, "
            f"{time.perf_counter() - t_start:.0f}s wall)", flush=True)
    pose = sim.pose()
    cmd = nav.command(*pose, t * robot.CTRL_DT)
    a_exp = walker(sim.walker_obs(cmd))
    proprio = sim.proprio()

    frame = None
    if t % VISION_EVERY == 0:
      rgb, dep = sim.render_ego()
      frame = (rgb, dep)
      vis_t.append(t)
      depth.append(downsample_depth(dep))
      jpegs.append(encode_jpeg(rgb))

    if student is None:
      a, expert_drove = a_exp, True
    else:
      t0 = time.perf_counter()
      a_student = student.act(proprio, frame)
      latency.append(time.perf_counter() - t0)
      if dagger and sim.sensors()["gravity"][2] > -TAKEOVER_TILT:
        takeover = TAKEOVER_STEPS
      expert_drove = takeover > 0
      a = a_exp if expert_drove else a_student
      takeover = max(0, takeover - 1)

    rec["proprio"].append(proprio)
    rec["action"].append(a_exp)
    rec["action_exec"].append(np.asarray(a, dtype=np.float32))
    rec["command"].append(cmd.astype(np.float32))
    rec["pose"].append(pose.astype(np.float32))
    rec["expert_drove"].append(expert_drove)

    if video and t % VIDEO_EVERY == 0:
      ego_rgb, ego_dep = frame if frame else sim.render_ego()
      status = (f"t={t * robot.CTRL_DT:5.1f}s  cmd=({cmd[0]:+.2f}, {cmd[2]:+.2f})  "
                f"{'expert' if student is None else 'VLA policy'}")
      video.add(compose(ego_rgb, ego_dep, sim.render_third(), task.instruction, status))

    sim.step(a)
    if sim.fallen():
      fell = True
      break
    if stop_when_done and nav.done and (t + 1) * robot.CTRL_DT - nav.done_time >= tasks.HOLD_AFTER_DONE:
      break

  if video:
    video.close()
  sim.close()
  out = {k: np.asarray(v, dtype=np.float32 if k != "expert_drove" else bool) for k, v in rec.items()}
  out["vis_t"] = np.asarray(vis_t, dtype=np.int32)
  out["depth"] = np.stack(depth)
  out["jpegs"] = jpegs
  result = tasks.evaluate(task, out["pose"], fell, sim.collided)
  out["meta"] = dict(task=task.to_dict(), seed=seed, result=result, fell=fell,
                     collided=bool(sim.collided), steps=len(out["action"]),
                     expert_done_time=nav.done_time, driver="expert" if student is None else "student",
                     dagger=dagger)
  if latency:
    out["meta"]["latency_ms"] = dict(mean=1e3 * float(np.mean(latency)),
                                     p99=1e3 * float(np.percentile(latency, 99)),
                                     max=1e3 * float(np.max(latency)))
  return out


def save_episode(path, ep: dict):
  offsets = np.cumsum([0] + [len(b) for b in ep["jpegs"]]).astype(np.int64)
  blob = np.frombuffer(b"".join(ep["jpegs"]), dtype=np.uint8)
  arrays = {k: v for k, v in ep.items() if k not in ("jpegs", "meta")}
  np.savez(path, **arrays, jpeg_blob=blob, jpeg_offsets=offsets,
           meta=np.array(json.dumps(ep["meta"])))


def load_episode(path, with_images: bool = False) -> dict:
  z = np.load(path)
  ep = {k: z[k] for k in z.files if k not in ("jpeg_blob", "jpeg_offsets", "meta")}
  ep["meta"] = json.loads(str(z["meta"]))
  if with_images:
    blob, off = z["jpeg_blob"], z["jpeg_offsets"]
    ep["rgb"] = [decode_jpeg(blob[off[i]:off[i + 1]].tobytes()) for i in range(len(off) - 1)]
  return ep
