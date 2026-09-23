# G1Nav report outline

Fill every `[[TODO]]` with your own measurements from `results.json`, `train_log.jsonl`, `progress.jsonl` and the manifests. Rewrite the rest in your own words and cut anything you can't defend in an interview. Export to `report.pdf`.

---

## 1. Problem and overview (½ page)

- Task: RGBD + sensors + histories + instruction → 15 lower-body joint targets for a Unitree G1 at 50 Hz in MuJoCo, in real time, physics only.
- Approach in one sentence: **privileged expert → demonstrations → student VLA on frozen GR00T features, refined with DAgger.**
- Figure suggestion: pipeline diagram (walker PPO → expert → data → features → BC → DAgger → eval).

## 2. Data

**Arena.** 9 × 9 m walled room. 3–6 static objects drawn from 6 colours × 4 shapes (ball, cube, cylinder, cone). Random floor, wall and light colours. Robot start pose sampled at random.

**Expert** (privileged, used only for labels):
- Walker: PPO policy trained from scratch in MuJoCo Playground's G1 Joystick task, restricted to the 15 lower-body joints, with domain randomization, pushes and sensor noise. [[TODO: steps, wall-clock, final reward]]
- Navigator: each instruction compiles to world-frame segments (go to object / go to point / turn to yaw). Tracking uses A* on a 10 cm cost grid with inflated obstacles, replanned at 2 Hz, plus pure pursuit. If the target is out of view, the expert always searches by turning counter-clockwise, which a camera-only student can imitate.
- Sim-to-sim: the walker is trained in MJX and deployed in the C MuJoCo engine. [[TODO: `check_walker.py` falls / tracking error]]

**Instructions.** Five families with paraphrase templates and referring expressions ("the red ball", "the red one", "the cylinder" when unique):

| family | example | success criterion (final pose) |
|---|---|---|
| goto | "follow the red ball" | within footprint + 0.85 m of the object |
| pass_turn | "go straight and turn right after passing the yellow cube" | within 0.7 m of the point 1.5 m past the turn |
| sequence | "go to the red ball, then go to the green cone" | visited the first object, ended at the second |
| turn | "turn left", "turn around" | yaw within 30°, moved < 0.6 m |
| forward | "walk forward two meters" | within 0.5 m of the target point |

All runs also fail on a fall or any contact with an object or wall. Held-out ("novel") templates are used only at test time.

**Amount.** [[TODO from manifests: expert episodes kept / attempted, hours, frames; DAgger episodes per round]]

## 3. Architecture

**Reused GR00T N1.6 parameters (all frozen):**

| component | params | why |
|---|---|---|
| Eagle vision tower (27-layer ViT, 1152-d, 224 px) | 412M | strong generic visual features; colour and shape recognition is the whole perception problem here |
| `mlp1` projector (pixel shuffle 2×2 → 2048-d) | 14M | maps vision into the LLM space GR00T already aligned with language |
| LLM, 16 layers (GR00T's `select_layer`) + embeddings | 1.1B | free-form instruction understanding. Runs once per episode (the instruction is fixed), so it costs no control-loop time |

**Not reused:** the 1.1B-parameter DiT action head, state/action encoders and embodiment projectors. Reasons: they predict 50-step flow-matching action chunks with 4 denoising steps for other embodiments; far too slow for a reactive 50 Hz humanoid loop on a T4; and G1 lower-body joint targets are not among their embodiments.

**Trained from scratch (~5.6M params):** projections; depth CNN (64 × 64 metric depth → 16 tokens); a 4-layer, 256-d transformer at 10 Hz over [CTRL] + instruction tokens + 8 frames × 16 vision tokens (history 0–3.2 s) + 2 depth frames + proprioception/odometry; an auxiliary velocity-command head; a 3-layer MLP at 50 Hz over [latent, predicted command, 4-step proprioception history] → 15 joint targets.

**Design points to discuss:**
- Why two rates: vision is expensive and slow-changing; balance needs 50 Hz.
- Why history: "after passing the yellow cube" needs memory once the cube leaves the field of view; IMU odometry covers "turn left" and "walk 2 m".
- Late fusion vs. running the LLM on image + text tokens every frame (GR00T style): the trade-off in latency and data size.
- The output is joint targets. The command head is auxiliary supervision only.

## 4. Training

- Loss: MSE on normalized joint targets + 0.5 × MSE on the normalized velocity command.
- Optimizer: AdamW (lr 3e-4, weight decay 0.05), 500 warm-up steps, cosine schedule, batch 512, fp16 autocast, gradient clip 1.0. Vision-frame dropout 0.1.
- Schedule: BC 30k steps → DAgger round 1 (200 episodes, fine-tune 15k) → round 2 (200 episodes, fine-tune 15k).
- Total compute: [[TODO: walker h, data h, features min, BC min, DAgger min; all on one free Colab T4]]
- Curves: [[TODO: val action MSE per stage from train_log.jsonl]]

**What did not work** (from your own runs). Candidates to check and report honestly:
- [[TODO: e.g. pure BC without DAgger: success X% vs Y% (falls from compounding errors)]]
- [[TODO: e.g. no vision history → pass_turn fails; no odometry → turn overshoots; walker trained too briefly → sim-to-sim falls]]

## 5. Real-time strategy

Budget: a 20 ms control period.

| component | rate | T4 latency [[TODO from results.json → component_latency]] |
|---|---|---|
| LLM on instruction | once per episode | |
| vision tower + projector (1 image, fp16) | 10 Hz | |
| slow transformer | 10 Hz | |
| fast MLP | 50 Hz | |
| MuJoCo physics (10 substeps) + RGBD render | 50 Hz / 10 Hz | |
| **end-to-end real-time factor** (sim s / wall s) | | [[TODO]] |

What was cut to make it real time: no DiT head; LLM moved out of the loop; 16 pooled vision tokens per frame instead of 64/256; vision at 10 Hz with cached history features (each frame encoded once); depth at 64 × 64; fp16.

## 6. Results

[[TODO: table per family × {train templates, novel templates}: student vs. expert success, n = 20 each]]

[[TODO: the two task-statement instructions (demo videos): outcome]]

**Three most interesting failure modes** (pick from `results.json` reasons and the videos):
1. [[TODO]]
2. [[TODO]]
3. [[TODO]]

## 7. Limitations and next steps

Static scenes, simple object vocabulary, template-based language, feet-only collision model, trained in simulation only. Next steps: [[TODO]]
