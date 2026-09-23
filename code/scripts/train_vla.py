"""Train the student VLA by behaviour cloning (and DAgger aggregation) on
precomputed frozen-GR00T features.

    python scripts/train_vla.py --data $G1NAV_DATA/episodes/expert --out $G1NAV_DATA/vla/bc
    python scripts/train_vla.py --data $G1NAV_DATA/episodes/expert $G1NAV_DATA/episodes/dagger1 \
        --init $G1NAV_DATA/vla/bc/student.pt --steps 10000 --out $G1NAV_DATA/vla/dagger1
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1nav.dataset import FeatureBank  # noqa: E402
from g1nav.model import StudentConfig, StudentVLA  # noqa: E402
from g1nav.runtime import pick_device  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", nargs="+", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--init", default=None, help="student checkpoint to fine-tune")
  ap.add_argument("--steps", type=int, default=30000)
  ap.add_argument("--batch", type=int, default=512)
  ap.add_argument("--lr", type=float, default=3e-4)
  ap.add_argument("--wd", type=float, default=0.05)
  ap.add_argument("--warmup", type=int, default=500)
  ap.add_argument("--cmd_weight", type=float, default=0.5)
  ap.add_argument("--frame_dropout", type=float, default=0.1)
  ap.add_argument("--eval_every", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  device = pick_device()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  model = StudentVLA.load(args.init) if args.init else StudentVLA(StudentConfig())
  model.to(device)
  bank = FeatureBank(args.data, model.cfg, device)
  if not args.init:
    model.set_normalization(*bank.prop_stats())
  n_params = sum(p.numel() for p in model.parameters())
  print(f"student params: {n_params / 1e6:.2f}M | train episodes "
        f"{bank.num_episodes - bank.num_val_episodes} | val episodes {bank.num_val_episodes}")

  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                            0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
  use_amp = device.type == "cuda"
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
  gen = torch.Generator(device=device)
  gen.manual_seed(args.seed)
  val_gen = torch.Generator(device=device)

  log = open(out / "train_log.jsonl", "a")
  best = float("inf")
  t0 = time.time()
  for step in range(1, args.steps + 1):
    model.train()
    batch = bank.sample(args.batch, "train", gen, args.frame_dropout)
    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
      loss, parts = model.loss(batch, args.cmd_weight)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % args.eval_every == 0 or step == args.steps:
      model.eval()
      val_gen.manual_seed(1234)  # same validation batches every time
      vals = []
      with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        for _ in range(20):
          if bank.num_val_episodes == 0:
            break
          _, vp = model.loss(bank.sample(args.batch, "val", val_gen), args.cmd_weight)
          vals.append({k: float(v) for k, v in vp.items()})
      rec = dict(step=step, lr=sched.get_last_lr()[0], train_action_mse=float(parts["action_mse"]),
                 train_cmd_mse=float(parts["cmd_mse"]), minutes=(time.time() - t0) / 60)
      if vals:
        rec.update({f"val_{k}": sum(v[k] for v in vals) / len(vals) for k in vals[0]})
      print(json.dumps(rec), flush=True)
      log.write(json.dumps(rec) + "\n")
      log.flush()
      meta = dict(args=vars(args), step=step, **{k: v for k, v in rec.items() if "mse" in k})
      model.save(out / "student_last.pt", meta)
      score = rec.get("val_action_mse", rec["train_action_mse"])
      if score < best:
        best = score
        model.save(out / "student.pt", meta)
  print(f"done in {(time.time() - t0) / 60:.1f} min, best val action MSE {best:.5f}")


if __name__ == "__main__":
  main()
