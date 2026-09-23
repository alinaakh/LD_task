# Regenerating the dataset

Each `manifest.json` records the generator, base seed, episode count, and the walker file's MD5 hash. Episode *i* of a split uses seed `base_seed + i` for the scene, the instruction, and the sensor noise. Expert episodes that fail (fall, collision, wrong final pose) are discarded and counted in `stats`.

Run from `code/`, with `checkpoint/` from this submission:

```bash
# expert demonstrations
python scripts/generate_data.py --walker ../checkpoint/walker.npz --out episodes/expert --n 600 --base_seed 0

# DAgger rounds: the student that collected each round is in checkpoint/intermediate/
python scripts/dagger.py --student ../checkpoint/intermediate/bc/student.pt --walker ../checkpoint/walker.npz --out episodes/dagger1 --n 200 --base_seed 1000000
python scripts/dagger.py --student ../checkpoint/intermediate/dagger1/student.pt --walker ../checkpoint/walker.npz --out episodes/dagger2 --n 200 --base_seed 2000000

# frozen GR00T features for all of them
python scripts/extract_features.py --dirs episodes/expert episodes/dagger1 episodes/dagger2
```

Determinism: physics, task sampling and sensor noise are deterministic given the seed on a given MuJoCo build. Rendered images can differ in the last bits across GPUs and OpenGL backends (EGL vs. others). DAgger episodes also depend on GPU floating-point results, because the student's actions feed back into the state. They regenerate to the same distribution but are not guaranteed to be bit-identical.
