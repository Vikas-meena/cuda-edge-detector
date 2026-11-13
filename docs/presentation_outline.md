# Presentation outline (5 to 10 minutes)

Suggested structure for the recorded demonstration. Timings are a guide.

## 1. Goal (about 1 minute)

- Canny edge detection: what it is and why it is a good GPU workload
  (five stages, four of them independent per pixel).
- Goal: a complete GPU pipeline that processes a directory of images in one
  run and proves its output is identical to a CPU implementation.

## 2. Pipeline and kernels (about 2 minutes)

- Show the stage table in `README.md`.
- Open `src/canny_gpu.cu`: the 2-D launch layout (`PixelFromThread`), one
  short kernel per stage, and the Thrust reduction that finds the maximum
  gradient.
- Open `src/pixel_ops.h`: the same functions compile for CPU and GPU.

## 3. The hard part: hysteresis (about 2 minutes)

- Why connectivity is not a per-pixel problem.
- The iterative kernel, the in-place update and why its race is harmless.
- Why the convergence flag is read only every four launches.

## 4. Live demonstration (about 2 minutes)

- `./run.sh` in a terminal: device line, one line per image, summary.
- Open a few files in `data/output/` next to their inputs.
- Run once more with different options, e.g. `./run.sh --sigma=2.5`, and
  show how the edge maps change.

## 5. Results (about 1 minute)

- `results/summary.md` and `results/speedup.png`: time per resolution, the
  split between transfer and kernel time, the mismatch count.
- Which resolutions benefit most from the GPU and why.

## 6. Lessons learned and next steps (about 1 minute)

- Bit-exact CPU/GPU comparison needs matching floating-point settings.
- Next steps from the README: union-find hysteresis, shared-memory blur,
  CUDA streams, comparison with NPP.
