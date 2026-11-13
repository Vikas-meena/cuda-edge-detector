# CUDA Canny Edge Detector

Capstone project for the GPU Programming Specialization (CUDA at Scale for
the Enterprise).

A command-line program that runs the complete Canny edge detection pipeline
on the GPU for every image in a directory, checks each result pixel by pixel
against a CPU reference implementation, and records per-image timings.

## What it does

Canny edge detection is a five-stage image-processing pipeline. Each stage is
a hand-written CUDA kernel that runs one thread per pixel:

| Stage | Kernel | What it computes |
|---|---|---|
| 1 | `GrayscaleKernel` | RGB to luma (BT.601) |
| 2 | `BlurHorizontalKernel`, `BlurVerticalKernel` | Separable Gaussian blur |
| 3 | `SobelKernel` | Gradient magnitude and direction quantised to 4 axes |
| 4 | `SuppressAndClassifyKernel` | Non-maximum suppression and double threshold (strong / weak / none) |
| 5 | `HysteresisKernel` | Weak pixels connected to a strong pixel become edges; repeated until nothing changes |

Between stages 3 and 4 the largest gradient magnitude is found with a Thrust
parallel reduction (`thrust::reduce` with `thrust::maximum`), because the two
thresholds are expressed as fractions of that maximum.

GPU usage is explicit: the image is copied to device memory with
`cudaMemcpy`, all stages run on the device, and only the final edge map is
copied back.

## Repository layout

```
src/pixel_ops.h      Per-pixel maths shared by the CPU and GPU code
src/canny_gpu.cu     CUDA kernels and the GpuCanny class
src/canny_cpu.cc     Single-threaded CPU reference
src/main.cc          Batch driver: load, detect, compare, save, log
src/options.cc       Command-line parsing
src/image.cc         PNG/JPEG loading and PNG writing (stb)
tests/               CPU-only unit test
scripts/             Dataset download and report generation
data/output/         Edge maps written by the program
results/             Proof of execution: log, timing CSV, summary, figures
third_party/         stb_image and stb_image_write (public domain)
```

## Requirements

- An NVIDIA GPU and the CUDA Toolkit (`nvcc`), version 11 or newer
- `g++` with C++17 support, `make`
- For the dataset and the report: `curl`, `unzip`, `python3` with Pillow and
  matplotlib

## Build and run

```bash
./run.sh
```

`run.sh` downloads the test images if `data/input` is empty, builds the
program, runs it and writes the report. The individual steps are:

```bash
./scripts/fetch_data.sh        # download and convert the test images
make build                     # compile bin/edge_detector
make run                       # run with default options
python3 scripts/make_report.py # summary table and figures in results/
make test                      # CPU-only unit test, needs no GPU
make clean
```

If the CUDA Toolkit is not in `/usr/local/cuda`, pass its location:
`make build CUDA_PATH=/path/to/cuda`.

### Command-line options

```
bin/edge_detector [options]
  --input_dir=DIR    Directory with input images (default data/input)
  --output_dir=DIR   Directory for the edge maps (default data/output)
  --csv=FILE         Per-image timing table (default results/timings.csv)
  --sigma=FLOAT      Gaussian blur sigma in pixels (default 1.4)
  --low=FLOAT        Low threshold, fraction of the maximum gradient (default 0.08)
  --high=FLOAT       High threshold, fraction of the maximum gradient (default 0.2)
  --block_size=INT   CUDA block edge length, 1-32 (default 16)
  --repeat=INT       Timed GPU runs per image (default 3)
  --skip_cpu         Do not run the CPU reference comparison
  --help             Show this message
```

Examples:

```bash
./run.sh --sigma=2.0 --high=0.3
make run ARGS="--block_size=32 --skip_cpu"
bin/edge_detector --input_dir=my_photos --output_dir=my_edges
```

## Data

The test set is the *Miscellaneous* and *Aerials* volumes of the
[USC-SIPI Image Database](https://sipi.usc.edu/database/database.php):
77 images from 256x256 to 2250x2250, about 44 megapixels in total.
`scripts/fetch_data.sh` downloads them and converts TIFF to PNG. The inputs
are not committed to this repository (about 75 MB); the outputs are.

## Proof of execution

One execution of the program processes all 77 images. Its artifacts are in
[`results/`](results) and [`data/output/`](data/output):

- [`results/run_log.txt`](results/run_log.txt): console output, one line per image plus a summary
- [`results/timings.csv`](results/timings.csv): per-image GPU upload / kernel / download times, CPU time, speedup, mismatching pixels
- [`results/summary.md`](results/summary.md): the same data averaged per resolution
- `data/output/*_edges.png`: one edge map per input image

![Inputs and their edge maps](results/montage.png)

![GPU versus CPU time per image](results/speedup.png)

## Design notes and lessons learned

**One definition of the maths for both processors.** The per-pixel functions
in `src/pixel_ops.h` are marked `__host__ __device__` when compiled by `nvcc`
and are plain inline functions otherwise. The CPU reference and the CUDA
kernels therefore run the same source. To make them also *round* the same,
fused multiply-add is disabled on both sides (`-fmad=false` for `nvcc`,
`-ffp-contract=off` for `g++`) and the gradient direction is quantised with
comparisons instead of `atan2`. With that, the program can report an exact
count of mismatching pixels instead of an approximate similarity score.

**Hysteresis does not fit the one-thread-per-pixel model.** Stages 1 to 4 are
embarrassingly parallel: each output pixel depends on a fixed neighbourhood.
Hysteresis is a connectivity problem: a weak pixel can become an edge because
of a strong pixel hundreds of pixels away along a contour. On the CPU this is
a flood fill with a stack. On the GPU it is an iterative kernel in which each
weak pixel looks at its eight neighbours, launched repeatedly until no pixel
changes. The number of launches is recorded per image in `timings.csv`
(`hysteresis_launches`).

**A deliberate, harmless data race.** The hysteresis kernel updates the label
buffer in place while other threads read it. Labels only ever change from
weak to strong, so a stale read can only delay a promotion by one launch; the
final result is the same fixed point that the CPU flood fill reaches. This
avoids a second buffer and a copy per launch.

**Convergence checks cost a synchronisation.** Reading the "something
changed" flag back to the host blocks until the device is idle. The flag is
therefore read only once every four launches.

**Measuring fairly.** The first kernel launch pays for CUDA context creation,
so one untimed warm-up run precedes the measurements. GPU times include the
host-to-device and device-to-host copies and are reported separately from
kernel time, which shows how much of the cost of small images is transfer
rather than computation. Device buffers are allocated once and reused across
the batch.

## Possible next steps

- Replace iterative hysteresis with GPU connected-component labelling
  (union-find), which converges in a number of steps that grows with the
  logarithm of the contour length instead of linearly.
- Use shared memory tiles for the blur so each block reads global memory once.
- Process several images concurrently with CUDA streams and pinned host
  memory to overlap transfers with computation.
- Compare against the NPP implementation
  (`nppiFilterCannyBorder_8u_C1R`).

## Third-party code

`third_party/stb_image.h` and `third_party/stb_image_write.h` by Sean Barrett
are public domain (https://github.com/nothings/stb).
