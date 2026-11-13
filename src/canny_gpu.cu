// Copyright 2026. Licensed under the MIT License.

#include "canny_gpu.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pixel_ops.h"

namespace canny {

namespace {

// Hysteresis kernel launches issued between two convergence checks. Reading
// the "changed" flag back forces a device synchronisation, so checking after
// every launch would waste time.
constexpr int kLaunchesPerCheck = 4;

void CheckCuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "CUDA error during %s: %s\n", what,
               cudaGetErrorString(status));
  std::exit(EXIT_FAILURE);
}

double MillisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

template <typename T>
void AllocateDevice(T** pointer, size_t count, const char* what) {
  void* raw = nullptr;
  CheckCuda(cudaMalloc(&raw, count * sizeof(T)), what);
  *pointer = static_cast<T*>(raw);
}

template <typename T>
void FreeDevice(T** pointer) {
  if (*pointer == nullptr) return;
  CheckCuda(cudaFree(*pointer), "cudaFree");
  *pointer = nullptr;
}

// Every image kernel uses the same 2-D launch layout: one thread per pixel.
// Returns false for the padding threads that fall outside the image.
__device__ inline bool PixelFromThread(int width, int height, int* x, int* y) {
  *x = blockIdx.x * blockDim.x + threadIdx.x;
  *y = blockIdx.y * blockDim.y + threadIdx.y;
  return *x < width && *y < height;
}

__global__ void GrayscaleKernel(const uint8_t* rgb, float* gray, int width,
                                int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  const int64_t index = static_cast<int64_t>(y) * width + x;
  gray[index] = RgbToGray(rgb[3 * index], rgb[3 * index + 1],
                          rgb[3 * index + 2]);
}

__global__ void BlurHorizontalKernel(const float* src, float* dst,
                                     const float* weights, int radius,
                                     int width, int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  dst[static_cast<int64_t>(y) * width + x] =
      BlurHorizontal(src, width, x, y, weights, radius);
}

__global__ void BlurVerticalKernel(const float* src, float* dst,
                                   const float* weights, int radius,
                                   int width, int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  dst[static_cast<int64_t>(y) * width + x] =
      BlurVertical(src, width, height, x, y, weights, radius);
}

__global__ void SobelKernel(const float* src, float* magnitude,
                            uint8_t* direction, int width, int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  const int64_t index = static_cast<int64_t>(y) * width + x;
  SobelAt(src, width, height, x, y, &magnitude[index], &direction[index]);
}

__global__ void SuppressAndClassifyKernel(const float* magnitude,
                                          const uint8_t* direction,
                                          uint8_t* labels, float low,
                                          float high, int width, int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  labels[static_cast<int64_t>(y) * width + x] = SuppressAndClassify(
      magnitude, direction, width, height, x, y, low, high);
}

// Promotes weak pixels that touch a strong pixel. The label buffer is updated
// in place while neighbouring threads read it. That race is harmless: labels
// only ever move from weak to strong, so a thread that sees a stale value
// merely promotes its pixel one launch later, and the final fixed point is the
// same regardless of the order in which threads run.
__global__ void HysteresisKernel(uint8_t* labels, int* changed, int width,
                                 int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  const int64_t index = static_cast<int64_t>(y) * width + x;
  if (labels[index] != kWeakEdge) return;
  if (HasStrongNeighbor(labels, width, height, x, y)) {
    labels[index] = kStrongEdge;
    *changed = 1;
  }
}

// Converts labels to the final edge map: strong -> 255, anything else -> 0.
__global__ void FinalizeKernel(uint8_t* labels, int width, int height) {
  int x;
  int y;
  if (!PixelFromThread(width, height, &x, &y)) return;
  const int64_t index = static_cast<int64_t>(y) * width + x;
  labels[index] = labels[index] == kStrongEdge ? kEdgeValue : 0;
}

}  // namespace

std::string DescribeGpu() {
  int device = 0;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties;
  CheckCuda(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
  char text[512];
  std::snprintf(text, sizeof(text),
                "%s (compute capability %d.%d, %.1f GiB, %d SMs)",
                properties.name, properties.major, properties.minor,
                static_cast<double>(properties.totalGlobalMem) /
                    (1024.0 * 1024.0 * 1024.0),
                properties.multiProcessorCount);
  return text;
}

GpuCanny::GpuCanny(int block_size) : block_size_(block_size) {
  AllocateDevice(&d_weights_, 2 * kMaxBlurRadius + 1, "weights allocation");
  AllocateDevice(&d_changed_, 1, "flag allocation");
}

GpuCanny::~GpuCanny() {
  Release();
  FreeDevice(&d_weights_);
  FreeDevice(&d_changed_);
}

void GpuCanny::Release() {
  FreeDevice(&d_rgb_);
  FreeDevice(&d_plane_a_);
  FreeDevice(&d_plane_b_);
  FreeDevice(&d_plane_c_);
  FreeDevice(&d_direction_);
  FreeDevice(&d_labels_);
  capacity_ = 0;
}

void GpuCanny::Reserve(size_t pixel_count) {
  if (pixel_count <= capacity_) return;
  Release();
  AllocateDevice(&d_rgb_, 3 * pixel_count, "RGB allocation");
  AllocateDevice(&d_plane_a_, pixel_count, "plane allocation");
  AllocateDevice(&d_plane_b_, pixel_count, "plane allocation");
  AllocateDevice(&d_plane_c_, pixel_count, "plane allocation");
  AllocateDevice(&d_direction_, pixel_count, "direction allocation");
  AllocateDevice(&d_labels_, pixel_count, "label allocation");
  capacity_ = pixel_count;
}

void GpuCanny::Detect(const Image& input, const CannyParams& params,
                      Image* edges, GpuStats* stats) {
  const int width = input.width;
  const int height = input.height;
  const size_t pixel_count = input.PixelCount();
  const std::vector<float> weights = BuildGaussianKernel(params.sigma);
  const int radius = static_cast<int>(weights.size() / 2);
  GpuStats local_stats;

  Reserve(pixel_count);

  // Stage 0: upload.
  auto start = std::chrono::steady_clock::now();
  CheckCuda(cudaMemcpy(d_rgb_, input.pixels.data(), input.ByteCount(),
                       cudaMemcpyHostToDevice),
            "image upload");
  CheckCuda(cudaMemcpy(d_weights_, weights.data(),
                       weights.size() * sizeof(float), cudaMemcpyHostToDevice),
            "weights upload");
  CheckCuda(cudaDeviceSynchronize(), "upload synchronisation");
  local_stats.upload_ms = MillisecondsSince(start);

  // Stages 1-4 run as one thread per pixel on a 2-D grid.
  start = std::chrono::steady_clock::now();
  const dim3 block(block_size_, block_size_);
  const dim3 grid((width + block_size_ - 1) / block_size_,
                  (height + block_size_ - 1) / block_size_);

  GrayscaleKernel<<<grid, block>>>(d_rgb_, d_plane_a_, width, height);
  BlurHorizontalKernel<<<grid, block>>>(d_plane_a_, d_plane_b_, d_weights_,
                                        radius, width, height);
  BlurVerticalKernel<<<grid, block>>>(d_plane_b_, d_plane_c_, d_weights_,
                                      radius, width, height);
  // The grayscale plane is no longer needed, so it receives the magnitudes.
  SobelKernel<<<grid, block>>>(d_plane_c_, d_plane_a_, d_direction_, width,
                               height);
  CheckCuda(cudaGetLastError(), "gradient kernels");

  // The thresholds are relative to the strongest gradient in the image. The
  // maximum is found on the device with a Thrust parallel reduction.
  const thrust::device_ptr<float> magnitude_begin(d_plane_a_);
  const float max_magnitude =
      thrust::reduce(thrust::device, magnitude_begin,
                     magnitude_begin + pixel_count, 0.0f,
                     thrust::maximum<float>());
  const float low = params.low_ratio * max_magnitude;
  const float high = params.high_ratio * max_magnitude;

  SuppressAndClassifyKernel<<<grid, block>>>(d_plane_a_, d_direction_,
                                             d_labels_, low, high, width,
                                             height);
  CheckCuda(cudaGetLastError(), "suppression kernel");

  // Stage 5: hysteresis, repeated until no pixel changes.
  int changed = 1;
  while (changed != 0) {
    CheckCuda(cudaMemset(d_changed_, 0, sizeof(int)), "flag reset");
    for (int i = 0; i < kLaunchesPerCheck; ++i) {
      HysteresisKernel<<<grid, block>>>(d_labels_, d_changed_, width, height);
    }
    local_stats.hysteresis_launches += kLaunchesPerCheck;
    CheckCuda(cudaMemcpy(&changed, d_changed_, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "flag download");
  }

  FinalizeKernel<<<grid, block>>>(d_labels_, width, height);
  CheckCuda(cudaGetLastError(), "finalize kernel");
  CheckCuda(cudaDeviceSynchronize(), "kernel synchronisation");
  local_stats.compute_ms = MillisecondsSince(start);

  // Stage 6: download.
  start = std::chrono::steady_clock::now();
  edges->width = width;
  edges->height = height;
  edges->channels = 1;
  edges->pixels.resize(pixel_count);
  CheckCuda(cudaMemcpy(edges->pixels.data(), d_labels_, pixel_count,
                       cudaMemcpyDeviceToHost),
            "edge map download");
  local_stats.download_ms = MillisecondsSince(start);

  if (stats != nullptr) *stats = local_stats;
}

}  // namespace canny
