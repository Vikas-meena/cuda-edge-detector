// Copyright 2026. Licensed under the MIT License.
//
// CUDA implementation of the Canny edge detector. This header is plain C++
// so that it can be included from translation units built by the host
// compiler; all CUDA specifics live in canny_gpu.cu.

#ifndef SRC_CANNY_GPU_H_
#define SRC_CANNY_GPU_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "canny_params.h"
#include "image.h"

namespace canny {

struct GpuStats {
  double upload_ms = 0.0;    // Host -> device copy of the RGB image.
  double compute_ms = 0.0;   // All kernels, including hysteresis.
  double download_ms = 0.0;  // Device -> host copy of the edge map.
  int hysteresis_launches = 0;

  double TotalMs() const { return upload_ms + compute_ms + download_ms; }
};

// Returns a one-line description of the active CUDA device, for logging.
std::string DescribeGpu();

// Owns the device buffers needed to process one image at a time. Buffers
// grow on demand and are reused across images, so a batch of same-sized
// images pays for allocation only once.
class GpuCanny {
 public:
  // `block_size` is the edge length of the square CUDA thread block.
  explicit GpuCanny(int block_size);
  ~GpuCanny();

  GpuCanny(const GpuCanny&) = delete;
  GpuCanny& operator=(const GpuCanny&) = delete;

  // Runs Canny on the 3-channel `input` and writes a 1-channel edge map
  // (0 or 255) to `edges`.
  void Detect(const Image& input, const CannyParams& params, Image* edges,
              GpuStats* stats);

 private:
  void Reserve(size_t pixel_count);
  void Release();

  int block_size_;
  size_t capacity_ = 0;  // Pixels that the current buffers can hold.
  uint8_t* d_rgb_ = nullptr;
  float* d_plane_a_ = nullptr;  // Grayscale, then gradient magnitude.
  float* d_plane_b_ = nullptr;  // Horizontally blurred image.
  float* d_plane_c_ = nullptr;  // Fully blurred image.
  uint8_t* d_direction_ = nullptr;
  uint8_t* d_labels_ = nullptr;  // Edge labels, then the final edge map.
  float* d_weights_ = nullptr;
  int* d_changed_ = nullptr;
};

}  // namespace canny

#endif  // SRC_CANNY_GPU_H_
