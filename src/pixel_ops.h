// Copyright 2026. Licensed under the MIT License.
//
// Per-pixel building blocks of the Canny edge detector.
//
// Every function here is compiled twice: once by the host compiler for the
// CPU reference implementation and once by nvcc as a __device__ function that
// the CUDA kernels call. Sharing one definition is what makes it possible to
// demand that the GPU output matches the CPU output pixel for pixel.

#ifndef SRC_PIXEL_OPS_H_
#define SRC_PIXEL_OPS_H_

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define CANNY_HOST_DEVICE __host__ __device__
#else
#define CANNY_HOST_DEVICE
#endif

namespace canny {

// Largest supported Gaussian radius (the kernel has 2 * radius + 1 taps).
constexpr int kMaxBlurRadius = 15;

// Pixel labels produced by the double-threshold stage.
constexpr uint8_t kNoEdge = 0;
constexpr uint8_t kWeakEdge = 1;
constexpr uint8_t kStrongEdge = 2;

// Value written to the final edge map for an edge pixel.
constexpr uint8_t kEdgeValue = 255;

// Gradient direction quantised to the four neighbour axes used by
// non-maximum suppression.
constexpr uint8_t kDirHorizontal = 0;    // Compare with left / right.
constexpr uint8_t kDirDiagonalDown = 1;  // Compare with (+1,+1) / (-1,-1).
constexpr uint8_t kDirVertical = 2;      // Compare with up / down.
constexpr uint8_t kDirDiagonalUp = 3;    // Compare with (+1,-1) / (-1,+1).

CANNY_HOST_DEVICE inline int ClampInt(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// ITU-R BT.601 luma from 8-bit RGB, as a float in [0, 255].
CANNY_HOST_DEVICE inline float RgbToGray(uint8_t r, uint8_t g, uint8_t b) {
  return 0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) +
         0.114f * static_cast<float>(b);
}

// One tap row of the separable Gaussian. Borders are handled by clamping.
CANNY_HOST_DEVICE inline float BlurHorizontal(const float* src, int width,
                                              int x, int y,
                                              const float* weights,
                                              int radius) {
  const float* row = src + static_cast<int64_t>(y) * width;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum += weights[k + radius] * row[ClampInt(x + k, 0, width - 1)];
  }
  return sum;
}

CANNY_HOST_DEVICE inline float BlurVertical(const float* src, int width,
                                            int height, int x, int y,
                                            const float* weights, int radius) {
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const int64_t row = ClampInt(y + k, 0, height - 1);
    sum += weights[k + radius] * src[row * width + x];
  }
  return sum;
}

// 3x3 Sobel operator. Writes the gradient magnitude and its quantised
// direction for pixel (x, y).
CANNY_HOST_DEVICE inline void SobelAt(const float* src, int width, int height,
                                      int x, int y, float* magnitude,
                                      uint8_t* direction) {
  const int x0 = ClampInt(x - 1, 0, width - 1);
  const int x2 = ClampInt(x + 1, 0, width - 1);
  const int64_t stride = width;
  const int64_t r0 = ClampInt(y - 1, 0, height - 1) * stride;
  const int64_t r1 = y * stride;
  const int64_t r2 = ClampInt(y + 1, 0, height - 1) * stride;

  const float gx = (src[r0 + x2] - src[r0 + x0]) +
                   2.0f * (src[r1 + x2] - src[r1 + x0]) +
                   (src[r2 + x2] - src[r2 + x0]);
  const float gy = (src[r2 + x0] - src[r0 + x0]) +
                   2.0f * (src[r2 + x] - src[r0 + x]) +
                   (src[r2 + x2] - src[r0 + x2]);

  *magnitude = sqrtf(gx * gx + gy * gy);

  // Quantise the angle without atan2: compare |gy| with |gx| * tan(22.5 deg)
  // and |gx| * tan(67.5 deg).
  const float abs_gx = fabsf(gx);
  const float abs_gy = fabsf(gy);
  if (abs_gy <= abs_gx * 0.41421356f) {
    *direction = kDirHorizontal;
  } else if (abs_gy >= abs_gx * 2.41421356f) {
    *direction = kDirVertical;
  } else if ((gx > 0.0f) == (gy > 0.0f)) {
    *direction = kDirDiagonalDown;
  } else {
    *direction = kDirDiagonalUp;
  }
}

// Returns the magnitude at (x, y), or 0 outside the image.
CANNY_HOST_DEVICE inline float MagnitudeOrZero(const float* magnitude,
                                               int width, int height, int x,
                                               int y) {
  if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
  return magnitude[static_cast<int64_t>(y) * width + x];
}

// Non-maximum suppression followed by the double threshold. A pixel survives
// only if it is the largest along its gradient direction; survivors are
// labelled strong (>= high) or weak (>= low).
CANNY_HOST_DEVICE inline uint8_t SuppressAndClassify(const float* magnitude,
                                                     const uint8_t* direction,
                                                     int width, int height,
                                                     int x, int y, float low,
                                                     float high) {
  const int64_t index = static_cast<int64_t>(y) * width + x;
  const float value = magnitude[index];
  if (value < low) return kNoEdge;

  int dx = 1;
  int dy = 0;
  const uint8_t dir = direction[index];
  if (dir == kDirDiagonalDown) {
    dx = 1;
    dy = 1;
  } else if (dir == kDirVertical) {
    dx = 0;
    dy = 1;
  } else if (dir == kDirDiagonalUp) {
    dx = 1;
    dy = -1;
  }
  const float ahead = MagnitudeOrZero(magnitude, width, height, x + dx, y + dy);
  const float behind =
      MagnitudeOrZero(magnitude, width, height, x - dx, y - dy);
  // The asymmetric comparison keeps exactly one pixel of a two-pixel plateau.
  if (value < ahead || value <= behind) return kNoEdge;
  return value >= high ? kStrongEdge : kWeakEdge;
}

// True when any of the eight neighbours of (x, y) is a strong edge.
CANNY_HOST_DEVICE inline bool HasStrongNeighbor(const uint8_t* labels,
                                                int width, int height, int x,
                                                int y) {
  for (int dy = -1; dy <= 1; ++dy) {
    const int ny = y + dy;
    if (ny < 0 || ny >= height) continue;
    for (int dx = -1; dx <= 1; ++dx) {
      const int nx = x + dx;
      if (nx < 0 || nx >= width || (dx == 0 && dy == 0)) continue;
      if (labels[static_cast<int64_t>(ny) * width + nx] == kStrongEdge) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace canny

#endif  // SRC_PIXEL_OPS_H_
