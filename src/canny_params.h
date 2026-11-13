// Copyright 2026. Licensed under the MIT License.
//
// Tunable parameters and helpers shared by the CPU and GPU detectors.

#ifndef SRC_CANNY_PARAMS_H_
#define SRC_CANNY_PARAMS_H_

#include <vector>

namespace canny {

struct CannyParams {
  // Standard deviation of the Gaussian pre-blur, in pixels.
  float sigma = 1.4f;
  // Thresholds as a fraction of the largest gradient magnitude in the image.
  float low_ratio = 0.08f;
  float high_ratio = 0.20f;
};

// Returns the normalised 1-D Gaussian taps for `sigma`. The result has
// 2 * radius + 1 entries with radius = ceil(3 * sigma), capped at
// kMaxBlurRadius.
std::vector<float> BuildGaussianKernel(float sigma);

}  // namespace canny

#endif  // SRC_CANNY_PARAMS_H_
