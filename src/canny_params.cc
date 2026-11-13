// Copyright 2026. Licensed under the MIT License.

#include "canny_params.h"

#include <algorithm>
#include <cmath>

#include "pixel_ops.h"

namespace canny {

std::vector<float> BuildGaussianKernel(float sigma) {
  const int radius = std::clamp(static_cast<int>(std::ceil(3.0f * sigma)), 1,
                                kMaxBlurRadius);
  std::vector<float> weights(2 * radius + 1);
  float total = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const float value = std::exp(-(k * k) / (2.0f * sigma * sigma));
    weights[k + radius] = value;
    total += value;
  }
  for (float& weight : weights) weight /= total;
  return weights;
}

}  // namespace canny
