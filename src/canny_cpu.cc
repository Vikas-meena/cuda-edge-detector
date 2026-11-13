// Copyright 2026. Licensed under the MIT License.

#include "canny_cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "pixel_ops.h"

namespace canny {

namespace {

// Promotes every weak pixel that is 8-connected to a strong pixel. A stack
// based flood fill reaches the same fixed point as the iterative GPU kernel.
void Hysteresis(int width, int height, std::vector<uint8_t>* labels) {
  std::vector<int64_t> stack;
  const int64_t pixel_count = static_cast<int64_t>(width) * height;
  for (int64_t i = 0; i < pixel_count; ++i) {
    if ((*labels)[i] == kStrongEdge) stack.push_back(i);
  }
  while (!stack.empty()) {
    const int64_t index = stack.back();
    stack.pop_back();
    const int x = static_cast<int>(index % width);
    const int y = static_cast<int>(index / width);
    for (int dy = -1; dy <= 1; ++dy) {
      const int ny = y + dy;
      if (ny < 0 || ny >= height) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx;
        if (nx < 0 || nx >= width) continue;
        const int64_t neighbor = static_cast<int64_t>(ny) * width + nx;
        if ((*labels)[neighbor] == kWeakEdge) {
          (*labels)[neighbor] = kStrongEdge;
          stack.push_back(neighbor);
        }
      }
    }
  }
}

}  // namespace

void DetectEdgesCpu(const Image& input, const CannyParams& params,
                    Image* edges, CpuStats* stats) {
  const auto start = std::chrono::steady_clock::now();

  const int width = input.width;
  const int height = input.height;
  const size_t pixel_count = input.PixelCount();
  const std::vector<float> weights = BuildGaussianKernel(params.sigma);
  const int radius = static_cast<int>(weights.size() / 2);

  std::vector<float> gray(pixel_count);
  const uint8_t* rgb = input.pixels.data();
  for (size_t i = 0; i < pixel_count; ++i) {
    gray[i] = RgbToGray(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
  }

  std::vector<float> temp(pixel_count);
  std::vector<float> blurred(pixel_count);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      temp[static_cast<size_t>(y) * width + x] =
          BlurHorizontal(gray.data(), width, x, y, weights.data(), radius);
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      blurred[static_cast<size_t>(y) * width + x] = BlurVertical(
          temp.data(), width, height, x, y, weights.data(), radius);
    }
  }

  std::vector<float>& magnitude = gray;  // The grayscale buffer is reused.
  std::vector<uint8_t> direction(pixel_count);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>(y) * width + x;
      SobelAt(blurred.data(), width, height, x, y, &magnitude[index],
              &direction[index]);
    }
  }

  const float max_magnitude =
      pixel_count == 0 ? 0.0f
                       : *std::max_element(magnitude.begin(), magnitude.end());
  const float low = params.low_ratio * max_magnitude;
  const float high = params.high_ratio * max_magnitude;

  std::vector<uint8_t> labels(pixel_count);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      labels[static_cast<size_t>(y) * width + x] =
          SuppressAndClassify(magnitude.data(), direction.data(), width,
                              height, x, y, low, high);
    }
  }

  Hysteresis(width, height, &labels);

  edges->width = width;
  edges->height = height;
  edges->channels = 1;
  edges->pixels.resize(pixel_count);
  for (size_t i = 0; i < pixel_count; ++i) {
    edges->pixels[i] = labels[i] == kStrongEdge ? kEdgeValue : 0;
  }

  if (stats != nullptr) {
    stats->total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  }
}

}  // namespace canny
