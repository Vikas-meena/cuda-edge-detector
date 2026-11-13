// Copyright 2026. Licensed under the MIT License.
//
// CPU-only checks of the shared per-pixel code, the reference detector and
// the option parser. They need neither a GPU nor a test framework:
//
//   make test

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "canny_cpu.h"
#include "canny_params.h"
#include "image.h"
#include "options.h"
#include "pixel_ops.h"

namespace {

int g_failures = 0;

void Expect(bool condition, const char* description) {
  std::printf("[%s] %s\n", condition ? " OK " : "FAIL", description);
  if (!condition) ++g_failures;
}

// A black RGB image with a white axis-aligned square in the middle.
canny::Image MakeSquareImage(int size, int square_begin, int square_end) {
  canny::Image image;
  image.width = size;
  image.height = size;
  image.channels = 3;
  image.pixels.assign(image.ByteCount(), 0);
  for (int y = square_begin; y < square_end; ++y) {
    for (int x = square_begin; x < square_end; ++x) {
      for (int c = 0; c < 3; ++c) {
        image.pixels[(static_cast<size_t>(y) * size + x) * 3 + c] = 255;
      }
    }
  }
  return image;
}

void TestGaussianKernel() {
  const std::vector<float> weights = canny::BuildGaussianKernel(1.4f);
  float total = 0.0f;
  for (float weight : weights) total += weight;
  Expect(weights.size() == 11, "sigma 1.4 gives an 11-tap kernel");
  Expect(std::fabs(total - 1.0f) < 1e-5f, "kernel weights sum to 1");
  Expect(weights.front() == weights.back(), "kernel is symmetric");
  Expect(canny::BuildGaussianKernel(100.0f).size() ==
             2 * canny::kMaxBlurRadius + 1,
         "kernel radius is capped");
}

void TestSobelDirections() {
  // A vertical step edge: left half 0, right half 100.
  const int size = 5;
  std::vector<float> step(size * size, 0.0f);
  for (int y = 0; y < size; ++y) {
    for (int x = 3; x < size; ++x) step[y * size + x] = 100.0f;
  }
  float magnitude = 0.0f;
  uint8_t direction = 255;
  canny::SobelAt(step.data(), size, size, 2, 2, &magnitude, &direction);
  Expect(magnitude == 400.0f, "Sobel magnitude of a vertical step is 400");
  Expect(direction == canny::kDirHorizontal,
         "vertical step has a horizontal gradient");

  canny::SobelAt(step.data(), size, size, 0, 2, &magnitude, &direction);
  Expect(magnitude == 0.0f, "flat region has zero gradient");
}

void TestUniformImageHasNoEdges() {
  canny::Image image = MakeSquareImage(32, 0, 32);
  canny::Image edges;
  canny::DetectEdgesCpu(image, canny::CannyParams(), &edges, nullptr);
  int edge_pixels = 0;
  for (uint8_t value : edges.pixels) edge_pixels += value != 0;
  Expect(edge_pixels == 0, "uniform image produces no edges");
}

void TestSquareOutline() {
  const int size = 96;
  const int begin = 24;
  const int end = 72;
  canny::Image edges;
  canny::CpuStats stats;
  canny::DetectEdgesCpu(MakeSquareImage(size, begin, end),
                        canny::CannyParams(), &edges, &stats);

  int edge_pixels = 0;
  int far_from_border = 0;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      if (edges.pixels[y * size + x] == 0) continue;
      ++edge_pixels;
      // Distance to the nearest side of the square's boundary.
      const int dx = std::min(std::abs(x - begin), std::abs(x - (end - 1)));
      const int dy = std::min(std::abs(y - begin), std::abs(y - (end - 1)));
      const bool near_vertical_side = dx <= 2 && y >= begin - 2 && y <= end + 1;
      const bool near_horizontal_side =
          dy <= 2 && x >= begin - 2 && x <= end + 1;
      if (!near_vertical_side && !near_horizontal_side) ++far_from_border;
    }
  }
  const int perimeter = 4 * (end - begin);
  Expect(edges.channels == 1 && edges.width == size, "edge map is 1-channel");
  Expect(far_from_border == 0, "all edges lie on the square's boundary");
  Expect(edge_pixels > perimeter * 3 / 4 && edge_pixels < perimeter * 3 / 2,
         "edge is thin: pixel count is close to the perimeter");
  Expect(edges.pixels[(size / 2) * size + size / 2] == 0,
         "interior of the square is not an edge");
  Expect(stats.total_ms > 0.0, "CPU timing is recorded");
}

void TestHysteresisKeepsConnectedWeakPixels() {
  // Labels: one strong pixel followed by a chain of weak ones, plus an
  // isolated weak pixel that must not survive.
  const int width = 8;
  const int height = 3;
  std::vector<uint8_t> labels(width * height, canny::kNoEdge);
  labels[1 * width + 0] = canny::kStrongEdge;
  labels[1 * width + 1] = canny::kWeakEdge;
  labels[0 * width + 2] = canny::kWeakEdge;
  labels[1 * width + 6] = canny::kWeakEdge;
  Expect(canny::HasStrongNeighbor(labels.data(), width, height, 1, 1),
         "weak pixel next to a strong pixel is promoted");
  Expect(!canny::HasStrongNeighbor(labels.data(), width, height, 2, 0),
         "second pixel of the chain waits for the first");
  Expect(!canny::HasStrongNeighbor(labels.data(), width, height, 6, 1),
         "isolated weak pixel is not promoted");
}

void TestOptions() {
  canny::Options options;
  std::string error;
  const char* good[] = {"prog", "--sigma=2.5", "--block_size", "8",
                        "--skip_cpu", "--input_dir=in"};
  Expect(canny::ParseOptions(6, const_cast<char**>(good), &options, &error),
         "valid options parse");
  Expect(options.params.sigma == 2.5f && options.block_size == 8 &&
             !options.compare_cpu && options.input_dir == "in",
         "parsed values are stored");

  const char* unknown[] = {"prog", "--bogus=1"};
  Expect(!canny::ParseOptions(2, const_cast<char**>(unknown), &options,
                              &error),
         "unknown option is rejected");
  const char* bad_block[] = {"prog", "--block_size=64"};
  Expect(!canny::ParseOptions(2, const_cast<char**>(bad_block), &options,
                              &error),
         "block size above 32 is rejected");
  const char* bad_number[] = {"prog", "--sigma=abc"};
  Expect(!canny::ParseOptions(2, const_cast<char**>(bad_number), &options,
                              &error),
         "non-numeric value is rejected");
}

}  // namespace

int main() {
  TestGaussianKernel();
  TestSobelDirections();
  TestUniformImageHasNoEdges();
  TestSquareOutline();
  TestHysteresisKeepsConnectedWeakPixels();
  TestOptions();
  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASSED" : "FAILED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
