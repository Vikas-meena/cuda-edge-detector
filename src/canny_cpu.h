// Copyright 2026. Licensed under the MIT License.
//
// Single-threaded CPU reference implementation of the Canny edge detector.
// It exists to validate the GPU output and to provide a timing baseline.

#ifndef SRC_CANNY_CPU_H_
#define SRC_CANNY_CPU_H_

#include "canny_params.h"
#include "image.h"

namespace canny {

struct CpuStats {
  double total_ms = 0.0;
};

// Runs Canny on the 3-channel `input` and writes a 1-channel edge map
// (0 or 255) to `edges`.
void DetectEdgesCpu(const Image& input, const CannyParams& params,
                    Image* edges, CpuStats* stats);

}  // namespace canny

#endif  // SRC_CANNY_CPU_H_
