// Copyright 2026. Licensed under the MIT License.
//
// Command-line options of the edge_detector program.

#ifndef SRC_OPTIONS_H_
#define SRC_OPTIONS_H_

#include <string>

#include "canny_params.h"

namespace canny {

struct Options {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  std::string csv_path = "results/timings.csv";
  CannyParams params;
  int block_size = 16;   // Edge length of the square CUDA thread block.
  int repeat = 3;        // Timed GPU runs per image; the fastest is reported.
  bool compare_cpu = true;
  bool show_help = false;
};

// Parses `argv` into `options`. Returns false and fills `error` when an
// argument is unknown or has an invalid value.
bool ParseOptions(int argc, char** argv, Options* options,
                  std::string* error);

// Returns the usage text shown by --help.
std::string UsageText(const char* program_name);

}  // namespace canny

#endif  // SRC_OPTIONS_H_
