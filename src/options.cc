// Copyright 2026. Licensed under the MIT License.

#include "options.h"

#include <cstdlib>
#include <sstream>

namespace canny {

namespace {

// Splits "--name=value" into its two halves. "--name" yields an empty value.
void SplitArgument(const std::string& argument, std::string* name,
                   std::string* value, bool* has_value) {
  const size_t equals = argument.find('=');
  *has_value = equals != std::string::npos;
  *name = argument.substr(0, equals);
  *value = *has_value ? argument.substr(equals + 1) : "";
}

bool ParseFloat(const std::string& text, float* result) {
  char* end = nullptr;
  const float value = std::strtof(text.c_str(), &end);
  if (text.empty() || *end != '\0') return false;
  *result = value;
  return true;
}

bool ParseInt(const std::string& text, int* result) {
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);  // NOLINT
  if (text.empty() || *end != '\0') return false;
  *result = static_cast<int>(value);
  return true;
}

}  // namespace

bool ParseOptions(int argc, char** argv, Options* options,
                  std::string* error) {
  for (int i = 1; i < argc; ++i) {
    std::string name;
    std::string value;
    bool has_value = false;
    SplitArgument(argv[i], &name, &value, &has_value);

    if (name == "--help" || name == "-h") {
      options->show_help = true;
      continue;
    }
    if (name == "--skip_cpu") {
      options->compare_cpu = false;
      continue;
    }
    // Every remaining option needs a value, given either as --name=value or
    // as --name value.
    if (!has_value) {
      if (i + 1 >= argc) {
        *error = "missing value for " + name;
        return false;
      }
      value = argv[++i];
    }

    bool ok = true;
    if (name == "--input_dir") {
      options->input_dir = value;
    } else if (name == "--output_dir") {
      options->output_dir = value;
    } else if (name == "--csv") {
      options->csv_path = value;
    } else if (name == "--sigma") {
      ok = ParseFloat(value, &options->params.sigma);
    } else if (name == "--low") {
      ok = ParseFloat(value, &options->params.low_ratio);
    } else if (name == "--high") {
      ok = ParseFloat(value, &options->params.high_ratio);
    } else if (name == "--block_size") {
      ok = ParseInt(value, &options->block_size);
    } else if (name == "--repeat") {
      ok = ParseInt(value, &options->repeat);
    } else {
      *error = "unknown option " + name;
      return false;
    }
    if (!ok) {
      *error = "invalid value '" + value + "' for " + name;
      return false;
    }
  }

  const CannyParams& params = options->params;
  if (params.sigma <= 0.0f) {
    *error = "--sigma must be positive";
    return false;
  }
  if (params.low_ratio < 0.0f || params.high_ratio > 1.0f ||
      params.low_ratio > params.high_ratio) {
    *error = "thresholds must satisfy 0 <= low <= high <= 1";
    return false;
  }
  // CUDA allows at most 1024 threads per block, i.e. 32 x 32.
  if (options->block_size < 1 || options->block_size > 32) {
    *error = "--block_size must be between 1 and 32";
    return false;
  }
  if (options->repeat < 1) {
    *error = "--repeat must be at least 1";
    return false;
  }
  return true;
}

std::string UsageText(const char* program_name) {
  const Options defaults;
  std::ostringstream text;
  text << "Usage: " << program_name << " [options]\n\n"
       << "Runs a CUDA Canny edge detector on every image in a directory.\n\n"
       << "Options:\n"
       << "  --input_dir=DIR    Directory with input images (default "
       << defaults.input_dir << ")\n"
       << "  --output_dir=DIR   Directory for the edge maps (default "
       << defaults.output_dir << ")\n"
       << "  --csv=FILE         Per-image timing table (default "
       << defaults.csv_path << ")\n"
       << "  --sigma=FLOAT      Gaussian blur sigma in pixels (default "
       << defaults.params.sigma << ")\n"
       << "  --low=FLOAT        Low threshold, fraction of the maximum "
       << "gradient (default " << defaults.params.low_ratio << ")\n"
       << "  --high=FLOAT       High threshold, fraction of the maximum "
       << "gradient (default " << defaults.params.high_ratio << ")\n"
       << "  --block_size=INT   CUDA block edge length, 1-32 (default "
       << defaults.block_size << ")\n"
       << "  --repeat=INT       Timed GPU runs per image (default "
       << defaults.repeat << ")\n"
       << "  --skip_cpu         Do not run the CPU reference comparison\n"
       << "  --help             Show this message\n";
  return text.str();
}

}  // namespace canny
