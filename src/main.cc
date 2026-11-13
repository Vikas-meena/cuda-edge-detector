// Copyright 2026. Licensed under the MIT License.
//
// edge_detector: batch Canny edge detection on the GPU.
//
// For every image in the input directory the program runs the CUDA detector,
// optionally runs the CPU reference and compares the two edge maps, writes
// the edge map as PNG and appends one row to a CSV timing table.

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "canny_cpu.h"
#include "canny_gpu.h"
#include "image.h"
#include "options.h"

namespace {

namespace fs = std::filesystem;

// Totals accumulated over the whole batch.
struct BatchSummary {
  int images = 0;
  int failed = 0;
  double megapixels = 0.0;
  double gpu_ms = 0.0;
  double cpu_ms = 0.0;
  int64_t mismatched_pixels = 0;
};

bool HasImageExtension(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  static const char* const kExtensions[] = {".png", ".jpg", ".jpeg", ".bmp",
                                            ".tga", ".pgm", ".ppm"};
  return std::any_of(
      std::begin(kExtensions), std::end(kExtensions),
      [&extension](const char* known) { return extension == known; });
}

// Returns the image files directly inside `directory`, sorted by name.
std::vector<fs::path> ListImages(const std::string& directory) {
  std::vector<fs::path> paths;
  std::error_code error;
  for (const fs::directory_entry& entry :
       fs::directory_iterator(directory, error)) {
    if (entry.is_regular_file() && HasImageExtension(entry.path())) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

// Number of pixels that differ between two edge maps of the same size.
int64_t CountMismatches(const canny::Image& a,
                          const canny::Image& b) {
  int64_t mismatches = 0;
  for (size_t i = 0; i < a.pixels.size(); ++i) {
    if (a.pixels[i] != b.pixels[i]) ++mismatches;
  }
  return mismatches;
}

// Processes one image and appends its row to `csv`. Returns false when the
// image cannot be read or written.
bool ProcessImage(const fs::path& path, const canny::Options& options,
                  canny::GpuCanny* detector, std::ofstream* csv,
                  BatchSummary* summary) {
  canny::Image input;
  std::string error;
  if (!canny::LoadRgbImage(path.string(), &input, &error)) {
    std::fprintf(stderr, "  skipping %s: %s\n", path.filename().c_str(),
                 error.c_str());
    return false;
  }

  // Run the GPU detector several times and keep the fastest run, which is
  // the one least disturbed by other activity on the machine.
  canny::Image gpu_edges;
  canny::GpuStats gpu_stats;
  for (int run = 0; run < options.repeat; ++run) {
    canny::GpuStats run_stats;
    detector->Detect(input, options.params, &gpu_edges, &run_stats);
    if (run == 0 || run_stats.TotalMs() < gpu_stats.TotalMs()) {
      gpu_stats = run_stats;
    }
  }

  canny::CpuStats cpu_stats;
  int64_t mismatches = 0;
  if (options.compare_cpu) {
    canny::Image cpu_edges;
    canny::DetectEdgesCpu(input, options.params, &cpu_edges, &cpu_stats);
    mismatches = CountMismatches(gpu_edges, cpu_edges);
  }

  const fs::path output_path =
      fs::path(options.output_dir) / (path.stem().string() + "_edges.png");
  if (!canny::SavePng(output_path.string(), gpu_edges)) {
    std::fprintf(stderr, "  cannot write %s\n", output_path.c_str());
    return false;
  }

  const int64_t edge_pixels =
      std::count(gpu_edges.pixels.begin(), gpu_edges.pixels.end(), 255);
  const double megapixels = static_cast<double>(input.PixelCount()) / 1.0e6;
  const double gpu_ms = gpu_stats.TotalMs();
  const double speedup = gpu_ms > 0.0 ? cpu_stats.total_ms / gpu_ms : 0.0;

  if (options.compare_cpu) {
    std::printf(
        "  %-18s %5dx%-5d gpu %8.3f ms  cpu %9.3f ms  speedup %6.1fx  "
        "mismatches %" PRId64 "\n",
        path.filename().c_str(), input.width, input.height, gpu_ms,
        cpu_stats.total_ms, speedup, mismatches);
  } else {
    std::printf("  %-18s %5dx%-5d gpu %8.3f ms  edge pixels %" PRId64 "\n",
                path.filename().c_str(), input.width, input.height, gpu_ms,
                edge_pixels);
  }

  *csv << path.filename().string() << ',' << input.width << ','
       << input.height << ',' << gpu_stats.upload_ms << ','
       << gpu_stats.compute_ms << ',' << gpu_stats.download_ms << ','
       << gpu_ms << ',' << gpu_stats.hysteresis_launches << ',';
  if (options.compare_cpu) {
    *csv << cpu_stats.total_ms << ',' << speedup << ',' << mismatches;
  } else {
    *csv << ",,";
  }
  *csv << ',' << edge_pixels << '\n';

  ++summary->images;
  summary->megapixels += megapixels;
  summary->gpu_ms += gpu_ms;
  summary->cpu_ms += cpu_stats.total_ms;
  summary->mismatched_pixels += mismatches;
  return true;
}

void PrintSummary(const canny::Options& options, const BatchSummary& summary) {
  std::printf("\nSummary\n");
  std::printf("  images processed : %d (%d failed)\n", summary.images,
              summary.failed);
  std::printf("  total megapixels : %.2f\n", summary.megapixels);
  std::printf("  GPU total        : %.2f ms (%.1f megapixels/s)\n",
              summary.gpu_ms,
              summary.gpu_ms > 0.0
                  ? summary.megapixels / (summary.gpu_ms / 1000.0)
                  : 0.0);
  if (options.compare_cpu) {
    std::printf("  CPU total        : %.2f ms (%.1f megapixels/s)\n",
                summary.cpu_ms,
                summary.cpu_ms > 0.0
                    ? summary.megapixels / (summary.cpu_ms / 1000.0)
                    : 0.0);
    std::printf("  overall speedup  : %.1fx\n",
                summary.gpu_ms > 0.0 ? summary.cpu_ms / summary.gpu_ms : 0.0);
    std::printf("  GPU/CPU mismatch : %" PRId64 " pixels\n",
                summary.mismatched_pixels);
  }
  std::printf("  edge maps        : %s\n", options.output_dir.c_str());
  std::printf("  timing table     : %s\n", options.csv_path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  canny::Options options;
  std::string error;
  if (!canny::ParseOptions(argc, argv, &options, &error)) {
    std::fprintf(stderr, "Error: %s\n\n%s", error.c_str(),
                 canny::UsageText(argv[0]).c_str());
    return EXIT_FAILURE;
  }
  if (options.show_help) {
    std::printf("%s", canny::UsageText(argv[0]).c_str());
    return EXIT_SUCCESS;
  }

  const std::vector<fs::path> paths = ListImages(options.input_dir);
  if (paths.empty()) {
    std::fprintf(stderr, "Error: no images found in %s\n",
                 options.input_dir.c_str());
    return EXIT_FAILURE;
  }

  std::error_code directory_error;
  fs::create_directories(options.output_dir, directory_error);
  const fs::path csv_directory = fs::path(options.csv_path).parent_path();
  if (!csv_directory.empty()) {
    fs::create_directories(csv_directory, directory_error);
  }
  std::ofstream csv(options.csv_path);
  if (!csv) {
    std::fprintf(stderr, "Error: cannot write %s\n", options.csv_path.c_str());
    return EXIT_FAILURE;
  }
  csv << "image,width,height,gpu_upload_ms,gpu_compute_ms,gpu_download_ms,"
         "gpu_total_ms,hysteresis_launches,cpu_ms,speedup,mismatched_pixels,"
         "edge_pixels\n";

  std::printf("CUDA Canny edge detector\n");
  std::printf("  device     : %s\n", canny::DescribeGpu().c_str());
  std::printf("  input      : %s (%zu images)\n", options.input_dir.c_str(),
              paths.size());
  std::printf("  parameters : sigma %.2f, low %.2f, high %.2f, block %dx%d, "
              "repeat %d\n\n",
              options.params.sigma, options.params.low_ratio,
              options.params.high_ratio, options.block_size,
              options.block_size, options.repeat);

  canny::GpuCanny detector(options.block_size);

  // The first kernel launches pay for CUDA context creation and module
  // loading. One untimed run keeps that cost out of the measurements.
  canny::Image warmup_input;
  if (canny::LoadRgbImage(paths.front().string(), &warmup_input, &error)) {
    canny::Image warmup_edges;
    detector.Detect(warmup_input, options.params, &warmup_edges, nullptr);
  }

  BatchSummary summary;
  for (const fs::path& path : paths) {
    if (!ProcessImage(path, options, &detector, &csv, &summary)) {
      ++summary.failed;
    }
  }

  PrintSummary(options, summary);
  return summary.images > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
