// Copyright 2026. Licensed under the MIT License.
//
// A minimal owning image container used by both the CPU and GPU code paths.

#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace canny {

// Interleaved 8-bit image. Pixel (x, y), channel c lives at
// pixels[(y * width + x) * channels + c].
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;

  size_t PixelCount() const {
    return static_cast<size_t>(width) * static_cast<size_t>(height);
  }
  size_t ByteCount() const {
    return PixelCount() * static_cast<size_t>(channels);
  }
  bool Empty() const { return pixels.empty(); }
};

// Loads `path` and converts it to 3-channel RGB. Returns false and fills
// `error` when the file cannot be decoded.
bool LoadRgbImage(const std::string& path, Image* image, std::string* error);

// Writes a 1- or 3-channel image as PNG. Returns false on failure.
bool SavePng(const std::string& path, const Image& image);

}  // namespace canny

#endif  // SRC_IMAGE_H_
