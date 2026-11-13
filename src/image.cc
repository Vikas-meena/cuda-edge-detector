// Copyright 2026. Licensed under the MIT License.

#include "image.h"

#include <cstring>

// stb is a third-party, warning-heavy, header-only library. Its
// implementation is compiled only in this translation unit.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace canny {

namespace {
constexpr int kRgbChannels = 3;
}  // namespace

bool LoadRgbImage(const std::string& path, Image* image, std::string* error) {
  int width = 0;
  int height = 0;
  int file_channels = 0;
  stbi_uc* data =
      stbi_load(path.c_str(), &width, &height, &file_channels, kRgbChannels);
  if (data == nullptr) {
    if (error != nullptr) *error = stbi_failure_reason();
    return false;
  }
  image->width = width;
  image->height = height;
  image->channels = kRgbChannels;
  image->pixels.resize(image->ByteCount());
  std::memcpy(image->pixels.data(), data, image->ByteCount());
  stbi_image_free(data);
  return true;
}

bool SavePng(const std::string& path, const Image& image) {
  if (image.Empty()) return false;
  const int stride = image.width * image.channels;
  return stbi_write_png(path.c_str(), image.width, image.height,
                        image.channels, image.pixels.data(), stride) != 0;
}

}  // namespace canny
