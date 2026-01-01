#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

#include <torch/torch.h>

namespace espcn {

inline void ensure(bool cond, std::string_view msg) {
  if (!cond) throw std::runtime_error(std::string(msg));
}

inline torch::Device pick_device(bool prefer_cuda) {
  if (prefer_cuda && torch::cuda::is_available()) return torch::Device(torch::kCUDA);
  return torch::Device(torch::kCPU);
}

inline std::uint64_t splitmix64(std::uint64_t x) {
  // Deterministic per-index RNG seed mixer.
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline int valid_crop_size(int img_size, int upscale) {
  return img_size - (img_size % upscale);
}

}  // namespace espcn
