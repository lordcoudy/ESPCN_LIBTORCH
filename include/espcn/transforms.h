#pragma once

#include <cstdint>
#include <random>

#include <opencv2/core.hpp>

namespace espcn {

struct RandomTransform {
  int crop_size = 256;
  int upscale = 2;
  double rot_deg_min = -10.0;
  double rot_deg_max = 10.0;

  // Applies in-place: flip/flip/rotate/center-crop/resize.
  void operator()(cv::Mat& y_img, cv::Mat& y_target, std::uint64_t seed) const;
};

}  // namespace espcn
