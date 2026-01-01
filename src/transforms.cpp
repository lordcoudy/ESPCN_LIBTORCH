#include "espcn/transforms.h"

#include "espcn/common.h"

#include <opencv2/imgproc.hpp>

namespace espcn
{

	static cv::Mat rotate_keep_size(const cv::Mat& src, double degrees)
	{
		const cv::Point2f center(static_cast<float>(src.cols) / 2.0F, static_cast<float>(src.rows) / 2.0F);
		cv::Mat rot = cv::getRotationMatrix2D(center, degrees, 1.0);
		cv::Mat dst;
		cv::warpAffine(src, dst, rot, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
		return dst;
	}

	static cv::Mat center_crop(const cv::Mat& src, int size)
	{
		ensure(size > 0, "center_crop: size must be > 0");
		ensure(src.cols >= size && src.rows >= size, "center_crop: image smaller than crop");
		const int x0 = (src.cols - size) / 2;
		const int y0 = (src.rows - size) / 2;
		return src(cv::Rect(x0, y0, size, size)).clone();
	}

	void RandomTransform::operator()(cv::Mat& y_img, cv::Mat& y_target, std::uint64_t seed) const
	{
		ensure(!y_img.empty() && !y_target.empty(), "RandomTransform: empty input");

		std::mt19937_64 rng(seed);
		std::uniform_real_distribution<double> uni01(0.0, 1.0);
		std::uniform_real_distribution<double> rot(rot_deg_min, rot_deg_max);

		if (uni01(rng) > 0.5)
		{
			cv::flip(y_img, y_img, 1);
			cv::flip(y_target, y_target, 1);
		}
		if (uni01(rng) > 0.5)
		{
			cv::flip(y_img, y_img, 0);
			cv::flip(y_target, y_target, 0);
		}

		const double angle = rot(rng);
		y_img = rotate_keep_size(y_img, angle);
		y_target = rotate_keep_size(y_target, angle);

		// Center crop
		y_img = center_crop(y_img, crop_size);
		y_target = center_crop(y_target, crop_size);

		// Resize
		const int lr = crop_size / upscale;
		ensure(lr > 0, "RandomTransform: invalid lr size");

		cv::resize(y_img, y_img, cv::Size(lr, lr), 0, 0, cv::INTER_LANCZOS4);
		cv::resize(y_target, y_target, cv::Size(crop_size, crop_size), 0, 0, cv::INTER_LANCZOS4);
	}

} // namespace espcn
