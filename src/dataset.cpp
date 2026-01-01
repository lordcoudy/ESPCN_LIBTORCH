#include "espcn/dataset.h"

#include "espcn/common.h"

#include <algorithm>
#include <cctype>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace espcn
{

	static bool is_image(const std::filesystem::path& p)
	{
		const auto ext = p.extension().string();
		std::string e;
		e.resize(ext.size());
		std::transform(ext.begin(), ext.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return e == ".png" || e == ".jpg" || e == ".jpeg";
	}

	std::filesystem::path Bsd500Dataset::split_dir_(const std::string& dataset_root, Split split)
	{
		auto base = std::filesystem::path(dataset_root);
		switch (split)
		{
		case Split::Train: return base / "train";
		case Split::Val: return base / "val";
		case Split::Test: return base / "test";
		}
		return base / "train";
	}

	Bsd500Dataset::Bsd500Dataset(const Config& cfg, Split split) : cfg_(cfg), split_(split)
	{
		const auto dir = split_dir_(cfg_.dataset_root, split_);
		ensure(std::filesystem::exists(dir), "Dataset split dir not found: " + dir.string());

		for (const auto& entry : std::filesystem::directory_iterator(dir))
		{
			if (!entry.is_regular_file())
				continue;
			if (is_image(entry.path()))
				files_.push_back(entry.path());
		}

		ensure(!files_.empty(), "No images found in: " + dir.string());
		std::sort(files_.begin(), files_.end());

		tf_.upscale = cfg_.upscale;
		tf_.crop_size = valid_crop_size(cfg_.img_size, cfg_.upscale);
	}

	torch::optional<std::size_t> Bsd500Dataset::size() const
	{
		return files_.size();
	}

	torch::data::Example<> Bsd500Dataset::get(std::size_t index)
	{
		ensure(index < files_.size(), "Dataset index out of range");

		// Read BGR
		cv::Mat bgr = cv::imread(files_[index].string(), cv::IMREAD_COLOR);
		ensure(!bgr.empty(), "Failed to load image: " + files_[index].string());

		// Convert to Y channel similar to PIL convert('YCbCr').split()[0]
		cv::Mat ycrcb;
		cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);
		std::vector<cv::Mat> ch;
		cv::split(ycrcb, ch);
		cv::Mat y = ch[0];

		cv::Mat target = y.clone();

		// Apply the same random transform to input and target.
		const std::uint64_t seed = splitmix64(cfg_.seed ^ static_cast<std::uint64_t>(index));
		tf_(y, target, seed);

		// To float [0..1]
		y.convertTo(y, CV_32F, 1.0 / 255.0);
		target.convertTo(target, CV_32F, 1.0 / 255.0);

		// To torch tensor: [1,H,W]
		auto in = torch::from_blob(y.data, { y.rows, y.cols, 1 }, torch::kFloat32).clone();
		auto gt = torch::from_blob(target.data, { target.rows, target.cols, 1 }, torch::kFloat32).clone();

		in = in.permute({ 2, 0, 1 });
		gt = gt.permute({ 2, 0, 1 });

		return { in, gt };
	}

} // namespace espcn
