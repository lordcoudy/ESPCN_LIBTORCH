#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "espcn/config.h"
#include "espcn/transforms.h"

namespace espcn
{

	class Bsd500Dataset final : public torch::data::datasets::Dataset<Bsd500Dataset>
	{
	public:
		enum class Split
		{
			Train,
			Val,
			Test
		};

		Bsd500Dataset(const Config& cfg, Split split);

		torch::data::Example<> get(std::size_t index) override;
		torch::optional<std::size_t> size() const override;

	private:
		Config cfg_;
		Split split_;
		std::vector<std::filesystem::path> files_;
		RandomTransform tf_;

		static std::filesystem::path split_dir_(const std::string& dataset_root, Split split);
	};

} // namespace espcn
