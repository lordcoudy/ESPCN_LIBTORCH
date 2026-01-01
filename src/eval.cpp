#include "espcn/eval.h"

#include "espcn/common.h"
#include "espcn/dataset.h"
#include "espcn/model.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <regex>
#include <string>

namespace espcn
{

	static double mse_to_psnr(double mse)
	{
		return 10.0 * std::log10(1.0 / mse);
	}

	static std::string find_latest_checkpoint(const Config& cfg) {
	  const auto dir = std::filesystem::path(cfg.model_dir) / run_name(cfg, /*tuning_tag=*/false);
	  if (!std::filesystem::exists(dir)) {
	    // Fallback: search entire model_dir for a checkpoint matching the requested flags.
	    const auto root = std::filesystem::path(cfg.model_dir);
	    if (!std::filesystem::exists(root)) return "";

	    std::regex re(".*_ckp([0-9]+)\\.pt$");
	    int best_epoch = -1;
	    std::string best;

	    const std::string must_have = std::to_string(cfg.upscale) + "x_epochs(";
	    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
	      if (!entry.is_regular_file()) continue;
	      const auto name = entry.path().filename().string();
	      if (name.find(must_have) == std::string::npos) continue;
	      if (cfg.separable && name.find("_separable") == std::string::npos) continue;
	      std::smatch m;
	      if (!std::regex_match(name, m, re)) continue;
	      const int epoch = std::stoi(m[1].str());
	      if (epoch > best_epoch) {
	        best_epoch = epoch;
	        best = entry.path().string();
	      }
	    }
	    return best;
	  }

	  std::regex re(".*_ckp([0-9]+)\\.pt$");
	  int best_epoch = -1;
	  std::string best;

	  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
	    if (!entry.is_regular_file()) continue;
	    const auto name = entry.path().filename().string();
	    std::smatch m;
	    if (!std::regex_match(name, m, re)) continue;
	    const int epoch = std::stoi(m[1].str());
	    if (epoch > best_epoch) {
	      best_epoch = epoch;
	      best = entry.path().string();
	    }
	  }
	  return best;
	}

	int run_eval(const Config& cfg)
	{
		torch::manual_seed(cfg.seed);
		torch::set_num_threads(cfg.threads);

		auto device = pick_device(cfg.cuda);

		ESPCN model(cfg.upscale, cfg.separable, cfg.dropout);
		model->to(device);
		model->eval();

	  std::string ckpt = cfg.checkpoint;
	  if (ckpt.empty()) ckpt = find_latest_checkpoint(cfg);
	  if (!ckpt.empty()) {
	    std::cout << "Loading checkpoint: " << ckpt << "\n";
	    torch::load(model, ckpt);
	    model->to(device);
	    model->eval();
	  } else {
	    std::cout << "Warning: no checkpoint provided/found; evaluating randomly initialized model.\n";
	  }

		auto test = Bsd500Dataset(cfg, Bsd500Dataset::Split::Test).map(torch::data::transforms::Stack<>());
		auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			  std::move(test), torch::data::DataLoaderOptions().batch_size(cfg.test_batch).workers(cfg.num_workers));

		double sum_psnr = 0.0;
		double max_mse = 0.0;
		double min_mse = std::numeric_limits<double>::infinity();
		std::size_t n = 0;

		torch::InferenceMode guard(true);
		for (auto& batch : *loader)
		{
			auto x = batch.data.to(device, /*non_blocking=*/cfg.cuda);
			auto y = batch.target.to(device, /*non_blocking=*/cfg.cuda);

			if (cfg.channels_last)
			{
				x = x.contiguous(torch::MemoryFormat::ChannelsLast);
				y = y.contiguous(torch::MemoryFormat::ChannelsLast);
			}

			auto out = model->forward(x);
			auto mse = torch::mse_loss(out, y).item<double>();
			max_mse = std::max(max_mse, mse);
			min_mse = std::min(min_mse, mse);
			sum_psnr += mse_to_psnr(mse);
			n += 1;
		}

		const double avg = (n == 0) ? 0.0 : (sum_psnr / static_cast<double>(n));
		std::cout << "Avg PSNR: " << avg << " dB\n";
		std::cout << "Max MSE:  " << max_mse << "\n";
		std::cout << "Min MSE:  " << min_mse << "\n";
		return 0;
	}

} // namespace espcn
