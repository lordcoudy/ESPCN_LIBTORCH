#include "espcn/tune.h"

#include "espcn/common.h"
#include "espcn/dataset.h"
#include "espcn/model.h"

#include <iostream>
#include <random>

namespace espcn
{

	static std::string pick_optimizer(std::mt19937_64& rng)
	{
		// Mirrors Python: Adam or AdamW.
		std::uniform_int_distribution<int> d(0, 1);
		return d(rng) == 0 ? "adam" : "adamw";
	}

	int run_tune(const Config& cfg)
	{
		torch::manual_seed(cfg.seed);
		torch::set_num_threads(cfg.threads);

		auto device = pick_device(cfg.cuda);

		// Cache dataset (like Python tuning cache).
		auto train_ds = Bsd500Dataset(cfg, Bsd500Dataset::Split::Train).map(torch::data::transforms::Stack<>());

		std::mt19937_64 rng(cfg.seed);
		std::uniform_real_distribution<double> log_lr(std::log(1e-5), std::log(1e-3));
		std::uniform_real_distribution<double> log_wd(std::log(1e-5), std::log(1e-4));
		std::uniform_int_distribution<int> bs_pick(0, 2);
		const int bs_choices[3] = { 16, 32, 64 };

		double best_loss = std::numeric_limits<double>::infinity();
		Config best = cfg;

		for (int t = 1; t <= cfg.trials; ++t)
		{
			Config trial = cfg;
			trial.lr = std::exp(log_lr(rng));
			trial.weight_decay = std::exp(log_wd(rng));
			trial.batch = bs_choices[bs_pick(rng)];
			trial.optimizer = pick_optimizer(rng);

			ESPCN model(trial.upscale, trial.separable, trial.dropout);
			model->to(device);
			model->train();

			std::unique_ptr<torch::optim::Optimizer> opt;
			if (trial.optimizer == "adamw")
			{
				opt = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(trial.lr).weight_decay(trial.weight_decay));
			}
			else
			{
				opt = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(trial.lr).weight_decay(trial.weight_decay));
			}

			auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
				train_ds, torch::data::DataLoaderOptions().batch_size(trial.batch).workers(trial.num_workers));

			double sum = 0.0;
			std::size_t it = 0;

			for (auto& batch : *loader)
			{
				auto x = batch.data.to(device, /*non_blocking=*/trial.cuda);
				auto y = batch.target.to(device, /*non_blocking=*/trial.cuda);
				if (trial.channels_last)
				{
					x = x.contiguous(torch::MemoryFormat::ChannelsLast);
					y = y.contiguous(torch::MemoryFormat::ChannelsLast);
				}

				opt->zero_grad();
				auto out = model->forward(x);
				auto loss = torch::mse_loss(out, y);
				loss.backward();
				opt->step();

				sum += loss.item<double>();
				it += 1;

				// Simple pruning: stop early if already much worse than best.
				if (it % 8 == 0 && best_loss < std::numeric_limits<double>::infinity())
				{
					const double cur = sum / static_cast<double>(it);
					if (cur > best_loss * 1.25)
						break;
				}
				if (static_cast<int>(it) >= cfg.tune_batches)
					break;
			}

			const double avg = (it == 0) ? std::numeric_limits<double>::infinity() : (sum / static_cast<double>(it));
			std::cout << "Trial " << t << ": loss=" << avg << " lr=" << trial.lr << " wd=" << trial.weight_decay
					  << " batch=" << trial.batch << " opt=" << trial.optimizer << "\n";

			if (avg < best_loss)
			{
				best_loss = avg;
				best = trial;
			}
		}

		std::cout << "Best: loss=" << best_loss << " lr=" << best.lr << " wd=" << best.weight_decay
				  << " batch=" << best.batch << " opt=" << best.optimizer << "\n";

		return 0;
	}

} // namespace espcn
