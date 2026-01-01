#include "espcn/config.h"
#include "espcn/eval.h"
#include "espcn/train.h"
#include "espcn/tune.h"

#include "CLI11.hpp"

#include <iostream>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace
{

	template <typename T>
	T parse_num(const std::string& s)
	{
		if constexpr (std::is_same_v<T, int>)
			return std::stoi(s);
		if constexpr (std::is_same_v<T, double>)
			return std::stod(s);
		if constexpr (std::is_same_v<T, std::uint64_t>)
			return static_cast<std::uint64_t>(std::stoull(s));
		throw std::runtime_error("unsupported parse type");
	}

	bool parse_bool(const std::string& s)
	{
		if (s == "1" || s == "true" || s == "True" || s == "yes")
			return true;
		if (s == "0" || s == "false" || s == "False" || s == "no")
			return false;
		return false;
	}

	void apply_kv(espcn::Config& cfg, const std::unordered_map<std::string, std::string>& kv)
	{
		auto get = [&](const char* k) -> std::optional<std::string> {
			auto it = kv.find(k);
			if (it == kv.end())
				return std::nullopt;
			if (it->second.empty())
				return std::string("1");
			return it->second;
		};

		if (auto v = get("dataset-root"))
			cfg.dataset_root = *v;
		if (auto v = get("img-size"))
			cfg.img_size = parse_num<int>(*v);
		if (auto v = get("upscale"))
			cfg.upscale = parse_num<int>(*v);
		if (auto v = get("separable"))
			cfg.separable = parse_bool(*v);
		if (auto v = get("dropout"))
			cfg.dropout = parse_num<double>(*v);
		if (auto v = get("channels-last"))
			cfg.channels_last = parse_bool(*v);

		if (auto v = get("epochs"))
			cfg.epochs = parse_num<int>(*v);
		if (auto v = get("batch"))
			cfg.batch = parse_num<int>(*v);
		if (auto v = get("test-batch"))
			cfg.test_batch = parse_num<int>(*v);
		if (auto v = get("lr"))
			cfg.lr = parse_num<double>(*v);
		if (auto v = get("weight-decay"))
			cfg.weight_decay = parse_num<double>(*v);
		if (auto v = get("momentum"))
			cfg.momentum = parse_num<double>(*v);
		if (auto v = get("optimizer"))
			cfg.optimizer = *v;
		if (auto v = get("scheduler"))
			cfg.scheduler = parse_bool(*v);
		if (auto v = get("onecycle-pct-start"))
			cfg.onecycle_pct_start = parse_num<double>(*v);
		if (auto v = get("onecycle-div-factor"))
			cfg.onecycle_div_factor = parse_num<double>(*v);
		if (auto v = get("onecycle-final-div-factor"))
			cfg.onecycle_final_div_factor = parse_num<double>(*v);

		if (auto v = get("psnr-delta"))
			cfg.psnr_delta = parse_num<double>(*v);
		if (auto v = get("stuck-level"))
			cfg.stuck_level = parse_num<int>(*v);
		if (auto v = get("target-min-psnr"))
			cfg.target_min_psnr = parse_num<double>(*v);

		if (auto v = get("model-dir"))
			cfg.model_dir = *v;
		if (auto v = get("checkpoint-frequency"))
			cfg.checkpoint_frequency = parse_num<int>(*v);
		if (auto v = get("checkpoint")) 
			cfg.checkpoint = *v;

		if (auto v = get("threads"))
			cfg.threads = parse_num<int>(*v);
		if (auto v = get("num-workers"))
			cfg.num_workers = parse_num<int>(*v);
		if (auto v = get("cuda"))
			cfg.cuda = parse_bool(*v);
		if (auto v = get("mixed-precision"))
			cfg.mixed_precision = parse_bool(*v);
		if (auto v = get("seed"))
			cfg.seed = parse_num<std::uint64_t>(*v);

		if (auto v = get("trials"))
			cfg.trials = parse_num<int>(*v);
		if (auto v = get("tune-batches"))
			cfg.tune_batches = parse_num<int>(*v);
	}

	void print_help()
	{
		std::cout << "Usage: espcn_libtorch <train|tune|eval> [--key value]\n\n";
		std::cout << "Common options:\n";
		std::cout << "  --dataset-root ./dataset/BSDS500/images\n";
		std::cout << "  --upscale 2|3|4|8\n";
		std::cout << "  --img-size 256\n";
		std::cout << "  --separable 0|1\n";
		std::cout << "  --channels-last 0|1\n";
		std::cout << "  --cuda 0|1\n";
		std::cout << "\nTrain options:\n";
		std::cout << "  --epochs 2000 --batch 16 --test-batch 8 --lr 1e-3 --weight-decay 5e-5\n";
		std::cout << "  --optimizer adam|adamw|sgd\n";
		std::cout << "\nTune options:\n";
		std::cout << "  --trials 150 --tune-batches 64\n";
	}

} // namespace

int main(int argc, const char** argv)
{
	try
	{
		cli11_lite::App app("ESPCN LibTorch");
		auto parsed = app.parse(argc, argv);

		espcn::Config cfg;
		apply_kv(cfg, parsed.kv);

		if (parsed.subcommand == "train")
		{
			return espcn::run_train(cfg);
		}
		if (parsed.subcommand == "tune")
		{
			return espcn::run_tune(cfg);
		}
		if (parsed.subcommand == "eval")
		{
			return espcn::run_eval(cfg);
		}

		print_help();
		return 2;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
