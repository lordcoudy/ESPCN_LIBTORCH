#include "espcn/train.h"

#include "espcn/common.h"
#include "espcn/dataset.h"
#include "espcn/model.h"
#include "espcn/scheduler.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

#include <ATen/autocast_mode.h>
#include <torch/cuda.h>

#if __has_include(<torch/cuda/amp/grad_scaler.h>)
#include <torch/cuda/amp/grad_scaler.h>
#define ESPCN_HAS_CUDA_AMP_GRAD_SCALER 1
#else
#define ESPCN_HAS_CUDA_AMP_GRAD_SCALER 0
#endif

namespace espcn {

class AutocastGuard {
 public:
  AutocastGuard(at::DeviceType device_type, bool enabled, at::ScalarType dtype)
      : device_type_(device_type), prev_enabled_(at::autocast::is_autocast_enabled(device_type)), prev_dtype_(at::autocast::get_autocast_dtype(device_type)) {
    at::autocast::set_autocast_enabled(device_type_, enabled);
    at::autocast::set_autocast_dtype(device_type_, dtype);
  }

  ~AutocastGuard() {
    at::autocast::set_autocast_enabled(device_type_, prev_enabled_);
    at::autocast::set_autocast_dtype(device_type_, prev_dtype_);
  }

  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;

 private:
  at::DeviceType device_type_;
  bool prev_enabled_;
  at::ScalarType prev_dtype_;
};

static double mse_to_psnr(double mse) {
  return 10.0 * std::log10(1.0 / mse);
}

static void save_checkpoint(const Config& cfg, const ESPCN& model, int epoch) {
  const auto out_dir = std::filesystem::path(cfg.model_dir) / run_name(cfg, /*tuning_tag=*/false);
  std::filesystem::create_directories(out_dir);
  const std::string path = (out_dir / (run_name(cfg, false) + std::string("_ckp") + std::to_string(epoch) + ".pt")).string();
  torch::save(model, path);
  std::cout << "  Saved checkpoint: " << path << "\n";
}

int run_train(const Config& cfg) {
  torch::manual_seed(cfg.seed);
  torch::set_num_threads(cfg.threads);

  auto device = pick_device(cfg.cuda);

  ESPCN model(cfg.upscale, cfg.separable, cfg.dropout);
  model->to(device);

  if (!cfg.checkpoint.empty()) {
    std::cout << "Loading checkpoint: " << cfg.checkpoint << "\n";
    torch::load(model, cfg.checkpoint);
    model->to(device);
  }

  std::unique_ptr<torch::optim::Optimizer> opt;
  if (cfg.optimizer == "sgd") {
    opt = std::make_unique<torch::optim::SGD>(model->parameters(), torch::optim::SGDOptions(cfg.lr).momentum(cfg.momentum).weight_decay(cfg.weight_decay));
  } else if (cfg.optimizer == "adamw") {
    opt = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else {
    opt = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(cfg.lr).weight_decay(cfg.weight_decay));
  }

  auto train_ds = Bsd500Dataset(cfg, Bsd500Dataset::Split::Train).map(torch::data::transforms::Stack<>());
  auto val_ds = Bsd500Dataset(cfg, Bsd500Dataset::Split::Val).map(torch::data::transforms::Stack<>());
  auto test_ds = Bsd500Dataset(cfg, Bsd500Dataset::Split::Test).map(torch::data::transforms::Stack<>());

  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(train_ds), torch::data::DataLoaderOptions().batch_size(cfg.batch).workers(cfg.num_workers));
  auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(val_ds), torch::data::DataLoaderOptions().batch_size(cfg.test_batch).workers(cfg.num_workers));
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(test_ds), torch::data::DataLoaderOptions().batch_size(cfg.test_batch).workers(cfg.num_workers));

  double best_psnr = -1e9;
  int slowdown = 0;

  // OneCycleLR: mimic Python Settings() usage (step per batch).
  std::unique_ptr<OneCycleScheduler> onecycle;
  if (cfg.scheduler) {
    const auto train_count = Bsd500Dataset(cfg, Bsd500Dataset::Split::Train).size().value();
    const std::size_t steps_per_epoch = (train_count + static_cast<std::size_t>(cfg.batch) - 1) / static_cast<std::size_t>(cfg.batch);
    const std::size_t total_steps = steps_per_epoch * static_cast<std::size_t>(cfg.epochs);
    onecycle = std::make_unique<OneCycleScheduler>(*opt, cfg.lr, total_steps, cfg.onecycle_pct_start, cfg.onecycle_div_factor,
                                                   cfg.onecycle_final_div_factor);
  }

  const bool use_amp = cfg.mixed_precision && device.is_cuda() && ESPCN_HAS_CUDA_AMP_GRAD_SCALER;
  if (cfg.mixed_precision && device.is_cuda() && !ESPCN_HAS_CUDA_AMP_GRAD_SCALER) {
    std::cout << "[warn] CUDA AMP GradScaler headers not available in this LibTorch build; falling back to FP32.\n";
  }

#if ESPCN_HAS_CUDA_AMP_GRAD_SCALER
  torch::cuda::amp::GradScaler scaler;
#endif

  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    model->train();

    double sum_loss = 0.0;
    std::size_t it = 0;

    for (auto& batch : *train_loader) {
      auto x = batch.data.to(device, /*non_blocking=*/cfg.cuda);
      auto y = batch.target.to(device, /*non_blocking=*/cfg.cuda);
      if (cfg.channels_last) {
        x = x.contiguous(torch::MemoryFormat::ChannelsLast);
        y = y.contiguous(torch::MemoryFormat::ChannelsLast);
      }

      opt->zero_grad();
      torch::Tensor loss;
      if (use_amp) {
        AutocastGuard autocast(at::kCUDA, /*enabled=*/true, /*dtype=*/at::kHalf);
        auto out = model->forward(x);
        loss = torch::mse_loss(out, y);
#if ESPCN_HAS_CUDA_AMP_GRAD_SCALER
        scaler.scale(loss).backward();
        scaler.step(*opt);
        scaler.update();
#else
        loss.backward();
        opt->step();
#endif
      } else {
        auto out = model->forward(x);
        loss = torch::mse_loss(out, y);
        loss.backward();
        opt->step();
      }

      if (onecycle) onecycle->step();

      sum_loss += loss.item<double>();
      it += 1;
    }

    // Validation
    model->eval();
    torch::InferenceMode guard(true);
    double sum_val = 0.0;
    std::size_t vit = 0;
    for (auto& batch : *val_loader) {
      auto x = batch.data.to(device, /*non_blocking=*/cfg.cuda);
      auto y = batch.target.to(device, /*non_blocking=*/cfg.cuda);
      if (cfg.channels_last) {
        x = x.contiguous(torch::MemoryFormat::ChannelsLast);
        y = y.contiguous(torch::MemoryFormat::ChannelsLast);
      }
      auto out = model->forward(x);
      sum_val += torch::mse_loss(out, y).item<double>();
      vit += 1;
    }

    const double avg_loss = (it == 0) ? 0.0 : (sum_loss / static_cast<double>(it));
    const double avg_val = (vit == 0) ? 0.0 : (sum_val / static_cast<double>(vit));
    std::cout << "Epoch " << epoch << "/" << cfg.epochs << " loss=" << avg_loss << " val=" << avg_val << "\n";

    // Test PSNR
    double sum_psnr = 0.0;
    std::size_t tn = 0;
    for (auto& batch : *test_loader) {
      auto x = batch.data.to(device, /*non_blocking=*/cfg.cuda);
      auto y = batch.target.to(device, /*non_blocking=*/cfg.cuda);
      if (cfg.channels_last) {
        x = x.contiguous(torch::MemoryFormat::ChannelsLast);
        y = y.contiguous(torch::MemoryFormat::ChannelsLast);
      }
      auto out = model->forward(x);
      const double mse = torch::mse_loss(out, y).item<double>();
      sum_psnr += mse_to_psnr(mse);
      tn += 1;
    }
    const double avg_psnr = (tn == 0) ? 0.0 : (sum_psnr / static_cast<double>(tn));
    std::cout << "  Test avg PSNR: " << avg_psnr << " dB\n";

    const double delta = avg_psnr - best_psnr;
    if (delta < cfg.psnr_delta) {
      slowdown += 1;
    } else {
      slowdown = 0;
    }

    if (avg_psnr > best_psnr) {
      best_psnr = avg_psnr;
      save_checkpoint(cfg, model, epoch);
    } else if (cfg.checkpoint_frequency > 0 && (epoch % cfg.checkpoint_frequency) == 0) {
      save_checkpoint(cfg, model, epoch);
    }

    if (slowdown >= cfg.stuck_level && avg_psnr < cfg.target_min_psnr) {
      std::cerr << "Training appears stuck (PSNR=" << avg_psnr << ").\n";
      return -2;
    }
  }

  return 0;
}

}  // namespace espcn
