#pragma once

#include <string>

namespace espcn {

struct Config {
  // Data
  std::string dataset_root = "./dataset/BSDS500/images";  // contains train/val/test
  int img_size = 256;

  // Model
  int upscale = 2;
  bool separable = false;
  double dropout = 0.5;
  bool channels_last = false;

  // Train
  int epochs = 10;
  int batch = 16;
  int test_batch = 8;
  double lr = 1e-3;
  double weight_decay = 5e-5;
  double momentum = 0.95;
  std::string optimizer = "adam";  // adam|adamw|sgd
  bool scheduler = false;
  double onecycle_pct_start = 0.3;
  double onecycle_div_factor = 25.0;
  double onecycle_final_div_factor = 1e4;

  // Re-run / stuck detection
  double psnr_delta = 0.0;
  int stuck_level = 30;
  double target_min_psnr = 26.0;

  // Checkpoints
  std::string model_dir = "./models";
  int checkpoint_frequency = 100;
  std::string checkpoint = "";  // explicit checkpoint path to load

  // Runtime
  int threads = 8;
  int num_workers = 0;
  bool cuda = false;
  bool mixed_precision = false;  // CUDA only
  std::uint64_t seed = 123;

  // Tuning
  int trials = 30;
  int tune_batches = 64;  // batches per trial
};

// Produces a stable, Python-like run name used for outputs/checkpoints.
std::string run_name(const Config& cfg, bool tuning_tag);

}  // namespace espcn
