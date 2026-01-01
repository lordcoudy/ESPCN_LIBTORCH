#pragma once

#include <cstddef>

#include <torch/torch.h>

#include "espcn/config.h"

namespace espcn {

class OneCycleScheduler {
 public:
  OneCycleScheduler(torch::optim::Optimizer& opt,
                   double max_lr,
                   std::size_t total_steps,
                   double pct_start = 0.3,
                   double div_factor = 25.0,
                   double final_div_factor = 1e4);

  void step();

 private:
  torch::optim::Optimizer* opt_{};
  double max_lr_{};
  double initial_lr_{};
  double min_lr_{};
  std::size_t total_steps_{};
  std::size_t step_{};
  double pct_start_{};

  static void set_lr_(torch::optim::Optimizer& opt, double lr);
  [[nodiscard]] double lr_at_(std::size_t step) const;
};

}  // namespace espcn
