#include "espcn/scheduler.h"

#include "espcn/common.h"

#include <algorithm>
#include <cmath>

namespace espcn {

static double cosine_anneal(double start, double end, double pct) {
  // pct in [0,1]
  pct = std::clamp(pct, 0.0, 1.0);
  const double cos_out = std::cos(M_PI * pct) + 1.0;
  return end + (start - end) / 2.0 * cos_out;
}

OneCycleScheduler::OneCycleScheduler(torch::optim::Optimizer& opt,
                                   double max_lr,
                                   std::size_t total_steps,
                                   double pct_start,
                                   double div_factor,
                                   double final_div_factor)
    : opt_(&opt),
      max_lr_(max_lr),
      initial_lr_(max_lr / div_factor),
      min_lr_(max_lr / final_div_factor),
      total_steps_(std::max<std::size_t>(1, total_steps)),
      step_(0),
      pct_start_(pct_start) {
  ensure(max_lr_ > 0.0, "OneCycle: max_lr must be > 0");
  ensure(div_factor > 0.0, "OneCycle: div_factor must be > 0");
  ensure(final_div_factor > 0.0, "OneCycle: final_div_factor must be > 0");
  ensure(pct_start_ > 0.0 && pct_start_ < 1.0, "OneCycle: pct_start must be in (0,1)");

  set_lr_(*opt_, initial_lr_);
}

void OneCycleScheduler::set_lr_(torch::optim::Optimizer& opt, double lr) {
  for (auto& group : opt.param_groups()) {
    // Try common optimizer options.
    if (auto* adam = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
      adam->lr(lr);
    } else if (auto* adamw = dynamic_cast<torch::optim::AdamWOptions*>(&group.options())) {
      adamw->lr(lr);
    } else if (auto* sgd = dynamic_cast<torch::optim::SGDOptions*>(&group.options())) {
      sgd->lr(lr);
    } else {
      // Fallback: try generic base (not all optimizers expose lr)
    }
  }
}

double OneCycleScheduler::lr_at_(std::size_t step) const {
  const std::size_t s = std::min(step, total_steps_ - 1);
  const double t = static_cast<double>(s) / static_cast<double>(total_steps_ - 1);
  const double cut = pct_start_;

  if (t <= cut) {
    const double pct = t / cut;
    // Warmup: initial -> max
    return cosine_anneal(initial_lr_, max_lr_, pct);
  }

  const double pct = (t - cut) / (1.0 - cut);
  // Cooldown: max -> min
  return cosine_anneal(max_lr_, min_lr_, pct);
}

void OneCycleScheduler::step() {
  const double lr = lr_at_(step_);
  set_lr_(*opt_, lr);
  step_ = std::min(step_ + 1, total_steps_);
}

}  // namespace espcn
