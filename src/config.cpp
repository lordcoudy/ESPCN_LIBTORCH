#include "espcn/config.h"

#include <cctype>
#include <sstream>

namespace espcn {

static std::string optimizer_tag(const std::string& opt) {
	if (opt.empty()) return "";
	std::string o = opt;
	for (auto& c : o) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (o == "adam") return "Adam";
	if (o == "adamw") return "AdamW";
	if (o == "sgd") return "SGD";
	return opt;
}

std::string run_name(const Config& cfg, bool tuning_tag) {
	std::ostringstream ss;
	ss << cfg.upscale << "x_epochs(" << cfg.epochs << ")";
	if (cfg.cuda) ss << "_cuda";
	if (tuning_tag) ss << "_tuning";
	if (cfg.mixed_precision) ss << "_mixed_precision";
	if (cfg.scheduler) ss << "_with_scheduler";
	if (cfg.separable) ss << "_separable";
	if (!cfg.optimizer.empty()) ss << "_optimizer(" << optimizer_tag(cfg.optimizer) << ")";
	ss << "_seed(" << cfg.seed << ")_batch_size(" << cfg.batch << ")";
	return ss.str();
}

}  // namespace espcn
