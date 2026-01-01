#include "espcn/icnr.h"

#include "espcn/common.h"

#include <torch/nn/init.h>

namespace espcn
{

	void icnr_(torch::Tensor weight, int upscale)
	{
		torch::NoGradGuard guard;
		ensure(weight.defined(), "ICNR: weight is undefined");
		ensure(weight.dim() == 4, "ICNR: expected 4D conv weight tensor");
		const int64_t out_ch = weight.size(0);
		const int64_t r2 = static_cast<int64_t>(upscale) * static_cast<int64_t>(upscale);
		ensure(out_ch % r2 == 0, "ICNR: out_channels must be divisible by upscale^2");

		auto sub = torch::empty({ out_ch / r2, weight.size(1), weight.size(2), weight.size(3) }, weight.options());
		torch::nn::init::kaiming_normal_(sub);

		// Repeat along dim0: [out_ch/r2, ...] -> [out_ch, ...]
		auto repeated = sub.repeat_interleave(r2, 0);
		ensure(repeated.sizes() == weight.sizes(), "ICNR: repeated shape mismatch");
		weight.copy_(repeated);
	}

} // namespace espcn
