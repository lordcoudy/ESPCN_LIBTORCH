#pragma once

#include <torch/torch.h>

namespace espcn
{

	// ICNR init for sub-pixel convolution weights.
	// Mirrors ESPCN_PYTORCH/ICNR.py behavior.
	//
	// weight shape: [out_channels, in_channels, kH, kW]
	// out_channels must be divisible by (upscale^2)
	void icnr_(torch::Tensor weight, int upscale);

} // namespace espcn
