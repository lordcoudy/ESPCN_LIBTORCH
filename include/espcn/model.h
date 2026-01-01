#pragma once

#include <torch/torch.h>

namespace espcn
{

	struct ESPCNImpl final : torch::nn::Module
	{
		static constexpr int kMaxUp = 8;

		ESPCNImpl(int upscale, bool separable, double dropout);

		torch::Tensor forward(torch::Tensor x);

		int upscale = 2;
		bool separable = false;
		double dropout = 0.5;

		torch::nn::Conv2d conv1{ nullptr };
		torch::nn::Conv2d conv2_dw{ nullptr };
		torch::nn::Conv2d conv2_pw{ nullptr };
		torch::nn::Conv2d conv2{ nullptr };
		torch::nn::Conv2d conv3_dw{ nullptr };
		torch::nn::Conv2d conv3_pw{ nullptr };
		torch::nn::Conv2d conv3{ nullptr };
		torch::nn::Conv2d conv4{ nullptr };
		torch::nn::PixelShuffle pixel_shuffle{ nullptr };
		torch::nn::Conv2d smoothing{ nullptr };

	private:
		void init_weights_();
		torch::Tensor reflect_pad_(torch::Tensor x, int pad) const;
	};

	TORCH_MODULE(ESPCN);

} // namespace espcn
