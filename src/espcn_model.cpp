#include "espcn/model.h"

#include "espcn/common.h"
#include "espcn/icnr.h"

#include <torch/nn/functional/padding.h>
#include <torch/nn/init.h>

namespace espcn
{

	static void orthogonal_relu_(torch::Tensor w)
	{
		const double gain = torch::nn::init::calculate_gain(torch::kReLU);
		torch::nn::init::orthogonal_(w, gain);
	}

	ESPCNImpl::ESPCNImpl(int upscale_, bool separable_, double dropout_)
	{
		upscale = upscale_;
		separable = separable_;
		dropout = dropout_;

		ensure(upscale == 2 || upscale == 3 || upscale == 4 || upscale == 8, "upscale must be 2,3,4,8");

		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).padding(0).bias(true)));

		if (separable)
		{
			conv2_dw = register_module("conv2_dw", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(0).groups(64).bias(true)));
			conv2_pw = register_module("conv2_pw", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).padding(0).bias(true)));

			conv3_dw = register_module("conv3_dw", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(0).groups(64).bias(true)));
			conv3_pw = register_module("conv3_pw", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).padding(0).bias(true)));
		}
		else
		{
			conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(0).bias(true)));
			conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 3).padding(0).bias(true)));
		}

		conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1 * (upscale * upscale), 3).padding(0).bias(true)));
		pixel_shuffle = register_module("pixel_shuffle", torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(upscale)));
		smoothing = register_module("smoothing", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 5).padding(0).bias(true)));

		init_weights_();
	}

	torch::Tensor ESPCNImpl::reflect_pad_(torch::Tensor x, int pad) const
	{
		if (pad <= 0)
			return x;
		// pad format: {left, right, top, bottom}
		return torch::nn::functional::pad(x,
										  torch::nn::functional::PadFuncOptions({ pad, pad, pad, pad }).mode(torch::kReflect));
	}

	void ESPCNImpl::init_weights_()
	{
		torch::NoGradGuard guard;
		orthogonal_relu_(conv1->weight);

		if (separable)
		{
			orthogonal_relu_(conv2_dw->weight);
			orthogonal_relu_(conv2_pw->weight);
			orthogonal_relu_(conv3_dw->weight);
			orthogonal_relu_(conv3_pw->weight);
		}
		else
		{
			orthogonal_relu_(conv2->weight);
			orthogonal_relu_(conv3->weight);
		}

		icnr_(conv4->weight, upscale);

		// Python: orthogonal_(smoothing.weight, 1/(5*5)), bias=0
		torch::nn::init::orthogonal_(smoothing->weight, 1.0 / (5.0 * 5.0));
		torch::nn::init::constant_(smoothing->bias, 0.0);
	}

	torch::Tensor ESPCNImpl::forward(torch::Tensor x)
	{
		// x: [N,1,H,W]
		x = torch::relu(conv1->forward(reflect_pad_(x, 2)));
		x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout).training(is_training()));

		if (separable)
		{
			x = torch::relu(conv2_pw->forward(conv2_dw->forward(reflect_pad_(x, 1))));
		}
		else
		{
			x = torch::relu(conv2->forward(reflect_pad_(x, 1)));
		}
		x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout).training(is_training()));

		if (separable)
		{
			x = torch::relu(conv3_pw->forward(conv3_dw->forward(reflect_pad_(x, 1))));
		}
		else
		{
			x = torch::relu(conv3->forward(reflect_pad_(x, 1)));
		}
		x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout).training(is_training()));

		x = conv4->forward(reflect_pad_(x, 1));
		x = pixel_shuffle->forward(x);
		x = smoothing->forward(reflect_pad_(x, 2));
		return x;
	}

} // namespace espcn
