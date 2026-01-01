#pragma once

#include <cstddef>
#include <limits>

namespace espcn
{

	struct PsnrStats
	{
		double avg_psnr = 0.0;
		double max_mse = 0.0;
		double min_mse = std::numeric_limits<double>::infinity();
		std::size_t count = 0;
	};

} // namespace espcn
