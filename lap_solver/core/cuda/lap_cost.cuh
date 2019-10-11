#pragma once

namespace lap
{
	namespace cuda
	{
		// Kernel for calculating the costs in a row
		template <class TC, typename GETCOST, typename STATE>
		__global__ void getCostRow_kernel(TC *cost, GETCOST getcost, STATE state, int x, int start, int count)
		{
			int y = threadIdx.x + blockIdx.x * blockDim.x;
			if (y < count) cost[y] = getcost(x, y + start, state);
		}

		// Kernel for calculating the costs of a rowsol
		template <class TC, typename GETCOST, typename STATE>
		__global__ void getCost_kernel(TC *cost, GETCOST getcost, STATE state, int *rowsol, int N)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			if (x < N) cost[x] = getcost(x, rowsol[x], state);
		}

	}
}
