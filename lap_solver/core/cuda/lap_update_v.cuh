#pragma once

namespace lap
{
	namespace cuda
	{
		template <class SC, class TC>
		__global__ void updateVSingle_kernel(TC *tt, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			__shared__ SC b_min_cost;
			if (threadIdx.x == 0) b_min_cost = (SC)tt[picked] - v[picked];
			__syncthreads();
			SC min_cost = b_min_cost;

			if (j == picked) taken[picked] = 0;
			else if (taken[j] != 0)
			{
				SC cost_l = (SC)tt[j] - v[j];
				if (cost_l < min_cost) v[j] -= min_cost - cost_l;
			}
		}

		template <class SC, class TC>
		__global__ void updateVMulti_kernel(TC *tt, SC *v, TC *tt2, SC *v2, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			__shared__ SC b_min_cost;
			if (threadIdx.x == 0) b_min_cost = (SC)tt2[picked] - v2[picked];
			__syncthreads();
			SC min_cost = b_min_cost;

			if (taken[j] != 0)
			{
				SC cost_l = (SC)tt[j] - v[j];
				if (cost_l < min_cost) v[j] -= min_cost - cost_l;
			}
		}

		template <class SC, class TC>
		__global__ void updateVMultiStart_kernel(TC *tt, SC *v, int *taken, SC *p_min_cost, int picked)
		{
			SC min_cost = (SC)tt[picked] - v[picked];
			*p_min_cost = min_cost;
			taken[picked] = 0;
		}

		template <class SC, class TC>
		__global__ void updateVMulti_kernel(TC *tt, SC *v, int *taken, SC min_cost, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			if (taken[j] != 0)
			{
				SC cost_l = (SC)tt[j] - v[j];
				if (cost_l < min_cost) v[j] -= min_cost - cost_l;
			}
		}
	}
}
