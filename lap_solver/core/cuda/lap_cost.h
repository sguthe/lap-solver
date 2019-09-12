#pragma once
#include <iostream>

namespace lap
{
	namespace cuda
	{
		// Wrapper around per-row cost funtion, e.g. CUDA, OpenCL or OpenMPI
		// Requires separate kernel to calculate costs for a given rowsol
		template <class TC, typename GETCOSTROW, typename GETCOST>
		class RowCostFunction
		{
		protected:
			GETCOSTROW getcostrow;
			GETCOST getcost;
			TC initialEpsilon;
			TC lowerEpsilon;
		public:
			RowCostFunction(GETCOSTROW &getcostrow, GETCOST &getcost) : getcostrow(getcostrow), getcost(getcost), initialEpsilon(0), lowerEpsilon(0) {}
			~RowCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline const TC getLowerEpsilon() const { return lowerEpsilon; }
			__forceinline void setLowerEpsilon(TC eps) { lowerEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end, bool async) const { getcostrow(row, t, stream, x, start, end, async); }
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim) const { getcost(row, stream, rowsol, dim); }
		};

		// Kernel for calculating the costs in a row
		template <class TC, typename GETCOST, typename STATE>
		__global__ void getCostRow_kernel(TC *cost, GETCOST getcost, STATE state, int x, int start, int end)
		{
			int y = start + threadIdx.x + blockIdx.x * blockDim.x;
			if (y >= end) return;

			cost[threadIdx.x + blockIdx.x * blockDim.x] = getcost(x, y, state);
		}

		// Kernel for calculating the costs of a rowsol
		template <class TC, typename GETCOST, typename STATE>
		__global__ void getCost_kernel(TC *cost, GETCOST getcost, STATE state, int *rowsol, int N)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			if (x >= N) return;
			int y = rowsol[x];

			cost[x] = getcost(x, y, state);
		}

		// Wrapper around simple cost function
		template <class TC, typename GETCOST, typename STATE>
		class SimpleCostFunction
		{
		protected:
			GETCOST getcost;
			STATE *state;
			TC initialEpsilon;
			TC lowerEpsilon;
		public:
			SimpleCostFunction(GETCOST &getcost, STATE *state) : getcost(getcost), state(state), initialEpsilon(0), lowerEpsilon(0) {}
			~SimpleCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline const TC getLowerEpsilon() const { return lowerEpsilon; }
			__forceinline void setLowerEpsilon(TC eps) { lowerEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end, bool async) const
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_kernel<<<grid_size, block_size, 0, stream>>>(row, getcost, state[t], x, start, end);
			}
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim) const
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (dim + block_size.x - 1) / block_size.x;
				getCost_kernel<<<grid_size, block_size, 0, stream>>>(row, getcost, state[0], rowsol, dim);
			}
		};
	}
}
