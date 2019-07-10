#pragma once

#include "../lap_solver.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef LAP_CUDA_OPENMP
#include <omp.h>
#endif

namespace lap
{
	namespace cuda
	{
		__device__ __forceinline__ void atomicMinExt(float *addr, float value) {
			if (value >= 0) atomicMin((int *)addr, __float_as_int(value));
			else atomicMax((unsigned int *)addr, __float_as_uint(value));
		}

		__device__ __forceinline__ void atomicMaxExt(float *addr, float value) {
			if (value >= 0) atomicMax((int *)addr, __float_as_int(value));
			else atomicMin((unsigned int *)addr, __float_as_uint(value));
		}

		__device__ __forceinline__ void atomicMinExt(double *addr, double value) {
			if (value >= 0) atomicMin((long long *)addr, __double_as_longlong(value));
			else atomicMax((unsigned long long *)addr, (unsigned long long)__double_as_longlong(value));
		}

		__device__ __forceinline__ void atomicMaxExt(double *addr, double value) {
			if (value >= 0) atomicMax((long long *)addr, __double_as_longlong(value));
			else atomicMin((unsigned long long *)addr, (unsigned long long)__double_as_longlong(value));
		}

		__device__ __forceinline__ void atomicMinExt(int *addr, int value) {
			atomicMin(addr, value);
		}

		__device__ __forceinline__ void atomicMaxExt(int *addr, int value) {
			atomicMax(addr, value);
		}

		__device__ __forceinline__ void atomicMinExt(long long *addr, long long value) {
			atomicMin(addr, value);
		}

		__device__ __forceinline__ void atomicMaxExt(long long *addr, long long value) {
			atomicMax(addr, value);
		}

		template <class C>
		__device__ __forceinline__ void minWarp(C &value)
		{
			C value2 = __shfl_xor_sync(0xffffffff, value, 1, 32);
			if (value2 < value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 2, 32);
			if (value2 < value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 4, 32);
			if (value2 < value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 8, 32);
			if (value2 < value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 16, 32);
			if (value2 < value) value = value2;
		}

		template <class C>
		__device__ __forceinline__ void maxWarp(C &value)
		{
			C value2 = __shfl_xor_sync(0xffffffff, value, 1, 32);
			if (value2 > value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 2, 32);
			if (value2 > value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 4, 32);
			if (value2 > value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 8, 32);
			if (value2 > value) value = value2;
			value2 = __shfl_xor_sync(0xffffffff, value, 16, 32);
			if (value2 > value) value = value2;
		}

		template <class C>
		__device__ __forceinline__ void minWarp8(C &value)
		{
			C value2 = __shfl_xor_sync(0xff, value, 1, 32);
			if (value2 < value) value = value2;
			value2 = __shfl_xor_sync(0xff, value, 2, 32);
			if (value2 < value) value = value2;
			value2 = __shfl_xor_sync(0xff, value, 4, 32);
			if (value2 < value) value = value2;
		}

		template <class C>
		__device__ __forceinline__ void maxWarp8(C &value)
		{
			C value2 = __shfl_xor_sync(0xff, value, 1, 32);
			if (value2 > value) value = value2;
			value2 = __shfl_xor_sync(0xff, value, 2, 32);
			if (value2 > value) value = value2;
			value2 = __shfl_xor_sync(0xff, value, 4, 32);
			if (value2 > value) value = value2;
		}

		template <class C>
		__device__ __forceinline__ void atomicMinWarp(C *addr, C value)
		{
			int laneId = threadIdx.x & 0x1f;
			minWarp(value);
			if (laneId == 0) atomicMinExt(addr, value);
		}

		template <class C>
		__device__ __forceinline__ void atomicMinWarp(C *addr, C value, C invalid)
		{
			int laneId = threadIdx.x & 0x1f;
			minWarp(value);
			if ((laneId == 0) && (value != invalid)) atomicMinExt(addr, value);
		}

		template <class C>
		__device__ __forceinline__ void atomicMaxWarp(C *addr, C value)
		{
			int laneId = threadIdx.x & 0x1f;
			maxWarp(value);
			if (laneId == 0) atomicMaxExt(addr, value);
		}

		template <class C>
		__device__ __forceinline__ void atomicMaxWarp(C *addr, C value, C invalid)
		{
			int laneId = threadIdx.x & 0x1f;
			maxWarp(value);
			if ((laneId == 0) && (value != invalid)) atomicMaxExt(addr, value);
		}

		template <class C>
		__device__ __forceinline__ void minWarpIndex(C &value, int &index)
		{
			C old_val = value;
			minWarp(value);
			if (old_val != value) index = 0x7fffffff;
			minWarp(index);
		}

		template <class C>
		__device__ __forceinline__ void minWarpIndex8(C &value, int &index)
		{
			C old_val = value;
			minWarp8(value);
			if (old_val != value) index = 0x7fffffff;
			minWarp8(index);
		}

		template <class C>
		__device__ __forceinline__ void minWarpIndex(C &value, int &index, int &old)
		{
			C old_val = value;
			minWarp(value);
			bool active = ((old < 0) && (old_val == value));
			int mask = __ballot_sync(0xffffffff, active);
			if (mask == 0)
			{
				active = (old_val == value);
				mask = __ballot_sync(0xffffffff, active);
			}
			int old_index = index;
			if (!active)
			{
				index = 0x7fffffff;
			}
			minWarp(index);
			int first = __ffs(__ballot_sync(0xffffffff, old_index == index)) - 1;
			old = __shfl_sync(0xffffffff, old, first, 32);
		}

		template <class C>
		__device__ __forceinline__ void minWarpIndex8(C &value, int &index, int &old)
		{
			C old_val = value;
			minWarp8(value);
			bool active = ((old < 0) && (old_val == value));
			int mask = __ballot_sync(0xff, active);
			if (mask == 0)
			{
				active = (old_val == value);
				mask = __ballot_sync(0xff, active);
			}
			int old_index = index;
			if (!active)
			{
				index = 0x7fffffff;
			}
			minWarp8(index);
			int first = __ffs(__ballot_sync(0xff, old_index == index)) - 1;
			old = __shfl_sync(0xff, old, first, 32);
		}

		template <class TC>
		__global__ void initMinMax_kernel(TC *minmax, TC min, TC max, int size)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;

			if (x >= size) return;
			x += x;
			minmax[x++] = min;
			minmax[x] = max;
		}

		template <class TC, class TC_IN>
		__global__ void minMax_kernel(TC *minmax, TC_IN *in, TC min, TC max, int size)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;

			TC v_min, v_max;
			v_min = min;
			v_max = max;
#pragma unroll 8
			while (x < size)
			{
				TC v = in[x];
				if (v < v_min) v_min = v;
				if (v > v_max) v_max = v;
				x += blockDim.x * gridDim.x;
			}

			atomicMinWarp(&(minmax[0]), v_min, min);
			atomicMaxWarp(&(minmax[1]), v_max, max);
		}

		template <class SC>
		class estimateEpsilon_struct
		{
		public:
			SC min;
			SC max;
			SC picked;
			int jmin;
			SC v_jmin;
		};

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			SC max;
			int jmin;
			int colsol;
		};

		template <class SC, class TC>
		__global__ void getMinMaxBest_kernel(SC *o_min_cost, SC *o_max_cost, SC *o_picked_cost, int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost = max;
			SC t_max_cost = min;
			SC t_picked_cost = max;
			int t_jmin = dim2;

#pragma unroll 8
			while (j < size)
			{
				SC t_cost = (SC)tt[j];
				if (t_cost < t_min_cost) t_min_cost = t_cost;
				if (i == j) t_max_cost = t_cost;
				if ((t_cost < t_picked_cost) && (picked[j] == 0))
				{
					t_jmin = j;
					t_picked_cost = t_cost;
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_picked_cost, t_jmin);
			minWarp(t_min_cost);
			maxWarp(t_max_cost);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min_cost[i] = t_min_cost;
				o_max_cost[i] = t_max_cost;
				o_picked_cost[i] = t_picked_cost;
				o_jmin[i] = t_jmin;
			}
		}

		template <class SC, class TC>
		__global__ void getMinSecondBest_kernel(SC *o_min_cost, SC *o_max_cost, SC *o_picked_cost, int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost = max;
			SC t_second_cost = max;
			SC t_picked_cost = max;
			int t_jmin = dim2;

#pragma unroll 8
			while (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				if (t_cost < t_min_cost)
				{
					t_second_cost = t_min_cost;
					t_min_cost = t_cost;
				}
				else if (t_cost < t_second_cost) t_second_cost = t_cost;
				if ((t_cost < t_picked_cost) && (picked[j] == 0))
				{
					t_jmin = j;
					t_picked_cost = t_cost;
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_picked_cost, t_jmin);
			SC old_min_cost = t_min_cost;
			minWarp(t_min_cost);
			if (t_min_cost < old_min_cost) t_second_cost = old_min_cost;
			minWarp(t_second_cost);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min_cost[i] = t_min_cost;
				o_max_cost[i] = t_second_cost;
				o_picked_cost[i] = t_picked_cost;
				o_jmin[i] = t_jmin;
			}
		}

		template <class SC, class TC>
		__global__ void getMinimalCost_kernel(SC *o_min_cost, int *o_jmin, SC *o_min_cost_real, TC *tt, SC *v, int *taken, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost_real = max;
			SC t_min_cost = max;
			int t_jmin = dim2;

#pragma unroll 8
			while (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				if (taken[j] == 0)
				{
					if (t_cost < t_min_cost)
					{
						t_min_cost = t_cost;
						t_jmin = j;
					}
				}
				if (t_cost < t_min_cost_real) t_min_cost_real = t_cost;
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_min_cost, t_jmin);
			minWarp(t_min_cost_real);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min_cost[i] = t_min_cost;
				o_jmin[i] = t_jmin;
				o_min_cost_real[i] = t_min_cost_real;
			}
		}

		template <class SC, class TC>
		__global__ void updateVSingle_kernel(TC *tt, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			SC min_cost = (SC)tt[picked] - v[picked];
			if (j == picked) taken[picked] = 0;
			else if (taken[j] != 0)
			{
				SC cost_l = (SC)tt[j] - v[j];
				if (cost_l < min_cost) v[j] -= min_cost - cost_l;
			}
		}

		template <class SC, class TC>
		__global__ void updateVMultiStart_kernel(TC *tt, SC *v, int *taken, SC *p_min_cost, int picked)
		{
			*p_min_cost = (SC)tt[picked] - v[picked];
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

		template <class SC, class TC>
		__global__ void getFinalCost_kernel(SC *o_min_cost, SC *o_picked_cost, SC *o_picked_v, TC *tt, SC *v, SC max, int j_picked, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost = max;
			SC t_picked_cost = max;
			SC t_picked_v = max;

#pragma unroll 8
			while (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				if (t_cost < t_min_cost) t_min_cost = t_cost;
				if (j == j_picked)
				{
					t_picked_cost = (SC)tt[j];
					t_picked_v = v[j];
				}
				j += blockDim.x * gridDim.x;
			}

			minWarp(t_min_cost);
			minWarp(t_picked_cost);
			minWarp(t_picked_v);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min_cost[i] = t_min_cost;
				o_picked_cost[i] = t_picked_cost;
				o_picked_v[i] = t_picked_v;
			}
		}

		template <class SC>
		__global__ void getMinMaxBestSmall_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *max_cost, SC *picked_cost, int *jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_max_cost = min;
			SC v_picked_cost = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_max_cost = max_cost[j];
				SC c_picked_cost = picked_cost[j];
				int c_jmin = jmin[j];
				if (c_min_cost < v_min_cost) v_min_cost = c_min_cost;
				if (c_max_cost > v_max_cost) v_max_cost = c_max_cost;
				if ((c_picked_cost < v_picked_cost) || ((c_picked_cost == v_picked_cost) && (c_jmin < v_jmin)))
				{
					v_picked_cost = c_picked_cost;
					v_jmin = c_jmin;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_picked_cost, v_jmin);
			minWarp(v_min_cost);
			maxWarp(v_max_cost);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->max = v_max_cost;
				s->picked = v_picked_cost;
				s->jmin = v_jmin;
			}
		}

		template <class SC>
		__global__ void getMinMaxBestMedium_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *max_cost, SC *picked_cost, int *jmin, SC min, SC max, int size, int dim2)
		{
			// 256 threads in 8 warps
			__shared__ SC b_min_cost[8];
			__shared__ SC b_max_cost[8];
			__shared__ SC b_picked_cost[8];
			__shared__ int b_jmin[8];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_max_cost = min;
			SC v_picked_cost = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_max_cost = max_cost[j];
				SC c_picked_cost = picked_cost[j];
				int c_jmin = jmin[j];
				if (c_min_cost < v_min_cost) v_min_cost = c_min_cost;
				if (c_max_cost > v_max_cost) v_max_cost = c_max_cost;
				if ((c_picked_cost < v_picked_cost) || ((c_picked_cost == v_picked_cost) && (c_jmin < v_jmin)))
				{
					v_picked_cost = c_picked_cost;
					v_jmin = c_jmin;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_picked_cost, v_jmin);
			minWarp(v_min_cost);
			maxWarp(v_max_cost);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_max_cost[bidx] = v_max_cost;
				b_picked_cost[bidx] = v_picked_cost;
				b_jmin[bidx] = v_jmin;
			}
			__syncthreads();
			if (threadIdx.x >= 8) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_max_cost = b_max_cost[threadIdx.x];
			v_picked_cost = b_picked_cost[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			minWarpIndex8(v_picked_cost, v_jmin);
			minWarp8(v_min_cost);
			maxWarp8(v_max_cost);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->max = v_max_cost;
				s->picked = v_picked_cost;
				s->jmin = v_jmin;
			}
		}

		template <class SC>
		__global__ void getMinMaxBestLarge_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *max_cost, SC *picked_cost, int *jmin, SC min, SC max, int size, int dim2)
		{
			// 1024 threads in 32 warps
			__shared__ SC b_min_cost[32];
			__shared__ SC b_max_cost[32];
			__shared__ SC b_picked_cost[32];
			__shared__ int b_jmin[32];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_max_cost = min;
			SC v_picked_cost = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_max_cost = max_cost[j];
				SC c_picked_cost = picked_cost[j];
				int c_jmin = jmin[j];
				if (c_min_cost < v_min_cost) v_min_cost = c_min_cost;
				if (c_max_cost > v_max_cost) v_max_cost = c_max_cost;
				if ((c_picked_cost < v_picked_cost) || ((c_picked_cost == v_picked_cost) && (c_jmin < v_jmin)))
				{
					v_picked_cost = c_picked_cost;
					v_jmin = c_jmin;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_picked_cost, v_jmin);
			minWarp(v_min_cost);
			maxWarp(v_max_cost);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_max_cost[bidx] = v_max_cost;
				b_picked_cost[bidx] = v_picked_cost;
				b_jmin[bidx] = v_jmin;
			}
			__syncthreads();
			if (threadIdx.x >= 32) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_max_cost = b_max_cost[threadIdx.x];
			v_picked_cost = b_picked_cost[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			minWarpIndex(v_picked_cost, v_jmin);
			minWarp(v_min_cost);
			maxWarp(v_max_cost);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->max = v_max_cost;
				s->picked = v_picked_cost;
				s->jmin = v_jmin;
			}
		}

		template <class SC>
		__global__ void getMinSecondBestSmall_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *second_cost, SC *picked_cost, int *jmin, SC *v, SC max, int size, int dim2)
		{
			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_second_cost = max;
			SC v_picked_cost = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_second_cost = second_cost[j];
				SC c_picked_cost = picked_cost[j];
				int c_jmin = jmin[j];
				if (c_min_cost < v_min_cost)
				{
					if (v_min_cost < c_second_cost) v_second_cost = v_min_cost;
					else v_second_cost = c_second_cost;
					v_min_cost = c_min_cost;
				}
				else if (c_min_cost < v_second_cost) v_second_cost = c_min_cost;
				if ((c_picked_cost < v_picked_cost) || ((c_picked_cost == v_picked_cost) && (c_jmin < v_jmin)))
				{
					v_picked_cost = c_picked_cost;
					v_jmin = c_jmin;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_picked_cost, v_jmin);
			SC old_min_cost = v_min_cost;
			minWarp(v_min_cost);
			if (v_min_cost < old_min_cost) v_second_cost = old_min_cost;
			minWarp(v_second_cost);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->max = v_second_cost;
				s->picked = v_picked_cost;
				s->jmin = v_jmin;
				if (v_jmin < dim2)
				{
					s->v_jmin = v[v_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class SC>
		__global__ void getMinSecondBestMedium_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *second_cost, SC *picked_cost, int *jmin, SC *v, SC max, int size, int dim2)
		{
			// 256 threads in 8 warps
			__shared__ SC b_min_cost[8];
			__shared__ SC b_second_cost[8];
			__shared__ SC b_picked_cost[8];
			__shared__ int b_jmin[8];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_second_cost = max;
			SC v_picked_cost = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_second_cost = second_cost[j];
				SC c_picked_cost = picked_cost[j];
				int c_jmin = jmin[j];
				if (c_min_cost < v_min_cost)
				{
					if (v_min_cost < c_second_cost) v_second_cost = v_min_cost;
					else v_second_cost = c_second_cost;
					v_min_cost = c_min_cost;
				}
				else if (c_min_cost < v_second_cost) v_second_cost = c_min_cost;
				if ((c_picked_cost < v_picked_cost) || ((c_picked_cost == v_picked_cost) && (c_jmin < v_jmin)))
				{
					v_picked_cost = c_picked_cost;
					v_jmin = c_jmin;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_picked_cost, v_jmin);
			SC old_min_cost = v_min_cost;
			minWarp(v_min_cost);
			if (v_min_cost < old_min_cost) v_second_cost = old_min_cost;
			minWarp(v_second_cost);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_second_cost[bidx] = v_second_cost;
				b_picked_cost[bidx] = v_picked_cost;
				b_jmin[bidx] = v_jmin;
			}
			__syncthreads();
			if (threadIdx.x >= 8) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_second_cost = b_second_cost[threadIdx.x];
			v_picked_cost = b_picked_cost[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			minWarpIndex8(v_picked_cost, v_jmin);
			old_min_cost = v_min_cost;
			minWarp8(v_min_cost);
			if (v_min_cost < old_min_cost) v_second_cost = old_min_cost;
			minWarp8(v_second_cost);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->max = v_second_cost;
				s->picked = v_picked_cost;
				s->jmin = v_jmin;
				if (v_jmin < dim2)
				{
					s->v_jmin = v[v_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class SC>
		__global__ void getMinSecondBestLarge_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *second_cost, SC *picked_cost, int *jmin, SC *v, SC max, int size, int dim2)
		{
			// 1024 threads in 32 warps
			__shared__ SC b_min_cost[32];
			__shared__ SC b_second_cost[32];
			__shared__ SC b_picked_cost[32];
			__shared__ int b_jmin[32];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_second_cost = max;
			SC v_picked_cost = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_second_cost = second_cost[j];
				SC c_picked_cost = picked_cost[j];
				int c_jmin = jmin[j];
				if (c_min_cost < v_min_cost)
				{
					if (v_min_cost < c_second_cost) v_second_cost = v_min_cost;
					else v_second_cost = c_second_cost;
					v_min_cost = c_min_cost;
				}
				else if (c_min_cost < v_second_cost) v_second_cost = c_min_cost;
				if ((c_picked_cost < v_picked_cost) || ((c_picked_cost == v_picked_cost) && (c_jmin < v_jmin)))
				{
					v_picked_cost = c_picked_cost;
					v_jmin = c_jmin;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_picked_cost, v_jmin);
			SC old_min_cost = v_min_cost;
			minWarp(v_min_cost);
			if (v_min_cost < old_min_cost) v_second_cost = old_min_cost;
			minWarp(v_second_cost);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_second_cost[bidx] = v_second_cost;
				b_picked_cost[bidx] = v_picked_cost;
				b_jmin[bidx] = v_jmin;
			}
			__syncthreads();
			if (threadIdx.x >= 32) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_second_cost = b_second_cost[threadIdx.x];
			v_picked_cost = b_picked_cost[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			minWarpIndex(v_picked_cost, v_jmin);
			old_min_cost = v_min_cost;
			minWarp(v_min_cost);
			if (v_min_cost < old_min_cost) v_second_cost = old_min_cost;
			minWarp(v_second_cost);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->max = v_second_cost;
				s->picked = v_picked_cost;
				s->jmin = v_jmin;
				if (v_jmin < dim2)
				{
					s->v_jmin = v[v_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class SC>
		__global__ void getMinimalCostSmall_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, int *jmin, SC *min_cost_real, SC *v, SC max, int size, int dim2)
		{
			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_min_cost_real = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_min_cost_real = min_cost_real[j];
				int c_jmin = jmin[j];
				if ((c_min_cost < v_min_cost) || ((c_min_cost == v_min_cost) && (c_jmin < v_jmin)))
				{
					v_min_cost = c_min_cost;
					v_jmin = c_jmin;
				}
				if (c_min_cost_real < v_min_cost_real) v_min_cost_real = c_min_cost_real;
				j += blockDim.x;
			}
			minWarpIndex(v_min_cost, v_jmin);
			minWarp(v_min_cost_real);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost_real;
				s->picked = v_min_cost;
				s->jmin = v_jmin;
				if (v_jmin < dim2)
				{
					s->v_jmin = v[v_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class SC>
		__global__ void getMinimalCostMedium_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, int *jmin, SC *min_cost_real, SC *v, SC max, int size, int dim2)
		{
			// 256 threads in 8 warps
			__shared__ SC b_min_cost[8];
			__shared__ SC b_min_cost_real[8];
			__shared__ int b_jmin[8];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_min_cost_real = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_min_cost_real = min_cost_real[j];
				int c_jmin = jmin[j];
				if ((c_min_cost < v_min_cost) || ((c_min_cost == v_min_cost) && (c_jmin < v_jmin)))
				{
					v_min_cost = c_min_cost;
					v_jmin = c_jmin;
				}
				if (c_min_cost_real < v_min_cost_real) v_min_cost_real = c_min_cost_real;
				j += blockDim.x;
			}
			minWarpIndex(v_min_cost, v_jmin);
			minWarp(v_min_cost_real);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_min_cost_real[bidx] = v_min_cost_real;
				b_jmin[bidx] = v_jmin;
			}
			__syncthreads();
			if (threadIdx.x >= 8) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_min_cost_real = b_min_cost_real[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			minWarpIndex8(v_min_cost, v_jmin);
			minWarp8(v_min_cost_real);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost_real;
				s->picked = v_min_cost;
				s->jmin = v_jmin;
				if (v_jmin < dim2)
				{
					s->v_jmin = v[v_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class SC>
		__global__ void getMinimalCostLarge_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, int *jmin, SC *min_cost_real, SC *v, SC max, int size, int dim2)
		{
			// 1024 threads in 32 warps
			__shared__ SC b_min_cost[32];
			__shared__ SC b_min_cost_real[32];
			__shared__ int b_jmin[32];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_min_cost_real = max;
			int v_jmin = dim2;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_min_cost_real = min_cost_real[j];
				int c_jmin = jmin[j];
				if ((c_min_cost < v_min_cost) || ((c_min_cost == v_min_cost) && (c_jmin < v_jmin)))
				{
					v_min_cost = c_min_cost;
					v_jmin = c_jmin;
				}
				if (c_min_cost_real < v_min_cost_real) v_min_cost_real = c_min_cost_real;
				j += blockDim.x;
			}
			minWarpIndex(v_min_cost, v_jmin);
			minWarp(v_min_cost_real);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_min_cost_real[bidx] = v_min_cost_real;
				b_jmin[bidx] = v_jmin;
			}
			__syncthreads();
			if (threadIdx.x >= 32) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_min_cost_real = b_min_cost_real[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			minWarpIndex(v_min_cost, v_jmin);
			minWarp(v_min_cost_real);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost_real;
				s->picked = v_min_cost;
				s->jmin = v_jmin;
				if (v_jmin < dim2)
				{
					s->v_jmin = v[v_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class SC>
		__global__ void getFinalCostSmall_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *picked_cost, SC *min_v, SC max, int size, int dim2)
		{
			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_picked_cost = max;
			SC v_min_v = max;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_picked_cost = picked_cost[j];
				SC c_min_v = min_v[j];
				if (c_min_cost < v_min_cost) v_min_cost = c_min_cost;
				if (c_picked_cost < v_picked_cost) v_picked_cost = c_picked_cost;
				if (c_min_v < v_min_v) v_min_v = c_min_v;
				j += blockDim.x;
			}
			minWarp(v_min_cost);
			minWarp(v_picked_cost);
			minWarp(v_min_v);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->picked = v_picked_cost;
				s->v_jmin = v_min_v;
			}
		}

		template <class SC>
		__global__ void getFinalCostMedium_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *picked_cost, SC *min_v, SC max, int size, int dim2)
		{
			// 256 threads in 8 warps
			__shared__ SC b_min_cost[8];
			__shared__ SC b_picked_cost[8];
			__shared__ SC b_min_v[8];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_picked_cost = max;
			SC v_min_v = max;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_picked_cost = picked_cost[j];
				SC c_min_v = min_v[j];
				if (c_min_cost < v_min_cost) v_min_cost = c_min_cost;
				if (c_picked_cost < v_picked_cost) v_picked_cost = c_picked_cost;
				if (c_min_v < v_min_v) v_min_v = c_min_v;
				j += blockDim.x;
			}
			minWarp(v_min_cost);
			minWarp(v_picked_cost);
			minWarp(v_min_v);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_picked_cost[bidx] = v_picked_cost;
				b_min_v[bidx] = v_min_v;
			}
			__syncthreads();
			if (threadIdx.x >= 8) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_picked_cost = b_picked_cost[threadIdx.x];
			v_min_v = b_min_v[threadIdx.x];
			minWarp8(v_min_cost);
			minWarp8(v_picked_cost);
			minWarp8(v_min_v);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->picked = v_picked_cost;
				s->v_jmin = v_min_v;
			}
		}

		template <class SC>
		__global__ void getFinalCostLarge_kernel(estimateEpsilon_struct<SC> *s, SC *min_cost, SC *picked_cost, SC *min_v, SC max, int size, int dim2)
		{
			// 1024 threads in 32 warps
			__shared__ SC b_min_cost[32];
			__shared__ SC b_picked_cost[32];
			__shared__ SC b_min_v[32];

			int j = threadIdx.x;

			SC v_min_cost = max;
			SC v_picked_cost = max;
			SC v_min_v = max;

			while (j < size)
			{
				SC c_min_cost = min_cost[j];
				SC c_picked_cost = picked_cost[j];
				SC c_min_v = min_v[j];
				if (c_min_cost < v_min_cost) v_min_cost = c_min_cost;
				if (c_picked_cost < v_picked_cost) v_picked_cost = c_picked_cost;
				if (c_min_v < v_min_v) v_min_v = c_min_v;
				j += blockDim.x;
			}
			minWarp(v_min_cost);
			minWarp(v_picked_cost);
			minWarp(v_min_v);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min_cost[bidx] = v_min_cost;
				b_picked_cost[bidx] = v_picked_cost;
				b_min_v[bidx] = v_min_v;
			}
			__syncthreads();
			if (threadIdx.x >= 32) return;
			v_min_cost = b_min_cost[threadIdx.x];
			v_picked_cost = b_picked_cost[threadIdx.x];
			v_min_v = b_min_v[threadIdx.x];
			minWarp(v_min_cost);
			minWarp(v_picked_cost);
			minWarp(v_min_v);
			if (threadIdx.x == 0)
			{
				s->min = v_min_cost;
				s->picked = v_picked_cost;
				s->v_jmin = v_min_v;
			}
		}

		template <class SC, class TC>
		__global__ void updateEstimatedVFirst_kernel(SC *min_v, TC *tt, int *picked, SC min_cost, int jmin, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			min_v[j] = (SC)tt[j] - min_cost;
			if (jmin == j) picked[j] = 1;
		}

		template <class SC, class TC>
		__global__ void updateEstimatedVSecond_kernel(SC *v, SC *min_v, TC *tt, int *picked, SC min_cost, int jmin, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			SC tmp = (SC)tt[j] - min_cost;
			if (tmp < min_v[j])
			{
				v[j] = min_v[j];
				min_v[j] = tmp;
			}
			else v[j] = tmp;
			if (jmin == j) picked[j] = 1;
		}

		template <class SC, class TC>
		__global__ void updateEstimatedV_kernel(SC *v, SC *min_v, TC *tt, int *picked, SC min_cost, int jmin, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			SC tmp = (SC)tt[j] - min_cost;
			if (tmp < min_v[j])
			{
				v[j] = min_v[j];
				min_v[j] = tmp;
			}
			else if (tmp < v[j]) v[j] = tmp;
			if (jmin == j) picked[j] = 1;
		}

		template <class SC, class TC>
		__global__ void initializeSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = (SC)tt[j] - v[j];
				int c_colsol = colsol[j];
				if ((v0 < t_min) || ((v0 == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
				{
					t_min = v0;
					t_jmin = j;
					t_colsol = c_colsol;
				}
				j += blockDim.x * gridDim.x;
			}


			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = t_min;
				o_jmin[i] = t_jmin;
				o_colsol[i] = t_colsol;
			}
		}

		template <class SC, class TC>
		__global__ void continueSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, SC tt_jmin, SC v_jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC h = d[j];
				SC v2 = ((SC)tt[j] - tt_jmin) - (v[j] - v_jmin) + min;
				bool is_active = (colactive[j] != 0);
				bool is_smaller = (v2 < h);
				if (is_active)
				{
					if (is_smaller)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					int c_colsol = colsol[j];
					if ((h < t_min) || ((h == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
					{
						t_min = h;
						t_jmin = j;
						t_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = t_min;
				o_jmin[i] = t_jmin;
				o_colsol[i] = t_colsol;
			}
		}

		template <class SC>
		__global__ void initializeSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = -v[j];
				int c_colsol = colsol[j];
				if (c_colsol < dim)
				{
					if ((v0 < t_min) || ((v0 == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
					{
						t_min = v0;
						t_jmin = j;
						t_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = t_min;
				o_jmin[i] = t_jmin;
				o_colsol[i] = t_colsol;
			}
		}

		template <class SC>
		__global__ void continueSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, SC v_jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC h = d[j];
				SC v2 = -(v[j] - v_jmin) + min;
				bool is_active = (colactive[j] != 0);
				bool is_smaller = (v2 < h);
				if (is_active)
				{
					if (is_smaller)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					int c_colsol = colsol[j];
					if ((h < t_min) || ((h == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
					{
						t_min = h;
						t_jmin = j;
						t_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = t_min;
				o_jmin[i] = t_jmin;
				o_colsol[i] = t_colsol;
			}
		}

		template <class SC, class TC>
		__global__ void continueSearchJMinMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min = max;
			int t_jmin = dim2;
			SC tt_jmin = (SC)tt[jmin];
			SC v_jmin = v[jmin];
			int t_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				if (j == jmin) colactive[jmin] = 0;
				else if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = ((SC)tt[j] - tt_jmin) - (v[j] - v_jmin) + min;
					bool is_smaller = (v2 < h);
					if (is_smaller)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					int c_colsol = colsol[j];
					if ((h < t_min) || ((h == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
					{
						t_min = h;
						t_jmin = j;
						t_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = t_min;
				o_jmin[i] = t_jmin;
				o_colsol[i] = t_colsol;
			}
		}

		template <class SC>
		__global__ void continueSearchJMinMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min = max;
			int t_jmin = dim2;
			SC v_jmin = v[jmin];
			int t_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				if (j == jmin) colactive[jmin] = 0;
				else if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = -(v[j] - v_jmin) + min;
					bool is_smaller = (v2 < h);
					if (is_smaller)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					int c_colsol = colsol[j];
					if ((h < t_min) || ((h == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
					{
						t_min = h;
						t_jmin = j;
						t_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = t_min;
				o_jmin[i] = t_jmin;
				o_colsol[i] = t_colsol;
			}
		}

		template <class MS, class SC>
		__global__ void findMaxSmall_kernel(MS *s, SC *max, SC min, int size)
		{
			int j = threadIdx.x;

			SC v_max = min;

			while (j < size)
			{
				SC c_max = max[j];
				if (c_max > v_max) v_max = c_max;
				j += blockDim.x;
			}
			maxWarp(v_max);
			if (threadIdx.x == 0) s->max = v_max;
		}

		template <class MS, class SC>
		__global__ void findMaxMedium_kernel(MS *s, SC *max, SC min, int size)
		{
			// 256 threads in 8 warps
			__shared__ SC b_max[8];

			int j = threadIdx.x;

			SC v_max = min;

			while (j < size)
			{
				SC c_max = max[j];
				if (c_max > v_max) v_max = c_max;
				j += blockDim.x;
			}
			maxWarp(v_max);
			if ((threadIdx.x & 0x1f) == 0) b_max[threadIdx.x >> 5] = v_max;
			__syncthreads();
			if (threadIdx.x >= 8) return;
			v_max = b_max[threadIdx.x];
			maxWarp8(v_max);
			if (threadIdx.x == 0) s->max = v_max;
		}

		template <class MS, class SC>
		__global__ void findMaxLarge_kernel(MS *s, SC *max, SC min, int size)
		{
			// 1024 threads in 32 warps
			__shared__ SC b_max[8];

			int j = threadIdx.x;

			SC v_max = min;

			while (j < size)
			{
				SC c_max = max[j];
				if (c_max > v_max) v_max = c_max;
				j += blockDim.x;
			}
			maxWarp(v_max);
			if ((threadIdx.x & 0x1f) == 0) b_max[threadIdx.x >> 5] = v_max;
			__syncthreads();
			if (threadIdx.x >= 32) return;
			v_max = b_max[threadIdx.x];
			maxWarp(v_max);
			if (threadIdx.x == 0) s->max = v_max;
		}

		template <class SC>
		__global__ void findMinSmall_kernel(min_struct<SC> *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
		{
			int j = threadIdx.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

			while (j < size)
			{
				SC c_min = min[j];
				int c_jmin = jmin[j];
				int c_colsol = colsol[j];
				bool is_better = (c_min < v_min);
				if (c_jmin < v_jmin)
				{
					is_better = is_better || ((c_min == v_min) && ((c_colsol < 0) || (v_colsol >= 0)));
				}
				else
				{
					is_better = is_better || ((c_min == v_min) && (c_colsol < 0) && (v_colsol >= 0));
				}
				if (is_better)
				{
					v_min = c_min;
					v_jmin = c_jmin;
					v_colsol = c_colsol;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_min, v_jmin, v_colsol);
			if (threadIdx.x == 0)
			{
				s->min = v_min;
				s->jmin = v_jmin;
				s->colsol = v_colsol;
			}
		}

		template <class SC>
		__global__ void findMinMedium_kernel(min_struct<SC> *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
		{
			// 256 threads in 8 warps
			__shared__ SC b_min[8];
			__shared__ int b_jmin[8];
			__shared__ int b_colsol[8];

			int j = threadIdx.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

			while (j < size)
			{
				SC c_min = min[j];
				int c_jmin = jmin[j];
				int c_colsol = colsol[j];
				bool is_better = (c_min < v_min);
				if (c_jmin < v_jmin)
				{
					is_better = is_better || ((c_min == v_min) && ((c_colsol < 0) || (v_colsol >= 0)));
				}
				else
				{
					is_better = is_better || ((c_min == v_min) && (c_colsol < 0) && (v_colsol >= 0));
				}
				if (is_better)
				{
					v_min = c_min;
					v_jmin = c_jmin;
					v_colsol = c_colsol;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min[bidx] = v_min;
				b_jmin[bidx] = v_jmin;
				b_colsol[bidx] = v_colsol;
			}
			__syncthreads();
			if (threadIdx.x >= 8) return;
			v_min = b_min[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			v_colsol = b_colsol[threadIdx.x];
			minWarpIndex8(v_min, v_jmin, v_colsol);
			if (threadIdx.x == 0)
			{
				s->min = v_min;
				s->jmin = v_jmin;
				s->colsol = v_colsol;
			}
		}

		template <class SC>
		__global__ void findMinLarge_kernel(min_struct<SC> *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
		{
			// 1024 threads in 32 warps
			__shared__ SC b_min[32];
			__shared__ int b_jmin[32];
			__shared__ int b_colsol[32];

			int j = threadIdx.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

			while (j < size)
			{
				SC c_min = min[j];
				int c_jmin = jmin[j];
				int c_colsol = colsol[j];
				bool is_better = (c_min < v_min);
				if (c_jmin < v_jmin)
				{
					is_better = is_better || ((c_min == v_min) && ((c_colsol < 0) || (v_colsol >= 0)));
				}
				else
				{
					is_better = is_better || ((c_min == v_min) && (c_colsol < 0) && (v_colsol >= 0));
				}
				if (is_better)
				{
					v_min = c_min;
					v_jmin = c_jmin;
					v_colsol = c_colsol;
				}
				j += blockDim.x;
			}
			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int bidx = threadIdx.x >> 5;
				b_min[bidx] = v_min;
				b_jmin[bidx] = v_jmin;
				b_colsol[bidx] = v_colsol;
			}
			__syncthreads();
			if (threadIdx.x >= 32) return;
			v_min = b_min[threadIdx.x];
			v_jmin = b_jmin[threadIdx.x];
			v_colsol = b_colsol[threadIdx.x];
			minWarpIndex(v_min, v_jmin, v_colsol);
			if (threadIdx.x == 0)
			{
				s->min = v_min;
				s->jmin = v_jmin;
				s->colsol = v_colsol;
			}
		}

		__global__ void setColInactive_kernel(char *colactive, int jmin)
		{
			colactive[jmin] = 0;
		}

		template <class SC>
		__global__ void setColInactive_kernel(char *colactive, int jmin, SC *v_jmin, SC *v_in)
		{
			*v_jmin = *v_in;
			colactive[jmin] = 0;
		}

		template <class TC, class TC2, class SC>
		__global__ void setColInactive_kernel(char *colactive, int jmin, TC *tt_jmin, TC2 *tt_in, SC *v_jmin, SC *v_in)
		{
			*tt_jmin = *tt_in;
			*v_jmin = *v_in;
			colactive[jmin] = 0;
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				v[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPricesClamp_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] -= dlt;
				if (eps > dlt) dlt = eps;
				v[j] -= dlt;
				total_eps[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] -= dlt;
				dlt += eps;
				v[j] -= dlt;
				total_eps[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size, int *colsol, int csol)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;
			if (j == 0) *colsol = csol;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				v[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPricesClamp_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int size, int *colsol, int csol)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;
			if (j == 0) *colsol = csol;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] -= dlt;
				if (eps > dlt) dlt = eps;
				v[j] -= dlt;
				total_eps[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int size, int *colsol, int csol)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;
			if (j == 0) *colsol = csol;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] -= dlt;
				dlt += eps;
				v[j] -= dlt;
				total_eps[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateUnassignedColumnPrices_kernel(int *colsol, SC *v, SC *total_eps, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colsol[j] < 0)
			{
				v[j] -= eps;
				total_eps[j] -= eps;
			}
		}

		template <class SC>
		__global__ void markedSkippedColumns_kernel(char *colactive, SC min_n, int jmin, int *colsol, SC *d, int dim, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			// ignore any columns assigned to virtual rows
			if ((j == jmin) || ((colsol[j] >= dim) && (d[j] <= min_n)))
			{
				colactive[j] = 0;
			}
		}

		template <class SC>
		__global__ void markedSkippedColumnsUpdate_kernel(char *colactive, SC min_n, int jmin, int *colsol, SC *d, int dim, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			// ignore any columns assigned to virtual rows
			if ((j == jmin) || ((colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min_n)))
			{
				colactive[j] = 0;
			}
		}

		template <class SC, class MS>
		__global__ void subtractMaximum_kernel(SC *v, MS *max_struct, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			v[j] -= max_struct->max;
		}

		template <class SC>
		__global__ void subtractMaximum_kernel(SC *v, SC max, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			v[j] -= max;
		}

		template <class SC, class MS>
		void findMaximum(SC *v_private, MS *max_struct, cudaStream_t &stream, int min_count)
		{
			// normalize v
			if (min_count <= 32) findMaxSmall_kernel<<<1, 32, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else if (min_count <= 256) findMaxMedium_kernel<<<1, 256, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else findMaxLarge_kernel<<<1, 1024, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
		}

		template <class SC, class MS>
		SC mergeMaximum(MS *max_struct, int devices)
		{
			SC max_cost = max_struct[0].max;
			for (int tx = 1; tx < devices; tx++)
			{
				max_cost = std::max(max_cost, max_struct[tx].max);
			}
			return max_cost;
		}

		template <class SC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC **v_private)
		{
			SC *mod_v;
			SC **mod_v_private;
			SC **min_cost_private;
			SC **max_cost_private;
			SC **picked_cost_private;
			int **jmin_private;
			int *perm;
			int **picked_private;
			int *picked;
			estimateEpsilon_struct<SC> *host_struct_private;

			int devices = (int)iterator.ws.device.size();
#ifdef LAP_CUDA_OPENMP
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#endif

			cudaMallocHost(&mod_v, dim2 * sizeof(SC));
			lapAlloc(mod_v_private, devices, __FILE__, __LINE__);
			lapAlloc(min_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(max_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(perm, dim, __FILE__, __LINE__);
			lapAlloc(picked_private, devices, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			cudaMallocHost(&host_struct_private, devices * sizeof(estimateEpsilon_struct<SC>));

#ifdef LAP_CUDA_OPENMP
			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (num_items + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
				int count = (grid_size_min.x * block_size.x) >> 5;
				cudaStream_t stream = iterator.ws.stream[0];
				cudaMalloc(&(mod_v_private[0]), num_items * std::max(sizeof(int), sizeof(SC)));
				cudaMalloc(&(picked_private[0]), num_items * sizeof(int));
				cudaMalloc(&(min_cost_private[0]), count * sizeof(SC));
				cudaMalloc(&(max_cost_private[0]), count * sizeof(SC));
				cudaMalloc(&(picked_cost_private[0]), count * sizeof(SC));
				cudaMalloc(&(jmin_private[0]), count * sizeof(int));
				cudaMemsetAsync(picked_private[0], 0, num_items * sizeof(int), stream);
			}
			else
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
					int count = (grid_size_min.x * block_size.x) >> 5;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMalloc(&(mod_v_private[t]), num_items * sizeof(SC));
					cudaMalloc(&(picked_private[t]), num_items * sizeof(int));
					cudaMalloc(&(min_cost_private[t]), count * sizeof(SC));
					cudaMalloc(&(max_cost_private[t]), count * sizeof(SC));
					cudaMalloc(&(picked_cost_private[t]), count * sizeof(SC));
					cudaMalloc(&(jmin_private[t]), count * sizeof(int));
					cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
				}
			}
#else
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (num_items + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
				int count = (grid_size_min.x * block_size.x) >> 5;
				cudaStream_t stream = iterator.ws.stream[t];
				cudaMalloc(&(mod_v_private[t]), num_items * sizeof(SC));
				cudaMalloc(&(picked_private[t]), num_items * sizeof(int));
				cudaMalloc(&(min_cost_private[t]), count * sizeof(SC));
				cudaMalloc(&(max_cost_private[t]), count * sizeof(SC));
				cudaMalloc(&(picked_cost_private[t]), count * sizeof(SC));
				cudaMalloc(&(jmin_private[t]), count * sizeof(int));
				cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
			}
#endif
			SC lower_bound = SC(0);
			SC greedy_bound = SC(0);
			SC upper_bound = SC(0);

			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				cudaStream_t stream = iterator.ws.stream[0];
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (dim2 + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
				int min_count = grid_size_min.x * (block_size.x >> 5);

				for (int i = 0; i < dim; i++)
				{
					auto *tt = iterator.getRow(0, i);

					getMinMaxBest_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					if (min_count <= 32) getMinMaxBestSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
					else if (min_count <= 256) getMinMaxBestMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
					else getMinMaxBestLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);

					checkCudaErrors(cudaStreamSynchronize(stream));

					SC min_cost = host_struct_private[0].min;
					int jmin = host_struct_private[0].jmin;

					if (i == 0) updateEstimatedVFirst_kernel<<<grid_size, block_size, 0, stream>>>(mod_v_private[0], tt, picked_private[0], min_cost, jmin, dim2);
					else if (i == 1) updateEstimatedVSecond_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], mod_v_private[0], tt, picked_private[0], min_cost, jmin, dim2);
					else updateEstimatedV_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], mod_v_private[0], tt, picked_private[0], min_cost, jmin, dim2);

					lower_bound += min_cost;
					upper_bound += host_struct_private[0].max;
					greedy_bound += host_struct_private[0].picked;
				}
				findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
				subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
			}
			else
			{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
					int min_count = grid_size_min.x * (block_size.x >> 5);

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(t, i);

						getMinMaxBest_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i - iterator.ws.part[t].first, num_items, dim2);
						if (min_count <= 32) getMinMaxBestSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) getMinMaxBestMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
						else getMinMaxBestLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);

						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC min_cost = host_struct_private[0].min;
						SC max_cost = host_struct_private[0].max;
						SC picked_cost = host_struct_private[0].picked;
						int jmin = host_struct_private[0].jmin;

						for (int tx = 1; tx < devices; tx++)
						{
							if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
							{
								picked_cost = host_struct_private[tx].picked;
								jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
							}
							min_cost = std::min(min_cost, host_struct_private[tx].min);
							max_cost = std::max(max_cost, host_struct_private[tx].max);
						}

						if (i == 0) updateEstimatedVFirst_kernel<<<grid_size, block_size, 0, stream>>>(mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else if (i == 1) updateEstimatedVSecond_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else updateEstimatedV_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);

						if (t == 0)
						{
							lower_bound += min_cost;
							upper_bound += max_cost;
							greedy_bound += picked_cost;
						}
					}
					findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					SC max_v = mergeMaximum<SC>(host_struct_private, devices);
					subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
				}
#else
				for (int i = 0; i < dim; i++)
				{
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto *tt = iterator.getRow(t, i);
						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
						getMinMaxBest_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], iterator.getRow(t, i), picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i - iterator.ws.part[t].first, num_items, dim2);
						int min_count = grid_size_min.x * (block_size.x >> 5);
						if (min_count <= 32) getMinMaxBestSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) getMinMaxBestMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
						else getMinMaxBestLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), min_count, dim2);
					}

					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

					SC min_cost = host_struct_private[0].min;
					SC max_cost = host_struct_private[0].max;
					SC picked_cost = host_struct_private[0].picked;
					int jmin = host_struct_private[0].jmin;

					for (int tx = 1; tx < devices; tx++)
					{
						if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
						{
							picked_cost = host_struct_private[tx].picked;
							jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
						}
						min_cost = std::min(min_cost, host_struct_private[tx].min);
						max_cost = std::max(max_cost, host_struct_private[tx].max);
					}

					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto *tt = iterator.getRow(t, i);
						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;

						if (i == 0) updateEstimatedVFirst_kernel<<<grid_size, block_size, 0, stream>>>(mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else if (i == 1) updateEstimatedVSecond_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else updateEstimatedV_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
					}

					lower_bound += min_cost;
					upper_bound += max_cost;
					greedy_bound += picked_cost;
				}
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
				}
				for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
				SC max_v = mergeMaximum<SC>(host_struct_private, devices);
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
				}
#endif
			}

			greedy_bound = std::min(greedy_bound, upper_bound);

			SC initial_gap = upper_bound - lower_bound;
			SC greedy_gap = greedy_bound - lower_bound;

#ifdef LAP_DEBUG
			lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap << std::endl;
			lapDebug << "  upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

			SC upper = std::numeric_limits<SC>::max();
			SC lower;

#ifdef LAP_CUDA_OPENMP
			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				cudaStream_t stream = iterator.ws.stream[0];
				cudaMemsetAsync(picked_private[0], 0, num_items * sizeof(int), stream);
			}
			else
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
				}
			}
#else
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				cudaStream_t stream = iterator.ws.stream[t];
				cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
			}
#endif

			lower_bound = SC(0);
			upper_bound = SC(0);

			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				cudaStream_t stream = iterator.ws.stream[0];
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (dim2 + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
				int min_count = grid_size_min.x * (block_size.x >> 5);

				for (int i = dim - 1; i >= 0; --i)
				{
					auto *tt = iterator.getRow(0, i);

					getMinSecondBest_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					if (min_count <= 32) getMinSecondBestSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], std::numeric_limits<SC>::max(), min_count, dim2);
					else if (min_count <= 256) getMinSecondBestMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], std::numeric_limits<SC>::max(), min_count, dim2);
					else getMinSecondBestLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], std::numeric_limits<SC>::max(), min_count, dim2);

					checkCudaErrors(cudaStreamSynchronize(stream));

					SC min_cost = host_struct_private[0].min;
					SC second_cost = host_struct_private[0].max;
					SC picked_cost = host_struct_private[0].picked;
					int jmin = host_struct_private[0].jmin;
					SC v_jmin = host_struct_private[0].v_jmin;

					cudaMemsetAsync(&(picked_private[0][jmin]), 1, 1, stream);
					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
			}
			else
			{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
					int min_count = grid_size_min.x * (block_size.x >> 5);

					for (int i = dim - 1; i >= 0; --i)
					{
						auto *tt = iterator.getRow(t, i);
#pragma omp barrier
						getMinSecondBest_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i - iterator.ws.part[t].first, num_items, dim2);
						if (min_count <= 32) getMinSecondBestSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) getMinSecondBestMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else getMinSecondBestLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);

						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC min_cost = host_struct_private[0].min;
						SC second_cost = host_struct_private[0].max;
						SC picked_cost = host_struct_private[0].picked;
						int jmin = host_struct_private[0].jmin;
						SC v_jmin = host_struct_private[0].v_jmin;

						for (int tx = 1; tx < devices; tx++)
						{
							if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
							{
								picked_cost = host_struct_private[tx].picked;
								jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
								v_jmin = host_struct_private[tx].v_jmin;
							}
							if (host_struct_private[tx].min < min_cost)
							{
								second_cost = std::min(min_cost, host_struct_private[tx].max);
								min_cost = host_struct_private[tx].min;
							}
							else
							{
								second_cost = std::min(second_cost, host_struct_private[tx].min);
							}
						}

						if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second)) cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, 1, stream);
						if (t == 0)
						{
							perm[i] = i;
							mod_v[i] = second_cost - min_cost;
							// need to use the same v values in total
							lower_bound += min_cost + v_jmin;
							upper_bound += picked_cost + v_jmin;
						}
					}
				}
#else
				for (int i = dim - 1; i >= 0; --i)
				{
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto *tt = iterator.getRow(t, i);
						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
						getMinSecondBest_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], iterator.getRow(t, i), v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i - iterator.ws.part[t].first, num_items, dim2);
						int min_count = grid_size_min.x * (block_size.x >> 5);
						if (min_count <= 32) getMinSecondBestSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) getMinSecondBestMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else getMinSecondBestLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
					}

					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

					SC min_cost = host_struct_private[0].min;
					SC second_cost = host_struct_private[0].max;
					SC picked_cost = host_struct_private[0].picked;
					int jmin = host_struct_private[0].jmin;
					SC v_jmin = host_struct_private[0].v_jmin;

					for (int tx = 1; tx < devices; tx++)
					{
						if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
						{
							picked_cost = host_struct_private[tx].picked;
							jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
							v_jmin = host_struct_private[tx].v_jmin;
						}
						if (host_struct_private[tx].min < min_cost)
						{
							second_cost = std::min(min_cost, host_struct_private[tx].max);
							min_cost = host_struct_private[tx].min;
						}
						else
						{
							second_cost = std::min(second_cost, host_struct_private[tx].min);
						}
					}

					for (int t = 0; t < devices; t++) if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
					{
						cudaSetDevice(iterator.ws.device[t]);
						cudaStream_t stream = iterator.ws.stream[t];
						cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, 1, stream);
					}

					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
#endif
			}
			upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

			if (initial_gap < SC(4) * greedy_gap)
			{
				// sort permutation by keys
				std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

				lower_bound = SC(0);
				upper_bound = SC(0);
				// greedy search
				if (devices == 1)
				{
					cudaSetDevice(iterator.ws.device[0]);
					cudaStream_t stream = iterator.ws.stream[0];
					int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
					int min_count = grid_size_min.x * (block_size.x >> 5);

					cudaMemsetAsync(picked_private[0], 0, dim2 * sizeof(int), stream);

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(0, perm[i]);

						getMinimalCost_kernel<<<grid_size_min, block_size, 0, stream>>>(picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
						if (min_count <= 32) getMinimalCostSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[0]), picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) getMinimalCostMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[0]), picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), min_count, dim2);
						else getMinimalCostLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[0]), picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), min_count, dim2);

						checkCudaErrors(cudaStreamSynchronize(stream));

						SC min_cost = host_struct_private[0].picked;
						SC min_cost_real = host_struct_private[0].min;
						int jmin = host_struct_private[0].jmin;
						SC v_jmin = host_struct_private[0].v_jmin;

						upper_bound += min_cost + v_jmin;
						// need to use the same v values in total
						lower_bound += min_cost_real + v_jmin;

						cudaMemsetAsync(&(picked_private[0][jmin]), 1, sizeof(int), stream);
						picked[i] = jmin;
					}
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
						int min_count = grid_size_min.x * (block_size.x >> 5);

						cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);

						for (int i = 0; i < dim; i++)
						{
#pragma omp barrier
							auto *tt = iterator.getRow(t, perm[i]);

							getMinimalCost_kernel<<<grid_size_min, block_size, 0, stream>>>(picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, num_items, dim2);
							if (min_count <= 32) getMinimalCostSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) getMinimalCostMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else getMinimalCostLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);

							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC min_cost = host_struct_private[0].picked;
							SC min_cost_real = host_struct_private[0].min;
							int jmin = host_struct_private[0].jmin;
							SC v_jmin = host_struct_private[0].v_jmin;
							for (int tx = 1; tx < devices; tx++)
							{
								if ((host_struct_private[tx].picked < min_cost) || ((host_struct_private[tx].picked == min_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
								{
									min_cost = host_struct_private[tx].picked;
									jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
									v_jmin = host_struct_private[tx].v_jmin;
								}
								min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
							}
							if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
							{
								cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, sizeof(int), stream);
							}
							if (t == 0)
							{
								upper_bound += min_cost + v_jmin;
								// need to use the same v values in total
								lower_bound += min_cost_real + v_jmin;
								picked[i] = jmin;
							}
						}
					}
#else
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];

						cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
					}
					for (int i = 0; i < dim; i++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size, grid_size_min;
							block_size.x = 256;
							grid_size.x = (num_items + block_size.x - 1) / block_size.x;
							grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
								std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
							int min_count = grid_size_min.x * (block_size.x >> 5);

							auto *tt = iterator.getRow(t, perm[i]);

							getMinimalCost_kernel<<<grid_size_min, block_size, 0, stream>>>(picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, num_items, dim2);
							if (min_count <= 32) getMinimalCostSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) getMinimalCostMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else getMinimalCostLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						}

						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

						SC min_cost = host_struct_private[0].picked;
						SC min_cost_real = host_struct_private[0].min;
						int jmin = host_struct_private[0].jmin;
						SC v_jmin = host_struct_private[0].v_jmin;
						for (int tx = 1; tx < devices; tx++)
						{
							if ((host_struct_private[tx].picked < min_cost) || ((host_struct_private[tx].picked == min_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
							{
								min_cost = host_struct_private[tx].picked;
								jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
								v_jmin = host_struct_private[tx].v_jmin;
							}
							min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
						}

						upper_bound += min_cost + v_jmin;
						// need to use the same v values in total
						lower_bound += min_cost_real + v_jmin;
						for (int t = 0; t < devices; t++)
						{
							if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
							{
								cudaSetDevice(iterator.ws.device[t]);
								cudaStream_t stream = iterator.ws.stream[t];

								cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, sizeof(int), stream);
							}
						}
						picked[i] = jmin;
					}
#endif
				}
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

				if (devices == 1)
				{
					cudaSetDevice(iterator.ws.device[0]);
					int start = iterator.ws.part[0].first;
					int end = iterator.ws.part[0].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[0];
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (size + block_size.x - 1) / block_size.x;
					for (int i = dim - 1; i >= 0; --i)
					{
						auto *tt = iterator.getRow(0, perm[i]);
						updateVSingle_kernel<<<grid_size, block_size, 0, stream>>>(tt, v_private[0], picked_private[0], picked[i], dim2);
					}
					findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
					subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						for (int i = dim - 1; i >= 0; --i)
						{
							auto *tt = iterator.getRow(t, perm[i]);
							if ((picked[i] >= start) && (picked[i] < end))
							{
								updateVMultiStart_kernel<<<1, 1, 0, stream>>>(tt, v_private[t], picked_private[t], &(mod_v[i]), picked[i] - start);
							}
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							updateVMulti_kernel<<<grid_size, block_size, 0, stream>>>(tt, v_private[t], picked_private[t], mod_v[i], size);
						}

						findMaximum(v_private[t], &(host_struct_private[t]), stream, size);
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC max_v = mergeMaximum<SC>(host_struct_private, devices);
						subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, size);
					}
#else
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						findMaximum(v_private[t], &(host_struct_private[t]), stream, size);
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
					SC max_v = mergeMaximum<SC>(host_struct_private, devices);
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
					}
#endif
				}

				SC old_upper_bound = upper_bound;
				SC old_lower_bound = lower_bound;
				upper_bound = SC(0);
				lower_bound = SC(0);
				if (devices == 1)
				{
					cudaSetDevice(iterator.ws.device[0]);
					cudaStream_t stream = iterator.ws.stream[0];
					int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
					int min_count = grid_size_min.x * (block_size.x >> 5);

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(0, perm[i]);

						getFinalCost_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items, dim2);
						if (min_count <= 32) getFinalCostSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], picked_cost_private[0], max_cost_private[0], std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) getFinalCostMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], picked_cost_private[0], max_cost_private[0], std::numeric_limits<SC>::max(), min_count, dim2);
						else getFinalCostLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[0]), min_cost_private[0], picked_cost_private[0], max_cost_private[0], std::numeric_limits<SC>::max(), min_count, dim2);

						checkCudaErrors(cudaStreamSynchronize(stream));

						SC picked_cost = host_struct_private[0].picked;
						SC v_picked = host_struct_private[0].v_jmin;
						SC min_cost_real = host_struct_private[0].min;

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
						int min_count = grid_size_min.x * (block_size.x >> 5);

						for (int i = 0; i < dim; i++)
						{
#pragma omp barrier
							auto *tt = iterator.getRow(t, perm[i]);

							getFinalCost_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items, dim2);
							if (min_count <= 32) getFinalCostSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], picked_cost_private[t], max_cost_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) getFinalCostMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], picked_cost_private[t], max_cost_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else getFinalCostLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], picked_cost_private[t], max_cost_private[t], std::numeric_limits<SC>::max(), min_count, dim2);

							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC picked_cost = host_struct_private[0].picked;
							SC v_picked = host_struct_private[0].v_jmin;
							SC min_cost_real = host_struct_private[0].min;
							for (int tx = 1; tx < devices; tx++)
							{
								picked_cost = std::min(picked_cost, host_struct_private[tx].picked);
								v_picked = std::min(v_picked, host_struct_private[tx].v_jmin);
								min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
							}

							if (t == 0)
							{
								// need to use all picked v for the lower bound as well
								upper_bound += picked_cost;
								lower_bound += min_cost_real + v_picked;
							}
						}
					}
#else
					for (int i = 0; i < dim; i++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size, grid_size_min;
							block_size.x = 256;
							grid_size.x = (num_items + block_size.x - 1) / block_size.x;
							grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
								std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
							int min_count = grid_size_min.x * (block_size.x >> 5);

							auto *tt = iterator.getRow(t, perm[i]);

							getFinalCost_kernel<<<grid_size_min, block_size, 0, stream>>>(min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items, dim2);
							if (min_count <= 32) getFinalCostSmall_kernel<<<1, 32, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], picked_cost_private[t], max_cost_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) getFinalCostMedium_kernel<<<1, 256, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], picked_cost_private[t], max_cost_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else getFinalCostLarge_kernel<<<1, 1024, 0, stream>>>(&(host_struct_private[t]), min_cost_private[t], picked_cost_private[t], max_cost_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						}

						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

						SC picked_cost = host_struct_private[0].picked;
						SC v_picked = host_struct_private[0].v_jmin;
						SC min_cost_real = host_struct_private[0].min;
						for (int tx = 1; tx < devices; tx++)
						{
							picked_cost = std::min(picked_cost, host_struct_private[tx].picked);
							v_picked = std::min(v_picked, host_struct_private[tx].v_jmin);
							min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
						}

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
#endif
				}
				upper_bound = std::min(upper_bound, old_upper_bound);
				lower_bound = std::max(lower_bound, old_lower_bound);
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif
			}

			getUpperLower(upper, lower, greedy_gap, initial_gap, dim2);

#ifdef LAP_CUDA_OPENMP
			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				cudaFree(mod_v_private[0]);
				cudaFree(picked_private[0]);
				cudaFree(min_cost_private[0]);
				cudaFree(max_cost_private[0]);
				cudaFree(picked_cost_private[0]);
				cudaFree(jmin_private[0]);
			}
			else
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					cudaFree(mod_v_private[t]);
					cudaFree(picked_private[t]);
					cudaFree(min_cost_private[t]);
					cudaFree(max_cost_private[t]);
					cudaFree(picked_cost_private[t]);
					cudaFree(jmin_private[t]);
				}
			}
#else
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(mod_v_private[t]);
				cudaFree(picked_private[t]);
				cudaFree(min_cost_private[t]);
				cudaFree(max_cost_private[t]);
				cudaFree(picked_cost_private[t]);
				cudaFree(jmin_private[t]);
			}
#endif
			cudaFreeHost(mod_v);
			lapFree(mod_v_private);
			lapFree(min_cost_private);
			lapFree(max_cost_private);
			lapFree(picked_cost_private);
			lapFree(jmin_private);
			lapFree(perm);
			lapFree(picked_private);
			lapFree(picked);
			cudaFreeHost(host_struct_private);


#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#endif
			return std::pair<SC, SC>((SC)upper, (SC)lower);
		}

		template <class SC, class TC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)

			// input:
			// dim        - problem size
			// costfunc - cost matrix
			// findcost   - searching cost matrix

			// output:
			// rowsol     - column assigned to row in solution
			// colsol     - row assigned to column in solution
			// u          - dual variables, row reduction numbers
			// v          - dual variables, column reduction numbers

		{
#ifndef LAP_QUIET
			auto start_time = std::chrono::high_resolution_clock::now();

			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;

			int elapsed = -1;
#else
#ifdef LAP_DISPLAY_EVALUATED
			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;
#endif
#endif

			int  endofpath;
#ifdef LAP_DEBUG
			SC *v;
#endif
			SC *h_total_d;
			SC *h_total_eps;
			// for calculating h2
			TC *tt_jmin;
			SC *v_jmin;

			int devices = (int)iterator.ws.device.size();
#ifdef LAP_CUDA_OPENMP
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#else
			const TC **tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);
#endif

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

#ifdef LAP_CUDA_COMPARE_CPU
			SC *d_tmp;
			unsigned char *colactive_tmp;
			lapAlloc(d_tmp, dim2, __FILE__, __LINE__);
			lapAlloc(colactive_tmp, dim2, __FILE__, __LINE__);
#endif

			// used for copying
			min_struct<SC> *host_min_private;
#ifdef LAP_DEBUG
			cudaMallocHost(&v, dim2 * sizeof(SC));
#endif
			cudaMallocHost(&h_total_d, dim2 * sizeof(SC));
			cudaMallocHost(&h_total_eps, dim2 * sizeof(SC));
			cudaMallocHost(&host_min_private, devices * sizeof(min_struct<SC>));
			cudaMallocHost(&tt_jmin, sizeof(TC));
			cudaMallocHost(&v_jmin, sizeof(SC));

			SC **min_private;
			int **jmin_private;
			int **csol_private;
			char **colactive_private;
			int **pred_private;
			SC **d_private;
			int *pred;
			int *colsol;
			int **colsol_private;
			SC **v_private;
			SC **total_eps_private;
			SC **total_d_private;
			// on device
			lapAlloc(min_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(csol_private, devices, __FILE__, __LINE__);
			lapAlloc(colactive_private, devices, __FILE__, __LINE__);
			lapAlloc(pred_private, devices, __FILE__, __LINE__);
			lapAlloc(d_private, devices, __FILE__, __LINE__);
			lapAlloc(colsol_private, devices, __FILE__, __LINE__);
			lapAlloc(v_private, devices, __FILE__, __LINE__);
			lapAlloc(total_d_private, devices, __FILE__, __LINE__);
			lapAlloc(total_eps_private, devices, __FILE__, __LINE__);
			cudaMallocHost(&colsol, sizeof(int) * dim2);
			cudaMallocHost(&pred, sizeof(int) * dim2);

			for (int t = 0; t < devices; t++)
			{
				// single device code
				cudaSetDevice(iterator.ws.device[t]);
				int start = iterator.ws.part[t].first;
				int end = iterator.ws.part[t].second;
				int size = end - start;
				cudaStream_t stream = iterator.ws.stream[t];
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (size + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
				int count = (grid_size_min.x * block_size.x) >> 5;
				cudaMalloc(&(min_private[t]), sizeof(SC) * count);
				cudaMalloc(&(jmin_private[t]), sizeof(int) * count);
				cudaMalloc(&(csol_private[t]), sizeof(int) * count);
				cudaMalloc(&(colactive_private[t]), sizeof(char) * size);
				cudaMalloc(&(d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(v_private[t]), sizeof(SC) * size);
				cudaMalloc(&(total_d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(total_eps_private[t]), sizeof(SC) * size);
				cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
				cudaMalloc(&(pred_private[t]), sizeof(int) * size);
				if (!use_epsilon) cudaMemsetAsync(v_private[t], 0, sizeof(SC) * size, stream);
			}

			SC epsilon_upper, epsilon_lower;

			if (use_epsilon)
			{
				std::pair<SC, SC> eps = estimateEpsilon(dim, dim2, iterator, v_private);
				epsilon_upper = eps.first;
				epsilon_lower = eps.second;
			}
			else
			{
				epsilon_upper = SC(0);
				epsilon_lower = SC(0);
			}


#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			SC epsilon = epsilon_upper;

			bool first = true;
			bool second = false;
			bool clamp = false;

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= SC(0))
			{
#ifdef LAP_DEBUG
				if (first)
				{
#ifdef LAP_CUDA_OPENMP
					if (devices == 1)
					{
						cudaSetDevice(iterator.ws.device[0]);
						int start = iterator.ws.part[0].first;
						int end = iterator.ws.part[0].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[0];
						cudaMemcpyAsync(&(v[start]), v_private[0], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						checkCudaErrors(cudaStreamSynchronize(stream));
					}
					else
					{
#pragma omp parallel
						{
							int t = omp_get_thread_num();
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
							checkCudaErrors(cudaStreamSynchronize(stream));
						}
					}
#else
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#endif
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
#endif
				getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);
				total_d = SC(0);
				total_eps = SC(0);
#ifndef LAP_QUIET
				{
					std::stringstream ss;
					ss << "eps = " << epsilon;
					const std::string tmp = ss.str();
					displayTime(start_time, tmp.c_str(), lapInfo);
				}
#endif
				// this is to ensure termination of the while statement
				if (epsilon == SC(0)) epsilon = SC(-1.0);
				memset(colsol, -1, dim2 * sizeof(int));

#ifdef LAP_CUDA_OPENMP
				if (devices == 1)
				{
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
				}
				else
				{
#pragma omp parallel for
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * size, stream);
						cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * size, stream);
						cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
					}
				}
#else
				for (int t = 0; t < devices; t++)
				{
					// upload v to devices
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
				}
#endif

				int jmin, colsol_old;
				SC min, min_n;
				bool unassignedfound;

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif
				long long count = 0ll;

				int dim_limit = ((epsilon > SC(0)) && (first)) ? dim : dim2;

				if (devices == 1)
				{
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[t];

					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (size + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));

					for (int f = 0; f < dim_limit; f++)
					{
						// start search and find minimum value
						if (f < dim)
							initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
						else
							initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim, dim2);
						// min is now set so we need to find the correspoding minima for free and taken columns
						int min_count = grid_size_min.x * (block_size.x >> 5);
						if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						checkCudaErrors(cudaStreamSynchronize(stream));
#ifndef LAP_QUIET
						if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
						if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
						scancount[f]++;
#endif
						count++;

						unassignedfound = false;

						// Dijkstra search
						min = host_min_private[t].min;
						jmin = host_min_private[t].jmin;
						colsol_old = host_min_private[t].colsol;

						// dijkstraCheck
						if (colsol_old < 0)
						{
							endofpath = jmin;
							unassignedfound = true;
						}
						else
						{
							unassignedfound = false;
						}

#ifdef LAP_CUDA_COMPARE_CPU
						{
							cudaMemcpyAsync(d_tmp, d_private[t], dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream);
							cudaMemcpyAsync(colactive_tmp, colactive_private[t], dim2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
							checkCudaErrors(cudaStreamSynchronize(stream));
							SC min_tmp = std::numeric_limits<SC>::max();
							int jmin_tmp = dim2;
							int colsol_old_tmp = 0;
							for (int j = 0; j < dim2; j++)
							{
								if (colactive_tmp[j] != 0)
								{
									if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
									{
										min_tmp = d_tmp[j];
										jmin_tmp = j;
										colsol_old_tmp = colsol[j];
									}
								}
							}
							if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
							{
								std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
							}
						}
#endif

						if (f >= dim) markedSkippedColumns_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, size);

						bool fast = unassignedfound;

						while (!unassignedfound)
						{
							int i = colsol_old;
							if (i < dim)
							{
								// get row
								auto tt = iterator.getRow(t, i);
								// continue search
								continueSearchJMinMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
							}
							else
							{
								// continue search
								continueSearchJMinMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
							}
							// min is now set so we need to find the correspoding minima for free and taken columns
							if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							checkCudaErrors(cudaStreamSynchronize(stream));
#ifndef LAP_QUIET
							if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
							if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
							scancount[i]++;
#endif
							count++;

							min_n = host_min_private[t].min;
							jmin = host_min_private[t].jmin;
							colsol_old = host_min_private[t].colsol;

							min = std::max(min, min_n);

							// dijkstraCheck
							if (colsol_old < 0)
							{
								endofpath = jmin;
								unassignedfound = true;
							}
							else
							{
								unassignedfound = false;
							}

#ifdef LAP_CUDA_COMPARE_CPU
							{
								cudaMemcpyAsync(d_tmp, d_private[t], dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream);
								cudaMemcpyAsync(colactive_tmp, colactive_private[t], dim2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
								checkCudaErrors(cudaStreamSynchronize(stream));
								SC min_tmp = std::numeric_limits<SC>::max();
								int jmin_tmp = dim2;
								int colsol_old_tmp = 0;
								for (int j = 0; j < dim2; j++)
								{
									if (colactive_tmp[j] != 0)
									{
										if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
										{
											min_tmp = d_tmp[j];
											jmin_tmp = j;
											colsol_old_tmp = colsol[j];
										}
									}
								}
								if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
								{
									std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
								}
							}
#endif

							if (i >= dim) markedSkippedColumnsUpdate_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, size);
						}

						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							if (epsilon > SC(0))
							{
								if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath]), colsol[endofpath]);
								else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
						}
						else
						{
							// update column prices. can increase or decrease
							if (epsilon > SC(0))
							{
								if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
								else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
							}
							// reset row and column assignments along the alternating path.
							cudaMemcpyAsync(pred, pred_private[t], dim2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
							checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_ROWS_SCANNED
							{
								int i;
								int eop = endofpath;
								do
								{
									i = pred[eop];
									eop = rowsol[i];
									if (i != f) pathlength[f]++;
								} while (i != f);
							}
#endif
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
							cudaMemcpyAsync(colsol_private[t], colsol, dim2 * sizeof(int), cudaMemcpyHostToDevice, stream);
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
							{
								long long hit, miss;
								iterator.getHitMiss(hit, miss);
								total_hit += hit;
								total_miss += miss;
								if ((hit != 0) || (miss != 0))
								{
									if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
									else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
								}
								old_complete = f + 1;
							}
						}
#endif
					}

					if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, size);

					// download updated v
					cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_DEBUG
					cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#endif
					checkCudaErrors(cudaStreamSynchronize(stream));
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];

						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));

						for (int f = 0; f < dim_limit; f++)
						{
							// start search and find minimum value
							if (f < dim)
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							else
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim, dim2);
							// min is now set so we need to find the correspoding minima for free and taken columns
							int min_count = grid_size_min.x * (block_size.x >> 5);
							if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							if (t == 0)
							{
#ifndef LAP_QUIET
								if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
								if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
								scancount[f]++;
#endif
								count++;

								unassignedfound = false;

								// Dijkstra search
								min = host_min_private[0].min;
								jmin = host_min_private[0].jmin;
								colsol_old = host_min_private[0].colsol;
								for (int t = 1; t < devices; t++)
								{
									if ((host_min_private[t].min < min) || ((host_min_private[t].min == min) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
									{
										min = host_min_private[t].min;
										jmin = host_min_private[t].jmin + iterator.ws.part[t].first;
										colsol_old = host_min_private[t].colsol;
									}
								}

								// dijkstraCheck
								if (colsol_old < 0)
								{
									endofpath = jmin;
									unassignedfound = true;
								}
								else
								{
									unassignedfound = false;
								}
							}
#pragma omp barrier
#ifdef LAP_CUDA_COMPARE_CPU
							{
								cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
								cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
								if (t == 0)
								{
									SC min_tmp = std::numeric_limits<SC>::max();
									int jmin_tmp = dim2;
									int colsol_old_tmp = 0;
									for (int j = 0; j < dim2; j++)
									{
										if (colactive_tmp[j] != 0)
										{
											if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
											{
												min_tmp = d_tmp[j];
												jmin_tmp = j;
												colsol_old_tmp = colsol[j];
											}
										}
									}
									if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
									{
										std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
									}
								}
							}
#endif
							// mark last column scanned
							if (f >= dim) markedSkippedColumns_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, size);

							bool fast = unassignedfound;

							while (!unassignedfound)
							{
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
								int i = colsol_old;
								if (i < dim)
								{
									// get row
									auto tt = iterator.getRow(t, i);
									// single device
									if ((jmin >= start) && (jmin < end))
									{
										setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, tt_jmin, &(tt[jmin - start]), v_jmin, &(v_private[t][jmin - start]));
										checkCudaErrors(cudaStreamSynchronize(stream));
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if ((jmin >= start) && (jmin < end))
									{
										setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, v_jmin, &(v_private[t][jmin - start]));
										checkCudaErrors(cudaStreamSynchronize(stream));
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
								// min is now set so we need to find the correspoding minima for free and taken columns
								int min_count = grid_size_min.x * (block_size.x >> 5);
								if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
								if (t == 0)
								{
#ifndef LAP_QUIET
									if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
									if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
									scancount[i]++;
#endif
									count++;

									min_n = host_min_private[0].min;
									jmin = host_min_private[0].jmin;
									colsol_old = host_min_private[0].colsol;
									for (int t = 1; t < devices; t++)
									{
										if ((host_min_private[t].min < min_n) || ((host_min_private[t].min == min_n) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
										{
											min_n = host_min_private[t].min;
											jmin = host_min_private[t].jmin + iterator.ws.part[t].first;
											colsol_old = host_min_private[t].colsol;
										}
									}

									min = std::max(min, min_n);

									// dijkstraCheck
									if (colsol_old < 0)
									{
										endofpath = jmin;
										unassignedfound = true;
									}
									else
									{
										unassignedfound = false;
									}
								}
#pragma omp barrier
#ifdef LAP_CUDA_COMPARE_CPU
								{
									cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
									cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
									checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
									if (t == 0)
									{
										SC min_tmp = std::numeric_limits<SC>::max();
										int jmin_tmp = dim2;
										int colsol_old_tmp = 0;
										for (int j = 0; j < dim2; j++)
										{
											if (colactive_tmp[j] != 0)
											{
												if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
												{
													min_tmp = d_tmp[j];
													jmin_tmp = j;
													colsol_old_tmp = colsol[j];
												}
											}
										}
										if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
										{
											std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
										}
									}
								}
#endif
								// mark last column scanned
								if (i >= dim) markedSkippedColumnsUpdate_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, size);
							}

							// update column prices. can increase or decrease
							if (fast)
							{
								if ((endofpath >= start) && (endofpath < end))
								{
									colsol[endofpath] = f;
									rowsol[f] = endofpath;
									if (epsilon > SC(0))
									{
										if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
										else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
								}
								else
								{
									if (epsilon > SC(0))
									{
										if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
										else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
									}
								}
							}
							else
							{
								if (epsilon > SC(0))
								{
									if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
									else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
								}
								else
								{
									updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
								}
								// reset row and column assignments along the alternating path.
								cudaMemcpyAsync(pred + start, pred_private[t], size * sizeof(int), cudaMemcpyDeviceToHost, stream);
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
#ifdef LAP_ROWS_SCANNED
								if (t == 0) {
									int i;
									int eop = endofpath;
									do
									{
										i = pred[eop];
										eop = rowsol[i];
										if (i != f) pathlength[f]++;
									} while (i != f);
								}
#endif
								if (t == 0) resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
#pragma omp barrier
								cudaMemcpyAsync(colsol_private[t], colsol + start, size * sizeof(int), cudaMemcpyHostToDevice, stream);
							}
#ifndef LAP_QUIET
							if (t == 0)
							{
								int level;
								if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
								{
									long long hit, miss;
									iterator.getHitMiss(hit, miss);
									total_hit += hit;
									total_miss += miss;
									if ((hit != 0) || (miss != 0))
									{
										if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
										else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
									}
									old_complete = f + 1;
								}
							}
#endif
						}

						if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, size);

						// download updated v
						cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_DEBUG
						cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#endif
						checkCudaErrors(cudaStreamSynchronize(stream));
					} // end of #pragma omp parallel
#else
					for (int f = 0; f < dim_limit; f++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size, grid_size_min;
							block_size.x = 256;
							grid_size.x = (size + block_size.x - 1) / block_size.x;
							grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
								std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
							// start search and find minimum value
							if (f < dim)
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							else
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim, dim2);
							// min is now set so we need to find the correspoding minima for free and taken columns
							int min_count = grid_size_min.x * (block_size.x >> 5);
							if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						}
						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#ifndef LAP_QUIET
						if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
						if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
						scancount[f]++;
#endif
						count++;

						unassignedfound = false;

						// Dijkstra search
						min = host_min_private[0].min;
						jmin = host_min_private[0].jmin;
						colsol_old = host_min_private[0].colsol;
						for (int t = 1; t < devices; t++)
						{
							if ((host_min_private[t].min < min) || ((host_min_private[t].min == min) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
							{
								min = host_min_private[t].min;
								jmin = host_min_private[t].jmin;
								colsol_old = host_min_private[t].colsol;
							}
						}

						// dijkstraCheck
						if (colsol_old < 0)
						{
							endofpath = jmin;
							unassignedfound = true;
						}
						else
						{
							unassignedfound = false;
						}
						if (f >= dim)
						{
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								markedSkippedColumns_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, size);
							}
						}

#ifdef LAP_CUDA_COMPARE_CPU
						{
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
								cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
							}
							for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
							SC min_tmp = std::numeric_limits<SC>::max();
							int jmin_tmp = dim2;
							int colsol_old_tmp = 0;
							for (int j = 0; j < dim2; j++)
							{
								if (colactive_tmp[j] != 0)
								{
									if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
									{
										min_tmp = d_tmp[j];
										jmin_tmp = j;
										colsol_old_tmp = colsol[j];
									}
								}
							}
							if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
							{
								std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
							}
						}
#endif
						bool fast = unassignedfound;
						while (!unassignedfound)
						{
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
							int i = colsol_old;
							if (i < dim)
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									// get row
									tt[t] = iterator.getRow(t, i);
									// initialize Search
									if ((jmin >= start) && (jmin < end))
									{
										setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, tt_jmin, &(tt[jmin - start]), v_jmin, &(v_private[t][jmin - start]));
									}
								}
								// single device
								checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[iterator.ws.find(jmin)]));
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size, grid_size_min;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
										std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							else
							{
								{
									int t = iterator.ws.find(jmin);
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									cudaStream_t stream = iterator.ws.stream[t];
									setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, v_jmin, &(v_private[t][jmin - start]));
									checkCudaErrors(cudaStreamSynchronize(stream));
								}
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size, grid_size_min;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
										std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size, grid_size_min;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t] *
									std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
								// min is now set so we need to find the correspoding minima for free and taken columns
								int min_count = grid_size_min.x * (block_size.x >> 5);
								if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							}
							for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#ifndef LAP_QUIET
							if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
							if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
							scancount[i]++;
#endif
							count++;

							min_n = host_min_private[0].min;
							jmin = host_min_private[0].jmin;
							colsol_old = host_min_private[0].colsol;
							for (int t = 1; t < devices; t++)
							{
								if ((host_min_private[t].min < min_n) || ((host_min_private[t].min == min_n) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
								{
									min_n = host_min_private[t].min;
									jmin = host_min_private[t].jmin;
									colsol_old = host_min_private[t].colsol;
								}
							}


							min = std::max(min, min_n);

							// dijkstraCheck
							if (colsol_old < 0)
							{
								endofpath = jmin;
								unassignedfound = true;
							}
							else
							{
								unassignedfound = false;
							}

							// mark last column scanned (single device)
							if (i >= dim)
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									markedSkippedColumnsUpdate_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, size);
								}
							}

#ifdef LAP_CUDA_COMPARE_CPU
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
									cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
								}
								for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
								SC min_tmp = std::numeric_limits<SC>::max();
								int jmin_tmp = dim2;
								int colsol_old_tmp = 0;
								for (int j = 0; j < dim2; j++)
								{
									if (colactive_tmp[j] != 0)
									{
										if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
										{
											min_tmp = d_tmp[j];
											jmin_tmp = j;
											colsol_old_tmp = colsol[j];
										}
									}
								}
								if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
								{
									std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
								}
							}
#endif
						}

						// update column prices. can increase or decrease
						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								if ((endofpath >= start) && (endofpath < end))
								{
									if (epsilon > SC(0))
									{
										if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
										else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
								}
								else
								{
									if (epsilon > SC(0))
									{
										if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
										else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
									}
								}
							}
						}
						else
						{
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								if (epsilon > SC(0))
								{
									if (clamp) updateColumnPricesClamp_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
									else updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
								}
								else
								{
									updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
								}
								cudaMemcpyAsync(pred + start, pred_private[t], size * sizeof(int), cudaMemcpyDeviceToHost, stream);
							}
							// reset row and column assignments along the alternating path.
							for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#ifdef LAP_ROWS_SCANNED
							{
								int i;
								int eop = endofpath;
								do
								{
									i = pred[eop];
									eop = rowsol[i];
									if (i != f) pathlength[f]++;
								} while (i != f);
							}
#endif
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								cudaMemcpyAsync(colsol_private[t], colsol + start, size * sizeof(int), cudaMemcpyHostToDevice, stream);
							}
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
							{
								long long hit, miss;
								iterator.getHitMiss(hit, miss);
								total_hit += hit;
								total_miss += miss;
								if ((hit != 0) || (miss != 0))
								{
									if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
									else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
								}
								old_complete = f + 1;
							}
						}
#endif
					}

					// download updated v
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, size);
						cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_DEBUG
						cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#endif
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#endif
				}
#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
					if (devices == 1)
					{
						cudaSetDevice(iterator.ws.device[0]);
						int start = iterator.ws.part[0].first;
						int end = iterator.ws.part[0].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[0];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						findMaximum(v_private[0], &(host_min_private[0]), stream, dim2);
						subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
					}
					else
					{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
						{
							int t = omp_get_thread_num();
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (size + block_size.x - 1) / block_size.x;
							findMaximum(v_private[t], &(host_min_private[t]), stream, size);
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC max_v = mergeMaximum<SC>(host_min_private, devices);
							subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, size);
						}
#else
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							findMaximum(v_private[t], &(host_min_private[t]), stream, size);
						}
						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
						SC max_v = mergeMaximum<SC>(host_min_private, devices);
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (num_items + block_size.x - 1) / block_size.x;
							subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
						}
#endif
					}
				}
#endif
				// get total_d and total_eps (total_eps was already fixed for the dim2 != dim_limit case
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel for reduction (+:total_d) reduction (+:total_eps)
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
				}
#else
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
				}
#endif

#ifdef LAP_DEBUG
				if (epsilon > SC(0))
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
				else
				{
					int count = (int)v_list.size();
					if (count > 0)
					{
						for (int l = 0; l < count; l++)
						{
							SC dlt(0), dlt2(0);
							for (int i = 0; i < dim2; i++)
							{
								SC diff = v_list[l][i] - v[i];
								dlt += diff;
								dlt2 += diff * diff;
							}
							dlt /= SC(dim2);
							dlt2 /= SC(dim2);
							lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
							lapFree(v_list[l]);
						}
					}
				}
#endif
				second = first;
				first = false;

#ifndef LAP_QUIET
				lapInfo << "  rows evaluated: " << total_rows;
				if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
				lapInfo << std::endl;
				if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
			}

#ifdef LAP_QUIET
#ifdef LAP_DISPLAY_EVALUATED
			iterator.getHitMiss(total_hit, total_miss);
			lapInfo << "  rows evaluated: " << total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
#endif

#ifdef LAP_ROWS_SCANNED
			lapInfo << "row\tscanned\tlength" << std::endl;
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << f << "\t" << scancount[f] << "\t" << pathlength[f] << std::endl;
			}

			lapFree(scancount);
#endif

			for (int j = 0; j < dim2; j++) rowsol[colsol[j]] = j;

			// free CUDA memory
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(min_private[t]);
				cudaFree(jmin_private[t]);
				cudaFree(csol_private[t]);
				cudaFree(colactive_private[t]);
				cudaFree(d_private[t]);
				cudaFree(v_private[t]);
				cudaFree(total_d_private[t]);
				cudaFree(total_eps_private[t]);
				cudaFree(pred_private[t]);
				cudaFree(colsol_private[t]);
			}

			// free reserved memory.
#ifdef LAP_DEBUG
			cudaFreeHost(v);
#endif
			cudaFreeHost(colsol);
			cudaFreeHost(pred);
			cudaFreeHost(h_total_d);
			cudaFreeHost(h_total_eps);
			cudaFreeHost(tt_jmin);
			cudaFreeHost(v_jmin);
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(csol_private);
			cudaFreeHost(host_min_private);
			lapFree(colactive_private);
			lapFree(pred_private);
			lapFree(d_private);
			lapFree(colsol_private);
			lapFree(v_private);
			lapFree(total_d_private);
			lapFree(total_eps_private);
#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#else
			lapFree(tt);
#endif
#ifdef LAP_CUDA_COMPARE_CPU
			lapFree(d_tmp);
			lapFree(colactive_tmp);
#endif
			// set device back to first one
			cudaSetDevice(iterator.ws.device[0]);
		}

		template <class SC, class TC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			TC *row = new TC[dim];
			int *d_rowsol;
			TC *d_row;
			cudaMalloc(&d_rowsol, dim * sizeof(int));
			cudaMalloc(&d_row, dim * sizeof(TC));
			cudaMemcpyAsync(d_rowsol, rowsol, dim * sizeof(int), cudaMemcpyHostToDevice, stream);
			costfunc.getCost(d_row, stream, d_rowsol, dim);
			cudaMemcpyAsync(row, d_row, dim * sizeof(TC), cudaMemcpyDeviceToHost, stream);
			checkCudaErrors(cudaStreamSynchronize(stream));
			cudaFree(d_row);
			cudaFree(d_rowsol);
			for (int i = 0; i < dim; i++) my_cost += row[i];
			delete[] row;
			return my_cost;
		}

		// shortcut for square problems
		template <class SC, class TC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
		{
			lap::cuda::solve<SC, TC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
		}

		// shortcut for square problems
		template <class SC, class TC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			return lap::cuda::cost<SC, TC, CF>(dim, dim, costfunc, rowsol, stream);
		}
	}
}
