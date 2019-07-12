#pragma once

#define NO_MIN_LOOP

// collection of all CUDA kernel required for the solver

namespace lap
{
	namespace cuda
	{
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

		template <class SC, class TC>
		__global__ void getMinMaxBest_kernel(SC *o_min_cost, SC *o_max_cost, SC *o_picked_cost, int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int size, int dim2)
		{
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min_cost = max;
			SC t_max_cost = min;
			SC t_picked_cost = max;
			int t_jmin = dim2;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
			{
				SC t_cost = (SC)tt[j];
				if (t_cost < t_min_cost) t_min_cost = t_cost;
				if (i == j) t_max_cost = t_cost;
				if ((t_cost < t_picked_cost) && (picked[j] == 0))
				{
					t_jmin = j;
					t_picked_cost = t_cost;
				}
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min_cost = max;
			SC t_second_cost = max;
			SC t_picked_cost = max;
			int t_jmin = dim2;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min_cost_real = max;
			SC t_min_cost = max;
			int t_jmin = dim2;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x) 
#endif
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min_cost = max;
			SC t_picked_cost = max;
			SC t_picked_v = max;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
			{
				SC t_cost = (SC)tt[j] - v[j];
				if (t_cost < t_min_cost) t_min_cost = t_cost;
				if (j == j_picked)
				{
					t_picked_cost = (SC)tt[j];
					t_picked_v = v[j];
				}
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

		template <class MS, class SC>
		__global__ void getMinMaxBestSmall_kernel(MS *s, SC *min_cost, SC *max_cost, SC *picked_cost, int *jmin, SC min, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinMaxBestMedium_kernel(MS *s, SC *min_cost, SC *max_cost, SC *picked_cost, int *jmin, SC min, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinMaxBestLarge_kernel(MS *s, SC *min_cost, SC *max_cost, SC *picked_cost, int *jmin, SC min, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinSecondBestSmall_kernel(MS *s, SC *min_cost, SC *second_cost, SC *picked_cost, int *jmin, SC *v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinSecondBestMedium_kernel(MS *s, SC *min_cost, SC *second_cost, SC *picked_cost, int *jmin, SC *v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinSecondBestLarge_kernel(MS *s, SC *min_cost, SC *second_cost, SC *picked_cost, int *jmin, SC *v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinimalCostSmall_kernel(MS *s, SC *min_cost, int *jmin, SC *min_cost_real, SC *v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinimalCostMedium_kernel(MS *s, SC *min_cost, int *jmin, SC *min_cost_real, SC *v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getMinimalCostLarge_kernel(MS *s, SC *min_cost, int *jmin, SC *min_cost_real, SC *v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getFinalCostSmall_kernel(MS *s, SC *min_cost, SC *picked_cost, SC *min_v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getFinalCostMedium_kernel(MS *s, SC *min_cost, SC *picked_cost, SC *min_v, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void getFinalCostLarge_kernel(MS *s, SC *min_cost, SC *picked_cost, SC *min_v, SC max, int size, int dim2)
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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
		__global__ void initializeSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = (SC)tt[j] - v[j];
				int c_colsol = colsol[j] = colsol_in[j];
				if ((v0 < t_min) || ((v0 == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
				{
					t_min = v0;
					t_jmin = j;
					t_colsol = c_colsol;
				}
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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
		__global__ void initializeSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim, int dim2)
		{
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = -v[j];
				int c_colsol = colsol[j] = colsol_in[j];
				if (c_colsol < dim)
				{
					if ((v0 < t_min) || ((v0 == t_min) && (c_colsol < 0) && (t_colsol >= 0)))
					{
						t_min = v0;
						t_jmin = j;
						t_colsol = c_colsol;
					}
				}
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			SC tt_jmin = (SC)tt[jmin];
			SC v_jmin = v[jmin];
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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
#ifdef NO_MIN_LOOP
			int j = threadIdx.x + blockIdx.x * blockDim.x;
#endif

			SC t_min = max;
			int t_jmin = dim2;
			SC v_jmin = v[jmin];
			int t_colsol = 0;

#ifdef NO_MIN_LOOP
			if (j < size)
#else
#pragma unroll 4
			for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
#endif
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

		template <class MS, class SC>
		__global__ void findMinSmall_kernel(MS *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void findMinMedium_kernel(MS *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
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

		template <class MS, class SC>
		__global__ void findMinLarge_kernel(MS *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
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
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int *dst, int *src, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			dst[j] = src[j];
			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				v[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int *dst, int *src, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			dst[j] = src[j];
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

		template <class C>
		__global__ void memcpy_kernel(C *dst, C *src, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

#pragma unroll 8
			for (int i = 0; i < 8; i++)
			{
				//if (j >= size) return;
				if (j < size)
				{
					dst[j] = src[j];
				}
				j += blockDim.x * gridDim.x;
			}
		}

		template <class C1, class C2>
		__global__ void memcpy2_kernel(C1 *dst1, C1 *src1, C2 *dst2, C2 *src2, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

#pragma unroll 4
			for (int i = 0; i < 4; i++)
			{
				//if (j >= size) return;
				if (j < size)
				{
					dst1[j] = src1[j];
					dst2[j] = src2[j];
				}
				j += blockDim.x * gridDim.x;
			}
		}
	}
}
