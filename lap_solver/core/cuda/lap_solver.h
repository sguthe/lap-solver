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

		template <class SC, class TC, class I>
		SC guessEpsilon(int x_size, int y_size, I& iterator)
		{
			SC epsilon(0);
			int devices = (int)iterator.ws.device.size();
#ifdef LAP_CUDA_OPENMP
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#endif
			TC* minmax_cost;
			TC** d_out;
			cudaMallocHost(&minmax_cost, 2 * x_size * devices * sizeof(TC));
			lapAlloc(d_out, devices, __FILE__, __LINE__);
			memset(minmax_cost, 0, 2 * sizeof(TC) * devices);
			memset(d_out, 0, sizeof(TC*) * devices);
			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				cudaStream_t stream = iterator.ws.stream[0];
				cudaMalloc(&(d_out[0]), 2 * x_size * sizeof(TC));
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (x_size + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min((num_items + block_size.x - 1) / block_size.x, (unsigned int)iterator.ws.sm_count[0] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), ((num_items + block_size.x - 1) / block_size.x) / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
				initMinMax_kernel<<<grid_size, block_size, 0, stream>>>(d_out[0], std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), x_size);
				for (int x = x_size - 1; x >= 0; --x)
				{
					auto tt = iterator.getRow(0, x);
					minMax_kernel<<<grid_size_min, block_size, 0, stream>>>(d_out[0] + 2 * x, tt, std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), num_items);
				}
				cudaMemcpyAsync(minmax_cost, d_out[0], 2 * x_size * sizeof(TC), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				for (int x = 0; x < x_size; x++)
				{
					SC min_cost, max_cost;
					min_cost = (SC)minmax_cost[2 * x];
					max_cost = (SC)minmax_cost[2 * x + 1];
					epsilon += max_cost - min_cost;
				}
			}
			else
			{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel for
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMalloc(&(d_out[t]), 2 * x_size * sizeof(TC));
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (x_size + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min((num_items + block_size.x - 1) / block_size.x, (unsigned int)iterator.ws.sm_count[t] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), ((num_items + block_size.x - 1) / block_size.x) / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
					initMinMax_kernel<<<grid_size, block_size, 0, stream>>>(d_out[t], std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), x_size);
					for (int x = x_size - 1; x >= 0; --x)
					{
						auto tt = iterator.getRow(t, x);
						minMax_kernel<<<grid_size_min, block_size, 0, stream>>>(d_out[t] + 2 * x, tt, std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), num_items);
					}
					cudaMemcpyAsync(&(minmax_cost[2 * t * x_size]), d_out[t], 2 * x_size * sizeof(TC), cudaMemcpyDeviceToHost, stream);
					cudaStreamSynchronize(stream);
				}
#else
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMalloc(&(d_out[t]), 2 * x_size * sizeof(TC));
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (x_size + block_size.x - 1) / block_size.x;
					initMinMax_kernel<<<grid_size, block_size, 0, stream>>>(d_out[t], std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), x_size);
				}
				for (int x = x_size - 1; x >= 0; --x)
				{
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto tt = iterator.getRow(t, x);
						dim3 block_size, grid_size_min;
						block_size.x = 256;
						grid_size_min.x = std::min((num_items + block_size.x - 1) / block_size.x, (unsigned int)iterator.ws.sm_count[t] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[t] / block_size.x), ((num_items + block_size.x - 1) / block_size.x) / (8u * (unsigned int)iterator.ws.sm_count[t])), 1u));
						minMax_kernel<<<grid_size_min, block_size, 0, stream>>>(d_out[t] + 2 * x, tt, std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), num_items);
					}
				}
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(&(minmax_cost[2 * t * x_size]), d_out[t], 2 * x_size * sizeof(TC), cudaMemcpyDeviceToHost, stream);
				}
				for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
#endif
				for (int x = 0; x < x_size; x++)
				{
					SC min_cost, max_cost;
					min_cost = (SC)minmax_cost[2 * x];
					max_cost = (SC)minmax_cost[2 * x + 1];
					for (int t = 1; t < devices; t++)
					{
						min_cost = std::min(min_cost, (SC)minmax_cost[2 * (t * x_size + x)]);
						max_cost = std::max(max_cost, (SC)minmax_cost[2 * (t * x_size + x) + 1]);
					}
					epsilon += max_cost - min_cost;
				}
			}
			cudaFreeHost(minmax_cost);
			lapFree(d_out);
#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#endif
			return epsilon / (SC(10) * SC(x_size));
		}

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			int jmin;
			int colsol;
		};

		class reset_struct
		{
		public:
			int i;
			int j;
		};

		template <class SC, class TC>
		__global__ void initializeSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = (SC)tt[j] - v[j];
				int c_colsol = colsol[j];
				if ((v0 < v_min) || ((v0 == v_min) && (c_colsol < 0) && (v_colsol >= 0)))
				{
					v_min = v0;
					v_jmin = j;
					v_colsol = c_colsol;
				}
				j += blockDim.x * gridDim.x;
			}


			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = v_min;
				o_jmin[i] = v_jmin;
				o_colsol[i] = v_colsol;
			}
		}

		template <class SC, class TC>
		__global__ void continueSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, SC h2, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC h = d[j];
				SC v2 = (SC)tt[j] - v[j] - h2;
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
					if ((h < v_min) || ((h == v_min) && (c_colsol < 0) && (v_colsol >= 0)))
					{
						v_min = h;
						v_jmin = j;
						v_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = v_min;
				o_jmin[i] = v_jmin;
				o_colsol[i] = v_colsol;
			}
		}

		template <class SC>
		__global__ void initializeSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = -v[j];
				int c_colsol = colsol[j];
				if ((v0 < v_min) || ((v0 == v_min) && (c_colsol < 0) && (v_colsol >= 0)))
				{
					v_min = v0;
					v_jmin = j;
					v_colsol = c_colsol;
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = v_min;
				o_jmin[i] = v_jmin;
				o_colsol[i] = v_colsol;
			}
		}

		template <class SC>
		__global__ void continueSearchMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, SC h2, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC h = d[j];
				SC v2 = -v[j] - h2;
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
					if ((h < v_min) || ((h == v_min) && (c_colsol < 0) && (v_colsol >= 0)))
					{
						v_min = h;
						v_jmin = j;
						v_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = v_min;
				o_jmin[i] = v_jmin;
				o_colsol[i] = v_colsol;
			}
		}

		template <class SC, class TC>
		__global__ void continueSearchJMinMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC v_min = max;
			int v_jmin = dim2;
			SC h2 = (SC)tt[jmin] - v[jmin] - min;
			int v_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC h = d[j];
				SC v2 = (SC)tt[j] - v[j] - h2;
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
					if ((h < v_min) || ((h == v_min) && (c_colsol < 0) && (v_colsol >= 0)))
					{
						v_min = h;
						v_jmin = j;
						v_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = v_min;
				o_jmin[i] = v_jmin;
				o_colsol[i] = v_colsol;
			}
		}

		template <class SC>
		__global__ void continueSearchJMinMin_kernel(SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC v_min = max;
			int v_jmin = dim2;
			SC h2 = -v[jmin] - min;
			int v_colsol = 0;

#pragma unroll 8
			while (j < size)
			{
				SC h = d[j];
				SC v2 = -v[j] - h2;
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
					if ((h < v_min) || ((h == v_min) && (c_colsol < 0) && (v_colsol >= 0)))
					{
						v_min = h;
						v_jmin = j;
						v_colsol = c_colsol;
					}
				}
				j += blockDim.x * gridDim.x;
			}

			minWarpIndex(v_min, v_jmin, v_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				int i = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
				o_min[i] = v_min;
				o_jmin[i] = v_jmin;
				o_colsol[i] = v_colsol;
			}
		}

		template <class SC>
		__global__ void findMinSmall_kernel(min_struct<SC> *s, SC *min, int *jmin, int *colsol, SC max, int size, int dim2)
		{
			int j = threadIdx.x;

			SC v_min = max;
			int v_jmin = dim2;
			int v_colsol = 0;

			if (j < size)
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

			if (j < size)
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
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				v[j] = v[j] + d[j] - min;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				v[j] = v[j] + d[j] - min - eps;
			}
		}

		__global__ void resetRowColumnAssignment_kernel(int *pred, int *colsol, int *rowsol, int start, int end, int endofpath, int f)
		{
			int j = endofpath;
			int i;
			do
			{
				i = pred[j - start];
				colsol[j - start] = i;
				int j1 = rowsol[i];
				rowsol[i] = j;
				j = j1;
			} while (i != f);
		}

		__global__ void resetRowColumnAssignment_kernel(int *pred, int *colsol, int *rowsol, reset_struct *state, int start, int end, int endofpath, int f)
		{
			int j = endofpath;
			int i;
			do
			{
				i = pred[j - start];
				colsol[j - start] = i;
				int j1 = rowsol[i];
				rowsol[i] = j;
				if ((i != f) && (j1 < 0))
				{
					// not last element but not in this list
					state->i = i;
					state->j = j;
					return;
				}
				j = j1;
			} while (i != f);
			state->i = -1;
		}

		__global__ void resetRowColumnAssignmentContinue_kernel(int *pred, int *colsol, int *rowsol, reset_struct *state_in, reset_struct *state_out, int start, int end, int f)
		{
			int i = state_in->i;
			int j = rowsol[i];
			// rowsol for i not found in this range
			if ((j < 0) || (j == state_in->j)) return;
			// clear old rowsol since it is no longer in this range
			rowsol[i] = -1;
			do 
			{
				i = pred[j - start];
				colsol[j - start] = i;
				int j1 = rowsol[i];
				rowsol[i] = j;
				if ((i != f) && (j1 < 0))
				{
					// not last element but not in this list
					state_out->i = i;
					state_out->j = j;
					return;
				}
				j = j1;
			} while (i != f);
			state_out->i = -1;
		}

		template <class SC, class TC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol)

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
			SC *v;
			SC *v2;
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
			int *colsol_tmp;
			lapAlloc(d_tmp, dim2, __FILE__, __LINE__);
			lapAlloc(colactive_tmp, dim2, __FILE__, __LINE__);
			lapAlloc(colsol_tmp, dim2, __FILE__, __LINE__);
#endif

			// used for copying
			min_struct<SC> *host_min_private;
			reset_struct *host_reset;
			cudaMallocHost(&host_reset, 2 * sizeof(reset_struct));
			cudaMallocHost(&v, dim2 * sizeof(SC));
			cudaMallocHost(&v2, dim2 * sizeof(SC));
			cudaMallocHost(&host_min_private, devices * sizeof(min_struct<SC>));
			cudaMallocHost(&tt_jmin, sizeof(TC));
			cudaMallocHost(&v_jmin, sizeof(SC));

			SC **min_private;
			int **jmin_private;
			int **csol_private;
			char **colactive_private;
			int **pred_private;
			SC **d_private;
			int **colsol_private;
			int **rowsol_private;
			SC **v_private;
			// on device
			lapAlloc(min_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(csol_private, devices, __FILE__, __LINE__);
			lapAlloc(colactive_private, devices, __FILE__, __LINE__);
			lapAlloc(pred_private, devices, __FILE__, __LINE__);
			lapAlloc(d_private, devices, __FILE__, __LINE__);
			lapAlloc(colsol_private, devices, __FILE__, __LINE__);
			lapAlloc(rowsol_private, devices, __FILE__, __LINE__);
			lapAlloc(v_private, devices, __FILE__, __LINE__);

			for (int t = 0; t < devices; t++)
			{
				// single device code
				cudaSetDevice(iterator.ws.device[t]);
				int start = iterator.ws.part[t].first;
				int end = iterator.ws.part[t].second;
				int size = end - start;
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (size + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
					std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
				int count = (grid_size_min.x * block_size.x) >> 5;
				cudaMalloc(&(min_private[t]), sizeof(SC) * count);
				cudaMalloc(&(jmin_private[t]), sizeof(int) * count);
				cudaMalloc(&(csol_private[t]), sizeof(int) * count);
				cudaMalloc(&(colactive_private[t]), sizeof(char) * size);
				cudaMalloc(&(d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(v_private[t]), sizeof(SC) * size);
				cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
				cudaMalloc(&(pred_private[t]), sizeof(int) * size);
				cudaMalloc(&(rowsol_private[t]), sizeof(int) * dim2);
			}

#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
#endif

			// this is the upper bound
			SC epsilon = (SC)costfunc.getInitialEpsilon();
			SC epsilon_lower = epsilon / SC(dim2);

			SC last_avg = SC(0);
			bool first = true;
			bool allow_reset = true;

			memset(v, 0, dim2 * sizeof(SC));

			while (epsilon >= SC(0))
			{
				lap::getNextEpsilon(epsilon, epsilon_lower, last_avg, first, allow_reset, v, dim2);
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

#ifdef LAP_CUDA_OPENMP
				if (devices == 1)
				{
					// upload v to devices
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
					cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
				}
				else
				{
					// upload v to devices
#pragma omp parallel for
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
						int start = iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
						cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
						cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
					}
				}
#else
				for (int t = 0; t < devices; t++)
				{
					// upload v to devices
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
					cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
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
				SC h2(0);

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
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
						std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));

					for (int f = 0; f < dim2; f++)
					{
						// start search and find minimum value
						if (f < dim)
							initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
						else
							initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
						// min is now set so we need to find the correspoding minima for free and taken columns
						int min_count = grid_size_min.x * (block_size.x >> 5);
						if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						cudaStreamSynchronize(stream);
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
							cudaMemcpyAsync(colsol_tmp, colsol_private[t], dim2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
							cudaStreamSynchronize(stream);
							SC min_tmp = std::numeric_limits<SC>::max();
							int jmin_tmp = dim2;
							int colsol_old_tmp = 0;
							for (int j = 0; j < dim2; j++)
							{
								if (colactive_tmp[j] != 0)
								{
									if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol_tmp[j] < 0) && (colsol_old_tmp >= 0)))
									{
										min_tmp = d_tmp[j];
										jmin_tmp = j;
										colsol_old_tmp = colsol_tmp[j];
									}
								}
							}
							if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
							{
								std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
							}
						}
#endif

						setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], jmin - start);

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
							cudaStreamSynchronize(stream);
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
								cudaMemcpyAsync(colsol_tmp, colsol_private[t], dim2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
								cudaStreamSynchronize(stream);
								SC min_tmp = std::numeric_limits<SC>::max();
								int jmin_tmp = dim2;
								int colsol_old_tmp = 0;
								for (int j = 0; j < dim2; j++)
								{
									if (colactive_tmp[j] != 0)
									{
										if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol_tmp[j] < 0) && (colsol_old_tmp >= 0)))
										{
											min_tmp = d_tmp[j];
											jmin_tmp = j;
											colsol_old_tmp = colsol_tmp[j];
										}
									}
								}
								if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
								{
									std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
								}
							}
#endif

							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], jmin - start);
						}

						// update column prices. can increase or decrease
						if ((epsilon > SC(0)) && (f + 1 < dim2))
						{
							updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], epsilon, size);
						}
						else
						{
							updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
						}
						// reset row and column assignments along the alternating path.
						resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], start, end, endofpath, f);
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim2, " rows")) != 0)
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
					cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					cudaStreamSynchronize(stream);
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
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
							std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));

						for (int f = 0; f < dim2; f++)
						{
							// start search and find minimum value
							if (f < dim)
								initializeSearchMin_kernel << <grid_size_min, block_size, 0, stream >> > (min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							else
								initializeSearchMin_kernel << <grid_size_min, block_size, 0, stream >> > (min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							// min is now set so we need to find the correspoding minima for free and taken columns
							int min_count = grid_size_min.x * (block_size.x >> 5);
							if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							cudaStreamSynchronize(stream);
#pragma omp barrier
#pragma omp master
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
								cudaMemcpyAsync(&(colsol_tmp[start]), colsol_private[t], size * sizeof(int), cudaMemcpyDeviceToHost, stream);
								cudaStreamSynchronize(stream);
#pragma omp barrier
#pragma omp master
								{
									SC min_tmp = std::numeric_limits<SC>::max();
									int jmin_tmp = dim2;
									int colsol_old_tmp = 0;
									for (int j = 0; j < dim2; j++)
									{
										if (colactive_tmp[j] != 0)
										{
											if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol_tmp[j] < 0) && (colsol_old_tmp >= 0)))
											{
												min_tmp = d_tmp[j];
												jmin_tmp = j;
												colsol_old_tmp = colsol_tmp[j];
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
							// mark last column scanned (single device)
							if ((jmin >= start) && (jmin < end))
							{
								setColInactive_kernel << <1, 1, 0, iterator.ws.stream[t] >> > (colactive_private[t], jmin - start);
							}
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
										cudaMemcpyAsync(tt_jmin, &(tt[jmin - start]), sizeof(TC), cudaMemcpyDeviceToHost, stream);
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
										cudaStreamSynchronize(stream);
										h2 = tt_jmin[0] - v_jmin[0] - min;
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
										cudaStreamSynchronize(stream);
										h2 = -v_jmin[0] - min;
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size, dim2);
								}
								// min is now set so we need to find the correspoding minima for free and taken columns
								int min_count = grid_size_min.x * (block_size.x >> 5);
								if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								cudaStreamSynchronize(stream);
#pragma omp barrier
#pragma omp master
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
									cudaMemcpyAsync(&(colsol_tmp[start]), colsol_private[t], size * sizeof(int), cudaMemcpyDeviceToHost, stream);
									cudaStreamSynchronize(stream);
#pragma omp barrier
#pragma omp master
									{
										SC min_tmp = std::numeric_limits<SC>::max();
										int jmin_tmp = dim2;
										int colsol_old_tmp = 0;
										for (int j = 0; j < dim2; j++)
										{
											if (colactive_tmp[j] != 0)
											{
												if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol_tmp[j] < 0) && (colsol_old_tmp >= 0)))
												{
													min_tmp = d_tmp[j];
													jmin_tmp = j;
													colsol_old_tmp = colsol_tmp[j];
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
								// mark last column scanned (single device)
								if ((jmin >= start) && (jmin < end))
								{
									setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], jmin - start);
								}
							}

							// update column prices. can increase or decrease
							if ((epsilon > SC(0)) && (f + 1 < dim2))
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], epsilon, size);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
							}
							// reset row and column assignments along the alternating path.
							if ((endofpath >= start) && (endofpath < end))
							{
								resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], host_reset, start, end, endofpath, f);
								cudaStreamSynchronize(stream);
							}
#pragma omp barrier
							while (host_reset[0].i != -1)
							{
								resetRowColumnAssignmentContinue_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], &(host_reset[0]), &(host_reset[1]), start, end, f);
								cudaStreamSynchronize(stream);
#pragma omp barrier
								if (host_reset[1].i != -1)
								{
									resetRowColumnAssignmentContinue_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], &(host_reset[1]), &(host_reset[0]), start, end, f);
									cudaStreamSynchronize(stream);
								}
								else
								{
									host_reset[0].i = -1;
								}
#pragma omp barrier
							}
#ifndef LAP_QUIET
#pragma omp master
							{
								int level;
								if ((level = displayProgress(start_time, elapsed, f + 1, dim2, " rows")) != 0)
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
						cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						cudaStreamSynchronize(stream);
					} // end of #pragma omp parallel
#else
					for (int f = 0; f < dim2; f++)
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
							grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
								std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
							// start search and find minimum value
							if (f < dim)
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							else
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							// min is now set so we need to find the correspoding minima for free and taken columns
							int min_count = grid_size_min.x * (block_size.x >> 5);
							if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
						}
						for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
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
						// mark last column scanned (single device)
						{
							int t = iterator.ws.find(jmin);
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], jmin - start);
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
								cudaMemcpyAsync(&(colsol_tmp[start]), colsol_private[t], size * sizeof(int), cudaMemcpyDeviceToHost, stream);
							}
							for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
							SC min_tmp = std::numeric_limits<SC>::max();
							int jmin_tmp = dim2;
							int colsol_old_tmp = 0;
							for (int j = 0; j < dim2; j++)
							{
								if (colactive_tmp[j] != 0)
								{
									if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol_tmp[j] < 0) && (colsol_old_tmp >= 0)))
									{
										min_tmp = d_tmp[j];
										jmin_tmp = j;
										colsol_old_tmp = colsol_tmp[j];
									}
								}
							}
							if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
							{
								std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
							}
						}
#endif
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
//									initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(tt_jmin, &(tt[t][jmin - start]), sizeof(TC), cudaMemcpyDeviceToHost, stream);
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
									}
								}
								// single device
								cudaStreamSynchronize(iterator.ws.stream[iterator.ws.find(jmin)]);
								h2 = tt_jmin[0] - v_jmin[0] - min;
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
									grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
										std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							else
							{
								{
									int t = iterator.ws.find(jmin);
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									cudaStream_t stream = iterator.ws.stream[t];
									cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
									cudaStreamSynchronize(stream);
								}
								h2 = -v_jmin[0] - min;
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
									grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
										std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size, dim2);
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
								grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0] *
									std::max(std::min((unsigned int)(iterator.ws.threads_per_sm[0] / block_size.x), grid_size.x / (8u * (unsigned int)iterator.ws.sm_count[0])), 1u));
								// min is now set so we need to find the correspoding minima for free and taken columns
								int min_count = grid_size_min.x * (block_size.x >> 5);
								if (min_count <= 32) findMinSmall_kernel<<<1, 32, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else if (min_count <= 256) findMinMedium_kernel<<<1, 256, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
								else findMinLarge_kernel<<<1, 1024, 0, stream>>>(&(host_min_private[t]), min_private[t], jmin_private[t], csol_private[t], std::numeric_limits<SC>::max(), min_count, dim2);
							}
							for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
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
							{
								int t = iterator.ws.find(jmin);
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								cudaStream_t stream = iterator.ws.stream[t];
								setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], jmin - start);
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
									cudaMemcpyAsync(&(colsol_tmp[start]), colsol_private[t], size * sizeof(int), cudaMemcpyDeviceToHost, stream);
								}
								for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
								SC min_tmp = std::numeric_limits<SC>::max();
								int jmin_tmp = dim2;
								int colsol_old_tmp = 0;
								for (int j = 0; j < dim2; j++)
								{
									if (colactive_tmp[j] != 0)
									{
										if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol_tmp[j] < 0) && (colsol_old_tmp >= 0)))
										{
											min_tmp = d_tmp[j];
											jmin_tmp = j;
											colsol_old_tmp = colsol_tmp[j];
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
						// need to copy pred from GPUs here
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
							if ((epsilon > SC(0)) && (f + 1 < dim2))
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], epsilon, size);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
							}
						}
						// reset row and column assignments along the alternating path.
						{
							{
								int t = iterator.ws.find(endofpath);
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								cudaStream_t stream = iterator.ws.stream[t];
								resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], host_reset, start, end, endofpath, f);
								cudaStreamSynchronize(stream);
							}
							while (host_reset->i != -1)
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									cudaStream_t stream = iterator.ws.stream[t];
									resetRowColumnAssignmentContinue_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], &(host_reset[0]), &(host_reset[1]), start, end, f);
								}
								for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
								std::swap(host_reset[0], host_reset[1]);
							}
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim2, " rows")) != 0)
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
						cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					}
					for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
#endif
				}

				// now we need to compare the previous v with the new one to get back all the changes
				SC total(0);
				SC scaled_epsilon = epsilon * SC(count) / SC(dim2);

				for (int i = 0; i < dim2; i++) total += v2[i] - v[i] + scaled_epsilon;

				if (count > 0) last_avg = total / SC(count);
				else last_avg = SC(0);
				std::swap(v, v2);

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
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << "row: " << f << " scanned: " << scancount[f] << std::endl;
			}

			lapFree(scancount);
#endif

			{
				int *colsol;
				cudaMallocHost(&colsol, dim2 * sizeof(int));
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(&(colsol[start]), colsol_private[t], sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
				}
				for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
				for (int j = 0; j < dim2; j++) rowsol[colsol[j]] = j;
				cudaFreeHost(colsol);
			}

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
				cudaFree(pred_private[t]);
				cudaFree(colsol_private[t]);
				cudaFree(rowsol_private[t]);
			}

			// free reserved memory.
			cudaFreeHost(host_reset);
			cudaFreeHost(v);
			cudaFreeHost(v2);
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
			lapFree(rowsol_private);
			lapFree(v_private);
#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#else
			lapFree(tt);
#endif
#ifdef LAP_CUDA_COMPARE_CPU
			lapFree(d_tmp);
			lapFree(colactive_tmp);
			lapFree(colsol_tmp);
#endif
			// set device back to first one
			cudaSetDevice(iterator.ws.device[0]);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			SC *row = new SC[dim2];
			int *d_rowsol;
			SC *d_row;
			cudaMalloc(&d_rowsol, dim2 * sizeof(int));
			cudaMalloc(&d_row, dim2 * sizeof(SC));
			cudaMemcpyAsync(d_rowsol, rowsol, dim2 * sizeof(int), cudaMemcpyHostToDevice, stream);
			costfunc.getCost(d_row, 0, d_rowsol, dim2);
			cudaMemcpyAsync(row, d_row, dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream);
			cudaFree(d_row);
			cudaFree(d_rowsol);
			cudaStreamSynchronize(stream);
			for (int i = 0; i < dim2; i++) my_cost += row[i];
			delete[] row;
			return my_cost;
		}

		// shortcut for square problems
		template <class SC, class TC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol)
		{
			lap::cuda::solve<SC, TC>(dim, dim, costfunc, iterator, rowsol);
		}

		// shortcut for square problems
		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			return lap::cuda::cost<SC, CF>(dim, dim, costfunc, rowsol, stream);
		}
	}
}
