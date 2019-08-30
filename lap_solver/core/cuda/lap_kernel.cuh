#pragma once

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
		__global__ void updateVMulti_kernel(TC *tt, SC *v, int *taken, SC *p_min_cost, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			__shared__ SC b_min_cost;
			if (threadIdx.x == 0) b_min_cost = *p_min_cost;
			__syncthreads();
			SC min_cost = b_min_cost;

			if (taken[j] != 0)
			{
				SC cost_l = (SC)tt[j] - v[j];
				if (cost_l < min_cost) v[j] -= min_cost - cost_l;
			}
		}

		__device__ __forceinline__ bool semaphoreWarp(unsigned int *semaphore)
		{
			int sem;
			if (threadIdx.x == 0) sem = atomicInc(semaphore, gridDim.x - 1);
			sem = __shfl_sync(0xffffffff, sem, 0, 32);
			return (sem == gridDim.x - 1);
		}

		__device__ __forceinline__ bool semaphoreBlock(unsigned int *semaphore)
		{
			__shared__ int b_active;
			__syncthreads();
			if (threadIdx.x == 0)
			{
				int sem = atomicInc(semaphore, gridDim.x - 1);
				if (sem == gridDim.x - 1) b_active = 1;
				else b_active = 0;
			}
			__syncthreads();
			return (b_active != 0);
		}

		template <class SC, class TC>
		__device__ __forceinline__ void getMinMaxBestRead(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin, int i, int j, TC *tt, int *picked, SC min, SC max, int size, int dim2)
		{
			t_min_cost = max;
			t_max_cost = min;
			t_picked_cost = max;
			t_jmin = dim2;

			if (j < size)
			{
				SC t_cost = (SC)tt[j];
				t_min_cost = t_cost;
				if (i == j) t_max_cost = t_cost;
				if (picked[j] == 0)
				{
					t_jmin = j;
					t_picked_cost = t_cost;
				}
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestCombineSmall(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin)
		{
			minWarpIndex(t_picked_cost, t_jmin);
			minWarp(t_min_cost);
			maxWarp(t_max_cost);
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestCombineTiny(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin)
		{
			minWarpIndex8(t_picked_cost, t_jmin);
			minWarp8(t_min_cost);
			maxWarp8(t_max_cost);
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestWriteShared(SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin, SC t_min_cost, SC t_max_cost, SC t_picked_cost, int t_jmin)
		{
			int bidx = threadIdx.x >> 5;
			b_min_cost[bidx] = t_min_cost;
			b_max_cost[bidx] = t_max_cost;
			b_picked_cost[bidx] = t_picked_cost;
			b_jmin[bidx] = t_jmin;
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestReadShared(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin, SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin)
		{
			t_min_cost = b_min_cost[threadIdx.x];
			t_max_cost = b_max_cost[threadIdx.x];
			t_picked_cost = b_picked_cost[threadIdx.x];
			t_jmin = b_jmin[threadIdx.x];
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestCombineMedium(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin, SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin)
		{
			getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getMinMaxBestWriteShared(b_min_cost, b_max_cost, b_picked_cost, b_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			}
			__syncthreads();
			if (threadIdx.x < 8)
			{
				getMinMaxBestReadShared(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinMaxBestCombineTiny(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestCombineLarge(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin, SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin)
		{
			getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getMinMaxBestWriteShared(b_min_cost, b_max_cost, b_picked_cost, b_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			}
			__syncthreads();
			if (threadIdx.x < 32)
			{
				getMinMaxBestReadShared(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestWriteTemp(volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, SC t_min_cost, SC t_max_cost, SC t_picked_cost, int t_jmin)
		{
			if (threadIdx.x == 0)
			{
				o_min_cost[blockIdx.x] = t_min_cost;
				o_max_cost[blockIdx.x] = t_max_cost;
				o_picked_cost[blockIdx.x] = t_picked_cost;
				o_jmin[blockIdx.x] = t_jmin;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestReadTemp(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, SC min, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_max_cost = o_max_cost[threadIdx.x];
				t_picked_cost = o_picked_cost[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
			}
			else
			{
				t_min_cost = max;
				t_max_cost = min;
				t_picked_cost = max;
				t_jmin = dim2;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinMaxBestReadTempLarge(SC &t_min_cost, SC &t_max_cost, SC &t_picked_cost, int &t_jmin, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, SC min, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_max_cost = o_max_cost[threadIdx.x];
				t_picked_cost = o_picked_cost[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < gridDim.x; i += blockDim.x)
				{
					SC c_min_cost = o_min_cost[i];
					SC c_max_cost = o_max_cost[i];
					SC c_picked_cost = o_picked_cost[i];
					int c_jmin = o_jmin[i];
					if (c_min_cost < t_min_cost) t_min_cost = c_min_cost;
					if (c_max_cost > t_max_cost) t_max_cost = c_max_cost;
					if ((c_picked_cost < t_picked_cost) || ((c_picked_cost == t_picked_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_picked_cost = c_picked_cost;
					}
				}
			}
			else
			{
				t_min_cost = max;
				t_max_cost = min;
				t_picked_cost = max;
				t_jmin = dim2;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void getMinMaxBestWrite(MS *s, SC t_min_cost, SC t_max_cost, SC t_picked_cost, int t_jmin, int * data_valid)
		{
			if (threadIdx.x == 0)
			{
				s->min = t_min_cost;
				s->max = t_max_cost;
				s->picked = t_picked_cost;
				s->jmin = t_jmin;

				__threadfence_system();
				data_valid[0] = 1;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void getMinMaxBestSingleWrite(MS *s, volatile SC *o_min_cost, volatile int *o_jmin, SC t_min_cost, SC t_max_cost, SC t_picked_cost, int t_jmin, int i)
		{
			if (threadIdx.x == 0)
			{
				s->min = t_min_cost;
				s->max = t_max_cost;
				s->picked = t_picked_cost;
				s->jmin = t_jmin;

				o_min_cost[0] = t_min_cost;
				o_jmin[0] = t_jmin;
			}
		}

		// 32 threads per block, up to 32 blocks, requires no shared memory and no thread synchronization
		template <class MS, class SC, class TC>
		__global__ void getMinMaxBestSmall_kernel(MS *s, unsigned int *semaphore, int * data_valid, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int start, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			getMinMaxBestRead(t_min_cost, t_max_cost, t_picked_cost, t_jmin, i - start, j, tt, picked, min, max, size, dim2);
			getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			getMinMaxBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (semaphoreWarp(semaphore))
			{
				getMinMaxBestReadTemp(t_min_cost, t_max_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, min, max, dim2);
				getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
				getMinMaxBestWrite(s, t_min_cost, t_max_cost, t_picked_cost, t_jmin + start, data_valid);
			}
		}

		// 256 threads per block, up to 256 blocks
		template <class MS, class SC, class TC>
		__global__ void getMinMaxBestMedium_kernel(MS *s, unsigned int *semaphore, int * data_valid, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int start, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_max_cost[8], b_picked_cost[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			getMinMaxBestRead(t_min_cost, t_max_cost, t_picked_cost, t_jmin, i - start, j, tt, picked, min, max, size, dim2);
			getMinMaxBestCombineMedium(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinMaxBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinMaxBestReadTemp(t_min_cost, t_max_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, min, max, dim2);
				getMinMaxBestCombineMedium(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinMaxBestWrite(s, t_min_cost, t_max_cost, t_picked_cost, t_jmin + start, data_valid);
			}
		}

		// 1024 threads per block, can be more than 1024 blocks
		template <class MS, class SC, class TC>
		__global__ void getMinMaxBestLarge_kernel(MS *s, unsigned int *semaphore, int * data_valid, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int start, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_max_cost[32], b_picked_cost[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			getMinMaxBestRead(t_min_cost, t_max_cost, t_picked_cost, t_jmin, i - start , j, tt, picked, min, max, size, dim2);
			getMinMaxBestCombineLarge(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinMaxBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinMaxBestReadTempLarge(t_min_cost, t_max_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, min, max, dim2);
				getMinMaxBestCombineLarge(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinMaxBestWrite(s, t_min_cost, t_max_cost, t_picked_cost, t_jmin + start, data_valid);
			}
		}

		// 32 threads per block, up to 32 blocks, requires no shared memory and no thread synchronization
		template <class MS, class SC, class TC>
		__global__ void getMinMaxBestSingleSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			getMinMaxBestRead(t_min_cost, t_max_cost, t_picked_cost, t_jmin, i, j, tt, picked, min, max, size, dim2);
			getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
			getMinMaxBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (semaphoreWarp(semaphore))
			{
				getMinMaxBestReadTemp(t_min_cost, t_max_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, min, max, dim2);
				getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);
				getMinMaxBestSingleWrite(s, o_min_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin, i);
			}
		}

		// 256 threads per block, up to 256 blocks
		template <class MS, class SC, class TC>
		__global__ void getMinMaxBestSingleMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_max_cost[8], b_picked_cost[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			getMinMaxBestRead(t_min_cost, t_max_cost, t_picked_cost, t_jmin, i, j, tt, picked, min, max, size, dim2);
			getMinMaxBestCombineMedium(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinMaxBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinMaxBestReadTemp(t_min_cost, t_max_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, min, max, dim2);
				getMinMaxBestCombineMedium(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinMaxBestSingleWrite(s, o_min_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin, i);
			}
		}

		// 1024 threads per block, can be more than 1024 blocks
		template <class MS, class SC, class TC>
		__global__ void getMinMaxBestSingleLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, int *picked, SC min, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_max_cost[32], b_picked_cost[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			getMinMaxBestRead(t_min_cost, t_max_cost, t_picked_cost, t_jmin, i, j, tt, picked, min, max, size, dim2);
			getMinMaxBestCombineLarge(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinMaxBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinMaxBestReadTempLarge(t_min_cost, t_max_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, min, max, dim2);
				getMinMaxBestCombineLarge(t_min_cost, t_max_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinMaxBestSingleWrite(s, o_min_cost, o_jmin, t_min_cost, t_max_cost, t_picked_cost, t_jmin, i);
			}
		}

#ifdef LAP_CUDA_COMBINE_KERNEL
		template <class MS, class SC>
		__global__ void combineMinMaxBest_kernel(MS *s, MS *o_s, SC *o_min_cost, int *o_jmin, int *start, int idx, SC min, SC max, int dim2, int devices)
		{
			SC t_min_cost, t_max_cost, t_picked_cost;
			int t_jmin;

			if (threadIdx.x >= devices)
			{
				t_min_cost = max;
				t_max_cost = min;
				t_picked_cost = max;
				t_jmin = dim2;
			}
			else
			{
				t_min_cost = s[threadIdx.x].min;
				t_max_cost = s[threadIdx.x].max;
				t_picked_cost = s[threadIdx.x].picked;
				t_jmin = s[threadIdx.x].jmin + start[threadIdx.x];

				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < devices; i += blockDim.x)
				{
					SC c_min_cost = s[i].min;
					SC c_max_cost = s[i].max;
					SC c_picked_cost = s[i].picked;
					int c_jmin = s[i].jmin + start[i];
					if (c_min_cost < t_min_cost) t_min_cost = c_min_cost;
					if (c_max_cost > t_max_cost) t_max_cost = c_max_cost;
					if ((c_picked_cost < t_picked_cost) || ((c_picked_cost == t_picked_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_picked_cost = c_picked_cost;
					}
				}
			}

			getMinMaxBestCombineSmall(t_min_cost, t_max_cost, t_picked_cost, t_jmin);

			if (threadIdx.x == 0)
			{
				o_min_cost[0] = t_min_cost;
				o_jmin[0] = t_jmin - start[idx];

				if (idx == 0)
				{
					o_s->min = t_min_cost;
					o_s->max = t_max_cost;
					o_s->picked = t_picked_cost;
				}
			}
		}
#endif

		template <class SC, class TC>
		__device__ __forceinline__ void getMinSecondBestRead(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, int i, int j, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			t_min_cost = max;
			t_second_cost = max;
			t_picked_cost = max;
			t_jmin = dim2;

			if (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				t_min_cost = t_cost;
				if (picked[j] == 0)
				{
					t_jmin = j;
					t_picked_cost = t_cost;
				}
			}
		}

#ifndef LAP_CUDA_COMBINE_KERNEL
		template <class SC, class TC>
		__device__ __forceinline__ void getMinSecondBestRead(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, int i, int j, TC *tt, SC *v, int *picked, int last_picked, SC max, int size, int dim2)
		{
			t_min_cost = max;
			t_second_cost = max;
			t_picked_cost = max;
			t_jmin = dim2;

			if (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				t_min_cost = t_cost;
				if (j == last_picked)
				{
					picked[j] = 1;
				}
				else if (picked[j] == 0)
				{
					t_jmin = j;
					t_picked_cost = t_cost;
				}
			}
		}
#endif


		template <class SC>
		__device__ __forceinline__ void getMinSecondBestCombineSmall(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin)
		{
			minWarpIndex(t_picked_cost, t_jmin);
			SC old_min_cost = t_min_cost;
			minWarp(t_min_cost);
			bool is_min = (t_min_cost == old_min_cost);
			int mask = __ballot_sync(0xffffffff, is_min);
			is_min &= (__clz(mask) + __ffs(mask) == 32);
			if (!is_min) t_second_cost = old_min_cost;
			minWarp(t_second_cost);
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestCombineTiny(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin)
		{
			minWarpIndex8(t_picked_cost, t_jmin);
			SC old_min_cost = t_min_cost;
			minWarp8(t_min_cost);
			bool is_min = (t_min_cost == old_min_cost);
			int mask = __ballot_sync(0xff, is_min);
			is_min &= (__clz(mask) + __ffs(mask) == 32);
			if (!is_min) t_second_cost = old_min_cost;
			minWarp8(t_second_cost);
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestWriteShared(SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin, SC t_min_cost, SC t_second_cost, SC t_picked_cost, int t_jmin)
		{
			int bidx = threadIdx.x >> 5;
			b_min_cost[bidx] = t_min_cost;
			b_max_cost[bidx] = t_second_cost;
			b_picked_cost[bidx] = t_picked_cost;
			b_jmin[bidx] = t_jmin;
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestReadShared(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin)
		{
			t_min_cost = b_min_cost[threadIdx.x];
			t_second_cost = b_max_cost[threadIdx.x];
			t_picked_cost = b_picked_cost[threadIdx.x];
			t_jmin = b_jmin[threadIdx.x];
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestCombineMedium(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin)
		{
			getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getMinSecondBestWriteShared(b_min_cost, b_max_cost, b_picked_cost, b_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			}
			__syncthreads();
			if (threadIdx.x < 8)
			{
				getMinSecondBestReadShared(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestCombineTiny(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestCombineLarge(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, SC *b_min_cost, SC *b_max_cost, SC *b_picked_cost, int *b_jmin)
		{
			getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getMinSecondBestWriteShared(b_min_cost, b_max_cost, b_picked_cost, b_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			}
			__syncthreads();
			if (threadIdx.x < 32)
			{
				getMinSecondBestReadShared(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestWriteTemp(volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, SC t_min_cost, SC t_second_cost, SC t_picked_cost, int t_jmin)
		{
			if (threadIdx.x == 0)
			{
				o_min_cost[blockIdx.x] = t_min_cost;
				o_max_cost[blockIdx.x] = t_second_cost;
				o_picked_cost[blockIdx.x] = t_picked_cost;
				o_jmin[blockIdx.x] = t_jmin;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestReadTemp(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_second_cost = o_max_cost[threadIdx.x];
				t_picked_cost = o_picked_cost[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
			}
			else
			{
				t_min_cost = max;
				t_second_cost = max;
				t_picked_cost = max;
				t_jmin = dim2;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinSecondBestReadTempLarge(SC &t_min_cost, SC &t_second_cost, SC &t_picked_cost, int &t_jmin, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_second_cost = o_max_cost[threadIdx.x];
				t_picked_cost = o_picked_cost[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < gridDim.x; i += blockDim.x)
				{
					SC c_min_cost = o_min_cost[i];
					SC c_second_cost = o_max_cost[i];
					SC c_picked_cost = o_picked_cost[i];
					int c_jmin = o_jmin[i];
					if (c_min_cost < t_min_cost)
					{
						if (t_min_cost < c_second_cost) t_second_cost = t_min_cost;
						else t_second_cost = c_second_cost;
						t_min_cost = c_min_cost;
					}
					else if (c_min_cost < t_second_cost) t_second_cost = c_min_cost;
					if ((c_picked_cost < t_picked_cost) || ((c_picked_cost == t_picked_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_picked_cost = c_picked_cost;
					}
				}
			}
			else
			{
				t_min_cost = max;
				t_second_cost = max;
				t_picked_cost = max;
				t_jmin = dim2;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void getMinSecondBestWrite(MS *s, SC t_min_cost, SC t_second_cost, SC t_picked_cost, int t_jmin, SC *v, SC max, int dim2)
		{
			if (threadIdx.x == 0)
			{
				s->min = t_min_cost;
				s->max = t_second_cost;
				s->picked = t_picked_cost;
				s->jmin = t_jmin;
				if (t_jmin < dim2)
				{
					s->v_jmin = v[t_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, max, size, dim2);
			getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreWarp(semaphore))
			{
				getMinSecondBestReadTemp(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_max_cost[8], b_picked_cost[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, max, size, dim2);
			getMinSecondBestCombineMedium(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinSecondBestReadTemp(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineMedium(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_max_cost[32], b_picked_cost[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, max, size, dim2);
			getMinSecondBestCombineLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinSecondBestReadTempLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
			}
		}

#ifdef LAP_CUDA_COMBINE_KERNEL
		template <class MS, class SC>
		__global__ void combineMinSecondBest_kernel(MS *s, MS *o_s, int *picked, int *start, int idx, SC max, int num_items, int dim2, int devices)
		{
			SC t_min_cost, t_second_cost, t_picked_cost, t_vjmin;
			int t_jmin;

			if (threadIdx.x >= devices)
			{
				t_min_cost = max;
				t_second_cost = max;
				t_picked_cost = max;
				t_vjmin = max;
				t_jmin = dim2;
			}
			else
			{
				t_min_cost = s[threadIdx.x].min;
				t_second_cost = s[threadIdx.x].max;
				t_picked_cost = s[threadIdx.x].picked;
				t_vjmin = s[threadIdx.x].v_jmin;
				t_jmin = s[threadIdx.x].jmin + start[threadIdx.x];
				if (t_jmin > dim2) t_jmin = dim2;

				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < devices; i += blockDim.x)
				{
					SC c_min_cost = s[i].min;
					SC c_second_cost = s[i].max;
					SC c_picked_cost = s[i].picked;
					SC c_vjmin = s[i].v_jmin;
					int c_jmin = s[i].jmin + start[i];
					if (c_jmin > dim2) c_jmin = dim2;
					if (c_min_cost < t_min_cost)
					{
						if (t_min_cost < c_second_cost) t_second_cost = t_min_cost;
						else t_second_cost = c_second_cost;
						t_min_cost = c_min_cost;
					}
					else if (c_min_cost < t_second_cost) t_second_cost = c_min_cost;
					if ((c_picked_cost < t_picked_cost) || ((c_picked_cost == t_picked_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_picked_cost = c_picked_cost;
						t_vjmin = c_vjmin;
					}
				}
			}

			int old_jmin = t_jmin;
			getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			if (old_jmin != t_jmin) t_vjmin = max;
			minWarp(t_vjmin);

			if (threadIdx.x == 0)
			{
				int t_idx = t_jmin - start[idx];
				if ((t_idx >= 0) && (t_idx < num_items)) picked[t_idx] = 1;
				if (idx == 0)
				{
					o_s->min = t_min_cost;
					o_s->max = t_second_cost;
					o_s->picked = t_picked_cost;
					o_s->jmin = t_jmin;
					o_s->v_jmin = t_vjmin;
				}
			}
		}
#else
		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, int last_picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, last_picked, max, size, dim2);
			getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreWarp(semaphore))
			{
				getMinSecondBestReadTemp(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, int last_picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_max_cost[8], b_picked_cost[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, last_picked, max, size, dim2);
			getMinSecondBestCombineMedium(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinSecondBestReadTemp(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineMedium(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, int last_picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_max_cost[32], b_picked_cost[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, last_picked, max, size, dim2);
			getMinSecondBestCombineLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinSecondBestReadTempLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
			}
		}
#endif

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestSingleSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, max, size, dim2);
			getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreWarp(semaphore))
			{
				getMinSecondBestReadTemp(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineSmall(t_min_cost, t_second_cost, t_picked_cost, t_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
				if (threadIdx.x == 0) picked[t_jmin] = 1;
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestSingleMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_max_cost[8], b_picked_cost[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, max, size, dim2);
			getMinSecondBestCombineMedium(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinSecondBestReadTemp(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineMedium(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
				if (threadIdx.x == 0) picked[t_jmin] = 1;
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinSecondBestSingleLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_max_cost, volatile SC *o_picked_cost, volatile int *o_jmin, TC *tt, SC *v, int *picked, SC max, int i, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_max_cost[32], b_picked_cost[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_second_cost, t_picked_cost;
			int t_jmin;

			getMinSecondBestRead(t_min_cost, t_second_cost, t_picked_cost, t_jmin, i, j, tt, v, picked, max, size, dim2);
			getMinSecondBestCombineLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
			getMinSecondBestWriteTemp(o_min_cost, o_max_cost, o_picked_cost, o_jmin, t_min_cost, t_second_cost, t_picked_cost, t_jmin);

			if (semaphoreBlock(semaphore))
			{
				getMinSecondBestReadTempLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, o_min_cost, o_max_cost, o_picked_cost, o_jmin, max, dim2);
				getMinSecondBestCombineLarge(t_min_cost, t_second_cost, t_picked_cost, t_jmin, b_min_cost, b_max_cost, b_picked_cost, b_jmin);
				getMinSecondBestWrite(s, t_min_cost, t_second_cost, t_picked_cost, t_jmin, v, max, dim2);
				if (threadIdx.x == 0) picked[t_jmin] = 1;
			}
		}

		template <class SC, class TC>
		__device__ __forceinline__ void getMinimalCostRead(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, int j, TC *tt, SC *v, int *taken, SC max, int size, int dim2)
		{
			t_min_cost_real = max;
			t_min_cost = max;
			t_jmin = dim2;

			if (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				if (taken[j] == 0)
				{
					t_min_cost = t_cost;
					t_jmin = j;
				}
				t_min_cost_real = t_cost;
			}
		}

#ifndef LAP_CUDA_COMBINE_KERNEL
		template <class SC, class TC>
		__device__ __forceinline__ void getMinimalCostRead(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, int j, TC *tt, SC *v, int *taken, int last_taken, SC max, int size, int dim2)
		{
			t_min_cost_real = max;
			t_min_cost = max;
			t_jmin = dim2;

			if (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				if (j == last_taken)
				{
					taken[j] = 1;
				}
				else if (taken[j] == 0)
				{
					t_min_cost = t_cost;
					t_jmin = j;
				}
				t_min_cost_real = t_cost;
			}
		}
#endif

		template <class SC>
		__device__ __forceinline__ void getMinimalCostCombineSmall(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real)
		{
			minWarpIndex(t_min_cost, t_jmin);
			minWarp(t_min_cost_real);
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostCombineTiny(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real)
		{
			minWarpIndex8(t_min_cost, t_jmin);
			minWarp8(t_min_cost_real);
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostWriteShared(SC *b_min_cost, int *b_jmin, SC *b_min_cost_real, SC t_min_cost, int t_jmin, SC t_min_cost_real)
		{
			int bidx = threadIdx.x >> 5;
			b_min_cost[bidx] = t_min_cost;
			b_min_cost_real[bidx] = t_min_cost_real;
			b_jmin[bidx] = t_jmin;
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostReadShared(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, SC *b_min_cost, int *b_jmin, SC *b_min_cost_real)
		{
			t_min_cost = b_min_cost[threadIdx.x];
			t_min_cost_real = b_min_cost_real[threadIdx.x];
			t_jmin = b_jmin[threadIdx.x];
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostCombineMedium(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, SC *b_min_cost, int *b_jmin, SC *b_min_cost_real)
		{
			getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getMinimalCostWriteShared(b_min_cost, b_jmin, b_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);
			}
			__syncthreads();
			if (threadIdx.x < 8)
			{
				getMinimalCostReadShared(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostCombineTiny(t_min_cost, t_jmin, t_min_cost_real);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostCombineLarge(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, SC *b_min_cost, int *b_jmin, SC *b_min_cost_real)
		{
			getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getMinimalCostWriteShared(b_min_cost, b_jmin, b_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);
			}
			__syncthreads();
			if (threadIdx.x < 32)
			{
				getMinimalCostReadShared(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostWriteTemp(volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, SC t_min_cost, int t_jmin, SC t_min_cost_real)
		{
			if (threadIdx.x == 0)
			{
				o_min_cost[blockIdx.x] = t_min_cost;
				o_jmin[blockIdx.x] = t_jmin;
				o_min_cost_real[blockIdx.x] = t_min_cost_real;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostReadTemp(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
				t_min_cost_real = o_min_cost_real[threadIdx.x];
			}
			else
			{
				t_min_cost = max;
				t_jmin = dim2;
				t_min_cost_real = max;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getMinimalCostReadTempLarge(SC &t_min_cost, int &t_jmin, SC &t_min_cost_real, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
				t_min_cost_real = o_min_cost_real[threadIdx.x];
				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < gridDim.x; i += blockDim.x)
				{
					SC c_min_cost = o_min_cost[i];
					int c_jmin = o_jmin[i];
					SC c_min_cost_real = o_min_cost_real[i];
					if ((c_min_cost < t_min_cost) || ((c_min_cost == t_min_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_min_cost = c_min_cost;
					}
					if (c_min_cost_real < t_min_cost_real) t_min_cost_real = c_min_cost_real;
				}
			}
			else
			{
				t_min_cost = max;
				t_jmin = dim2;
				t_min_cost_real = max;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void getMinimalCostWrite(MS *s, SC t_min_cost, int t_jmin, SC t_min_cost_real, SC *v, SC max, int dim2)
		{
			if (threadIdx.x == 0)
			{
				s->min = t_min_cost_real;
				s->picked = t_min_cost;
				s->jmin = t_jmin;
				if (t_jmin < dim2)
				{
					s->v_jmin = v[t_jmin];
				}
				else
				{
					s->v_jmin = max;
				}
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, max, size, dim2);
			getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreWarp(semaphore))
			{
				getMinimalCostReadTemp(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_min_cost_real[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, max, size, dim2);
			getMinimalCostCombineMedium(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreBlock(semaphore))
			{
				getMinimalCostReadTemp(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineMedium(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_min_cost_real[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, max, size, dim2);
			getMinimalCostCombineLarge(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreBlock(semaphore))
			{
				getMinimalCostReadTempLarge(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineLarge(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
			}
		}

#ifdef LAP_CUDA_COMBINE_KERNEL
		template <class MS, class SC>
		__global__ void combineMinimalCost_kernel(MS *s, MS *o_s, int *picked, int *start, int idx, SC max, int num_items, int dim2, int devices)
		{
			SC t_min_cost, t_min_cost_real, t_vjmin;
			int t_jmin;

			if (threadIdx.x >= devices)
			{
				t_min_cost = max;
				t_min_cost_real = max;
				t_jmin = dim2;
				t_vjmin = max;
			}
			else
			{
				t_min_cost = s[threadIdx.x].picked;
				t_min_cost_real = s[threadIdx.x].min;
				t_jmin = s[threadIdx.x].jmin + start[threadIdx.x];
				t_vjmin = s[threadIdx.x].v_jmin;
				if (t_jmin > dim2) t_jmin = dim2;

				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < devices; i += blockDim.x)
				{
					SC c_min_cost = s[i].picked;
					SC c_min_cost_real = s[i].min;
					int c_jmin = s[i].jmin + start[i];
					SC c_vjmin = s[i].v_jmin;
					if (c_jmin > dim2) c_jmin = dim2;

					if ((c_min_cost < t_min_cost) || ((c_min_cost == t_min_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_min_cost = c_min_cost;
						t_vjmin = c_vjmin;
					}
					if (c_min_cost_real < t_min_cost_real) t_min_cost_real = c_min_cost_real;
				}
			}

			int old_jmin = t_jmin;
			getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			if (old_jmin != t_jmin) t_vjmin = max;
			minWarp(t_vjmin);

			if (threadIdx.x == 0)
			{
				int t_idx = t_jmin - start[idx];
				if ((t_idx >= 0) && (t_idx < num_items)) picked[t_idx] = 1;
				if (idx == 0)
				{
					o_s->picked = t_min_cost;
					o_s->min = t_min_cost_real;
					o_s->jmin = t_jmin;
					o_s->v_jmin = t_vjmin;
				}
			}
		}
#else
		template <class MS, class SC, class TC>
		__global__ void getMinimalCostSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, int last_picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, last_picked, max, size, dim2);
			getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreWarp(semaphore))
			{
				getMinimalCostReadTemp(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, int last_picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_min_cost_real[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, last_picked, max, size, dim2);
			getMinimalCostCombineMedium(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreBlock(semaphore))
			{
				getMinimalCostReadTemp(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineMedium(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, int last_picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_min_cost_real[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, last_picked, max, size, dim2);
			getMinimalCostCombineLarge(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreBlock(semaphore))
			{
				getMinimalCostReadTempLarge(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineLarge(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
			}
		}
#endif

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostSingleSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, max, size, dim2);
			getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreWarp(semaphore))
			{
				getMinimalCostReadTemp(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineSmall(t_min_cost, t_jmin, t_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
				if (threadIdx.x == 0) picked[t_jmin] = 1;
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostSingleMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_min_cost_real[8];
			__shared__ int b_jmin[8];

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, max, size, dim2);
			getMinimalCostCombineMedium(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreBlock(semaphore))
			{
				getMinimalCostReadTemp(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineMedium(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
				if (threadIdx.x == 0) picked[t_jmin] = 1;
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getMinimalCostSingleLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile int *o_jmin, volatile SC *o_min_cost_real, TC *tt, SC *v, int *picked, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_min_cost_real[32];
			__shared__ int b_jmin[32];

			SC t_min_cost, t_min_cost_real;
			int t_jmin;

			getMinimalCostRead(t_min_cost, t_jmin, t_min_cost_real, j, tt, v, picked, max, size, dim2);
			getMinimalCostCombineLarge(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
			getMinimalCostWriteTemp(o_min_cost, o_jmin, o_min_cost_real, t_min_cost, t_jmin, t_min_cost_real);

			if (semaphoreBlock(semaphore))
			{
				getMinimalCostReadTempLarge(t_min_cost, t_jmin, t_min_cost_real, o_min_cost, o_jmin, o_min_cost_real, max, dim2);
				getMinimalCostCombineLarge(t_min_cost, t_jmin, t_min_cost_real, b_min_cost, b_jmin, b_min_cost_real);
				getMinimalCostWrite(s, t_min_cost, t_jmin, t_min_cost_real, v, max, dim2);
				if (threadIdx.x == 0) picked[t_jmin] = 1;
			}
		}

		template <class SC, class TC>
		__device__ __forceinline__ void getFinalCostRead(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v, int j, TC *tt, SC *v, int j_picked, SC max, int size)
		{
			t_min_cost = max;
			t_picked_cost = max;
			t_picked_v = max;

			if (j < size)
			{
				SC t_cost = (SC)tt[j] - v[j];
				t_min_cost = t_cost;
				if (j == j_picked)
				{
					t_picked_cost = (SC)tt[j];
					t_picked_v = v[j];
				}
			}
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostCombineSmall(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v)
		{
			minWarp(t_min_cost);
			minWarp(t_picked_cost);
			minWarp(t_picked_v);
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostCombineTiny(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v)
		{
			minWarp8(t_min_cost);
			minWarp8(t_picked_cost);
			minWarp8(t_picked_v);
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostWriteShared(SC *b_min_cost, SC *b_picked_cost, SC *b_picked_v, SC t_min_cost, SC t_picked_cost, SC t_picked_v)
		{
			int bidx = threadIdx.x >> 5;
			b_min_cost[bidx] = t_min_cost;
			b_picked_cost[bidx] = t_picked_cost;
			b_picked_v[bidx] = t_picked_v;
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostReadShared(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v, SC *b_min_cost, SC *b_picked_cost, SC *b_picked_v)
		{
			t_min_cost = b_min_cost[threadIdx.x];
			t_picked_cost = b_picked_cost[threadIdx.x];
			t_picked_v = b_picked_v[threadIdx.x];
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostCombineMedium(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v, SC *b_min_cost, SC *b_picked_cost, SC *b_picked_v)
		{
			getFinalCostCombineSmall(t_min_cost, t_picked_cost, t_picked_v);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getFinalCostWriteShared(b_min_cost, b_picked_cost, b_picked_v, t_min_cost, t_picked_cost, t_picked_v);
			}
			__syncthreads();
			if (threadIdx.x < 8)
			{
				getFinalCostReadShared(t_min_cost, t_picked_cost, t_picked_v, b_min_cost, b_picked_cost, b_picked_v);
				getFinalCostCombineTiny(t_min_cost, t_picked_cost, t_picked_v);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostCombineLarge(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v, SC *b_min_cost, SC *b_picked_cost, SC *b_picked_v)
		{
			getFinalCostCombineSmall(t_min_cost, t_picked_cost, t_picked_v);
			if ((threadIdx.x & 0x1f) == 0)
			{
				getFinalCostWriteShared(b_min_cost, b_picked_cost, b_picked_v, t_min_cost, t_picked_cost, t_picked_v);
			}
			__syncthreads();
			if (threadIdx.x < 32)
			{
				getFinalCostReadShared(t_min_cost, t_picked_cost, t_picked_v, b_min_cost, b_picked_cost, b_picked_v);
				getFinalCostCombineSmall(t_min_cost, t_picked_cost, t_picked_v);
			}
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostWriteTemp(volatile SC *o_min_cost, volatile SC *o_picked_cost, volatile SC *o_picked_v, SC t_min_cost, SC t_picked_cost, SC t_picked_v)
		{
			if (threadIdx.x == 0)
			{
				o_min_cost[blockIdx.x] = t_min_cost;
				o_picked_cost[blockIdx.x] = t_picked_cost;
				o_picked_v[blockIdx.x] = t_picked_v;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostReadTemp(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v, volatile SC *o_min_cost, volatile SC *o_picked_cost, volatile SC *o_picked_v, SC max)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_picked_cost = o_picked_cost[threadIdx.x];
				t_picked_v = o_picked_v[threadIdx.x];
			}
			else
			{
				t_min_cost = max;
				t_picked_cost = max;
				t_picked_v = max;
			}
		}

		template <class SC>
		__device__ __forceinline__ void getFinalCostReadTempLarge(SC &t_min_cost, SC &t_picked_cost, SC &t_picked_v, volatile SC *o_min_cost, volatile SC *o_picked_cost, volatile SC *o_picked_v, SC max)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min_cost = o_min_cost[threadIdx.x];
				t_picked_cost = o_picked_cost[threadIdx.x];
				t_picked_v = o_picked_v[threadIdx.x];
				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < gridDim.x; i += blockDim.x)
				{
					SC c_min_cost = o_min_cost[i];
					SC c_picked_cost = o_picked_cost[i];
					SC c_picked_v = o_picked_v[i];
					if (c_min_cost < t_min_cost) t_min_cost = c_min_cost;
					if (c_picked_cost < t_picked_cost) t_picked_cost = c_picked_cost;
					if (c_picked_v < t_picked_v) t_picked_v = c_picked_v;
				}
			}
			else
			{
				t_min_cost = max;
				t_picked_cost = max;
				t_picked_v = max;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void getFinalCostWrite(MS *s, SC t_min_cost, SC t_picked_cost, SC t_picked_v)
		{
			if (threadIdx.x == 0)
			{
				s->min = t_min_cost;
				s->picked = t_picked_cost;
				s->v_jmin = t_picked_v;
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getFinalCostSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_picked_cost, volatile SC *o_picked_v, TC *tt, SC *v, SC max, int j_picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min_cost, t_picked_cost, t_picked_v;

			getFinalCostRead(t_min_cost, t_picked_cost, t_picked_v, j, tt, v, j_picked, max, size);
			getFinalCostCombineSmall(t_min_cost, t_picked_cost, t_picked_v);
			getFinalCostWriteTemp(o_min_cost, o_picked_cost, o_picked_v, t_min_cost, t_picked_cost, t_picked_v);

			if (semaphoreWarp(semaphore))
			{
				getFinalCostReadTemp(t_min_cost, t_picked_cost, t_picked_v, o_min_cost, o_picked_cost, o_picked_v, max);
				getFinalCostCombineSmall(t_min_cost, t_picked_cost, t_picked_v);
				getFinalCostWrite(s, t_min_cost, t_picked_cost, t_picked_v);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getFinalCostMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_picked_cost, volatile SC *o_picked_v, TC *tt, SC *v, SC max, int j_picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[8], b_picked_cost[8], b_picked_v[8];

			SC t_min_cost, t_picked_cost, t_picked_v;

			getFinalCostRead(t_min_cost, t_picked_cost, t_picked_v, j, tt, v, j_picked, max, size);
			getFinalCostCombineMedium(t_min_cost, t_picked_cost, t_picked_v, b_min_cost, b_picked_cost, b_picked_v);
			getFinalCostWriteTemp(o_min_cost, o_picked_cost, o_picked_v, t_min_cost, t_picked_cost, t_picked_v);

			if (semaphoreBlock(semaphore))
			{
				getFinalCostReadTemp(t_min_cost, t_picked_cost, t_picked_v, o_min_cost, o_picked_cost, o_picked_v, max);
				getFinalCostCombineMedium(t_min_cost, t_picked_cost, t_picked_v, b_min_cost, b_picked_cost, b_picked_v);
				getFinalCostWrite(s, t_min_cost, t_picked_cost, t_picked_v);
			}
		}

		template <class MS, class SC, class TC>
		__global__ void getFinalCostLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min_cost, volatile SC *o_picked_cost, volatile SC *o_picked_v, TC *tt, SC *v, SC max, int j_picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost[32], b_picked_cost[32], b_picked_v[32];

			SC t_min_cost, t_picked_cost, t_picked_v;

			getFinalCostRead(t_min_cost, t_picked_cost, t_picked_v, j, tt, v, j_picked, max, size);
			getFinalCostCombineLarge(t_min_cost, t_picked_cost, t_picked_v, b_min_cost, b_picked_cost, b_picked_v);
			getFinalCostWriteTemp(o_min_cost, o_picked_cost, o_picked_v, t_min_cost, t_picked_cost, t_picked_v);

			if (semaphoreBlock(semaphore))
			{
				getFinalCostReadTempLarge(t_min_cost, t_picked_cost, t_picked_v, o_min_cost, o_picked_cost, o_picked_v, max);
				getFinalCostCombineLarge(t_min_cost, t_picked_cost, t_picked_v, b_min_cost, b_picked_cost, b_picked_v);
				getFinalCostWrite(s, t_min_cost, t_picked_cost, t_picked_v);
			}
		}

#ifdef LAP_CUDA_COMBINE_KERNEL
		template <class MS, class SC>
		__global__ void combineFinalCost_kernel(MS *s, MS *o_s, SC max, int devices)
		{
			SC t_min_cost, t_picked_cost, t_picked_v;

			if (threadIdx.x >= devices)
			{
				t_min_cost = max;
				t_picked_cost = max;
				t_picked_v = max;
			}
			else
			{
				t_min_cost = s[threadIdx.x].min;
				t_picked_cost = s[threadIdx.x].picked;
				t_picked_v = s[threadIdx.x].v_jmin;

				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < devices; i += blockDim.x)
				{
					SC c_min_cost = s[i].min;
					SC c_picked_cost = s[i].picked;
					SC c_picked_v = s[i].v_jmin;
					if (c_min_cost < t_min_cost) t_min_cost = c_min_cost;
					if (c_picked_cost < t_picked_cost) t_picked_cost = c_picked_cost;
					if (c_picked_v < t_picked_v) t_picked_v = c_picked_v;
				}
			}

			getFinalCostCombineSmall(t_min_cost, t_picked_cost, t_picked_v);

			if (threadIdx.x == 0)
			{
				o_s->min = t_min_cost;
				o_s->picked = t_picked_cost;
				o_s->v_jmin = t_picked_v;
			}
		}
#endif

		template <class SC, class TC>
		__global__ void updateEstimatedVFirst_kernel(SC *min_v, TC *tt, int *picked, SC *min_cost, int *jmin, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			min_v[j] = (SC)tt[j] - min_cost[0];
			if (jmin[0] == j) picked[j] = 1;
		}

		template <class SC, class TC>
		__global__ void updateEstimatedVSecond_kernel(SC *v, SC *min_v, TC *tt, int *picked, SC *min_cost, int *jmin, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			SC tmp = (SC)tt[j] - min_cost[0];
			if (tmp < min_v[j])
			{
				v[j] = min_v[j];
				min_v[j] = tmp;
			}
			else v[j] = tmp;
			if (jmin[0] == j) picked[j] = 1;
		}

		template <class SC, class TC>
		__global__ void updateEstimatedV_kernel(SC *v, SC *min_v, TC *tt, int *picked, SC *min_cost, int *jmin, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			SC tmp = (SC)tt[j] - min_cost[0];
			if (tmp < min_v[j])
			{
				v[j] = min_v[j];
				min_v[j] = tmp;
			}
			else if (tmp < v[j]) v[j] = tmp;
			if (jmin[0] == j) picked[j] = 1;
		}

		template <class MS, class SC>
		__device__ __forceinline__ void updateEstimateVGetMin(int &jmin, SC &min_cost, volatile int *data_valid, MS *s, int start, int dim, SC max, int devices)
		{
			__shared__ int b_jmin;
			__shared__ SC b_min_cost;

			if (threadIdx.x < 32)
			{
				bool is_valid = false;
				do
				{
					if (!is_valid)
					{
						is_valid = true;
						for (int t = threadIdx.x; t < devices; t += 32) is_valid &= (data_valid[t] != 0);
					}
				} while (!__all_sync(0xffffffff, is_valid));

				int t_jmin = dim;
				SC t_min_cost = max;

				for (int t = threadIdx.x; t < devices; t += 32)
				{
					int c_jmin = s[t].jmin;
					SC c_min_cost = s[t].min;
					if ((c_min_cost < t_min_cost) || ((c_min_cost == t_min_cost) && (c_jmin < t_jmin)))
					{
						t_jmin = c_jmin;
						t_min_cost = t_min_cost;
					}
				}
				minWarpIndex(t_min_cost, t_jmin);
				b_jmin = t_jmin;
				b_min_cost = t_min_cost;
			}

			__syncthreads();
			jmin = b_jmin;
			min_cost = b_min_cost;
		}

		template <class MS, class SC, class TC>
		__global__ void updateEstimatedVFirst_kernel(SC *min_v, TC *tt, int *picked, volatile int *data_valid, MS *s, int start, int size, int dim, SC max, int devices)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			int jmin;
			SC min_cost;
			updateEstimateVGetMin(jmin, min_cost, data_valid, s, start, dim, max, devices);

			min_v[j] = (SC)tt[j] - min_cost;
			if (jmin == j) picked[j] = 1;
		}

		template <class MS, class SC, class TC>
		__global__ void updateEstimatedVSecond_kernel(SC *v, SC *min_v, TC *tt, int *picked, volatile int *data_valid, MS *s, int start, int size, int dim, SC max, int devices)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			int jmin;
			SC min_cost;
			updateEstimateVGetMin(jmin, min_cost, data_valid, s, start, dim, max, devices);

			SC tmp = (SC)tt[j] - min_cost;
			if (tmp < min_v[j])
			{
				v[j] = min_v[j];
				min_v[j] = tmp;
			}
			else v[j] = tmp;
			if (jmin == j) picked[j] = 1;
		}

		template <class MS, class SC, class TC>
		__global__ void updateEstimatedV_kernel(SC *v, SC *min_v, TC *tt, int *picked, volatile int *data_valid, MS *s, int start, int size, int dim, SC max, int devices)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;

			int jmin;
			SC min_cost;
			updateEstimateVGetMin(jmin, min_cost, data_valid, s, start, dim, max, devices);

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
		__device__ __forceinline__ void initializeSearchMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *pred, SC *d, int j, int f, TC *tt, SC *v, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				colactive[j] = 1;
				pred[j] = f;

				d[j] = t_min = (SC)tt[j] - v[j];
				t_jmin = j;
				t_colsol = colsol[j];
			}
		}

		template <class SC, class TC>
		__device__ __forceinline__ void initializeSearchMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *colsol_in, int *pred, SC *d, int j, int f, TC *tt, SC *v, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				colactive[j] = 1;
				pred[j] = f;

				d[j] = t_min = (SC)tt[j] - v[j];
				t_jmin = j;
				t_colsol = colsol[j] = colsol_in[j];
			}
		}

		template <class SC>
		__device__ __forceinline__ void initializeSearchMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *pred, SC *d, int j, int f, SC *v, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				colactive[j] = 1;
				pred[j] = f;

				d[j] = t_min = -v[j];
				t_jmin = j;
				t_colsol = colsol[j];
			}
		}

		template <class SC>
		__device__ __forceinline__ void initializeSearchMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *colsol_in, int *pred, SC *d, int j, int f, SC *v, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				colactive[j] = 1;
				pred[j] = f;

				d[j] = t_min = -v[j];
				t_jmin = j;
				t_colsol = colsol[j] = colsol_in[j];
			}
		}

		template <class SC, class TC>
		__device__ __forceinline__ void continueSearchJMinMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *pred, SC *d, int j, int i, TC *tt, SC *v, SC min, int jmin, SC tt_jmin, SC v_jmin,SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
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

					t_min = h;
					t_jmin = j;
					t_colsol = colsol[j];
				}
			}
		}

		template <class SC>
		__device__ __forceinline__ void continueSearchJMinMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *pred, SC *d, int j, int i, SC *v, SC min, int jmin, SC v_jmin, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				if (j == jmin) colactive[j] = 0;
				else if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = (v[j] - v_jmin) + min;

					bool is_smaller = (v2 < h);
					if (is_smaller)
					{
						pred[j] = i;
						d[j] = h = v2;
					}

					t_min = h;
					t_jmin = j;
					t_colsol = colsol[j];
				}
			}
		}

		template <class SC, class TC>
		__device__ __forceinline__ void continueSearchMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *pred, SC *d, int j, int i, TC *tt, SC *v, SC min, SC tt_jmin, SC v_jmin, int jmin, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				if (j == jmin) colactive[j] = 0;
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

					t_min = h;
					t_jmin = j;
					t_colsol = colsol[j];
				}
			}
		}

		template <class SC>
		__device__ __forceinline__ void continueSearchMinRead(SC &t_min, int &t_jmin, int &t_colsol, char *colactive, int *colsol, int *pred, SC *d, int j, int i, SC *v, SC min, SC v_jmin, int jmin, SC max, int size, int dim2)
		{
			t_min = max;
			t_jmin = dim2;
			t_colsol = 0;

			if (j < size)
			{
				if (j == jmin) colactive[j] = 1;
				else if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = (v[j] - v_jmin) + min;

					bool is_smaller = (v2 < h);
					if (is_smaller)
					{
						pred[j] = i;
						d[j] = h = v2;
					}

					t_min = h;
					t_jmin = j;
					t_colsol = colsol[j];
				}
			}
		}

		template <class SC>
		__device__ __forceinline__ void searchWriteShared(SC *b_min, int *b_jmin, int *b_colsol, SC t_min, int t_jmin, int t_colsol)
		{
			int bidx = threadIdx.x >> 5;
			b_min[bidx] = t_min;
			b_jmin[bidx] = t_jmin;
			b_colsol[bidx] = t_colsol;
		}

		template <class SC>
		__device__ __forceinline__ void searchReadShared(SC &t_min, int &t_jmin, int &t_colsol, SC *b_min, int *b_jmin, int *b_colsol)
		{
			t_min = b_min[threadIdx.x];
			t_jmin = b_jmin[threadIdx.x];
			t_colsol = b_colsol[threadIdx.x];
		}

		template <class SC>
		__device__ __forceinline__ void searchCombineMedium(SC &t_min, int &t_jmin, int &t_colsol, SC *b_min, int *b_jmin, int *b_colsol)
		{
			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				searchWriteShared(b_min, b_jmin, b_colsol, t_min, t_jmin, t_colsol);
			}
			__syncthreads();
			if (threadIdx.x < 8)
			{
				searchReadShared(t_min, t_jmin, t_colsol, b_min, b_jmin, b_colsol);
				minWarpIndex8(t_min, t_jmin, t_colsol);
			}
		}

		template <class SC>
		__device__ __forceinline__ void searchCombineLarge(SC &t_min, int &t_jmin, int &t_colsol, SC *b_min, int *b_jmin, int *b_colsol)
		{
			minWarpIndex(t_min, t_jmin, t_colsol);
			if ((threadIdx.x & 0x1f) == 0)
			{
				searchWriteShared(b_min, b_jmin, b_colsol, t_min, t_jmin, t_colsol);
			}
			__syncthreads();
			if (threadIdx.x < 32)
			{
				searchReadShared(t_min, t_jmin, t_colsol, b_min, b_jmin, b_colsol);
				minWarpIndex(t_min, t_jmin, t_colsol);
			}
		}

		template <class SC>
		__device__ __forceinline__ void searchWriteTemp(volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC t_min, int t_jmin, int t_colsol)
		{
			if (threadIdx.x == 0)
			{
				o_min[blockIdx.x] = t_min;
				o_jmin[blockIdx.x] = t_jmin;
				o_colsol[blockIdx.x] = t_colsol;
			}
		}

		template <class SC>
		__device__ __forceinline__ void searchReadTemp(SC &t_min, int &t_jmin, int &t_colsol, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min = o_min[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
				t_colsol = o_colsol[threadIdx.x];
			}
			else
			{
				t_min = max;
				t_jmin = dim2;
				t_colsol = 0;
			}
		}

		template <class SC>
		__device__ __forceinline__ void searchReadTempLarge(SC &t_min, int &t_jmin, int &t_colsol, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC max, int dim2)
		{
			if (threadIdx.x < gridDim.x)
			{
				t_min = o_min[threadIdx.x];
				t_jmin = o_jmin[threadIdx.x];
				t_colsol = o_colsol[threadIdx.x];
				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < gridDim.x; i += blockDim.x)
				{
					SC c_min = o_min[threadIdx.x];
					int c_jmin = o_jmin[threadIdx.x];
					int c_colsol = o_colsol[threadIdx.x];
					if ((c_min < t_min) || ((c_min == t_min) && ((c_colsol < t_colsol) || ((c_colsol == t_colsol) && (c_jmin < t_jmin)))))
					{
						t_min = c_min;
						t_jmin = c_jmin;
						t_colsol = c_colsol;
					}
				}
			}
			else
			{
				t_min = max;
				t_jmin = dim2;
				t_colsol = 0;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void searchWrite(MS *s, SC t_min, int t_jmin, int t_colsol)
		{
			if (threadIdx.x == 0)
			{
				s->min = t_min;
				s->jmin = t_jmin;
				s->colsol = t_colsol;
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void searchSmall(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC &t_min, int &t_jmin, int &t_colsol, SC max, int size, int dim2)
		{
			minWarpIndex(t_min, t_jmin, t_colsol);
			searchWriteTemp(o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol);

			if (semaphoreWarp(semaphore))
			{
				searchReadTemp(t_min, t_jmin, t_colsol, o_min, o_jmin, o_colsol, max, dim2);
				minWarpIndex(t_min, t_jmin, t_colsol);
				searchWrite(s, t_min, t_jmin, t_colsol);
			}
		}
		
		template <class MS, class SC>
		__device__ __forceinline__ void searchMedium(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC &t_min, int &t_jmin, int &t_colsol, SC max, int size, int dim2)
		{
			__shared__ SC b_min[8];
			__shared__ int b_jmin[8], b_colsol[8];

			searchCombineMedium(t_min, t_jmin, t_colsol, b_min, b_jmin, b_colsol);
			searchWriteTemp(o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol);

			if (semaphoreBlock(semaphore))
			{
				searchReadTemp(t_min, t_jmin, t_colsol, o_min, o_jmin, o_colsol, max, dim2);
				searchCombineMedium(t_min, t_jmin, t_colsol, b_min, b_jmin, b_colsol);
				searchWrite(s, t_min, t_jmin, t_colsol);
			}
		}

		template <class MS, class SC>
		__device__ __forceinline__ void searchLarge(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC &t_min, int &t_jmin, int &t_colsol, SC max, int size, int dim2)
		{
			__shared__ SC b_min[32];
			__shared__ int b_jmin[32], b_colsol[32];

			searchCombineLarge(t_min, t_jmin, t_colsol, b_min, b_jmin, b_colsol);
			searchWriteTemp(o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol);

			if (semaphoreBlock(semaphore))
			{
				searchReadTempLarge(t_min, t_jmin, t_colsol, o_min, o_jmin, o_colsol, max, dim2);
				searchCombineLarge(t_min, t_jmin, t_colsol, b_min, b_jmin, b_colsol);
				searchWrite(s, t_min, t_jmin, t_colsol);
			}
		}

		// normal
		template <class MS, class SC, class TC>
		__global__ void initializeSearchMinSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, f, tt, v, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC>
		__global__ void initializeSearchMinMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, f, tt, v, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC>
		__global__ void initializeSearchMinLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, f, tt, v, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		// copy colsol
		template <class MS, class SC, class TC>
		__global__ void initializeSearchMinSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, colsol_in, pred, d, j, f, tt, v, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC>
		__global__ void initializeSearchMinMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, colsol_in, pred, d, j, f, tt, v, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC>
		__global__ void initializeSearchMinLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, colsol_in, pred, d, j, f, tt, v, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		// virtual row
		template <class MS, class SC>
		__global__ void initializeSearchMinSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, f, v, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void initializeSearchMinMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, f, v, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void initializeSearchMinLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, f, v, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		// copy colsol, virtual row
		template <class MS, class SC>
		__global__ void initializeSearchMinSmall_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, colsol_in, pred, d, j, f, v, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void initializeSearchMinMedium_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, colsol_in, pred, d, j, f, v, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void initializeSearchMinLarge_kernel(MS *s, unsigned int *semaphore, volatile SC *o_min, volatile int *o_jmin, volatile int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *colsol_in, int *pred, int f, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			initializeSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, colsol_in, pred, d, j, f, v, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

#ifdef LAP_CUDA_COMBINE_KERNEL
		template <class MS, class SC>
		__global__ void combineSearchMin(MS *s, MS *o_s, int *start, SC max, int dim2, int devices)
		{
			SC t_min;
			int t_jmin;
			int t_colsol;

			if (threadIdx.x >= devices)
			{
				t_min = max;
				t_jmin = dim2;
				t_colsol = dim2;
			}
			else
			{
				t_min = s[threadIdx.x].min;
				t_jmin = s[threadIdx.x].jmin + start[threadIdx.x];
				t_colsol = s[threadIdx.x].colsol;

				// read additional values
				for (int i = threadIdx.x + blockDim.x; i < devices; i += blockDim.x)
				{
					SC c_min = s[i].min;
					int c_jmin = s[i].jmin + start[i];
					int c_colsol = s[i].colsol;
					if ((c_min < t_min) || ((c_min == t_min) && (t_colsol >= 0) && (c_colsol < 0)))
					{
						t_min = c_min;
						t_jmin = c_jmin;
						t_colsol = c_colsol;
					}
				}
			}
			minWarpIndex(t_min, t_jmin, t_colsol);
			if (threadIdx.x == 0)
			{
				o_s->min = t_min;
				o_s->jmin = t_jmin;
				o_s->colsol = t_colsol;
			}
		}
#endif

		template <class MS, class SC, class TC>
		__global__ void continueSearchJMinMinSmall_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_tt_jmin, b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt[jmin];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v[jmin];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			SC t_min;
			int t_jmin, t_colsol;

			continueSearchJMinMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, jmin, tt_jmin, v_jmin, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC>
		__global__ void continueSearchJMinMinMedium_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_tt_jmin, b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt[jmin];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v[jmin];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			SC t_min;
			int t_jmin, t_colsol;

			continueSearchJMinMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, jmin, tt_jmin, v_jmin, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC>
		__global__ void continueSearchJMinMinLarge_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_tt_jmin, b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt[jmin];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v[jmin];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			SC t_min;
			int t_jmin, t_colsol;

			continueSearchJMinMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, jmin, tt_jmin, v_jmin, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void continueSearchJMinMinSmall_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_v_jmin = v[jmin];
			}
			__syncthreads();
			SC v_jmin = b_v_jmin;

			SC t_min;
			int t_jmin, t_colsol;

			continueSearchJMinMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, v, min, jmin, v_jmin, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void continueSearchJMinMinMedium_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_v_jmin = v[jmin];
			}
			__syncthreads();
			SC v_jmin = b_v_jmin;

			SC t_min;
			int t_jmin, t_colsol;

			continueSearchJMinMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, v, min, jmin, v_jmin, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void continueSearchJMinMinLarge_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_v_jmin = v[jmin];
			}
			__syncthreads();
			SC v_jmin = b_v_jmin;

			SC t_min;
			int t_jmin, t_colsol;

			continueSearchJMinMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, v, min, jmin, v_jmin, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC, class TC2>
		__global__ void continueSearchMinPeerSmall_kernel(MS *s, unsigned int *semaphore, volatile int *data_valid, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, TC2 *tt2, SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_tt_jmin, b_v_jmin;
			if ((jmin >= 0) && (jmin < size))
			{
				if (threadIdx.x == 0) data_valid[0] = 1;
			}
			else
			{
				if (threadIdx.x == 0) while (data_valid[0] == 0) {}
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt2[0];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, tt_jmin, v_jmin, jmin, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC, class TC2>
		__global__ void continueSearchMinPeerMedium_kernel(MS *s, unsigned int *semaphore, volatile int *data_valid, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, TC2 *tt2, SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_tt_jmin, b_v_jmin;
			if ((jmin >= 0) && (jmin < size))
			{
				if (threadIdx.x == 0) data_valid[0] = 1;
			}
			else
			{
				if (threadIdx.x == 0) while (data_valid[0] == 0) {}
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt2[0];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, tt_jmin, v_jmin, jmin, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC, class TC2>
		__global__ void continueSearchMinPeerLarge_kernel(MS *s, unsigned int *semaphore, volatile int *data_valid, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, TC2 *tt2, SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_tt_jmin, b_v_jmin;
			if ((jmin >= 0) && (jmin < size))
			{
				if (threadIdx.x == 0) data_valid[0] = 1;
			}
			else
			{
				if (threadIdx.x == 0) while (data_valid[0] == 0) {}
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt2[0];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, tt_jmin, v_jmin, jmin, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void continueSearchMinSmall_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, v, min, v_jmin, jmin, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void continueSearchMinMedium_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, v, min, v_jmin, jmin, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC>
		__global__ void continueSearchMinLarge_kernel(MS *s, unsigned int *semaphore, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, char *colactive, int *colsol, int *pred, int i, SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_v_jmin;
			if (threadIdx.x == 0)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, v, min, v_jmin, jmin, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC, class TC2>
		__global__ void continueSearchMinSmall_kernel(MS *s, unsigned int *semaphore, volatile int *data_valid, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, volatile TC2 *tt2, volatile SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_tt_jmin, b_v_jmin;

			if ((jmin >= 0) && (jmin < size))
			{
				if (threadIdx.x == 0)
				{
					if (data_valid[0] == 0)
					{
						tt2[0] = tt[jmin];
						v2[0] = v[jmin];
						__threadfence_system();
						data_valid[0] = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0) while (data_valid[0] == 0) {}
			}
			__syncthreads();

			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt2[0];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, tt_jmin, v_jmin, jmin, max, size, dim2);
			searchSmall(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC, class TC2>
		__global__ void continueSearchMinMedium_kernel(MS *s, unsigned int *semaphore, volatile int *data_valid, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, volatile TC2 *tt2, volatile SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_tt_jmin, b_v_jmin;

			if ((jmin >= 0) && (jmin < size))
			{
				if (threadIdx.x == 0)
				{
					if (data_valid[0] == 0)
					{
						tt2[0] = (TC2)tt[jmin];
						v2[0] = v[jmin];
						__threadfence_system();
						data_valid[0] = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0) while (data_valid[0] == 0) {}
			}
			__syncthreads();

			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt2[0];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, tt_jmin, v_jmin, jmin, max, size, dim2);
			searchMedium(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
		}

		template <class MS, class SC, class TC, class TC2>
		__global__ void continueSearchMinLarge_kernel(MS *s, unsigned int *semaphore, volatile int *data_valid, SC *o_min, int *o_jmin, int *o_colsol, SC *v, SC *d, TC *tt, char *colactive, int *colsol, int *pred, int i, volatile TC2 *tt2, volatile SC *v2, int jmin, SC min, SC max, int size, int dim2)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC t_min;
			int t_jmin, t_colsol;

			__shared__ SC b_tt_jmin, b_v_jmin;

			if ((jmin >= 0) && (jmin < size))
			{
				if (threadIdx.x == 0)
				{
					if (data_valid[0] == 0)
					{
						tt2[0] = (TC2)tt[jmin];
						v2[0] = v[jmin];
						__threadfence_system();
						data_valid[0] = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0) while (data_valid[0] == 0) {}
			}
			__syncthreads();

			if (threadIdx.x == 0)
			{
				b_tt_jmin = (SC)tt2[0];
			}
			else if (threadIdx.x == 1)
			{
				b_v_jmin = v2[0];
			}
			__syncthreads();
			SC tt_jmin = b_tt_jmin;
			SC v_jmin = b_v_jmin;

			continueSearchMinRead(t_min, t_jmin, t_colsol, colactive, colsol, pred, d, j, i, tt, v, min, tt_jmin, v_jmin, jmin, max, size, dim2);
			searchLarge(s, semaphore, o_min, o_jmin, o_colsol, t_min, t_jmin, t_colsol, max, size, dim2);
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
	}
}
