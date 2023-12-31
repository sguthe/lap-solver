#pragma once

namespace lap
{
	namespace cuda
	{
		template <class MS, class SC, class TC>
		__device__ __forceinline__ void updateVExchange(volatile MS *s, volatile MS *s2, unsigned int *semaphore, SC *v, TC *tt, SC &min_cost, int picked, int size)
		{
			__shared__ SC b_min_cost;

			if ((picked >= 0) && (picked < size))
			{
				if (threadIdx.x == 0)
				{
					b_min_cost = min_cost = (SC)tt[picked] - v[picked];
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						s->min = min_cost;
						__threadfence_system();
						s->jmin = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0)
				{
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						while (s->jmin == 0) {}
						__threadfence_system();
						s2->min = min_cost = s->min;
						__threadfence();
						s2->jmin = 1;
					}
					else
					{
						while (s2->jmin == 0) {}
						__threadfence();
						min_cost = s2->min;
					}
					b_min_cost = min_cost;
				}
			}
			__syncthreads();
			min_cost = b_min_cost;
		}

		template <class MS, class SC, class I, class ISTATE, class STATE>
		__device__ __forceinline__ void updateVExchange(volatile MS* s, volatile MS* s2, unsigned int* semaphore, SC* v, I &iterator, ISTATE &istate, STATE &state, int i, int j, int start, int idx, SC& min_cost, int picked, int size)
		{
			__shared__ SC b_min_cost;

			if ((picked >= 0) && (picked < size))
			{
				if (threadIdx.x == 0)
				{
					b_min_cost = min_cost = (SC)iterator.getCostForced(i, j, start, istate, state, idx) - v[picked];
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						s->min = min_cost;
						__threadfence_system();
						s->jmin = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0)
				{
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						while (s->jmin == 0) {}
						__threadfence_system();
						s2->min = min_cost = s->min;
						__threadfence();
						s2->jmin = 1;
					}
					else
					{
						while (s2->jmin == 0) {}
						__threadfence();
						min_cost = s2->min;
					}
					b_min_cost = min_cost;
				}
			}
			__syncthreads();
			min_cost = b_min_cost;
		}

		template <class MS, class SC, class TC>
		__device__ __forceinline__ void updateVExchangeSmall(volatile MS *s, volatile MS *s2, unsigned int *semaphore, SC *v, TC *tt, SC &min_cost, int picked, int size)
		{
			if ((picked >= 0) && (picked < size))
			{
				if (threadIdx.x == 0)
				{
					min_cost = (SC)tt[picked] - v[picked];
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						s->min = min_cost;
						__threadfence_system();
						s->jmin = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0)
				{
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						while (s->jmin == 0) {}
						__threadfence_system();
						s2->min = min_cost = s->min;
						__threadfence();
						s2->jmin = 1;
					}
					else
					{
						while (s2->jmin == 0) {}
						__threadfence();
						min_cost = s2->min;
					}
				}
			}
			min_cost = __shfl_sync(0xffffffff, min_cost, 0, 32);
		}

		template <class MS, class SC, class I, class ISTATE, class STATE>
		__device__ __forceinline__ void updateVExchangeSmall(volatile MS* s, volatile MS* s2, unsigned int* semaphore, SC* v, I &iterator, ISTATE &istate, STATE &state, int i, int j, int start, int idx, SC& min_cost, int picked, int size)
		{
			if ((picked >= 0) && (picked < size))
			{
				if (threadIdx.x == 0)
				{
					min_cost = (SC)iterator.getCostForced(i, j, start, istate, state, idx) - v[picked];
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						s->min = min_cost;
						__threadfence_system();
						s->jmin = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0)
				{
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						while (s->jmin == 0) {}
						__threadfence_system();
						s2->min = min_cost = s->min;
						__threadfence();
						s2->jmin = 1;
					}
					else
					{
						while (s2->jmin == 0) {}
						__threadfence();
						min_cost = s2->min;
					}
				}
			}
			min_cost = __shfl_sync(0xffffffff, min_cost, 0, 32);
		}

		template <class MS, class SC>
		__device__ __forceinline__ void updateVExchange(volatile MS *s, volatile MS *s2, unsigned int *semaphore, SC *v, SC &min_cost, int picked, int size)
		{
			__shared__ SC b_min_cost;

			if ((picked >= 0) && (picked < size))
			{
				if (threadIdx.x == 0)
				{
					b_min_cost = min_cost = -v[picked];
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						s->min = min_cost;
						__threadfence_system();
						s->jmin = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0)
				{
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						while (s->jmin == 0) {}
						__threadfence_system();
						s2->min = min_cost = s->min;
						__threadfence();
						s2->jmin = 1;
					}
					else
					{
						while (s2->jmin == 0) {}
						__threadfence();
						min_cost = s2->min;
					}
					b_min_cost = min_cost;
				}
			}
			__syncthreads();
			min_cost = b_min_cost;
		}

		template <class MS, class SC>
		__device__ __forceinline__ void updateVExchangeSmall(volatile MS *s, volatile MS *s2, unsigned int *semaphore, SC *v, SC &min_cost, int picked, int size)
		{
			if ((picked >= 0) && (picked < size))
			{
				if (threadIdx.x == 0)
				{
					min_cost = -v[picked];
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						s->min = min_cost;
						__threadfence_system();
						s->jmin = 1;
					}
				}
			}
			else
			{
				if (threadIdx.x == 0)
				{
					int sem = atomicInc(semaphore, gridDim.x - 1);
					if (sem == 0)
					{
						while (s->jmin == 0) {}
						__threadfence_system();
						s2->min = min_cost = s->min;
						__threadfence();
						s2->jmin = 1;
					}
					else
					{
						while (s2->jmin == 0) {}
						__threadfence();
						min_cost = s2->min;
					}
				}
			}
			min_cost = __shfl_sync(0xffffffff, min_cost, 0, 32);
		}

		template <class SC, class TC>
		__global__ void updateVSingleSmall_kernel(TC *tt, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;
			if (threadIdx.x == 0) min_cost = (SC)tt[picked] - v[picked];
			min_cost = __shfl_sync(0xffffffff, min_cost, 0, 32);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = (SC)tt[j] - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}
		}

		template <class SC, class I, class ISTATE, class STATE>
		__global__ void updateVSingleSmall_kernel(I iterator, ISTATE istate, STATE state, int i, SC* v, int* taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			int idx;
			iterator.openRowWarp(i, j, 0, istate, state, idx);

			SC min_cost;
			if (threadIdx.x == 0) min_cost = (SC)iterator.getCostForced(i, picked, 0, istate, state, idx) - v[picked];
			min_cost = __shfl_sync(0xffffffff, min_cost, 0, 32);

			if (j < size)
			{
				SC t = (SC)iterator.getCost(i, j, 0, istate, state, idx);

				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = t - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}
			iterator.closeRow(istate);
		}

		template <class SC, class TC>
		__global__ void updateVSingle_kernel(TC *tt, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost;
			if (threadIdx.x == 0) b_min_cost = (SC)tt[picked] - v[picked];
			__syncthreads();
			SC min_cost = b_min_cost;

			if (j < size) {
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = (SC)tt[j] - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}
		}

		template <class SC, class I, class ISTATE, class STATE>
		__global__ void updateVSingle_kernel(I iterator, ISTATE istate, STATE state, int i, SC* v, int* taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			int idx;
			iterator.openRowBlock(i, j, 0, istate, state, idx);

			SC t = (SC)iterator.getCost(i, j, 0, istate, state, idx);

			__shared__ SC b_min_cost;
			if (threadIdx.x == 0) b_min_cost = (SC)iterator.getCostForced(i, picked, 0, istate, state, idx) - v[picked];
			__syncthreads();
			SC min_cost = b_min_cost;

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = t - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}
			iterator.closeRow(istate);
		}

		template <class SC, class TC, class MS>
		__global__ void updateVMultiSmall_kernel(volatile MS *s, volatile MS *s2, unsigned int *semaphore, TC *tt, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;

			updateVExchangeSmall(s, s2, semaphore + 1, v, tt, min_cost, picked, size);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = (SC)tt[j] - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}

			if (semaphoreWarp(semaphore))
			{
				if (threadIdx.x == 0) s2->jmin = 0;
			}
		}

		template <class SC, class I, class ISTATE, class STATE, class MS>
		__global__ void updateVMultiSmall_kernel(volatile MS* s, volatile MS* s2, unsigned int* semaphore, I iterator, ISTATE istate, STATE state, int i, int start, SC* v, int* taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;

			int idx;
			iterator.openRowWarp(i, j, start, istate, state, idx);

			SC t;
			if (j < size)
			{
				t = (SC)iterator.getCost(i, j, start, istate, state, idx);
			}

			updateVExchangeSmall(s, s2, semaphore + 1, v, iterator, istate, state, i, j, start, idx, min_cost, picked, size);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = t - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}

			if (semaphoreWarp(semaphore))
			{
				if (threadIdx.x == 0) s2->jmin = 0;
			}

			iterator.closeRow(istate);
		}

		template <class SC, class TC, class MS>
		__global__ void updateVMulti_kernel(volatile MS *s, volatile MS *s2, unsigned int *semaphore, TC *tt, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;

			updateVExchange(s, s2, semaphore + 1, v, tt, min_cost, picked, size);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = (SC)tt[j] - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}

			if (semaphoreBlock(semaphore))
			{
				if (threadIdx.x == 0) s2->jmin = 0;
			}
		}

		template <class SC, class I, class ISTATE, class STATE, class MS>
		__global__ void updateVMulti_kernel(volatile MS* s, volatile MS* s2, unsigned int* semaphore, I iterator, ISTATE istate, STATE state, int i, int start, SC* v, int* taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;

			int idx;
			iterator.openRowBlock(i, j, start, istate, state, idx);

			SC t;
			if (j < size)
			{
				t = (SC)iterator.getCost(i, j, start, istate, state, idx);
			}

			updateVExchange(s, s2, semaphore + 1, v, iterator, istate, state, i, j, start, idx, min_cost, picked, size);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = t - v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}

			if (semaphoreBlock(semaphore))
			{
				if (threadIdx.x == 0) s2->jmin = 0;
			}

			iterator.closeRow(istate);
		}

		template <class SC>
		__global__ void updateVSingleSmallVirtual_kernel(SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;
			if (threadIdx.x == 0) min_cost = -v[picked];
			min_cost = __shfl_sync(0xffffffff, min_cost, 0, 32);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = -v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}
		}

		template <class SC>
		__global__ void updateVSingleVirtual_kernel(SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			__shared__ SC b_min_cost;
			if (threadIdx.x == 0) b_min_cost = -v[picked];
			__syncthreads();
			SC min_cost = b_min_cost;

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = -v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}
		}

		template <class SC, class MS>
		__global__ void updateVMultiSmallVirtual_kernel(volatile MS *s, volatile MS *s2, unsigned int *semaphore, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;

			updateVExchangeSmall(s, s2, semaphore + 1, v, min_cost, picked, size);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = -v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}

			if (semaphoreWarp(semaphore))
			{
				if (threadIdx.x == 0) s2->jmin = 0;
			}
		}

		template <class SC, class MS>
		__global__ void updateVMultiVirtual_kernel(volatile MS *s, volatile MS *s2, unsigned int *semaphore, SC *v, int *taken, int picked, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			SC min_cost;

			updateVExchange(s, s2, semaphore + 1, v, min_cost, picked, size);

			if (j < size)
			{
				if (j == picked) taken[j] = 0;
				else if (taken[j] != 0)
				{
					SC cost_l = -v[j];
					if (cost_l < min_cost) v[j] -= min_cost - cost_l;
				}
			}

			if (semaphoreBlock(semaphore))
			{
				if (threadIdx.x == 0) s2->jmin = 0;
			}
		}
	}
}
