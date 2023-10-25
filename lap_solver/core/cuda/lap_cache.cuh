#pragma once

namespace lap
{
	namespace cuda
	{
		// Kernel for initializing SLRU cache
		__global__ void initCacheSLRU_kernel(int* map, lap::CacheListNode<int>* list, int* first, int* last, int* id, char* priv, int* priv_avail, int* open_row, int* dirty, long long* chit, long long* cmiss, int dim, int entries)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i < dim) map[i] = -1;
			if (i < entries)
			{
				list[i].prev = i - 1;
				list[i].next = (i == entries - 1) ? -1 : i + 1;
				id[i] = -1;
				priv[i] = 0;
			}
			if (i == 0)
			{
				chit[0] = 0ll;
				cmiss[0] = 0ll;
				first[0] = 0;
				last[0] = entries - 1;
				first[1] = -1;
				last[1] = -1;
				priv_avail[0] = entries >> 1;
				open_row[0] = -1;
				dirty[0] = 0;
			}
		}

		// Kernel for resetting SLRU cache
		__global__ void resetCacheSLRU_kernel(lap::CacheListNode<int>* list, int* first, int* last, char* priv, int* priv_avail, int entries)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i < entries)
			{
				list[i].prev = i - 1;
				list[i].next = (i == entries - 1) ? -1 : i + 1;
				priv[i] = 0;
			}
			if (i == 0)
			{
				first[0] = 0;
				last[0] = entries - 1;
				first[1] = -1;
				last[1] = -1;
				priv_avail[0] = entries >> 1;
			}
		}

		// Kernel for initializing LFU cache
		__global__ void initCacheLFU_kernel(int* map, int* count, int* order, int* pos, int* id, int* open_row, int* dirty, long long* chit, long long* cmiss, int dim, int entries)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i < dim)
			{
				map[i] = -1;
				count[i] = 0;
			}
			if (i < entries)
			{
				order[i] = i;
				pos[i] = i;
				id[i] = -1;
			}
			if (i == 0)
			{
				chit[0] = 0;
				cmiss[0] = 0;
			}
		}

	}
}
