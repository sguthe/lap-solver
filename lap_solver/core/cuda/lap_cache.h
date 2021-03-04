#pragma once
#include "../lap_cache.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "lap_cache.cuh"

namespace lap
{
	namespace cuda
	{
		// segmented least recently used
		class CacheSLRU
		{
		private:
			// actual buffer for storing data
			char* buffer;
		protected:
			// everything needs to be stored in pointers so it survives kernel calls
			lap::CacheListNode<int> *list;
			int *first;
			int *last;
			int *id;
			char *priv;
			int *priv_avail;
			long long *chit, *cmiss;
			int *map;
			int* open_row;
			int* dirty;
			unsigned int* semaphore;
			// this will never be changed from GPU side
			int entries;

			__forceinline __device__ void remove_entry(int i)
			{
				int l = priv[i];
				int prev = list[i].prev;
				int next = list[i].next;
				if (prev != -1) list[prev].next = next;
				if (next != -1) list[next].prev = prev;
				if (first[l] == i) first[l] = next;
				if (last[l] == i) last[l] = prev;
			}

			__forceinline __device__ void remove_first(int i)
			{
				int l = priv[i];
				first[l] = list[i].next;
				list[first[l]].prev = -1;
			}

			__forceinline __device__ void push_last(int i)
			{
				int l = priv[i];
				list[i].prev = last[l];
				list[i].next = -1;
				if (last[l] == -1)
				{
					first[l] = i;
				}
				else
				{
					list[last[l]].next = i;
				}
				last[l] = i;
			}

		public:
			CacheSLRU() { entries = 0; }
			~CacheSLRU() {}

			__forceinline void setSize(int entries, int dim)
			{
				this->entries = entries;

				size_t buffer_size = sizeof(*chit) + sizeof(*cmiss) + sizeof(*list) * entries + sizeof(*map) * dim + sizeof(*id) * entries + sizeof(*semaphore) +
									 sizeof(*first) * 2 + sizeof(last) * 2 + sizeof(*priv_avail) + sizeof(*open_row) + sizeof(*dirty) + sizeof(*priv) * entries;
				lapAllocDevice(buffer, buffer_size, __FILE__, __LINE__);

				buffer_size = 0;
				chit = (long long*)(buffer + buffer_size);
				buffer_size += sizeof(*chit);
				cmiss = (long long*)(buffer + buffer_size);
				buffer_size += sizeof(*cmiss);
				list = (lap::CacheListNode<int>*)(buffer + buffer_size);
				buffer_size += sizeof(*list) * entries;
				map = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*map) * dim;
				id = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*id) * entries;
				first = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*first) * 2;
				last = (int*)(buffer + buffer_size);
				buffer_size += sizeof(last) * 2;
				priv_avail = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*priv_avail);
				open_row = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*open_row);
				dirty = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*dirty);
				semaphore = (unsigned int*)(buffer + buffer_size);
				buffer_size += sizeof(*semaphore);
				priv = buffer + buffer_size;

				initCacheSLRU_kernel<<<(std::max(dim, entries) + 255) >> 8, 256>>>(map, list, first, last, id, priv, priv_avail, open_row, dirty, chit, cmiss, dim, entries);
			}

			__forceinline void destroy()
			{
				lapFreeDevice(buffer);
			}

			__forceinline __device__ bool open(int& idx, int i)
			{
				idx = atomicCAS(open_row, -1, -2);
				if (idx == -1)
				{
					// first to arrive
					if (map[i] == -1)
					{
						// replace
						idx = first[0];
						dirty[0] = 1;
						__threadfence();
						atomicCAS(open_row, -2, idx);

						if (id[idx] != -1) map[id[idx]] = -1;
						id[idx] = i;
						map[i] = idx;
						remove_first(idx);
						push_last(idx);
						cmiss[0]++;
						return false;
					}
					else
					{
						// found
						idx = map[i];
						dirty[0] = 0;
						__threadfence();
						atomicCAS(open_row, -2, idx);

						if (priv[idx] == -1)
						{
							priv[idx] = 0;
							remove_entry(idx);
						}
						else
						{
							remove_entry(idx);
							if (priv[idx] == 0)
							{
								priv[idx] = 1;
								if (priv_avail[0] > 0)
								{
									priv_avail[0]--;
								}
								else
								{
									int idx1 = first[1];
									remove_first(idx1);
									priv[idx1] = 0;
									push_last(idx1);
								}
							}
						}
						push_last(idx);
						chit[0]++;
						return true;
					}
				}
				else
				{
					while (idx < 0) idx = atomicCAS(open_row, -1, -2);
					__threadfence();
					return (((volatile int*)dirty)[0] == 0);
				}
			}

			__forceinline __device__ bool findWarp(int& idx, int i)
			{
				bool ret;
				if (threadIdx.x == 0) ret = open(idx, i);

				idx = __shfl_sync(0xffffffff, idx, 0, 32);
				ret = __shfl_sync(0xffffffff, ret, 0, 32);
				return ret;
			}

			__forceinline __device__ bool findBlock(int& idx, int i)
			{
				__shared__ bool b_ret;
				__shared__ int b_idx;
				if (threadIdx.x == 0) b_ret = open(b_idx, i);
				__syncthreads();

				idx = b_idx;
				return b_ret;
			}

			__forceinline __device__ void close()
			{
				if (semaphoreOnce(semaphore))
				{
					open_row[0] = -1;
					dirty[0] = 0;
				}
			}

			__forceinline void restart()
			{
				resetCacheSLRU_kernel<<<(entries + 255) >> 8, 256>>>(list, first, last, priv, priv_avail, entries);
			}

			__forceinline void getHitMiss(long long& hit, long long& miss)
			{
				cudaMemcpyAsync(&hit, chit, sizeof(long long), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(&miss, cmiss, sizeof(long long), cudaMemcpyDeviceToHost);
				cudaMemsetAsync(chit, 0, sizeof(long long));
				cudaMemsetAsync(cmiss, 0, sizeof(long long));
				cudaDeviceSynchronize();
			}

			__forceinline int getEntries() { return entries; }
		};

		// least frequently used
		class CacheLFU
		{
		private:
			// actual buffer for storing data
			char* buffer;
		protected:
			// everything needs to be stored in pointers so it survives kernel calls
			long long* chit;
			long long *cmiss;
			int* map;
			int* id;
			int* count;
			int* order;
			int* pos;
			int* open_row;
			int* dirty;
			unsigned int* semaphore;
			// these will never be changed on the GPU
			int entries, dim;

			template <typename T>
			__forceinline __device__ void swap(T& a, T& b)
			{
				T c = a;
				a = b;
				b = c;
			}

			__forceinline __device__ void advance(int start)
			{
				// this uses a heap now
				bool done = false;
				int i = start;
				int ii = id[order[i]];
				int ci = count[ii];
				while (!done)
				{
					int l = i + i + 1;
					int r = l + 1;
					if (l >= entries) done = true;
					else
					{
						if (r >= entries)
						{
							int il = id[order[l]];
							int cl;
							if (il == -1) cl = -1; else cl = count[il];
							if ((ci > cl) || ((ci == cl) && (ii < il)))
							{
								swap(order[i], order[l]);
								pos[order[i]] = i;
								pos[order[l]] = l;
								i = l;
								ii = id[order[i]];
								ci = count[ii];
							}
							else
							{
								done = true;
							}
						}
						else
						{
							int il = id[order[l]];
							int ir = id[order[r]];
							int cl, cr;
							if (il == -1) cl = -1; else cl = count[il];
							if (ir == -1) cr = -1; else cr = count[ir];
							if ((cr > cl) || ((cr == cl) && (ir < il)))
							{
								// left
								if ((ci > cl) || ((ci == cl) && (ii < il)))
								{
									swap(order[i], order[l]);
									pos[order[i]] = i;
									pos[order[l]] = l;
									i = l;
								}
								else
								{
									done = true;
								}
							}
							else
							{
								// right
								if ((ci > cr) || ((ci == cr) && (ii < ir)))
								{
									swap(order[i], order[r]);
									pos[order[i]] = i;
									pos[order[r]] = r;
									i = r;
								}
								else
								{
									done = true;
								}
							}
						}
					}
				}
			}

		public:
			CacheLFU() {}
			~CacheLFU() {}

			__forceinline void setSize(int entries, int dim)
			{
				this->entries = entries;
				this->dim = dim;

				size_t buffer_size = sizeof(*chit) + sizeof(*cmiss) + sizeof(*map) * dim + sizeof(*count) * dim + sizeof(*id) * entries +
					sizeof(*order) * entries + sizeof(pos) * entries + sizeof(*open_row) + sizeof(*dirty) + sizeof(*semaphore);
				lapAllocDevice(buffer, buffer_size, __FILE__, __LINE__);

				buffer_size = 0;
				chit = (long long*)(buffer + buffer_size);
				buffer_size += sizeof(*chit);
				cmiss = (long long*)(buffer + buffer_size);
				buffer_size += sizeof(*cmiss);
				map = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*map) * dim;
				count = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*count) * dim;
				id = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*id) * entries;
				order = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*order) * entries;
				pos = (int*)(buffer + buffer_size);
				buffer_size += sizeof(pos) * entries;
				open_row = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*open_row);
				dirty = (int*)(buffer + buffer_size);
				buffer_size += sizeof(*dirty);
				semaphore = (unsigned int*)(buffer + buffer_size);

				initCacheLFU_kernel<<<(std::max(dim, entries) + 255) >> 8, 256>>>(map, count, order, pos, id, open_row, dirty, chit, cmiss, dim, entries);
			}

			__forceinline void destroy()
			{
				lapFreeDevice(buffer);
			}

			__forceinline __device__ bool open(int& idx, int i)
			{
				idx = atomicCAS(open_row, -1, -2);
				if (idx == -1)
				{
					// first to arrive
					if (map[i] == -1)
					{
						// replace
						idx = order[0];
						dirty[0] = 1;
						__threadfence();
						atomicCAS(open_row, -2, idx);
						if (id[idx] != -1) map[id[idx]] = -1;
						id[idx] = i;
						map[i] = idx;
						count[i]++;
						advance(0);
						cmiss[0]++;
						return false;
					}
					else
					{
						idx = map[i];
						dirty[0] = 0;
						__threadfence();
						atomicCAS(open_row, -2, idx);
						count[i]++;
						advance(pos[idx]);
						chit[0]++;
						return true;
					}
				}
				else
				{
					while (idx < 0)
					{
						idx = atomicCAS(open_row, -1, -2);
					}
					__threadfence();
					return  (((volatile int*)dirty)[0] == 0);
				}
			}

			__forceinline __device__ bool findWarp(int& idx, int i)
			{
				bool ret;
				if (threadIdx.x == 0) ret = open(idx, i);

				idx = __shfl_sync(0xffffffff, idx, 0, 32);
				ret = __shfl_sync(0xffffffff, ret, 0, 32);
				return ret;
			}

			__forceinline __device__ bool findBlock(int& idx, int i)
			{
				__shared__ bool b_ret;
				__shared__ int b_idx;
				if (threadIdx.x == 0) b_ret = open(b_idx, i);
				__syncthreads();

				idx = b_idx;
				return b_ret;
			}

			__forceinline __device__ void close()
			{
				if (semaphoreOnce(semaphore))
				{
					open_row[0] = -1;
					dirty[0] = 0;
				}
			}

			__forceinline void restart()
			{
				cudaMemset(count, 0, sizeof(int) * dim);
			}

			__forceinline void getHitMiss(long long& hit, long long& miss)
			{
				cudaMemcpyAsync(&hit, chit, sizeof(long long), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(&miss, cmiss, sizeof(long long), cudaMemcpyDeviceToHost);
				cudaMemsetAsync(chit, 0, sizeof(long long));
				cudaMemsetAsync(cmiss, 0, sizeof(long long));
				cudaDeviceSynchronize();
			}

			__forceinline int getEntries() { return entries; }
		};
	};
}
