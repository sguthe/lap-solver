#pragma once

#include "lap_direct_iterator.h"
#include "lap_solver.h"
#include <cuda.h>

namespace lap
{
	namespace cuda
	{
		template <class TC, class CF, class CACHE>
		class CachingIterator
		{
		protected:
			TC** rows;
			CACHE** cache;
		public:
			CF& costfunc;
			Worksharing& ws;
			static const bool GPU = false;

		public:
			CachingIterator(int dim, int dim2, long long max_memory, CF& costfunc, Worksharing& ws)
				: costfunc(costfunc), ws(ws)
			{
				int devices = (int)ws.device.size();
				lapAlloc(cache, devices, __FILE__, __LINE__);
				lapAlloc(rows, devices, __FILE__, __LINE__);
				for (int t = 0; t < devices; t++)
				{
					lapAlloc(cache[t], 1, __FILE__, __LINE__);
					int size = ws.part[t].second - ws.part[t].first;
					// entries potentially vary between GPUs
					cudaSetDevice(ws.device[t]);
					cache[t]->setSize((int)std::min((long long)dim2, max_memory / size), dim);
					lapAllocDevice(rows[t], (long long)cache[t]->getEntries() * (long long)size, __FILE__, __LINE__);
				}
			}

			~CachingIterator()
			{
				int devices = (int)ws.device.size();
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(ws.device[t]);
					lapFreeDevice(rows[t]);
					lapFree(cache[t]);
				}
				lapFree(rows);
				lapFree(cache);
			}

			__forceinline CACHE& getCache(int i) { return *cache[i]; }
			__forceinline TC* getCacheRows(int i) { return rows[i]; }

			__forceinline void getHitMiss(long long& hit, long long& miss) { cache[0]->getHitMiss(hit, miss); }

			__forceinline const TC* getRow(int t, int i, bool async)
			{
				int size = ws.part[t].second - ws.part[t].first;
				int idx;
				bool found = cache[t]->find(idx, i);
				if (!found)
				{
					costfunc.getCostRow(rows[t] + (long long)size * (long long)idx, t, ws.stream[t], i, ws.part[t].first, ws.part[t].second, 1, async);
				}
				return rows[t] + (long long)size * (long long)idx;
			}

			__forceinline void fillRows(int t, int row_count)
			{
				costfunc.getCostRow(rows[t], t, ws.stream[t], 0, ws.part[t].first, ws.part[t].second, row_count, false);
			}
		};

		template <class TC, class CACHE>
		struct DeviceCachingIteratorState
		{
			TC* rows;
			CACHE cache;
			int size;

			void setSize(int dim, int entries, int size)
			{
				this->size = size;
				cache.setSize(entries, dim);
				lapAllocDevice(rows, (long long)entries * (long long)size, __FILE__, __LINE__);
			}

			void destroy()
			{
				cache.destroy();
				lapFreeDevice(rows);
			}
		};

		template <class TC, class CF>
		class DeviceCachingIteratorObject
		{
		protected:
			CF& costfunc;
		public:
			DeviceCachingIteratorObject(CF& costfunc) : costfunc(costfunc) { }

			~DeviceCachingIteratorObject() { }

			// once per block, pointer should be in shared memory
			template <class ISTATE, class STATE>
			__forceinline __device__ void openRowWarp(int i, int j, int start, ISTATE& istate, STATE& state, int& idx)
			{
				bool found = istate.cache.findWarp(idx, i);
				if ((!found) && (j < istate.size)) istate.rows[(size_t)idx * (size_t)istate.size + j] = costfunc(i, j + start, state);
			}

			template <class ISTATE, class STATE>
			__forceinline __device__ void openRowBlock(int i, int j, int start, ISTATE& istate, STATE& state, int& idx)
			{
				bool found = istate.cache.findBlock(idx, i);
				if ((!found) && (j < istate.size)) istate.rows[(size_t)idx * (size_t)istate.size + j] = costfunc(i, j + start, state);
			}

			// once per grid
			template <class ISTATE>
			__forceinline __device__ void closeRow(ISTATE& istate)
			{
				istate.cache.close();
			}

			template <class ISTATE, class STATE>
			__forceinline __device__ TC getCost(int i, int j, int start, ISTATE& istate, STATE& state, int& idx)
			{
				return istate.rows[(size_t)idx * (size_t)istate.size + j];
			}

			template <class ISTATE, class STATE>
			__forceinline __device__ TC getCostForced(int i, int j, int start, ISTATE& istate, STATE& state, int& idx)
			{
				return costfunc(i, j + start, state);
			}
		};

		template <class TC, class CF, class GETCOST, class CACHE>
		class DeviceCachingIterator
		{
		protected:
			DeviceCachingIteratorState<TC, CACHE> *istate;
			DeviceCachingIteratorObject<TC, GETCOST> iobject;
			CF& costfunc;
		public:
			Worksharing& ws;
			static const bool GPU = true;

			DeviceCachingIterator(int dim, int dim2, long long max_memory, CF& costfunc, GETCOST& getcost, Worksharing& ws)
				: iobject(getcost), costfunc(costfunc), ws(ws)
			{
				int devices = (int)ws.device.size();
				lapAlloc(istate, devices, __FILE__, __LINE__);
				for (int t = 0; t < devices; t++)
				{
					int size = ws.part[t].second - ws.part[t].first;
					int entries = (int)std::min((long long)dim2, max_memory / size);
					// entries potentially vary between GPUs
					cudaSetDevice(ws.device[t]);
					istate[t].setSize(dim, entries, size);
				}
			}

			~DeviceCachingIterator()
			{
				int devices = (int)ws.device.size();
				for (int t = 0; t < devices; t++) istate[t].destroy();
				lapFree(istate);
			}

			__forceinline void getHitMiss(long long& hit, long long& miss) { istate->cache.getHitMiss(hit, miss); }

			__forceinline decltype(costfunc.getState(0))& getState(int t) { return costfunc.getState(t); }

			__forceinline DeviceCachingIteratorState<TC, CACHE>& getIState(int t) { return istate[t]; }

			__forceinline DeviceCachingIteratorObject<TC, GETCOST>& getIObject() { return iobject; }
		};
	}
}
