#pragma once

#include "lap_direct_iterator.h"
#include "lap_solver.h"
#include <cuda.h>

namespace lap
{
	namespace cuda
	{
		template <class SC, class TC, class CF, class CACHE>
		class CachingIterator
		{
		protected:
			int dim, dim2;
			long long max_memory;
			TC** rows;
			CACHE* cache;
		public:
			CF &costfunc;
			Worksharing &ws;

		public:
			CachingIterator(int dim, int dim2, long long max_memory, CF &costfunc, Worksharing &ws)
				: dim(dim), dim2(dim2), max_memory(max_memory), costfunc(costfunc), ws(ws)
			{
				int devices = (int)ws.device.size();
				lapAlloc(cache, devices, __FILE__, __LINE__);
				lapAlloc(rows, devices, __FILE__, __LINE__);
				for (int t = 0; t < devices; t++)
				{
					int size = ws.part[t].second - ws.part[t].first;
					// entries potentially vary between GPUs
					cache[t].setSize((int)std::min((long long) dim2, max_memory / size), dim);
					cudaSetDevice(ws.device[t]);
					lapAllocDevice(rows[t], (long long)cache[t].getEntries() * (long long)size, __FILE__, __LINE__);
				}
			}

			~CachingIterator()
			{
				int devices = (int)ws.device.size();
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(ws.device[t]);
					lapFreeDevice(rows[t]);
				}
				lapFree(rows);
				lapFree(cache);
			}

			__forceinline void getHitMiss(long long &hit, long long &miss) { cache[0].getHitMiss(hit, miss); }

			__forceinline const TC *getRow(int t, int i, bool async)
			{
				int size = ws.part[t].second - ws.part[t].first;
				int idx;
				bool found = cache[t].find(idx, i);
				if (!found)
				{
					costfunc.getCostRow(rows[t] + (long long)size * (long long)idx, t, ws.stream[t], i, ws.part[t].first, ws.part[t].second, async);
				}
				return rows[t] + (long long)size * (long long)idx;
			}

			__forceinline bool checkRow(int t, int i)
			{
				return cache[t].present(i);
			}
		};
	}
}
