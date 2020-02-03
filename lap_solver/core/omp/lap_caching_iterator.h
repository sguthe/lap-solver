#pragma once

#include <iostream>
#include <chrono>
#include "lap_worksharing.h"
#include "../lap_caching_iterator.h"

namespace lap
{
	namespace omp
	{
		template <class SC, class TC, class CF, class CACHE>
		class CachingIterator
		{
		protected:
			int dim, dim2;
			int entries;
			TC** rows;
			CACHE *cache;
			bool *tc;
		public:
			CF &costfunc;
			Worksharing &ws;

		public:
			CachingIterator(int dim, int dim2, int entries, CF &costfunc, Worksharing &ws)
				: dim(dim), dim2(dim2), entries(entries), costfunc(costfunc), ws(ws)
			{
				int max_threads = omp_get_max_threads();
				lapAlloc(cache, max_threads, __FILE__, __LINE__);
				lapAlloc(rows, max_threads, __FILE__, __LINE__);
				lapAlloc(tc, max_threads, __FILE__, __LINE__);
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int size = ws.part[t].second - ws.part[t].first;
					cache[t].setSize(entries, dim);
					lapAlloc(rows[t], (size_t)entries * (size_t)size, __FILE__, __LINE__);
					// first touch
					rows[t][0] = TC(0);
				}
			}

			~CachingIterator()
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					lapFree(rows[t]);
				}
				lapFree(rows);
				lapFree(cache);
				lapFree(tc);
			}

			__forceinline void getHitMiss(long long &hit, long long &miss) { cache[0].getHitMiss(hit, miss); }

			__forceinline const TC *getRow(int t, int i)
			{
				int size = ws.part[t].second - ws.part[t].first;
				int idx;
				bool found = cache[t].find(idx, i);
				if (!found)
				{
					costfunc.getCostRow(rows[t] + (long long)size * (long long)idx, i, ws.part[t].first, ws.part[t].second);
				}
				return rows[t] + (long long)size * (long long)idx;
			}
		};
	}
}
