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
			struct table_t
			{
				TC* rows;
				CACHE cache;
				char dummy[256 - sizeof(TC*) - sizeof(CACHE)];
			};
			table_t* table;
		public:
			CF &costfunc;
			Worksharing &ws;

		public:
			CachingIterator(int dim, int dim2, int entries, CF &costfunc, Worksharing &ws)
				: dim(dim), dim2(dim2), entries(entries), costfunc(costfunc), ws(ws)
			{
				int max_threads = omp_get_max_threads();
				lapAlloc(table, max_threads, __FILE__, __LINE__);
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int size = ws.part[t].second - ws.part[t].first;
					table[t].cache.setSize(entries, dim);
					lapAlloc(table[t].rows, (size_t)entries * (size_t)size, __FILE__, __LINE__);
					// first touch
					table[t].rows[0] = TC(0);
				}
			}

			~CachingIterator()
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					lapFree(table[t].rows);
				}
				lapFree(table);
			}

			__forceinline void getHitMiss(long long &hit, long long &miss) { table[0].cache.getHitMiss(hit, miss); }

			__forceinline const TC *getRow(int t, int i)
			{
				int size = ws.part[t].second - ws.part[t].first;
				int idx;
				bool found = table[t].cache.find(idx, i);
				if (!found)
				{
					costfunc.getCostRow(table[t].rows + (long long)size * (long long)idx, i, ws.part[t].first, ws.part[t].second);
				}
				return table[t].rows + (long long)size * (long long)idx;
			}
		};
	}
}
