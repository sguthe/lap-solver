#pragma once

#include <iostream>
#include <chrono>
#include "lap_worksharing.h"
#include "../lap_caching_iterator.h"

namespace lap
{
	namespace sparse
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
				int** col_idx;
				int** row_length;
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
					lapAlloc(col_idx, max_threads, __FILE__, __LINE__);
					lapAlloc(row_length, max_threads, __FILE__, __LINE__);
					lapAlloc(tc, max_threads, __FILE__, __LINE__);
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						int size = ws.part[t].second - ws.part[t].first;
						cache[t].setSize(entries, dim, size);
						lapAlloc(rows[t], entries * size, __FILE__, __LINE__);
						lapAlloc(col_idx[t], entries * size, __FILE__, __LINE__);
						lapAlloc(row_length[t], dim2, __FILE__, __LINE__);
						// first touch
						rows[t][0] = TC(0);
						col_idx[t][0] = 0;
						std::fill(row_length[t], row_length[t] + dim2, -1);
					}
				}

				~CachingIterator()
				{
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						lapFree(rows[t]);
						lapFree(col_idx[t]);
						lapFree(row_length[t]);
					}
					lapFree(rows);
					lapFree(col_idx);
					lapFree(row_length);
					lapFree(cache);
					lapFree(tc);
				}

				__forceinline void getHitMiss(long long &hit, long long &miss) { cache[0].getHitMiss(hit, miss); }

				__forceinline const std::tuple<int, int *, TC *> getRow(int t, int i)
				{
					int size = ws.part[t].second - ws.part[t].first;
					int idx;
					int off;
					if (row_length[t][i] < 0) row_length[t][i] = costfunc.getRowLength(i, ws.part[t].first, ws.part[t].second);
					bool found = cache[t].find(idx, off, i, row_length[t][i]);
					long long offset = (long long)size * (long long)idx + (long long)off;
					if (!found)
					{
						costfunc.getCostRow(col_idx[t] + offset, rows[t] + offset, i, ws.part[t].first, ws.part[t].second);
					}
					return std::tuple<int, int *, TC *>(row_length[t][i], col_idx[t] + offset, rows[t] + offset);
				}
			};
		}
	}
}
