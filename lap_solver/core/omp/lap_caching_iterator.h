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
			CF &costfunc;
			TC** rows;
			CACHE *cache;
			bool *tc;
		public:
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
					lapAlloc(rows[t], entries * size, __FILE__, __LINE__);
					// first touch
					//memset(rows[t], 0, entries * size * sizeof(TC));
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
					// everyone will go in here since all intances of the cache hold the same information
					if (costfunc.allEnabled())
					{
						costfunc.getCostRow(rows[t] + (long long)size * (long long)idx, i, ws.part[t].first, ws.part[t].second);
					}
					else
					{
						tc[t] = costfunc.enabled(t);
#pragma omp barrier
						if (costfunc.enabled(t))
						{
							int t_local = t;
							while (t_local < omp_get_max_threads())
							{
								int size_local = ws.part[t_local].second - ws.part[t_local].first;
								costfunc.getCostRow(rows[t_local] + (long long)size_local * (long long)idx, i, ws.part[t_local].first, ws.part[t_local].second);
								// find next
#pragma omp critical
								{
									do
									{
										t_local++;
										if (t_local >= omp_get_max_threads()) t_local = 0;
									} while ((t_local != t) && (tc[t_local] == true));
									if (t_local == t) t_local = omp_get_max_threads();
									else tc[t_local] = true;
								}
							}
						}
#pragma omp barrier
					}
				}
				return rows[t] + (long long)size * (long long)idx;
			}
		};
	}
}
