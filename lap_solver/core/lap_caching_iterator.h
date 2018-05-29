#pragma once

#include "lap_direct_iterator.h"

namespace lap
{
	template <class SC, class TC, class CF, class CACHE>
	class CachingIterator
	{
	protected:
		int dim, dim2;
		int entries;
		CF &costfunc;
		TC** rows;
		CACHE cache;

	public:
		CachingIterator(int dim, int dim2, int entries, CF &costfunc)
			: dim(dim), dim2(dim2), entries(entries), costfunc(costfunc)
		{
			cache.setSize(entries, dim);
			lapAlloc(rows, entries, __FILE__, __LINE__);
			for (int i = 0; i < entries; i++)
			{
				lapAlloc(rows[i], dim2, __FILE__, __LINE__);
			}
		}

		~CachingIterator()
		{
			for (int i = 0; i < entries; i++)
			{
				lapFree(rows[i]);
			}
			lapFree(rows);
		}

		__forceinline void getHitMiss(long long &hit, long long &miss) { cache.getHitMiss(hit, miss); }

		__forceinline const TC *getRow(int i)
		{
			int idx;
			bool found = cache.find(idx, i);
			if (!found)
			{
				costfunc.getCostRow(rows[idx], i, 0, dim2);
			}
			return rows[idx];
		}
	};
}
