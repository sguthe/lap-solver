#pragma once

#include "lap_direct_iterator.h"

namespace lap
{
	namespace sparse
	{
		template <class SC, class TC, class CF, class CACHE>
		class CachingIterator
		{
		protected:
			int dim, dim2;
			int entries;
			TC* rows;
			int* col_idx;
			int* row_length;
			CACHE cache;
		public:
			CF &costfunc;
		public:
			// dim2 is the maximum number of elements in a row
			CachingIterator(int dim, int dim2, int entries, CF &costfunc)
				: dim(dim), dim2(dim2), entries(entries), costfunc(costfunc)
			{
				cache.setSize(entries, dim, dim2);
				lapAlloc(rows, (long long)entries * (long long)dim2, __FILE__, __LINE__);
				lapAlloc(col_idx, (long long)entries * (long long)dim2, __FILE__, __LINE__);
				lapAlloc(row_length, (long long)dim2, __FILE__, __LINE__);
			}

			~CachingIterator()
			{
				lapFree(rows);
				lapFree(col_idx);
				lapFree(row_length);
			}

			__forceinline void getHitMiss(long long &hit, long long &miss) { cache.getHitMiss(hit, miss); }

			__forceinline const std::tuple<int, int *, TC *> getRow(int i)
			{
				int idx, off;
				if (row_length[i] < 0) row_length[i] = costfunc.getRowLength(i, 0, dim2);
				bool found = cache.find(idx, off, i, row_length[i]);
				long long offset = (long long)dim2 * (long long)idx + (long long)off;
				if (!found)
				{
					costfunc.getCostRow(col_idx + offset, rows + offset, i, 0, dim2);
				}
				return std::tuple<int, int *, TC *>(row_length[i], col_idx + offset, rows + offset);
			}
		};
	}
}
