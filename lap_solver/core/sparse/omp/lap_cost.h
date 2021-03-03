#pragma once

#include "lap_worksharing.h"
#include "../lap_cost.h"

namespace lap
{
	namespace sparse
	{
		namespace omp
		{
			// Wrapper around simple cost function, scheduling granularity is assumed to be 8 for load balancing
			template <class TC, typename GETROWLENGTH, typename GETCOST>
			class SimpleCostFunction : public lap::sparse::SimpleCostFunction<TC, GETROWLENGTH, GETCOST>
			{
			protected:
				bool sequential;
			public:
				SimpleCostFunction(GETROWLENGTH &getrowlength, GETCOST &getcost, bool sequential = false) : lap::sparse::SimpleCostFunction<TC, GETROWLENGTH, GETCOST>(getrowlength, getcost), sequential(sequential) {}
				~SimpleCostFunction() {}
			public:
				__forceinline int getMultiple() const { return 8; }
				__forceinline bool isSequential() const { return sequential; }
			};

			// Costs stored in a table. Used for conveniency only
			// This can be constructed using a CostFunction from above or by specifying an array that holds the data (does not copy the data in this case).
			template <class TC>
			class TableCost
			{
			protected:
				int x_size;
				int y_size;
				size_t **row_start;
				int** row_length;
				int** col_idx;
				TC** rows;
				bool free_in_destructor;
				Worksharing &ws;
			protected:
				template <class DirectCost>
				void initTable(DirectCost &cost)
				{
					free_in_destructor = true;
					// needs to get the size of the table first
					lapAlloc(row_start, omp_get_max_threads(), __FILE__, __LINE__);
					lapAlloc(row_length, omp_get_max_threads(), __FILE__, __LINE__);
					lapAlloc(col_idx, omp_get_max_threads(), __FILE__, __LINE__);
					lapAlloc(rows, omp_get_max_threads(), __FILE__, __LINE__);
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						size_t table_size(0);
						for (int x = 0; x < x_size; x++) table_size += (size_t)cost.getRowLength(x, ws.part[t].first, ws.part[t].second);
						lapAlloc(rows[t], table_size, __FILE__, __LINE__);
						lapAlloc(col_idx[t], table_size, __FILE__, __LINE__);
						lapAlloc(row_length[t], x_size, __FILE__, __LINE__);
						lapAlloc(row_start[t], x_size, __FILE__, __LINE__);
						// first touch
						rows[t][0] = TC(0);
						col_idx[t][0] = 0;
						row_length[t][0] = 0;
						row_start[t][0] = 0;
					}
					if (cost.isSequential())
					{
						// cost table needs to be initialized sequentially
						for (int x = 0; x < x_size; x++)
						{
							std::vector<int> idx(omp_get_max_threads());
							for (int t = 0; t < omp_get_max_threads(); t++)
							{
								row_start[t][x] = idx[t];
								idx[t] += row_length[t][x] = cost.getCostRow(&(col_idx[t][idx[t]]), &(rows[t][idx[t]]), x, ws.part[t].first, ws.part[t].second);
							}
						}
					}
					else
					{
						// create and initialize in parallel
#pragma omp parallel
						{
							int t = omp_get_thread_num();
							int idx = 0;
							for (int x = 0; x < x_size; x++)
							{
								row_start[t][x] = idx;
								idx += row_length[t][x] = cost.getCostRow(&(col_idx[t][idx]), &(rows[t][idx]), x, ws.part[t].first, ws.part[t].second);
							}
						}
					}
				}

			public:
				template <class DirectCost> TableCost(int x_size, int y_size, DirectCost &cost, Worksharing &ws) :
					x_size(x_size), y_size(y_size), ws(ws) {
					initTable(cost);
				}
				template <class DirectCost> TableCost(int size, DirectCost &cost, Worksharing &ws) :
					x_size(size), y_size(size), ws(ws) {
					initTable(cost);
				}
				~TableCost()
				{
					if (free_in_destructor)
					{
#pragma omp parallel
						{
							int t = omp_get_thread_num();
							lapFree(row_start[t]);
							lapFree(row_length[t]);
							lapFree(col_idx[t]);
							lapFree(rows[t]);
						}
					}
					lapFree(row_start);
					lapFree(row_length);
					lapFree(col_idx);
					lapFree(rows);
				}
			public:
				__forceinline const std::tuple<int, int *, TC *> getRow(int t, int x) const { return std::tuple<int, int *, TC *>(row_length[t][x], &(col_idx[t][row_start[t][x]]), &(rows[t][row_start[t][x]])); }
				__forceinline const TC getCost(int x, int y) const
				{
					int t = 0;
					while (y >= ws.part[t].second) t++;
					auto row = getRow(t, x);
					for (int yy = 0; yy < std::get<0>(row); yy++)
					{
						if (std::get<1>(row)[yy] == y) return std::get<2>(row)[yy];
					}
					return std::numeric_limits<TC>::infinity();
				}
			};
		}
	}
}
