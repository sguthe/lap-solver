#pragma once
#include <iostream>

#include "lap_kernel.cuh"
#include "lap_worksharing.h"

namespace lap
{
	namespace cuda
	{
		// Wrapper around per-row cost funtion, e.g. CUDA, OpenCL or OpenMPI
		// Requires separate kernel to calculate costs for a given rowsol
		template <class TC, typename GETCOSTROW, typename GETCOST>
		class RowCostFunction
		{
		protected:
			GETCOSTROW getcostrow;
			GETCOST getcost;
			TC initialEpsilon;
			TC lowerEpsilon;
		public:
			RowCostFunction(GETCOSTROW &getcostrow, GETCOST &getcost) : getcostrow(getcostrow), getcost(getcost), initialEpsilon(0), lowerEpsilon(0) {}
			~RowCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline const TC getLowerEpsilon() const { return lowerEpsilon; }
			__forceinline void setLowerEpsilon(TC eps) { lowerEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end, bool async) const { getcostrow(row, t, stream, x, start, end, async); }
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim) const { getcost(row, stream, rowsol, dim); }
		};

		// Wrapper around simple cost function
		template <class TC, typename GETCOST, typename STATE>
		class SimpleCostFunction
		{
		protected:
			GETCOST getcost;
			STATE *state;
			TC initialEpsilon;
			TC lowerEpsilon;
		public:
			SimpleCostFunction(GETCOST &getcost, STATE *state) : getcost(getcost), state(state), initialEpsilon(0), lowerEpsilon(0) {}
			~SimpleCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline const TC getLowerEpsilon() const { return lowerEpsilon; }
			__forceinline void setLowerEpsilon(TC eps) { lowerEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end, bool async) const
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_kernel<<<grid_size, block_size, 0, stream>>>(row, getcost, state[t], x, start, end - start);
			}
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim) const
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (dim + block_size.x - 1) / block_size.x;
				getCost_kernel<<<grid_size, block_size, 0, stream>>>(row, getcost, state[0], rowsol, dim);
			}
		};

		// Costs stored in a table. Used for conveniency only
		// This can be constructed using a CostFunction from above or by specifying an array that holds the data (does not copy the data in this case).
		template <class TC>
		class PinnedTableCost
		{
		protected:
			int x_size;
			int y_size;
			TC** cc;
			int* stride;
			bool free_in_destructor;
			Worksharing& ws;
		protected:
			void referenceTable(TC* tab)
			{
				free_in_destructor = false;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
				for (int t = 0; t < omp_get_max_threads(); t++)
				{
					stride[t] = y_size;
					cc[t] = &(tab[ws.part[t].first]);
				}
			}

			template <class DirectCost>
			void initTable(DirectCost& cost)
			{
				free_in_destructor = true;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
				if (cost.isSequential())
				{
					// cost table needs to be initialized sequentially
#pragma omp parallel
					{
						const int t = omp_get_thread_num();
						stride[t] = ws.part[t].second - ws.part[t].first;
						lapAllocPinned(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						// first touch
						cc[t][0] = TC(0);
					}
					for (int x = 0; x < x_size; x++)
					{
						for (int t = 0; t < omp_get_max_threads(); t++)
						{
							cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
						}
					}
				}
				else
				{
					// create and initialize in parallel
#pragma omp parallel
					{
						const int t = omp_get_thread_num();
						stride[t] = ws.part[t].second - ws.part[t].first;
						lapAllocPinned(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						// first touch
						cc[t][0] = TC(0);
						for (int x = 0; x < x_size; x++)
						{
							cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
						}
					}
				}
			}

			void createTable()
			{
				free_in_destructor = true;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
#pragma omp parallel
				{
					const int t = omp_get_thread_num();
					stride[t] = ws.part[t].second - ws.part[t].first;
					lapAllocPinned(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
					// first touch
					cc[t][0] = TC(0);
				}
			}
		public:
			template <class DirectCost> PinnedTableCost(int x_size, int y_size, DirectCost& cost, Worksharing& ws) :
				x_size(x_size), y_size(y_size), ws(ws) {
				initTable(cost);
			}
			template <class DirectCost> PinnedTableCost(int size, DirectCost& cost, Worksharing& ws) :
				x_size(size), y_size(size), ws(ws) {
				initTable(cost);
			}
			PinnedTableCost(int x_size, int y_size, Worksharing& ws) : x_size(x_size), y_size(y_size), ws(ws) { createTable(); }
			PinnedTableCost(int size, Worksharing& ws) : x_size(size), y_size(size), ws(ws) { createTable(); }
			PinnedTableCost(int x_size, int y_size, TC* tab, Worksharing& ws) : x_size(x_size), y_size(y_size), ws(ws) { referenceTable(tab); }
			PinnedTableCost(int size, TC* tab, Worksharing& ws) : x_size(size), y_size(size), ws(ws) { referenceTable(tab); }
			~PinnedTableCost()
			{
				if (free_in_destructor)
				{
#pragma omp parallel
					lapFreePinned(cc[omp_get_thread_num()]);
				}
				lapFree(cc);
				lapFree(stride);
			}
		public:
			__forceinline const TC* getRow(int t, int x) const { return cc[t] + (long long)x * (long long)stride[t]; }
			__forceinline const TC getCost(int x, int y) const
			{
				int t = 0;
				while (y >= ws.part[t].second) t++;
				long long off_y = y - ws.part[t].first;
				long long off_x = x;
				off_x *= stride[t];
				return cc[t][off_x + off_y];
			}
			__forceinline void setRow(int x, TC* v)
			{
				for (int t = 0; t < omp_get_max_threads(); t++)
				{
					long long off_x = x;
					off_x *= stride[t];
					memcpy(&(cc[t][off_x]), &(v[ws.part[t].first]), (ws.part[t].second - ws.part[t].first) * sizeof(TC));
				}
			}
		};
	}
}
