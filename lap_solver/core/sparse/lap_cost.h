#pragma once

#include <string.h>
#include <limits>
#include <tuple>

namespace lap
{
	namespace sparse
	{
		// Wrapper around simple cost function (return infinite for forbidden connection)
		template <class TC, typename GETROWLENGTH, typename GETCOST>
		class SimpleCostFunction
		{
		protected:
			GETCOST getcost;
			GETROWLENGTH getrowlength;
			TC initialEpsilon;
			TC lowerEpsilon;
		public:
			SimpleCostFunction(GETROWLENGTH &getrowlength, GETCOST &getcost) : getrowlength(getrowlength), getcost(getcost), initialEpsilon(0), lowerEpsilon(0) {}
			~SimpleCostFunction() {}
		public:
			__forceinline const TC getCost(int x, int y) const { return getcost(x, y); }
			__forceinline int getCostRow(int *col_idx, TC *row, int x, int start, int end) const {
				int idx = 0;
				for (int y = start; y < end; y++)
				{
					TC c = getCost(x, y);
					if (!std::isinf(c))
					{
						col_idx[idx] = y;
						row[idx++] = c;
					}
				}
				return idx;
			}
			__forceinline int getRowLength(int x, int start, int end) const { return getrowlength(x, start, end); }
		};

		// Wrapper around per-row cost funtion, e.g. CUDA, OpenCL or OpenMPI
		template <class TC, typename GETROWLENGTH, typename GETCOSTROW>
		class RowCostFunction
		{
		protected:
			GETROWLENGTH getrowlength;
			GETCOSTROW getcostrow;
			TC initialEpsilon;
		public:
			RowCostFunction(GETROWLENGTH &getrowlength, GETCOSTROW &getcostrow) : getrowlength(getrowlength), getcostrow(getcostrow), initialEpsilon(0) {}
			~RowCostFunction() {}
		public:
			__forceinline const TC getCost(int x, int y) const {
				TC r;
				getcostrow(&r, x, y, y + 1);
				return r;
			}
			__forceinline int getCostRow(int *col_idx, TC *row, int x, int start, int end) const { return getcostrow(col_idx, row, x, start, end); }
			__forceinline int getRowLength(int x, int start, int end) const { return getrowlength(x, start, end); }
		};

		// Costs stored in a table. Used for conveniency only
		// This can be constructed using a CostFunction from above or by specifying an array that holds the data (does not copy the data in this case).
		template <class TC>
		class TableCost
		{
		protected:
			int x_size;
			int y_size;
			size_t *row_start;
			int* row_length;
			int* col_idx;
			TC* rows;
			TC initialEpsilon;
			TC lowerEpsilon;
			bool free_in_destructor;
		protected:
			template <class DirectCost>
			void initTable(DirectCost &cost)
			{
				// needs to get the size of the table first
				size_t table_size(0);
				for (int x = 0; x < x_size; x++) table_size += (size_t)cost.getRowLength(x, 0, y_size);
				lapAlloc(rows, table_size, __FILE__, __LINE__);
				lapAlloc(col_idx, table_size, __FILE__, __LINE__);
				lapAlloc(row_length, x_size, __FILE__, __LINE__);
				lapAlloc(row_start, x_size, __FILE__, __LINE__);
				free_in_destructor = true;
				size_t idx(0);
				for (int x = 0; x < x_size; x++)
				{
					row_start[x] = idx;
					idx += row_length[x] = cost.getCostRow(&(col_idx[idx]), &(rows[idx]), x, 0, y_size);
				}
			}
		public:
			template <class DirectCost> TableCost(int x_size, int y_size, DirectCost &cost) : x_size(x_size), y_size(y_size), initialEpsilon(0), lowerEpsilon(0) { initTable(cost); }
			template <class DirectCost> TableCost(int size, DirectCost &cost) : x_size(size), y_size(size), initialEpsilon(0), lowerEpsilon(0) { initTable(cost); }
			TableCost(int x_size, int y_size, size_t *row_start, int *row_length, int *col_idx, TC* rows) : x_size(x_size), y_size(y_size), row_start(row_start), row_length(row_length), col_idx(col_idx), rows(rows), initialEpsilon(0), lowerEpsilon(0) { free_in_destructor = false; }
			TableCost(int size, size_t *row_start, int *row_length, int *col_idx, TC* rows) : x_size(size), y_size(size), row_start(row_start), row_length(row_length), col_idx(col_idx), rows(rows), initialEpsilon(0), lowerEpsilon(0) { free_in_destructor = false; }
			~TableCost() {
				if (free_in_destructor) {
					lapFree(rows);
					lapFree(col_idx);
					lapFree(row_length);
					lapFree(row_start);
				}
			}
		public:
			__forceinline const std::tuple<int, int *, TC *> getRow(int x) const { return std::tuple<int, int *, TC *>(row_length[x], &(col_idx[row_start[x]]), &(rows[row_start[x]])); }
			__forceinline const TC getCost(int x, int y) const {
				auto row = getRow(x);
				for (int yy = 0; yy < std::get<0>(row); yy++)
				{
					if (std::get<1>(row)[yy] == y) return std::get<2>(row)[yy];
				}
				return std::numeric_limits<TC>::infinity();
			}
		};
	}
}
