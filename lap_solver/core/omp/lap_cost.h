#pragma once

#include "lap_worksharing.h"
#include "../lap_cost.h"

namespace lap
{
	namespace omp
	{
		// Wrapper around simple cost function, scheduling granularity is assumed to be 8 for load balancing
		template <class TC, typename GETCOST>
		class SimpleCostFunction : public lap::SimpleCostFunction<TC, GETCOST>
		{
		protected:
			bool sequential;
		public:
			SimpleCostFunction(GETCOST &getcost, bool sequential = false) : lap::SimpleCostFunction<TC, GETCOST>(getcost), sequential(sequential) {}
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
			struct table_t
			{
				TC* cc;
				int stride;
				char dummy[256 - sizeof (TC *) - sizeof(int)];
			};
			table_t* table;
			bool free_in_destructor;
			Worksharing &ws;
		protected:
			void referenceTable(TC *tab)
			{
				free_in_destructor = false;
				lapAlloc(table, omp_get_max_threads(), __FILE__, __LINE__);
				for (int t = 0; t < omp_get_max_threads(); t++)
				{
					table[t].stride = y_size;
					table[t].cc = &(tab[ws.part[t].first]);
				}
			}
			
			template <class DirectCost>
			void initTable(DirectCost &cost)
			{
				free_in_destructor = true;
				lapAlloc(table, omp_get_max_threads(), __FILE__, __LINE__);
				if (cost.isSequential())
				{
					// cost table needs to be initialized sequentially
#pragma omp parallel
					{
						const int t = omp_get_thread_num();
						table[t].stride = ws.part[t].second - ws.part[t].first;
						lapAlloc(table[t].cc, (long long)(table[t].stride) * (long long)x_size, __FILE__, __LINE__);
						for (int x = 0; x < x_size; x++)
						{
							for (int tt = 0; tt < omp_get_max_threads(); tt++)
							{
#pragma omp barrier
								if (tt == t) cost.getCostRow(table[t].cc + (long long)x * (long long)table[t].stride, x, ws.part[t].first, ws.part[t].second);
							}
						}
					}
				}
				else
				{
					// create and initialize in parallel
#pragma omp parallel
					{
						const int t = omp_get_thread_num();
						table[t].stride = ws.part[t].second - ws.part[t].first;
						lapAlloc(table[t].cc, (long long)(table[t].stride) * (long long)x_size, __FILE__, __LINE__);
						// first touch
						table[t].cc[0] = TC(0);
						for (int x = 0; x < x_size; x++)
						{
							cost.getCostRow(table[t].cc + (long long)x * (long long)table[t].stride, x, ws.part[t].first, ws.part[t].second);
						}
					}
				}
			}

			void createTable()
			{
				free_in_destructor = true;
				lapAlloc(table, omp_get_max_threads(), __FILE__, __LINE__);
#pragma omp parallel
				{
					const int t = omp_get_thread_num();
					table[t].stride = ws.part[t].second - ws.part[t].first;
					lapAlloc(table[t].cc, (long long)(table[t].stride) * (long long)x_size, __FILE__, __LINE__);
					// first touch
					table[t].cc[0] = TC(0);
				}
			}
		public:
			template <class DirectCost> TableCost(int x_size, int y_size, DirectCost &cost, Worksharing &ws) :
				x_size(x_size), y_size(y_size), ws(ws) { initTable(cost); }
			template <class DirectCost> TableCost(int size, DirectCost &cost, Worksharing &ws) :
				x_size(size), y_size(size), ws(ws) { initTable(cost); }
			TableCost(int x_size, int y_size, Worksharing &ws) : x_size(x_size), y_size(y_size), ws(ws) { createTable(); }
			TableCost(int size, Worksharing &ws) : x_size(size), y_size(size), ws(ws) { createTable(); }
			TableCost(int x_size, int y_size, TC* tab, Worksharing &ws) : x_size(x_size), y_size(y_size), ws(ws) { referenceTable(tab); }
			TableCost(int size, TC* tab, Worksharing &ws) : x_size(size), y_size(size), ws(ws) { referenceTable(tab); }
			~TableCost()
			{
				if (free_in_destructor)
				{
#pragma omp parallel
					lapFree(table[omp_get_thread_num()].cc);
				}
				lapFree(table);
			}
			public:
			__forceinline const TC *getRow(int t, int x) const { return table[t].cc + (long long)x * (long long)table[t].stride; }
			__forceinline const TC getCost(int x, int y) const
			{
				int t = 0;
				while (y >= ws.part[t].second) t++;
				long long off_y = y - (long long)ws.part[t].first;
				long long off_x = x;
				off_x *= table[t].stride;
				return table[t].cc[off_x + off_y];
			}
			__forceinline void setRow(int x, TC *v)
			{
				for (int t = 0; t < omp_get_max_threads(); t++)
				{
					long long off_x = x;
					off_x *= table[t].stride;
					memcpy(&(table[t].cc[off_x]), &(v[ws.part[t].first]), (ws.part[t].second - ws.part[t].first) * sizeof(TC));
				}
			}
		};
	}
}
