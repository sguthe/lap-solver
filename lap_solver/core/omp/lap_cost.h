#pragma once

#include "../omp/lap_worksharing.h"
#include "../lap_cost.h"

namespace lap
{
	namespace omp
	{
		// Wrapper around simple cost function, scheduling granularity is assumed to be 1 for load balancing
		template <class TC, typename GETCOST>
		class SimpleCostFunction : public lap::SimpleCostFunction<TC, GETCOST>
		{
		public:
			SimpleCostFunction(GETCOST &getcost) : lap::SimpleCostFunction<TC, GETCOST>(getcost) {}
			~SimpleCostFunction() {}
		public:
			__forceinline bool allEnabled() const { return true; }
			__forceinline bool enabled(int t) const { return true; }
			__forceinline int getMultiple() const { return 8; }
		};

		// Wrapper around enabled cost funtion, e.g. CUDA, OpenCL or OpenMPI where only a subset of threads takes part in calculating the cost function
		// getCost is not supported here
		// Scheduling granularity can be set for for load balancing but cost function code has to handle arbitray subsets
		template <class TC, typename GETENABLED, typename GETCOSTROW>
		class RowCostFunction : public lap::RowCostFunction<TC, GETCOSTROW>
		{
		protected:
			GETENABLED getenabled;
			// scheduling granularity
			int multiple;
		public:
			RowCostFunction(GETENABLED &getenabled, GETCOSTROW &getcostrow, int multiple = 1) : lap::RowCostFunction<TC, GETCOSTROW>(getcostrow), getenabled(getenabled), multiple(multiple) {}
			~RowCostFunction() {}
		public:
			__forceinline bool enabled(int t) const { return getenabled(t); }
			__forceinline bool allEnabled() const { for (int i = 0; i < omp_get_max_threads(); i++) if (!enabled(i)) return false; return true; }
			__forceinline int getMultiple() const { return multiple; }
		};

		// Costs stored in a table. Used for conveniency only
		// This can be constructed using a CostFunction from above or by specifying an array that holds the data (does not copy the data in this case).
		template <class TC>
		class TableCost
		{
		protected:
			int x_size;
			int y_size;
			TC **cc;
			int *stride;
			TC initialEpsilon;
			bool free_in_destructor;
			Worksharing &ws;
		protected:
			void referenceTable(TC *tab)
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
			void initTable(DirectCost &cost)
			{
				free_in_destructor = true;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
				// used in case not all threads take part in the calculation
				bool *tc;
				lapAlloc(tc, omp_get_max_threads(), __FILE__, __LINE__);
				memset(tc, 0, omp_get_max_threads() * sizeof(bool));
#pragma omp parallel
				{
					const int t = omp_get_thread_num();
					stride[t] = ws.part[t].second - ws.part[t].first;
					lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
					// first touch
					memset(cc[t], 0, (long long)(stride[t]) * (long long)x_size * sizeof(TC));
					cc[t][0] = TC(0);
					if (cost.allEnabled())
					{
						for (int x = 0; x < x_size; x++)
						{
							cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
						}
					}
					else
					{
						int t_local = t;
						tc[t] = cost.enabled(t);
						if (cost.enabled(t))
						{
							while (t_local < omp_get_max_threads())
							{
								for (int x = 0; x < x_size; x++)
								{
									cost.getCostRow(cc[t_local] + (long long)x * (long long)stride[t_local], x, ws.part[t_local].first, ws.part[t_local].second);
								}
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
					}
				}
				lapFree(tc);
			}
		public:
			template <class DirectCost> TableCost(int x_size, int y_size, DirectCost &cost, Worksharing &ws) : 
				x_size(x_size), y_size(y_size), initialEpsilon(0), ws(ws) { initTable(cost); }
			template <class DirectCost> TableCost(int size, DirectCost &cost, Worksharing &ws) : 
				x_size(size), y_size(size), initialEpsilon(0), ws(ws) { initTable(cost); }
			TableCost(int x_size, int y_size, TC* tab, Worksharing &ws) : x_size(x_size), y_size(y_size), initialEpsilon(0), ws(ws) { referenceTable(tab); }
			TableCost(int size, TC* tab, Worksharing &ws) : x_size(size), y_size(size), initialEpsilon(0), ws(ws) { referenceTable(tab); }
			~TableCost()
			{
				if (free_in_destructor)
				{
#pragma omp parallel
					lapFree(cc[omp_get_thread_num()]);
				}
				lapFree(cc);
				lapFree(stride);
			}
			public:
			__forceinline bool allEnabled() const { return true; }
			__forceinline bool enabled(int t) const { return true; }
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			// These should never be used so it's commented out
			//__forceinline void getCostRow(TC *row, int x, int start, int end) const { memcpy(row, &(getRow(x)[start]), (end - start) * sizeof(TC)); }
			__forceinline const TC *getRow(int t, int x) const { return cc[t] + (long long)x * (long long)stride[t]; }
			__forceinline const TC getCost(int x, int y) const
			{
				int t = 0;
				while (y >= ws.part[t].second) t++;
				long long off_y = y - ws.part[t].first;
				long long off_x = x;
				off_x *= stride[t];
				return cc[t][off_x + off_y];
			}
		};
	}
}
