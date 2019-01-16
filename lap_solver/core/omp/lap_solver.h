#pragma once

#include "../lap_solver.h"

namespace lap
{
	namespace omp
	{
		template <class TC, class I>
		TC guessEpsilon(int x_size, int y_size, I& iterator, int step)
		{
			TC epsilon(0);
			int x_count = x_size / step;
			TC *min_cost;
			TC *max_cost;
			lapAlloc(min_cost, omp_get_max_threads() * x_count, __FILE__, __LINE__);
			lapAlloc(max_cost, omp_get_max_threads() * x_count, __FILE__, __LINE__);
#pragma omp parallel
			{
				int t = omp_get_thread_num();
				for (int x = 0; x < x_size; x += step)
				{
					int xx = x / step;
					const TC *tt = iterator.getRow(t, x);
					TC min_cost_l, max_cost_l;
					min_cost_l = max_cost_l = tt[0];
					for (int y = 1; y < iterator.ws.part[t].second - iterator.ws.part[t].first; y++)
					{
						TC cost_l = tt[y];
						min_cost_l = std::min(min_cost_l, cost_l);
						max_cost_l = std::max(max_cost_l, cost_l);
					}
					min_cost[xx + x_count * t] = min_cost_l;
					max_cost[xx + x_count * t] = max_cost_l;
				}
#pragma omp barrier
#pragma omp for
				for (int x = 0; x < x_count; x++)
				{
					TC max_c = max_cost[x];
					TC min_c = min_cost[x];
					for (int xx = 0; xx < omp_get_max_threads(); xx++)
					{
						max_c = std::max(max_c, max_cost[x + x_count * xx]);
						min_c = std::min(min_c, min_cost[x + x_count * xx]);
					}
					max_cost[x] = max_c;
					min_cost[x] = min_c;
				}
			}
			for (int x = 0; x < x_count; x++)
			{
				epsilon += max_cost[x] - min_cost[x];
			}
			lapFree(min_cost);
			lapFree(max_cost);
			return (epsilon / TC(10 * (x_size + step - 1) / step));
		}

		template <class SC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol)

			// input:
			// dim        - problem size
			// costfunc - cost matrix
			// findcost   - searching cost matrix

			// output:
			// rowsol     - column assigned to row in solution
			// colsol     - row assigned to column in solution
			// u          - dual variables, row reduction numbers
			// v          - dual variables, column reduction numbers

		{
#ifndef LAP_QUIET
			auto start_time = std::chrono::high_resolution_clock::now();

			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;

			int elapsed = -1;
#else
#ifdef LAP_DISPLAY_EVALUATED
			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;
#endif
#endif

			int  *pred;
			int  endofpath;
			char *colactive;
			int *colcomplete;
			int completecount;
			SC *d;
			int *colsol;
			SC *v;

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

			lapAlloc(colactive, dim2, __FILE__, __LINE__);
			lapAlloc(colcomplete, dim2, __FILE__, __LINE__);
			lapAlloc(d, dim2, __FILE__, __LINE__);
			lapAlloc(pred, dim2, __FILE__, __LINE__);
			lapAlloc(v, dim2, __FILE__, __LINE__);
			lapAlloc(colsol, dim2, __FILE__, __LINE__);

			SC *min_private;
			int *jmin_private;
			lapAlloc(min_private, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(jmin_private, omp_get_max_threads(), __FILE__, __LINE__);

#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			// this is the upper bound
			SC epsilon = costfunc.getInitialEpsilon();
			SC epsilon_lower = epsilon / SC(dim2);

			SC last_avg = SC(0);
			bool first = true;
			bool allow_reset = true;

			memset(v, 0, dim2 * sizeof(SC));

			while (epsilon >= SC(0))
			{
				SC total = SC(0);
				unsigned long long count = 0ULL;
				if (epsilon > SC(0))
				{
					if (epsilon < SC(2) * epsilon_lower) epsilon = SC(0);
					else
					{
						if (!first)
						{
#ifdef LAP_DEBUG
							lapDebug << "  v_d = " << -last_avg << " v_eps = " << epsilon << std::endl;
#endif
							if ((allow_reset) && (-last_avg <= SC(0.1) * epsilon))
							{
#ifdef LAP_DEBUG
								lapDebug << "modification mostly based on epsilon -> reverting v." << std::endl;
#endif
								memset(v, 0, dim2 * sizeof(SC));
								if (last_avg == SC(0.0)) epsilon *= SC(0.1);
								else epsilon = -last_avg;
							}
							else
							{
								epsilon = std::max(SC(0.1) * epsilon, SC(0.5) * (epsilon + last_avg));
								allow_reset = false;
							}
							if ((epsilon > SC(0)) && (epsilon < epsilon_lower)) epsilon = epsilon_lower;
						}
					}
				}
				SC eps = epsilon;
#ifndef LAP_QUIET
				{
					std::stringstream ss;
					ss << "eps = " << eps;
					const std::string tmp = ss.str();
					displayTime(start_time, tmp.c_str(), lapInfo);
				}
#endif
				// this is to ensure termination of the while statement
				if (epsilon == SC(0)) epsilon = SC(-1.0);
				memset(rowsol, -1, dim2 * sizeof(int));
				memset(colsol, -1, dim2 * sizeof(int));

				int jmin;
				SC min, min_n;
				bool unassignedfound;

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif
				jmin = dim2;
				min = std::numeric_limits<SC>::max();
				SC h2_global;
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;

					for (int f = 0; f < dim2; f++)
					{
						int jmin_local = dim2;
						SC min_local = std::numeric_limits<SC>::max();
						if (f < dim)
						{
							auto tt = iterator.getRow(t, f);
							for (int j = start; j < end; j++)
							{
								int j_local = j - start;
								colactive[j] = 1;
								pred[j] = f;
								SC h = d[j] = tt[j_local] - v[j];
								if (h < min_local)
								{
									// better
									jmin_local = j;
									min_local = h;
								}
								else if (h == min_local)
								{
									// same, do only update if old was used and new is free
									if ((colsol[jmin_local] >= 0) && (colsol[j] < 0)) jmin_local = j;
								}
							}
						}
						else
						{
							for (int j = start; j < end; j++)
							{
								colactive[j] = 1;
								pred[j] = f;
								SC h = d[j] = -v[j];
								if (h < min_local)
								{
									// better
									jmin_local = j;
									min_local = h;
								}
								else if (h == min_local)
								{
									// same, do only update if old was used and new is free
									if ((colsol[jmin_local] >= 0) && (colsol[j] < 0)) jmin_local = j;
								}
							}
						}
						min_private[t] = min_local;
						jmin_private[t] = jmin_local;
#pragma omp barrier
#pragma omp master
						{
#ifndef LAP_QUIET
							if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
							if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
							scancount[f]++;
#endif
							min = min_private[0];
							jmin = jmin_private[0];
							for (int tt = 1; tt < omp_get_num_threads(); tt++)
							{
								if (min_private[tt] < min)
								{
									// better than previous
									min = min_private[tt];
									jmin = jmin_private[tt];
								}
								else if (min_private[tt] == min)
								{
									if ((colsol[jmin] >= 0) && (colsol[jmin_private[tt]] < 0)) jmin = jmin_private[tt];
								}
							}
							unassignedfound = false;
							completecount = 0;
							dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
						}
#pragma omp barrier
						while (!unassignedfound)
						{
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
							int i = colsol[jmin];
							//updateDistance(i, dim, dim2, iterator, min, jmin, min_n, jmin_n, colactive, pred, colsol, d, v);
							jmin_local = dim2;
							min_local = std::numeric_limits<SC>::max();
							if (i < dim)
							{
								auto tt = iterator.getRow(t, i);
								SC h2;
								if ((jmin >= start) && (jmin < end)) h2_global = h2 = tt[jmin - start] - v[jmin] - min;
#pragma omp barrier
								if ((jmin < start) || (jmin >= end)) h2 = h2_global;
								for (int j = start; j < end; j++)
								{
									int j_local = j - start;
									if (colactive[j] != 0)
									{
										SC v2 = tt[j_local] - v[j] - h2;
										SC h = d[j];
										if (v2 < h)
										{
											pred[j] = i;
											d[j] = v2;
											h = v2;
										}
										if (h < min_local)
										{
											// better
											jmin_local = j;
											min_local = h;
										}
										else if (h == min_local)
										{
											// same, do only update if old was used and new is free
											if ((colsol[jmin_local] >= 0) && (colsol[j] < 0)) jmin_local = j;
										}
									}
								}
							}
							else
							{
								SC h2 = -v[jmin] - min;
								for (int j = start; j < end; j++)
								{
									if (colactive[j] != 0)
									{
										SC v2 = -v[j] - h2;
										SC h = d[j];
										if (v2 < h)
										{
											pred[j] = i;
											d[j] = v2;
											h = v2;
										}
										if (h < min_local)
										{
											// better
											jmin_local = j;
											min_local = h;
										}
										else if (h == min_local)
										{
											// same, do only update if old was used and new is free
											if ((colsol[jmin_local] >= 0) && (colsol[j] < 0)) jmin_local = j;
										}
									}
								}
							}
							min_private[t] = min_local;
							jmin_private[t] = jmin_local;
#pragma omp barrier
#pragma omp master
							{
#ifndef LAP_QUIET
								if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
								if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
								scancount[i]++;
#endif
								min_n = min_private[0];
								jmin = jmin_private[0];
								for (int tt = 1; tt < omp_get_num_threads(); tt++)
								{
									if (min_private[tt] < min_n)
									{
										// better than previous
										min_n = min_private[tt];
										jmin = jmin_private[tt];
									}
									else if (min_private[tt] == min_n)
									{
										if ((colsol[jmin] >= 0) && (colsol[jmin_private[tt]] < 0)) jmin = jmin_private[tt];
									}
								}
								min = std::max(min, min_n);
								dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
								h2_global = std::numeric_limits<SC>::infinity();
							}
#pragma omp barrier
						}
#pragma omp master
						{
							// update column prices. can increase or decrease
							if (eps > SC(0))
							{
								updateColumnPrices(colcomplete, completecount, min, v, d, eps, total, count);
							}
							else
							{
								updateColumnPrices(colcomplete, completecount, min, v, d);
							}
#ifdef LAP_ROWS_SCANNED
//							scancount[f] += completecount;
							{
								int i;
								int eop = endofpath;
								do
								{
									i = pred[eop];
									eop = rowsol[i];
									pathlength[f]++;
								} while (i != f);
							}
#endif

							// reset row and column assignments along the alternating path.
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
							// for next iteration
							jmin = dim2;
							min = std::numeric_limits<SC>::max();
#ifndef LAP_QUIET
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim2, " rows")) != 0)
							{
								long long hit, miss;
								iterator.getHitMiss(hit, miss);
								total_hit += hit;
								total_miss += miss;
								if ((hit != 0) || (miss != 0))
								{
									{
										if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
										else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
									}
								}
								old_complete = f + 1;
							}
#endif
						}
#pragma omp barrier
					}
				}

				if (count > 0) last_avg = total / SC(count);
				else last_avg = SC(0);

#ifdef LAP_DEBUG
				if (eps > SC(0))
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(eps);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
				else
				{
					int count = (int)v_list.size();
					if (count > 0)
					{
						for (int l = 0; l < count; l++)
						{
							SC dlt(0), dlt2(0);
							for (int i = 0; i < dim2; i++)
							{
								SC diff = v_list[l][i] - v[i];
								dlt += diff;
								dlt2 += diff * diff;
							}
							dlt /= SC(dim2);
							dlt2 /= SC(dim2);
							lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
							lapFree(v_list[l]);
						}
					}
				}
#endif
				first = false;

#ifndef LAP_QUIET
				lapInfo << "  rows evaluated: " << total_rows;
				if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
				lapInfo << std::endl;
				if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
			}

#ifdef LAP_QUIET
#ifdef LAP_DISPLAY_EVALUATED
			iterator.getHitMiss(total_hit, total_miss);
			lapInfo << "  rows evaluated: " << total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
#endif

#ifdef LAP_ROWS_SCANNED
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << "row: " << f << " scanned: " << scancount[f] << " length: " << pathlength[f] << std::endl;
			}

			lapFree(scancount);
			lapFree(pathlength);
#endif

			// free reserved memory.
			lapFree(pred);
			lapFree(colactive);
			lapFree(colcomplete);
			lapFree(d);
			lapFree(v);
			lapFree(colsol);
			lapFree(min_private);
			lapFree(jmin_private);
		}

		// shortcut for square problems
		template <class SC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol)
		{
			lap::omp::solve<SC>(dim, dim, costfunc, iterator, rowsol);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
		{
			SC total = SC(0);
			if (costfunc.allEnabled())
			{
#pragma omp parallel
				{
					SC total_local = SC(0);
#pragma omp for nowait schedule(static)
					for (int i = 0; i < dim; i++) total_local += costfunc.getCost(i, rowsol[i]);
#pragma omp critical
					total += total_local;
				}
			}
			else
			{
				int i = 0;
#pragma omp parallel
				{
					SC total_local = SC(0);
					if (costfunc.enabled(omp_get_thread_num()))
					{
						int i_local;
						do
						{
#pragma omp critical
							i_local = i++;
							if (i_local < dim)
							{
								total_local += costfunc.getCost(i_local, rowsol[i_local]);
							}
						} while (i_local < dim);
					}
#pragma omp critical
					total += total_local;
				}
			}
			return total;
		}

		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol)
		{
			return lap::omp::cost<SC, CF>(dim, dim, costfunc, rowsol);
		}
	}
}
