#pragma once

#include "../lap_solver.h"

namespace lap
{
	namespace omp
	{
		// dim2 is not actually used in this function
		template <class SC, class I>
		std::pair<SC, SC> guessEpsilon(int dim, int dim2, I& iterator)
		{
			SC epsilon(0);
			SC *min_cost;
			SC *max_cost;
			unsigned long long *min_count;
			lapAlloc(min_cost, omp_get_max_threads() * dim, __FILE__, __LINE__);
			lapAlloc(max_cost, omp_get_max_threads() * dim, __FILE__, __LINE__);
			lapAlloc(min_count, dim2, __FILE__, __LINE__);
			memset(min_count, 0, dim2 * sizeof(unsigned long long));
#pragma omp parallel
			{
				int t = omp_get_thread_num();
				// reverse order to avoid cachethrashing
				for (int i = dim - 1; i >= 0; --i)
				{
					auto *tt = iterator.getRow(t, i);
					SC min_cost_l, max_cost_l;
					min_cost_l = max_cost_l = (SC)tt[0];
					for (int j = 1; j < iterator.ws.part[t].second - iterator.ws.part[t].first; j++)
					{
						SC cost_l = (SC)tt[j];
						min_cost_l = std::min(min_cost_l, cost_l);
						max_cost_l = std::max(max_cost_l, cost_l);
					}
					min_cost[i + dim * t] = min_cost_l;
					max_cost[i + dim * t] = max_cost_l;
#pragma omp barrier
#pragma omp master
					{
						SC max_c = max_cost[i];
						SC min_c = min_cost[i];
						for (int xx = 0; xx < omp_get_max_threads(); xx++)
						{
							max_c = std::max(max_c, max_cost[i + dim * xx]);
							min_c = std::min(min_c, min_cost[i + dim * xx]);
						}
						max_cost[i] = max_c;
						min_cost[i] = min_c;
					}
#pragma omp barrier
					min_cost_l = min_cost[i];
					for (int j = 0; j < iterator.ws.part[t].second - iterator.ws.part[t].first; j++)
					{
						SC cost_l = (SC)tt[j];
						if (cost_l == min_cost_l) min_count[j]++;
					}

				}
			}
			for (int i = 0; i < dim; i++)
			{
				epsilon += max_cost[i] - min_cost[i];
			}
			unsigned long long coll = min_count[0];
			unsigned long long total = min_count[0];
			unsigned long long zero = (min_count[0] == 0) ? 1 : 0;
			for (int j = 1; j < dim2; j++)
			{
				coll = std::max(coll, min_count[j]);
				total += min_count[j];
				if (min_count[j] == 0) zero++;
			}
			lapFree(min_cost);
			lapFree(max_cost);
			lapFree(min_count);
			long double r_col = ((long double)coll - (long double)total / (long double)dim2) / (long double)total;
			long double r_zero = 0.5l + 0.5l * (long double)(zero + dim - dim2) / (long double)dim2;
			long double r_eps = (long double)epsilon / (long double)(4 * dim2);
			return std::pair<SC, SC>((SC)std::max(0.0l, r_col * r_zero * r_eps), (SC)std::max(0.0l, r_eps / SC(32 * dim2)));
		}

		template <class SC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, SC *initial_v)

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
			// use << 3 to avoid false sharing
			lapAlloc(min_private, omp_get_max_threads() << 3, __FILE__, __LINE__);
			lapAlloc(jmin_private, omp_get_max_threads() << 3, __FILE__, __LINE__);

#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			// this is the upper bound
			SC epsilon = (SC)costfunc.getInitialEpsilon();
			SC epsilon_lower = (SC)costfunc.getLowerEpsilon();

			bool first = true;
			bool allow_continue = true;
			bool clamp = true;

			if (initial_v == 0) memset(v, 0, dim2 * sizeof(SC));
			else memcpy(v, initial_v, dim2 * sizeof(SC));

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= SC(0))
			{
				lap::getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, allow_continue, v, dim2, initial_v);
				//if ((!first) && (allow_continue)) clamp = false;
				total_d = SC(0);
				total_eps = SC(0);
#ifndef LAP_QUIET
				{
					std::stringstream ss;
					ss << "eps = " << epsilon;
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
				//SC h2_global;
				SC tt_jmin_global;
				int dim_limit = ((epsilon > SC(0)) && (first)) ? dim : dim2;

#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;

					for (int f = 0; f < dim_limit; f++)
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
							min_local = std::numeric_limits<SC>::max();
							for (int j = start; j < end; j++)
							{
								colactive[j] = 1;
								pred[j] = f;
								SC h = d[j] = -v[j];
								// ignore any columns assigned to virtual rows
								if (colsol[j] < dim)
								{
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
						min_private[t << 3] = min_local;
						jmin_private[t << 3] = jmin_local;
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
								if (min_private[tt << 3] < min)
								{
									// better than previous
									min = min_private[tt  << 3];
									jmin = jmin_private[tt << 3];
								}
								else if (min_private[tt << 3] == min)
								{
									if ((colsol[jmin] >= 0) && (colsol[jmin_private[tt << 3]] < 0)) jmin = jmin_private[tt << 3];
								}
							}
							unassignedfound = false;
							completecount = 0;
							dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
							// marked skipped columns that were cheaper
							if (f >= dim)
							{
								for (int j = 0; j < dim2; j++)
								{
									// ignore any columns assigned to virtual rows
									if ((colsol[j] >= dim) && (d[j] <= min))
									{
										colcomplete[completecount++] = j;
										colactive[j] = 0;
									}
								}
							}
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
								//SC h2;
								SC tt_jmin;
								SC v_jmin = v[jmin];
								auto tt = iterator.getRow(t, i);
								if ((jmin >= start) && (jmin < end))
								{
									//h2_global = h2 = tt[jmin - start] - v[jmin] - min;
									tt_jmin_global = tt_jmin = (SC)tt[jmin - start];
								}
#pragma omp barrier
								if ((jmin < start) || (jmin >= end))
								{
									//h2 = h2_global;
									tt_jmin = tt_jmin_global;
								}
								for (int j = start; j < end; j++)
								{
									int j_local = j - start;
									if (colactive[j] != 0)
									{
										//SC v2 = tt[j_local] - v[j] - h2;
										SC v2 = (tt[j_local] - tt_jmin) - (v[j] - v_jmin) + min;
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
								SC v_jmin = v[jmin];
								//SC h2 = -v[jmin] - min;
								for (int j = start; j < end; j++)
								{
									if (colactive[j] != 0)
									{
										SC v2 = -(v[j] - v_jmin) + min;
										SC h = d[j];
										if (v2 < h)
										{
											pred[j] = i;
											d[j] = v2;
											h = v2;
										}
										// ignore any columns assigned to virtual rows
										if (colsol[j] < dim)
										{
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
							}
							min_private[t << 3] = min_local;
							jmin_private[t << 3] = jmin_local;
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
									if (min_private[tt << 3] < min_n)
									{
										// better than previous
										min_n = min_private[tt << 3];
										jmin = jmin_private[tt << 3];
									}
									else if (min_private[tt << 3] == min_n)
									{
										if ((colsol[jmin] >= 0) && (colsol[jmin_private[tt << 3]] < 0)) jmin = jmin_private[tt << 3];
									}
								}
								min = std::max(min, min_n);
								dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
								// marked skipped columns that were cheaper
								if (i >= dim)
								{
									for (int j = 0; j < dim2; j++)
									{
										// ignore any columns assigned to virtual rows
										if ((colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min_n))
										{
											colcomplete[completecount++] = j;
											colactive[j] = 0;
										}
									}
								}
							}
#pragma omp barrier
						}
#pragma omp master
						{
							// update column prices. can increase or decrease
							if (epsilon > SC(0))
							{
								if (clamp) updateColumnPricesClamp(colcomplete, completecount, min, v, d, epsilon, total_d, total_eps);
								else updateColumnPrices(colcomplete, completecount, min, v, d, epsilon, total_d, total_eps);
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
									if (i != f) pathlength[f]++;
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
							if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
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

				if (dim_limit < dim2)
				{
					total_eps -= SC(dim2 - dim_limit) * epsilon;
					// fix v in unassigned columns
					for (int j = 0; j < dim2; j++)
					{
						if (colsol[j] < 0) v[j] -= epsilon;
					}
				}

#ifdef LAP_DEBUG
				if (epsilon > SC(0))
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
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
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, SC *initial_v)
		{
			lap::omp::solve<SC>(dim, dim, costfunc, iterator, rowsol, initial_v);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
		{
			SC total = SC(0);
#pragma omp parallel for reduction(+:total)
			for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
			return total;
		}

		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol)
		{
			return lap::omp::cost<SC, CF>(dim, dim, costfunc, rowsol);
		}
	}
}
