#pragma once

#include "../lap_solver.h"

namespace lap
{
	namespace omp
	{
		template <class SC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC *v, bool estimate_v)
		{
			SC *merge_cost;
			lapAlloc(merge_cost, omp_get_max_threads() << 3, __FILE__, __LINE__);
			int *merge_idx;
			lapAlloc(merge_idx, omp_get_max_threads() << 3, __FILE__, __LINE__);
			double *merge_moments;
			lapAlloc(merge_moments, omp_get_max_threads() << 3, __FILE__, __LINE__);
			if (estimate_v)
			{
				SC *min_v;
				lapAlloc(min_v, dim2, __FILE__, __LINE__);
				// initialize using mean
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					for (int i = 0; i < dim; i++)
					{
#pragma omp barrier
						auto *tt = iterator.getRow(t, i);
						SC min_cost_l;
						min_cost_l = (SC)tt[0];
						for (int j = 1; j < iterator.ws.part[t].second - iterator.ws.part[t].first; j++)
						{
							SC cost_l = (SC)tt[j];
							min_cost_l = std::min(min_cost_l, cost_l);
						}
						merge_cost[t << 3] = min_cost_l;
#pragma omp barrier
						min_cost_l = merge_cost[0];
						for (int xx = 1; xx < omp_get_max_threads(); xx++) min_cost_l = std::min(min_cost_l, merge_cost[xx << 3]);

						if (i == 0)
						{
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								min_v[j] = (SC)tt[j - iterator.ws.part[t].first] - min_cost_l;
							}
						}
						else if (i == 1)
						{
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								SC tmp = (SC)tt[j - iterator.ws.part[t].first] - min_cost_l;
								if (tmp < min_v[j])
								{
									v[j] = min_v[j];
									min_v[j] = tmp;
								}
								else v[j] = tmp;
							}
						}
						else
						{
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								SC tmp = (SC)tt[j - iterator.ws.part[t].first] - min_cost_l;
								if (tmp < min_v[j])
								{
									v[j] = min_v[j];
									min_v[j] = tmp;
								}
								else v[j] = std::min(v[j], tmp);
							}
						}
					}
				}
				// make sure all j are < 0
				SC max_v = v[0];
				for (int j = 1; j < dim2; j++) max_v = std::max(max_v, v[j]);
				for (int j = 0; j < dim2; j++) v[j] = v[j] - max_v;
				lapFree(min_v);
			}

			double *moments;
			lapAlloc(moments, 5 * dim, __FILE__, __LINE__);
			memset(moments, 0, 5 * dim * sizeof(double));

			int *flag;
			lapAlloc(flag, dim2, __FILE__, __LINE__);
			memset(flag, 0, dim2 * sizeof(int));

			int valid_dim = dim;

#pragma omp parallel
			{
				int t = omp_get_thread_num();
				// reverse order to avoid cache thrashing
				for (int i = dim - 1; i >= 0; --i)
				{
					const auto *tt = iterator.getRow(t, i);
					SC min_cost_l, max_cost_l, second_cost_l = SC(0);
					int min_idx = iterator.ws.part[t].first;
					int second_idx = -1;
					min_cost_l = max_cost_l = (SC)tt[0] - v[iterator.ws.part[t].first];
					double moments_l[4] = { 0.0, 0.0, 0.0, 0.0 };
					for (int j = iterator.ws.part[t].first + 1; j < iterator.ws.part[t].second; j++)
					{
						SC cost_l = (SC)tt[j - iterator.ws.part[t].first] - v[j];
						if (cost_l < min_cost_l)
						{
							second_cost_l = min_cost_l;
							min_cost_l = cost_l;
							second_idx = min_idx;
							min_idx = j;
						}
						else
						{
							if (second_idx < 0)
							{
								if (cost_l > min_cost_l)
								{
									second_cost_l = cost_l;
									second_idx = j;
								}
							}
							else
							{
								if ((cost_l > min_cost_l) && (cost_l < second_cost_l))
								{
									second_cost_l = cost_l;
									second_idx = j;
								}
							}
						}
						max_cost_l = std::max(max_cost_l, cost_l);
					}
					merge_cost[t << 3] = min_cost_l;
					merge_cost[(t << 3) + 1] = second_cost_l;
					merge_cost[(t << 3) + 2] = max_cost_l;
					merge_idx[t << 3] = min_idx;
					merge_idx[(t << 3) + 1] = second_idx;
#pragma omp barrier
					min_cost_l = merge_cost[0];
					second_cost_l = merge_cost[1];
					max_cost_l = merge_cost[2];
					min_idx = merge_idx[0];
					second_idx = merge_idx[1];
					for (int xx = 1; xx < omp_get_num_threads(); xx++)
					{
						SC cost_l = merge_cost[xx << 3];
						int idx_l = merge_idx[xx << 3];
						if (cost_l < min_cost_l)
						{
							int tmp_idx = merge_idx[(xx << 3) + 1];
							SC tmp_cost = merge_cost[(xx << 3) + 1];
							if ((tmp_idx < 0) || (min_cost_l <= tmp_cost))
							{
								second_cost_l = min_cost_l;
								second_idx = min_idx;
							}
							else
							{
								second_cost_l = tmp_cost;
								second_idx = tmp_idx;
							}
							min_cost_l = cost_l;
							min_idx = idx_l;
						}
						else
						{
							if (second_idx < 0)
							{
								if (cost_l > min_cost_l)
								{
									second_cost_l = cost_l;
									second_idx = idx_l;
								}
							}
							else
							{
								if ((cost_l > min_cost_l) && (cost_l < second_cost_l))
								{
									second_cost_l = cost_l;
									second_idx = idx_l;
								}
							}
						}
						max_cost_l = std::max(max_cost_l, merge_cost[(xx << 3) + 2]);
					}
					if (t == 0)
					{
						flag[min_idx] |= 1;
						if (second_idx >= 0) flag[second_idx] |= 2;
						if (max_cost_l == min_cost_l)
						{
							valid_dim--;
						}
					}
					if (max_cost_l != min_cost_l)
					{
						double var_l = 0.0;
						for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
						{
							SC cost_l = (SC)tt[j - iterator.ws.part[t].first] - v[j] - min_cost_l;
							double x = (double)(cost_l - min_cost_l);
							moments_l[0] += x;
							moments_l[1] += x * x;
							moments_l[2] += x * x * x;
							moments_l[3] += x * x * x * x;
						}
						for (int j = 0; j < 4; j++) moments_l[j] /= (double)(dim2 - 1);
					}
					for (int j = 0; j < 4; j++) merge_moments[(t << 3) + j] = moments_l[j];
#pragma omp barrier
					if (t == 0)
					{
						for (int j = 0; j < 4; j++) moments_l[j] = merge_moments[j];
						for (int xx = 1; xx < omp_get_num_threads(); xx++) for (int j = 0; j < 4; j++) moments_l[j] += merge_moments[(xx << 3) + j];
						for (int j = 0; j < 4; j++) moments[i + dim * j] = moments_l[j];
						moments[i + dim * 4] = (double)(max_cost_l - min_cost_l);
					}
				}
			}
			double lower, upper;

			lap::estimateBounds(lower, upper, moments, flag, valid_dim, dim, dim2);

			lapFree(flag);
			lapFree(moments);
			lapFree(merge_cost);
			lapFree(merge_idx);
			lapFree(merge_moments);
			return std::pair<SC, SC>((SC)upper, (SC)lower);
		}

		template <class SC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon, SC epsilon_upper, SC epsilon_lower, SC *v)

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

			long long last_rows = 0LL;
			long long last_virtual = 0LL;

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

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

			lapAlloc(colactive, dim2, __FILE__, __LINE__);
			lapAlloc(colcomplete, dim2, __FILE__, __LINE__);
			lapAlloc(d, dim2, __FILE__, __LINE__);
			lapAlloc(pred, dim2, __FILE__, __LINE__);
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
			bool initialize_v = (v == 0);
			if (initialize_v) lapAlloc(v, dim2, __FILE__, __LINE__);

			SC epsilon;

			if (use_epsilon)
			{
				if (epsilon_upper == SC(0))
				{
					std::pair<SC, SC> eps = lap::omp::estimateEpsilon(dim, dim2, iterator, v, initialize_v);
					epsilon_upper = eps.first;
					epsilon_lower = eps.second;
				}
				epsilon = epsilon_upper;
			}
			else
			{
				if (initialize_v) memset(v, 0, dim2 * sizeof(SC));
				epsilon = SC(0);
			}

			bool first = true;
			bool allow_continue = true;
			bool clamp = true;

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= SC(0))
			{
#ifdef LAP_DEBUG
				if (first)
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
#endif
				lap::getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, dim2);
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
						if (t == 0)
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
							if (t == 0)
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
						if (t == 0)
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

#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
					SC min_v = v[0];
					for (int i = 1; i < dim2; i++) min_v = std::max(min_v, v[i]);
					for (int i = 0; i < dim2; i++) v[i] -= min_v;
				}
#endif

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
				if (last_rows > 0LL) lapInfo << " (+" << total_rows - last_rows << ")";
				last_rows = total_rows;
				if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
				if (last_virtual > 0LL) lapInfo << " (+" << total_virtual - last_virtual << ")";
				last_virtual = total_virtual;
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
			if (initialize_v) lapFree(v);
			lapFree(colsol);
			lapFree(min_private);
			lapFree(jmin_private);
		}

		// shortcut for square problems
		template <class SC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon, SC epsilon_upper, SC epsilon_lower, SC *initial_v)
		{
			lap::omp::solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon, epsilon_upper, epsilon_lower, initial_v);
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
