#pragma once
#pragma once

#include <chrono>
#include <sstream>
#include <iostream>
#include <cstring>
#ifndef LAP_QUIET
#include <deque>
#include <mutex>
#endif
#include <math.h>

namespace lap
{
	namespace sparse
	{
		template <class SC, typename COST, typename INDEX>
		void getMinSecondBest(SC &min_cost_l, SC &second_cost_l, COST &cost, INDEX &index, int sparse_count, int count)
		{
			min_cost_l = std::numeric_limits<SC>::max();
			second_cost_l = std::numeric_limits<SC>::max();
			for (int jj = 0; jj < sparse_count; jj++)
			{
				int j = index(jj);
				SC cost_l = cost(jj);
				if (cost_l < min_cost_l)
				{
					second_cost_l = min_cost_l;
					min_cost_l = cost_l;
				}
				else second_cost_l = std::min(second_cost_l, cost_l);
			}
		}

		template <class SC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC *v, int *perm)
		{
#ifdef LAP_DEBUG
			auto start_time = std::chrono::high_resolution_clock::now();
#endif
			SC lower_bound = std::numeric_limits<SC>::max();
			SC upper_bound = SC(0);

			SC *mod_v;

			lapAlloc(mod_v, dim2, __FILE__, __LINE__);

			for (int i = 0; i < dim2; i++)
			{
				perm[i] = i;
				v[i] = SC(0);
			}
			// reverse order
			for (int i = dim - 1; i >= 0; --i)
			{
				SC min_cost_l, second_cost_l;
				auto tt = iterator.getRow(i);
				auto index = [&tt](int j) -> int { return std::get<1>(tt)[j]; };
				auto cost = [&tt](int j) -> SC { return (SC)std::get<2>(tt)[j]; };
				getMinSecondBest(min_cost_l, second_cost_l, cost, index, std::get<0>(tt), dim2);

				if (second_cost_l == std::numeric_limits<SC>::max())
				{
					mod_v[i] = SC(-1);
				}
				else
				{
					mod_v[i] = second_cost_l - min_cost_l;
					// need to use the same v values in total
					lower_bound = std::min(lower_bound, second_cost_l - min_cost_l);
					upper_bound = std::max(upper_bound, second_cost_l - min_cost_l);
				}
			}

			upper_bound = (SC)(0.125 * (double)upper_bound * (double)dim / (double)dim2);
			lower_bound = (SC)(8.0 * (double)upper_bound / (double)dim);

			// sort permutation by keys
			std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });
			lapFree(mod_v);

			return std::pair<SC, SC>(upper_bound, lower_bound);
		}

		template <class SC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)

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
			SC *d;
			int *colsol;
			SC epsilon_upper;
			SC epsilon_lower;
			SC *v;
			int *perm;

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

			lapAlloc(colactive, dim2, __FILE__, __LINE__);
			lapAlloc(d, dim2, __FILE__, __LINE__);
			lapAlloc(pred, dim2, __FILE__, __LINE__);
			lapAlloc(colsol, dim2, __FILE__, __LINE__);
			lapAlloc(v, dim2, __FILE__, __LINE__);
			lapAlloc(perm, dim2, __FILE__, __LINE__);

#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			SC epsilon;

			// can't be done like this
			if (use_epsilon)
			{
				std::pair<SC, SC> eps = lap::sparse::estimateEpsilon(dim, dim2, iterator, v, perm);
				epsilon_upper = eps.first;
				epsilon_lower = eps.second;
			}
			else
			{
				memset(v, 0, dim2 * sizeof(SC));
				epsilon_upper = SC(0);
				epsilon_lower = SC(0);
			}
			epsilon = epsilon_upper;

			bool first = true;
			bool second = false;
			bool reverse = true;

			if ((!use_epsilon) || (epsilon > SC(0)))
			{
				for (int i = 0; i < dim2; i++) perm[i] = i;
				reverse = false;
			}

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
				getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);

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
				int jmin, jmin_n;
				SC min, min_n;
				bool unassignedfound;

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

#ifdef LAP_MINIMIZE_V
//				int dim_limit = ((reverse) || (epsilon < SC(0))) ? dim2 : dim;
				int dim_limit = dim2;
#else
				int dim_limit = dim2;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim_limit, " rows");
#endif
				for (int fc = 0; fc < dim_limit; fc++)
				{
					int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
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

					unassignedfound = false;

					// Dijkstra search
					min = std::numeric_limits<SC>::max();
					jmin = dim2;
					if (f < dim)
					{
						memset(pred, -1, dim2 * sizeof(int));
						std::fill(d, d + dim2, std::numeric_limits<SC>::max());
						std::fill(colactive, colactive + dim2, 1);
						auto tt = iterator.getRow(f);
						for (int jj = 0; jj < std::get<0>(tt); jj++)
						{
							int j = std::get<1>(tt)[jj];
							pred[j] = f;
							SC h = d[j] = std::get<2>(tt)[jj] - v[j];
							if (h < min)
							{
								// better
								jmin = j;
								min = h;
							}
							else if ((h == min) && (h < std::numeric_limits<SC>::max()))
							{
								// same, do only update if old was used and new is free
								if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
							}
						}
					}
					else
					{
						for (int j = 0; j < dim2; j++)
						{
							colactive[j] = 1;
							pred[j] = f;
							SC h = d[j] = -v[j];
							if (colsol[j] < dim)
							{
								if (h < min)
								{
									// better
									jmin = j;
									min = h;
								}
								else if ((h == min) && (h < std::numeric_limits<SC>::max()))
								{
									// same, do only update if old was used and new is free
									if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
								}
							}
						}
					}

					dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive);
					// marked skipped columns that were cheaper
					if (f >= dim)
					{
						for (int j = 0; j < dim2; j++)
						{
							// ignore any columns assigned to virtual rows
							if ((colsol[j] >= dim) && (d[j] <= min))
							{
								colactive[j] = 0;
							}
						}
					}

					while (!unassignedfound)
					{
						// update 'distances' between freerow and all unscanned columns, via next scanned column.
						int i = colsol[jmin];
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

						jmin_n = dim2;
						min_n = std::numeric_limits<SC>::max();
						if (i < dim)
						{
							auto tt = iterator.getRow(i);
							SC tt_jmin;
							for (int jj = 0; jj < std::get<0>(tt); jj++) if (std::get<1>(tt)[jj] == jmin) {
								tt_jmin = (SC)std::get<2>(tt)[jj];
								break;
							}
							SC v_jmin = v[jmin];
							for (int jj = 0; jj < std::get<0>(tt); jj++)
							{
								int j = std::get<1>(tt)[jj];
								if (colactive[j] != 0)
								{
									SC v2 = (std::get<2>(tt)[jj] - tt_jmin) - (v[j] - v_jmin) + min;
									if (v2 < d[j])
									{
										pred[j] = i;
										d[j] = v2;
									}
								}
							}
							for (int j = 0; j < dim2; j++)
							{
								if (colactive[j] != 0)
								{
									SC h = d[j];
									if (h < min_n)
									{
										// better
										jmin_n = j;
										min_n = h;
									}
									else if ((h == min_n) && (h < std::numeric_limits<SC>::max()))
									{
										// same, do only update if old was used and new is free
										if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
									}
								}
							}
						}
						else
						{
							SC v_jmin = v[jmin];
							for (int j = 0; j < dim2; j++)
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
									if (colsol[j] < dim)
									{
										if (h < min_n)
										{
											// better
											jmin_n = j;
											min_n = h;
										}
										else if ((h == min_n) && (h < std::numeric_limits<SC>::max()))
										{
											// same, do only update if old was used and new is free
											if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
										}
									}
								}
							}
						}

						min = std::max(min, min_n);
						jmin = jmin_n;
						dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive);

						// marked skipped columns that were cheaper
						if (i >= dim)
						{
							for (int j = 0; j < dim2; j++)
							{
								// ignore any columns assigned to virtual rows
								if ((colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min))
								{
									colactive[j] = 0;
								}
							}
						}
					}

					// update column prices. can increase or decrease
					if (epsilon > SC(0))
					{
						updateColumnPrices(colactive, 0, dim2, min, v, d, epsilon, total_d, total_eps);
					}
					else
					{
						updateColumnPrices(colactive, 0, dim2, min, v, d);
					}
#ifdef LAP_ROWS_SCANNED
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
#ifndef LAP_QUIET
					int level;
					if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
					{
						long long hit, miss;
						iterator.getHitMiss(hit, miss);
						total_hit += hit;
						total_miss += miss;
						if ((hit != 0) || (miss != 0))
						{
							if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
							else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
						}
						old_complete = f + 1;
					}
#endif
				}

#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
#if 0
					if (dim_limit < dim2) lap::normalizeV(v, dim2, colsol);
					else lap::normalizeV(v, dim2);
#else
					if (dim_limit < dim2) for (int i = 0; i < dim2; i++) if (colsol[i] < 0) v[i] -= SC(2) * epsilon;
					lap::normalizeV(v, dim2);
#endif
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
				second = first;
				first = false;
				reverse = !reverse;
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
			lapInfo << "row\tscanned\tlength" << std::endl;
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << f << "\t" << scancount[f] << "\t" << pathlength[f] << std::endl;
			}

			lapFree(scancount);
			lapFree(pathlength);
#endif

#ifdef LAP_VERIFY_RESULT
			SC slack = SC(0);
			bool correct = true;
			for (int f = 0; f < dim2; f++)
			{
				auto tt = iterator.getRow(f);
				int jmin = rowsol[f];
				SC ref_min = tt[jmin] - v[jmin];
				SC min = ref_min;
				for (int j = 0; j < dim2; j++)
				{
					SC h = tt[j] - v[j];
					if (h < min)
					{
						// better
						jmin = j;
						min = h;
					}
				}
				if (jmin != rowsol[f])
				{
					slack += ref_min - min;
					correct = false;
				}
			}
			if (correct)
			{
				lapInfo << "Solution accurate." << std::endl;
			}
			else
			{
				lapInfo << "Solution might be inaccurate (slack = " << slack << ")." << std::endl;
			}
#endif

			// free reserved memory.
			lapFree(pred);
			lapFree(colactive);
			lapFree(d);
			lapFree(v);
			lapFree(colsol);
			lapFree(perm);
		}

		// shortcut for square problems
		template <class SC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
		{
			solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
		{
			SC total = SC(0);
			for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
			return total;
		}

		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol)
		{
			return cost<SC, CF>(dim, dim, costfunc, rowsol);
		}
	}
}