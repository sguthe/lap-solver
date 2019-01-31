#pragma once

#include "../lap_solver.h"
#include <algorithm>
#include <cuda.h>
#include <cub/cub.cuh>

namespace lap
{
	namespace cuda
	{
		template <class SC, class I>
		SC guessEpsilon(int x_size, int y_size, I& iterator, int step = 1)
		{
			SC epsilon(0);
			int devices = (int)iterator.ws.device.size();
			SC* minmax_cost;
			SC** d_temp_storage;
			size_t *temp_storage_bytes;
			SC** d_out;
			lapAlloc(minmax_cost, 2 * (x_size / step) * devices, __FILE__, __LINE__);
			lapAlloc(d_temp_storage, devices, __FILE__, __LINE__);
			lapAlloc(temp_storage_bytes, devices, __FILE__, __LINE__);
			lapAlloc(d_out, devices, __FILE__, __LINE__);
			memset(minmax_cost, 0, 2 * sizeof(SC) * devices);
			memset(d_temp_storage, 0, sizeof(SC*) * devices);
			memset(temp_storage_bytes, 0, sizeof(size_t) * devices);
			memset(d_out, 0, sizeof(SC*) * devices);
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				for (int x = 0; x < x_size; x += step)
				{
					const auto *tt = iterator.getRow(t, x);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					if (temp_storage_bytes[t] == 0)
					{
						cudaMalloc(&(d_out[t]), 2 * (x_size / step) * sizeof(SC));
						size_t temp_size(0);
						cub::DeviceReduce::Min(d_temp_storage[t], temp_size, tt, d_out[t], num_items);
						cub::DeviceReduce::Min(d_temp_storage[t], temp_storage_bytes[t], tt, d_out[t], num_items);
						temp_storage_bytes[t] = std::max(temp_storage_bytes[t], temp_size);
						cudaMalloc(&(d_temp_storage[t]), 2 * temp_storage_bytes[t]);
					}
					cub::DeviceReduce::Min(d_temp_storage[t], temp_storage_bytes[t], tt, d_out[t] + 2 * (x / step), num_items);
					cub::DeviceReduce::Max(d_temp_storage[t] + temp_storage_bytes[t], temp_storage_bytes[t], tt, d_out[t] + 2 * (x / step) + 1, num_items);
				}
				cudaMemcpyAsync(&(minmax_cost[2 * t *  (x_size / step)]), d_out[t], 2 * (x_size / step) * sizeof(SC), cudaMemcpyDeviceToHost);
			}
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaDeviceSynchronize();
			}
			for (int x = 0; x < x_size; x += step)
			{
				SC min_cost, max_cost;
				for (int t = 0; t < devices; t++)
				{
					if (t == 0)
					{
						min_cost = minmax_cost[2 * t *  (x_size / step) + 2 * (x / step)];
						max_cost = minmax_cost[2 * t *  (x_size / step) + 2 * (x / step) + 1];
					}
					else
					{
						max_cost = std::max(max_cost, minmax_cost[2 * t *  (x_size / step) + 2 * (x / step)]);
						min_cost = std::min(min_cost, minmax_cost[2 * t *  (x_size / step) + 2 * (x / step) + 1]);
					}
				}
				epsilon += max_cost - min_cost;
			}
			lapFree(minmax_cost);
			for (int t = 0; t < devices; t++)
			{
				if (temp_storage_bytes[t] != 0)
				{
					cudaFree(d_out[t]);
					cudaFree(d_temp_storage[t]);
				}
			}
			lapFree(d_temp_storage);
			lapFree(temp_storage_bytes);
			lapFree(d_out);
			return (epsilon / SC(10 * (x_size + step - 1) / step));
		}

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			int jmin_free;
			int jmin_taken;
		};

		template <class SC>
		__global__
		void initializeMin_kernel(min_struct<SC> *s, int dim2)
		{
			s->jmin_free = dim2;
			s->jmin_taken = dim2;
		}

		template <class SC, class TC>
		__global__
		void initializeSearch_kernel(SC *v, SC *d, TC *tt, char *colactive, int *pred, int f, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			colactive[j] = 1;
			pred[j] = f;
			d[j] = tt[j] - v[j];
		}

		template <class SC, class TC>
		__global__
		void continueSearch_kernel(SC *v, SC *d, TC *tt, char *colactive, int *pred, int i, SC h2, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] != 0)
			{
				SC v2 = tt[j] - v[j] - h2;
				SC h = d[j];
				if (v2 < h)
				{
					pred[j] = i;
					d[j] = v2;
				}
			}
		}

		template <class SC>
		__global__
		void initializeSearch_kernel(SC *v, SC *d, char *colactive, int *pred, int f, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			colactive[j] = 1;
			pred[j] = f;
			d[j] = -v[j];
		}

		template <class SC>
		__global__
		void continueSearch_kernel(SC *v, SC *d, char *colactive, int *pred, int i, SC h2, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] != 0)
			{
				SC v2 = -v[j] - h2;
				SC h = d[j];
				if (v2 < h)
				{
					pred[j] = i;
					d[j] = v2;
				}
			}
		}

		template <class SC>
		__global__
		void findMin_kernel(min_struct<SC> *s, SC *d, int *colsol, int start, int end)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j + start >= end) return;

			if (d[j] == s->min)
			{
				if (colsol[j] < 0) atomicMin(&(s->jmin_free), j + start);
				else atomicMin(&(s->jmin_taken), j + start);
			}
		}

		template <class SC>
		__global__
		void setColInactive_kernel(char *colactive, SC *d, SC *d2, SC max, int jmin)
		{
			colactive[jmin] = 0;
			d2[jmin] = d[jmin];
			d[jmin] = max;
		}

		template <class SC>
		__global__
		void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				v[j] = v[j] + d[j] - min;
			}
		}

		template <class SC>
		__global__
		void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				v[j] = v[j] + d[j] - min - eps;
			}
		}


		template <class SC, class TC, class CF, class I>
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
			// needs to be removed
			char *colactive;
			// needs to be removed
			SC *d;
			int *colsol;
			SC *v;
			SC *v2;

			int devices = (int)iterator.ws.device.size();

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

			lapAlloc(colactive, dim2, __FILE__, __LINE__);
			lapAlloc(d, dim2, __FILE__, __LINE__);
			lapAlloc(pred, dim2, __FILE__, __LINE__);
			lapAlloc(v, dim2, __FILE__, __LINE__);
			lapAlloc(v2, dim2, __FILE__, __LINE__);
			lapAlloc(colsol, dim2, __FILE__, __LINE__);

			min_struct<SC> **min_private;
			min_struct<SC> *host_min_private;
			char **colactive_private;
			int **pred_private;
			SC **d_private;
			SC **d2_private;
			int **colsol_private;
			SC **v_private;
			SC **temp_storage;
			size_t *temp_storage_bytes;
			lapAlloc(host_min_private, devices, __FILE__, __LINE__);
			lapAlloc(min_private, devices, __FILE__, __LINE__);
			lapAlloc(colactive_private, devices, __FILE__, __LINE__);
			lapAlloc(pred_private, devices, __FILE__, __LINE__);
			lapAlloc(d_private, devices, __FILE__, __LINE__);
			lapAlloc(d2_private, devices, __FILE__, __LINE__);
			lapAlloc(colsol_private, devices, __FILE__, __LINE__);
			lapAlloc(v_private, devices, __FILE__, __LINE__);
			lapAlloc(temp_storage, devices, __FILE__, __LINE__);
			lapAlloc(temp_storage_bytes, devices, __FILE__, __LINE__);
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
				cudaMalloc(&(min_private[t]), sizeof(min_struct<SC>));
				cudaMalloc(&(colactive_private[t]), sizeof(char) * size);
				cudaMalloc(&(pred_private[t]), sizeof(int) * size);
				cudaMalloc(&(d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(d2_private[t]), sizeof(SC) * size);
				cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
				cudaMalloc(&(v_private[t]), sizeof(SC) * size);
				temp_storage[t] = 0;
				temp_storage_bytes[t] = 0;
			}

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
			SC epsilon_lower = epsilon / SC(dim2);

			SC last_avg = SC(0);
			bool first = true;
			bool allow_reset = true;

			memset(v, 0, dim2 * sizeof(SC));

			while (epsilon >= SC(0))
			{
				lap::getNextEpsilon(epsilon, epsilon_lower, last_avg, first, allow_reset, v, dim2);
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

				// upload v to devices
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice);
				}

				int jmin, jmin_free, jmin_taken;
				SC min, min_n;
				bool unassignedfound;

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif
				long long count = 0ll;
				for (int f = 0; f < dim2; f++)
				{
					// upload colsol to devices
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
						int start = iterator.ws.part[t].first;
						cudaMemcpyAsync(colsol_private[t], &(colsol[start]), sizeof(int) * size, cudaMemcpyHostToDevice);
					}
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
					count++;

					unassignedfound = false;

					// Dijkstra search
					min = std::numeric_limits<SC>::max();
					jmin_free = dim2;
					jmin_taken = dim2;
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						// initialize Search
						initializeMin_kernel<<<1, 1>>>(min_private[t], dim2);
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						if (f < dim)
							initializeSearch_kernel<<<grid_size, block_size>>>(v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], pred_private[t], f, size);
						else
							initializeSearch_kernel<<<grid_size, block_size>>>(v_private[t], d_private[t], colactive_private[t], pred_private[t], f, size);
						if (temp_storage_bytes[t] == 0)
						{
							cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size);
							cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
						}
						cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size);
						// min is now set so we need to find the correspoding minima for free and taken columns
						findMin_kernel<<<grid_size, block_size>>>(min_private[t], d_private[t], colsol_private[t], start, end);
						cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost);
					}
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						cudaDeviceSynchronize();
						if (host_min_private[t].min < min)
						{
							min = host_min_private[t].min;
							jmin_free = host_min_private[t].jmin_free;
							jmin_taken = host_min_private[t].jmin_taken;
						}
						else if (host_min_private[t].min == min)
						{
							jmin_free = std::min(jmin_free, host_min_private[t].jmin_free);
							jmin_taken = std::min(jmin_taken, host_min_private[t].jmin_taken);
						}
					}

					if (jmin_free < dim2) jmin = jmin_free;
					else jmin = jmin_taken;

					// dijkstraCheck
					for (int t = 0; t < devices; t++)
					{
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						if ((jmin >= start) && (jmin < end))
						{
							cudaSetDevice(iterator.ws.device[t]);
							setColInactive_kernel<<<1, 1>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
						}
					}
					if (colsol[jmin] < 0)
					{
						endofpath = jmin;
						unassignedfound = true;
					}

					while (!unassignedfound)
					{
						// update 'distances' between freerow and all unscanned columns, via next scanned column.
						int i = colsol[jmin];
#ifndef LAP_QUIET
						if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
						if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
						scancount[i]++;
#endif
						count++;

						jmin_free = dim2;
						jmin_taken = dim2;
						if (f < dim)
						{
							SC h2;
							for (int t = 0; t < devices; t++)
							{
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								if ((jmin >= start) && (jmin < end))
								{
									cudaSetDevice(iterator.ws.device[t]);
									auto tt = iterator.getRow(t, i);
									TC tt_jmin;
									SC v_jmin;
									cudaMemcpyAsync(&(tt_jmin), &(tt[jmin - start]), sizeof(tt_jmin), cudaMemcpyDeviceToHost);
									cudaMemcpyAsync(&(v_jmin), &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost);
									cudaDeviceSynchronize();
									h2 = tt_jmin - v_jmin - min;
								}
							}
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								// initialize Search
								initializeMin_kernel<<<1, 1>>>(min_private[t], dim2);
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								continueSearch_kernel<<<grid_size, block_size>>>(v_private[t], d_private[t], iterator.getRow(t, i), colactive_private[t], pred_private[t], i, h2, size);
								if (temp_storage_bytes[t] == 0)
								{
									cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size);
									cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
								}
								cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size);
								// min is now set so we need to find the correspoding minima for free and taken columns
								findMin_kernel<<<grid_size, block_size>>>(min_private[t], d_private[t], colsol_private[t], start, end);
								cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost);
							}
						}
						else
						{
							SC h2;
							for (int t = 0; t < devices; t++)
							{
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								if ((jmin >= start) && (jmin < end))
								{
									cudaSetDevice(iterator.ws.device[t]);
									SC v_jmin;
									cudaMemcpyAsync(&(v_jmin), &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost);
									cudaDeviceSynchronize();
									h2 = -v_jmin - min;
								}
							}
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								// initialize Search
								initializeMin_kernel<<<1, 1>>>(min_private[t], dim2);
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								continueSearch_kernel<<<grid_size, block_size>>> (v_private[t], d_private[t], colactive_private[t], pred_private[t], i, h2, size);
								if (temp_storage_bytes[t] == 0)
								{
									cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size);
									cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
								}
								cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size);
								// min is now set so we need to find the correspoding minima for free and taken columns
								findMin_kernel<<<grid_size, block_size>>>(min_private[t], d_private[t], colsol_private[t], start, end);
								cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost);
							}
						}
						min_n = std::numeric_limits<SC>::max();
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							cudaDeviceSynchronize();
							if (host_min_private[t].min < min_n)
							{
								min_n = host_min_private[t].min;
								jmin_free = host_min_private[t].jmin_free;
								jmin_taken = host_min_private[t].jmin_taken;
							}
							else if (host_min_private[t].min == min_n)
							{
								jmin_free = std::min(jmin_free, host_min_private[t].jmin_free);
								jmin_taken = std::min(jmin_taken, host_min_private[t].jmin_taken);
							}
						}

						if (jmin_free < dim2) jmin = jmin_free;
						else jmin = jmin_taken;

						min = std::max(min, min_n);

						// dijkstraCheck
						for (int t = 0; t < devices; t++)
						{
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							if ((jmin >= start) && (jmin < end))
							{
								cudaSetDevice(iterator.ws.device[t]);
								setColInactive_kernel<<<1, 1>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
							}
						}
						if (colsol[jmin] < 0)
						{
							endofpath = jmin;
							unassignedfound = true;
						}
					}

					// update column prices. can increase or decrease
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						// initialize Search
						initializeMin_kernel<<<1, 1>>>(min_private[t], dim2);
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						if (epsilon > SC(0))
						{
							updateColumnPrices_kernel<<<grid_size, block_size>>>(colactive_private[t], min, v_private[t], d2_private[t], epsilon, size);
						}
						else
						{
							updateColumnPrices_kernel<<<grid_size, block_size>>>(colactive_private[t], min, v_private[t], d2_private[t], size);
						}
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

					// need to copy pred from GPUs here
					// upload data to devices
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
						int start = iterator.ws.part[t].first;
						cudaMemcpyAsync(&(pred[start]), pred_private[t], sizeof(int) * size, cudaMemcpyDeviceToHost);
					}
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						cudaDeviceSynchronize();
					}
					// reset row and column assignments along the alternating path.
					lap::resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
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
							if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
							else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
						}
						old_complete = f + 1;
					}
#endif
				}

				// now we need to compare the previous v with the new one to get back all the changes
				SC total(0);
				SC scaled_epsilon = epsilon * SC(count) / SC(dim2);

				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost);
				}
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					cudaDeviceSynchronize();
				}
				for (int i = 0; i < dim2; i++) total += v2[i] - v[i] + scaled_epsilon;

				if (count > 0) last_avg = total / SC(count);
				else last_avg = SC(0);
				std::swap(v, v2);

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

			// free CUDA memory
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(min_private[t]);
				cudaFree(colactive_private[t]);
				cudaFree(pred_private[t]);
				cudaFree(d_private[t]);
				cudaFree(colsol_private[t]);
				cudaFree(v_private[t]);
				if (temp_storage[t] != 0) cudaFree(temp_storage[t]);
			}

			// free reserved memory.
			lapFree(pred);
			lapFree(colactive);
			lapFree(d);
			lapFree(v);
			lapFree(v2);
			lapFree(colsol);
			lapFree(min_private);
			lapFree(host_min_private);
			lapFree(colactive_private);
			lapFree(pred_private);
			lapFree(d_private);
			lapFree(colsol_private);
			lapFree(v_private);
			lapFree(temp_storage_bytes);
			lapFree(temp_storage);
		}

		// shortcut for square problems
		template <class SC, class TC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol)
		{
			lap::cuda::solve<SC, TC>(dim, dim, costfunc, iterator, rowsol);
		}

#if 0
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
			return lap::cuda::cost<SC, CF>(dim, dim, costfunc, rowsol);
		}
#endif
	}
}
