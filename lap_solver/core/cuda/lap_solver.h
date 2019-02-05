#pragma once

#include "../lap_solver.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <omp.h>

namespace lap
{
	namespace cuda
	{
		template <class SC, class I>
		SC guessEpsilon(int x_size, int y_size, I& iterator, int step = 1)
		{
			SC epsilon(0);
			int devices = (int)iterator.ws.device.size();
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
			SC* minmax_cost;
			SC** d_out;
			cudaMallocHost(&minmax_cost, 2 * (x_size / step) * devices * sizeof(SC));
			lapAlloc(d_out, devices, __FILE__, __LINE__);
			memset(minmax_cost, 0, 2 * sizeof(SC) * devices);
			memset(d_out, 0, sizeof(SC*) * devices);
			SC** d_temp_storage;
			size_t *temp_storage_bytes;
			lapAlloc(d_temp_storage, devices, __FILE__, __LINE__);
			lapAlloc(temp_storage_bytes, devices, __FILE__, __LINE__);
			memset(d_temp_storage, 0, sizeof(SC*) * devices);
			memset(temp_storage_bytes, 0, sizeof(size_t) * devices);
#pragma omp parallel for
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				cudaStream_t stream = iterator.ws.stream[t];
				for (int x = step * ((x_size - 1) / step); x >= 0; x -= step)
				{
					auto tt = iterator.getRow(t, x);
					if (temp_storage_bytes[t] == 0)
					{
						cudaMalloc(&(d_out[t]), 2 * (x_size / step) * sizeof(SC));
						size_t temp_size(0);
						cub::DeviceReduce::Min(d_temp_storage[t], temp_size, tt, d_out[t], num_items, stream);
						cub::DeviceReduce::Max(d_temp_storage[t], temp_storage_bytes[t], tt, d_out[t], num_items, stream);
						temp_storage_bytes[t] = std::max(temp_storage_bytes[t], temp_size);
						cudaMalloc(&(d_temp_storage[t]), 2 * temp_storage_bytes[t]);
					}
					cub::DeviceReduce::Min(d_temp_storage[t], temp_storage_bytes[t], tt, d_out[t] + 2 * (x / step), num_items, stream);
					cub::DeviceReduce::Max(d_temp_storage[t] + temp_storage_bytes[t], temp_storage_bytes[t], tt, d_out[t] + 2 * (x / step) + 1, num_items, stream);
				}
				cudaMemcpyAsync(&(minmax_cost[2 * t *  (x_size / step)]), d_out[t], 2 * (x_size / step) * sizeof(SC), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
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
			cudaFreeHost(minmax_cost);
#pragma omp parallel for
			for (int t = 0; t < devices; t++)
			{
				if (temp_storage_bytes[t] != 0)
				{
					cudaSetDevice(iterator.ws.device[t]);
					cudaFree(d_out[t]);
					cudaFree(d_temp_storage[t]);
				}
			}
			lapFree(d_temp_storage);
			lapFree(temp_storage_bytes);
			lapFree(d_out);
			omp_set_num_threads(old_threads);
			return (epsilon / SC(10 * (x_size + step - 1) / step));
		}

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			int jmin_free;
			int jmin_taken;
			int colsol_take;
		};

		class reset_struct
		{
		public:
			int i;
			int j;
		};

		template <class SC>
		__global__ void initializeMin_kernel(min_struct<SC> *s, int dim2)
		{
			s->jmin_free = dim2;
			s->jmin_taken = dim2;
		}

		template <class SC, class TC>
		__global__ void initializeSearch_kernel(SC *v, SC *d, TC *tt, char *colactive, int *pred, int f, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			colactive[j] = 1;
			pred[j] = f;
			d[j] = tt[j] - v[j];
		}

		template <class SC, class TC>
		__global__ void continueSearch_kernel(SC *v, SC *d, TC *tt, char *colactive, int *pred, int i, SC h2, int size)
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
		__global__ void initializeSearch_kernel(SC *v, SC *d, char *colactive, int *pred, int f, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			colactive[j] = 1;
			pred[j] = f;
			d[j] = -v[j];
		}

		template <class SC>
		__global__ void continueSearch_kernel(SC *v, SC *d, char *colactive, int *pred, int i, SC h2, int size)
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

		template <class SC, class TC>
		__global__ void continueSearchJMin_kernel(SC *v, SC *d, TC *tt, char *colactive, int *pred, int i, int jmin, SC min, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] != 0)
			{
				SC h2 = tt[jmin] - v[jmin] - min;
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
		__global__ void continueSearchJMin_kernel(SC *v, SC *d, char *colactive, int *pred, int i, int jmin, SC min, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] != 0)
			{
				SC h2 = -v[jmin] - min;
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
		__global__ void findMin_kernel(min_struct<SC> *s, SC *d, int *colsol, int start, int end)
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
		__global__ void setColInactive_kernel(char *colactive, SC *d, SC *d2, SC max, int jmin)
		{
			colactive[jmin] = 0;
			d2[jmin] = d[jmin];
			d[jmin] = max;
		}

		template <class SC>
		__global__ void setColInactive_kernel(char *colactive, SC *d, SC *d2, min_struct<SC> *s, int *colsol, SC max, int jmin)
		{
			colactive[jmin] = 0;
			d2[jmin] = d[jmin];
			d[jmin] = max;
			s->colsol_take = colsol[jmin];
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				v[j] = v[j] + d[j] - min;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				v[j] = v[j] + d[j] - min - eps;
			}
		}

		__global__ void resetRowColumnAssignment_kernel(int *pred, int *colsol, int *rowsol, int start, int end, int endofpath, int f)
		{
			int j = endofpath;
			int i;
			do
			{
				i = pred[j - start];
				colsol[j - start] = i;
				int j1 = rowsol[i];
				rowsol[i] = j;
				j = j1;
			} while (i != f);
		}

		__global__ void resetRowColumnAssignment_kernel(int *pred, int *colsol, int *rowsol, reset_struct *state, int start, int end, int endofpath, int f)
		{
			int j = endofpath;
			int i;
			do
			{
				i = pred[j - start];
				colsol[j - start] = i;
				int j1 = rowsol[i];
				rowsol[i] = j;
				if ((i != f) && (j1 < 0))
				{
					// not last element but not in this list
					state->i = i;
					state->j = j;
					return;
				}
				j = j1;
			} while (i != f);
			state->i = -1;
		}

		__global__ void resetRowColumnAssignmentContinue_kernel(int *pred, int *colsol, int *rowsol, reset_struct *state_in, reset_struct *state_out, int start, int end, int f)
		{
			int i = state_in->i;
			int j = rowsol[i];
			// rowsol for i not found in this range
			if ((j < 0) || (j == state_in->j)) return;
			// clear old rowsol since it is no longer in this range
			rowsol[i] = -1;
			do 
			{
				i = pred[j - start];
				colsol[j - start] = i;
				int j1 = rowsol[i];
				rowsol[i] = j;
				if ((i != f) && (j1 < 0))
				{
					// not last element but not in this list
					state_out->i = i;
					state_out->j = j;
					return;
				}
				j = j1;
			} while (i != f);
			state_out->i = -1;
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

			int  endofpath;
#ifndef LAP_CUDA_LOCAL_ROWSOL
			int  *pred;
			int *colsol;
#endif
			SC *v;
			SC *v2;
			// for calculating h2
			TC *tt_jmin;
			SC *v_jmin;

			int devices = (int)iterator.ws.device.size();
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

			// used for copying
			min_struct<SC> *host_min_private;
#ifdef LAP_CUDA_LOCAL_ROWSOL
			reset_struct *host_reset;
			cudaMallocHost(&host_reset, 2 * sizeof(reset_struct));
#else
			cudaMallocHost(&pred, dim2 * sizeof(int));
			cudaMallocHost(&colsol, dim2 * sizeof(int));
#endif
			cudaMallocHost(&v, dim2 * sizeof(SC));
			cudaMallocHost(&v2, dim2 * sizeof(SC));
			cudaMallocHost(&host_min_private, devices * sizeof(min_struct<SC>));
			cudaMallocHost(&tt_jmin, sizeof(TC));
			cudaMallocHost(&v_jmin, sizeof(SC));

			min_struct<SC> **min_private;
			char **colactive_private;
			int **pred_private;
			SC **d_private;
			SC **d2_private;
			int **colsol_private;
#ifdef LAP_CUDA_LOCAL_ROWSOL
			int **rowsol_private;
#endif
			SC **v_private;
			SC **temp_storage;
			size_t *temp_storage_bytes;
			// on device
			lapAlloc(min_private, devices, __FILE__, __LINE__);
			lapAlloc(colactive_private, devices, __FILE__, __LINE__);
			lapAlloc(pred_private, devices, __FILE__, __LINE__);
			lapAlloc(d_private, devices, __FILE__, __LINE__);
			lapAlloc(d2_private, devices, __FILE__, __LINE__);
			lapAlloc(colsol_private, devices, __FILE__, __LINE__);
#ifdef LAP_CUDA_LOCAL_ROWSOL
			lapAlloc(rowsol_private, devices, __FILE__, __LINE__);
#endif
			lapAlloc(v_private, devices, __FILE__, __LINE__);
			lapAlloc(temp_storage, devices, __FILE__, __LINE__);
			lapAlloc(temp_storage_bytes, devices, __FILE__, __LINE__);

			if (devices == 1)
			{
				// single device code
				int t = 0;
				cudaSetDevice(iterator.ws.device[t]);
				int start = iterator.ws.part[t].first;
				int end = iterator.ws.part[t].second;
				int size = end - start;
				cudaMalloc(&(min_private[t]), sizeof(min_struct<SC>));
				cudaMalloc(&(colactive_private[t]), sizeof(char) * size);
				cudaMalloc(&(d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(d2_private[t]), sizeof(SC) * size);
				cudaMalloc(&(v_private[t]), sizeof(SC) * size);
				temp_storage[t] = 0;
				temp_storage_bytes[t] = 0;
#ifdef LAP_CUDA_AVOID_MEMCPY
				colsol_private[t] = &(colsol[start]);
				pred_private[t] = &(pred[start]);
#else
				cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
				cudaMalloc(&(pred_private[t]), sizeof(int) * size);
#endif
#ifdef LAP_CUDA_LOCAL_ROWSOL
				cudaMalloc(&(rowsol_private[t]), sizeof(int) * dim2);
#endif
			}
			else
			{
#pragma omp parallel for
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaMalloc(&(min_private[t]), sizeof(min_struct<SC>));
					cudaMalloc(&(colactive_private[t]), sizeof(char) * size);
					cudaMalloc(&(d_private[t]), sizeof(SC) * size);
					cudaMalloc(&(d2_private[t]), sizeof(SC) * size);
					cudaMalloc(&(v_private[t]), sizeof(SC) * size);
					temp_storage[t] = 0;
					temp_storage_bytes[t] = 0;
#ifdef LAP_CUDA_AVOID_MEMCPY
					colsol_private[t] = &(colsol[start]);
					pred_private[t] = &(pred[start]);
#else
					cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
					cudaMalloc(&(pred_private[t]), sizeof(int) * size);
#endif
#ifdef LAP_CUDA_LOCAL_ROWSOL
					cudaMalloc(&(rowsol_private[t]), sizeof(int) * dim2);
#endif
				}
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
#ifndef LAP_CUDA_LOCAL_ROWSOL
				memset(rowsol, -1, dim2 * sizeof(int));
				memset(colsol, -1, dim2 * sizeof(int));
#endif

				if (devices == 1)
				{
					// upload v to devices
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
#ifdef LAP_CUDA_EVENT_SYNC
					cudaEvent_t event = iterator.ws.event[t];
#endif
					cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
#ifdef LAP_CUDA_LOCAL_ROWSOL
					cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
#endif
				}
				else
				{
					// upload v to devices
#pragma omp parallel for
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
						int start = iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
#ifdef LAP_CUDA_EVENT_SYNC
						cudaEvent_t event = iterator.ws.event[t];
#endif
						cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
#ifdef LAP_CUDA_LOCAL_ROWSOL
						cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
						cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
#endif
					}
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
				SC h2(0);

				if (devices == 1)
				{
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[t];
#ifdef LAP_CUDA_EVENT_SYNC
					cudaEvent_t event = iterator.ws.event[t];
#endif

					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (size + block_size.x - 1) / block_size.x;

					for (int f = 0; f < dim2; f++)
					{
#if (!defined LAP_CUDA_AVOID_MEMCPY) && (!defined LAP_CUDA_LOCAL_ROWSOL)
						// upload colsol to devices
						cudaMemcpyAsync(colsol_private[t], &(colsol[start]), sizeof(int) * size, cudaMemcpyHostToDevice, stream);
#endif
						// initialize Search
						initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], dim2);
						if (f < dim)
							initializeSearch_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], pred_private[t], f, size);
						else
							initializeSearch_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], colactive_private[t], pred_private[t], f, size);
						if (temp_storage_bytes[t] == 0)
						{
							cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
							cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
						}
						cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
						// min is now set so we need to find the correspoding minima for free and taken columns
						findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
						cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
						cudaEventRecord(event, stream);
						cudaEventSynchronize(event);
#else
						cudaStreamSynchronize(stream);
#endif
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
						min = host_min_private[t].min;
						jmin_free = host_min_private[t].jmin_free;
						jmin_taken = host_min_private[t].jmin_taken;

						// dijkstraCheck
						if (jmin_free < dim2)
						{
							jmin = jmin_free;
							endofpath = jmin;
							unassignedfound = true;
						}
						else
						{
							jmin = jmin_taken;
							unassignedfound = false;
						}

						while (!unassignedfound)
						{
							// mark last column scanned (single device)
#ifdef LAP_CUDA_LOCAL_ROWSOL
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#ifdef LAP_CUDA_EVENT_SYNC
							cudaEventRecord(event, stream);
							cudaEventSynchronize(event);
#else
							cudaStreamSynchronize(stream);
#endif
#else
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
#ifdef LAP_CUDA_LOCAL_ROWSOL
							int i = host_min_private[0].colsol_take;
#else
							int i = colsol[jmin];
#endif
							if (f < dim)
							{
								// get row
								auto tt = iterator.getRow(t, i);
								// initialize Search
								initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], dim2);
								// continue search
								continueSearchJMin_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], tt, colactive_private[t], pred_private[t], i, jmin, min, size);
							}
							else
							{
								// initialize Search
								initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], dim2);
								// continue search
								continueSearchJMin_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], colactive_private[t], pred_private[t], i, jmin, min, size);
							}
							// find minimum
							if (temp_storage_bytes[t] == 0)
							{
								cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
								cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
							}
							cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
							// min is now set so we need to find the correspoding minima for free and taken columns
							findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
							cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
							cudaEventRecord(event, stream);
							cudaEventSynchronize(event);
#else
							cudaStreamSynchronize(stream);
#endif
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

							min_n = host_min_private[t].min;
							jmin_free = host_min_private[t].jmin_free;
							jmin_taken = host_min_private[t].jmin_taken;

							min = std::max(min, min_n);

							// dijkstraCheck
							if (jmin_free < dim2)
							{
								jmin = jmin_free;
								endofpath = jmin;
								unassignedfound = true;
							}
							else
							{
								jmin = jmin_taken;
								unassignedfound = false;
							}
						}

						// update column prices. can increase or decrease
						// need to copy pred from GPUs here
#if (!defined LAP_CUDA_AVOID_MEMCPY) && (!defined LAP_CUDA_LOCAL_ROWSOL)
						cudaMemcpyAsync(&(pred[start]), pred_private[t], sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
#endif
						if (epsilon > SC(0))
						{
							updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d2_private[t], epsilon, size);
						}
						else
						{
							updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d2_private[t], size);
						}
						// synchronization only required due to copy
#if (!defined LAP_CUDA_AVOID_MEMCPY) && (!defined LAP_CUDA_LOCAL_ROWSOL)
#ifdef LAP_CUDA_EVENT_SYNC
						cudaEventRecord(event, stream);
						cudaEventSynchronize(event);
#else
						cudaStreamSynchronize(stream);
#endif
#endif
						// reset row and column assignments along the alternating path.
#ifdef LAP_CUDA_LOCAL_ROWSOL
						resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], start, end, endofpath, f);
#ifndef LAP_QUIET
						{
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
						}
#endif
#else
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
						lap::resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
#ifndef LAP_QUIET
						{
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
						}
#endif
#endif
					}

					// download updated v
					cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
					cudaEventRecord(event, stream);
					cudaEventSynchronize(event);
#else
					cudaStreamSynchronize(stream);
#endif
				}
				else
				{
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
#ifdef LAP_CUDA_EVENT_SYNC
						cudaEvent_t event = iterator.ws.event[t];
#endif

						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;

						for (int f = 0; f < dim2; f++)
						{
#if (!defined LAP_CUDA_AVOID_MEMCPY) && (!defined LAP_CUDA_LOCAL_ROWSOL)
							// upload colsol to devices
							cudaMemcpyAsync(colsol_private[t], &(colsol[start]), sizeof(int) * size, cudaMemcpyHostToDevice, stream);
#endif
							// initialize Search
							initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], dim2);
							if (f < dim)
								initializeSearch_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], pred_private[t], f, size);
							else
								initializeSearch_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], colactive_private[t], pred_private[t], f, size);
							if (temp_storage_bytes[t] == 0)
							{
								cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
								cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
							}
							cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
							// min is now set so we need to find the correspoding minima for free and taken columns
							findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
							cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
							cudaEventRecord(event, stream);
							cudaEventSynchronize(event);
#else
							cudaStreamSynchronize(stream);
#endif
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
								count++;

								unassignedfound = false;

								// Dijkstra search
								min = std::numeric_limits<SC>::max();
								jmin_free = dim2;
								jmin_taken = dim2;
								for (int t = 0; t < devices; t++)
								{
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

								// dijkstraCheck
								if (jmin_free < dim2)
								{
									jmin = jmin_free;
									endofpath = jmin;
									unassignedfound = true;
								}
								else
								{
									jmin = jmin_taken;
									unassignedfound = false;
								}
							}
#pragma omp barrier

							while (!unassignedfound)
							{
								// mark last column scanned (single device)
								if ((jmin >= start) && (jmin < end))
								{
#ifdef LAP_CUDA_LOCAL_ROWSOL
									setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#ifdef LAP_CUDA_EVENT_SYNC
									cudaEventRecord(event, stream);
									cudaEventSynchronize(event);
#else
									cudaStreamSynchronize(stream);
#endif
#else
									setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
								}
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
#ifdef LAP_CUDA_LOCAL_ROWSOL
#pragma omp barrier
								int i = host_min_private[iterator.ws.find(jmin)].colsol_take;
#else
								int i = colsol[jmin];
#endif
								if (f < dim)
								{
									// get row
									auto tt = iterator.getRow(t, i);
									// initialize Search
									initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], dim2);
									// single device
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(tt_jmin, &(tt[jmin - start]), sizeof(TC), cudaMemcpyDeviceToHost, stream);
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
										cudaEventRecord(event, stream);
										cudaEventSynchronize(event);
#else
										cudaStreamSynchronize(stream);
#endif
										h2 = tt_jmin[0] - v_jmin[0] - min;
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearch_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], tt, colactive_private[t], pred_private[t], i, h2, size);
								}
								else
								{
									// initialize Search
									initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], dim2);
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
										cudaEventRecord(event, stream);
										cudaEventSynchronize(event);
#else
										cudaStreamSynchronize(stream);
#endif
										h2 = -v_jmin[0] - min;
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearch_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], d_private[t], colactive_private[t], pred_private[t], i, h2, size);
								}
								// find minimum
								if (temp_storage_bytes[t] == 0)
								{
									cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
									cudaMalloc(&(temp_storage[t]), temp_storage_bytes[t]);
								}
								cub::DeviceReduce::Min(temp_storage[t], temp_storage_bytes[t], d_private[t], &(min_private[t]->min), size, stream);
								// min is now set so we need to find the correspoding minima for free and taken columns
								findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
								cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
								cudaEventRecord(event, stream);
								cudaEventSynchronize(event);
#else
								cudaStreamSynchronize(stream);
#endif
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
									scancount[i]++;
#endif
									count++;

									jmin_free = dim2;
									jmin_taken = dim2;
									min_n = std::numeric_limits<SC>::max();
									for (int t = 0; t < devices; t++)
									{
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

									min = std::max(min, min_n);

									// dijkstraCheck
									if (jmin_free < dim2)
									{
										jmin = jmin_free;
										endofpath = jmin;
										unassignedfound = true;
									}
									else
									{
										jmin = jmin_taken;
										unassignedfound = false;
									}
								}
#pragma omp barrier
							}

							// update column prices. can increase or decrease
							// need to copy pred from GPUs here
#if (!defined LAP_CUDA_AVOID_MEMCPY) && (!defined LAP_CUDA_LOCAL_ROWSOL)
							cudaMemcpyAsync(&(pred[start]), pred_private[t], sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
#endif
							if (epsilon > SC(0))
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d2_private[t], epsilon, size);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d2_private[t], size);
							}
							// synchronization only required due to copy
#if (!defined LAP_CUDA_AVOID_MEMCPY) && (!defined LAP_CUDA_LOCAL_ROWSOL)
#ifdef LAP_CUDA_EVENT_SYNC
							cudaEventRecord(event, stream);
							cudaEventSynchronize(event);
#else
							cudaStreamSynchronize(stream);
#endif
#pragma omp barrier
#endif
							// reset row and column assignments along the alternating path.
#ifdef LAP_CUDA_LOCAL_ROWSOL
							{
								if ((endofpath >= start) && (endofpath < end))
								{
									resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], host_reset, start, end, endofpath, f);
#ifdef LAP_CUDA_EVENT_SYNC
									cudaEventRecord(event, stream);
									cudaEventSynchronize(event);
#else
									cudaStreamSynchronize(stream);
#endif
								}
#pragma omp barrier
								while (host_reset->i != -1)
								{
									resetRowColumnAssignmentContinue_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], &(host_reset[0]), &(host_reset[1]), start, end, f);
#ifdef LAP_CUDA_EVENT_SYNC
									cudaEventRecord(event, stream);
									cudaEventSynchronize(event);
#else
									cudaStreamSynchronize(stream);
#endif
#pragma omp barrier
#pragma omp master
									{
										std::swap(host_reset[0], host_reset[1]);
									}
#pragma omp barrier
								}
							}
#ifndef LAP_QUIET
#pragma omp master
							{
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
							}
#endif
#else
#pragma omp master
							{
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
#endif
#pragma omp barrier
						}

						// download updated v
						cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_CUDA_EVENT_SYNC
						cudaEventRecord(event, stream);
						cudaEventSynchronize(event);
#else
						cudaStreamSynchronize(stream);
#endif
					} // end of #pragma omp parallel
				}

				// now we need to compare the previous v with the new one to get back all the changes
				SC total(0);
				SC scaled_epsilon = epsilon * SC(count) / SC(dim2);

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

#ifdef LAP_CUDA_LOCAL_ROWSOL
			{
				int *colsol;
				cudaMallocHost(&colsol, dim2 * sizeof(int));
#pragma omp parallel for
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(&(colsol[start]), colsol_private[t], sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
					cudaStreamSynchronize(stream);
				}
				for (int j = 0; j < dim2; j++) rowsol[colsol[j]] = j;
				cudaFreeHost(colsol);
			}
#endif
			// free CUDA memory
#pragma omp parallel for
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(min_private[t]);
				cudaFree(colactive_private[t]);
				cudaFree(d_private[t]);
				cudaFree(v_private[t]);
				if (temp_storage[t] != 0) cudaFree(temp_storage[t]);
#ifndef LAP_CUDA_AVOID_MEMCPY
				cudaFree(pred_private[t]);
				cudaFree(colsol_private[t]);
#endif
#ifdef LAP_CUDA_LOCAL_ROWSOL
				cudaFree(rowsol_private[t]);
#endif
			}

			// free reserved memory.
#ifdef LAP_CUDA_LOCAL_ROWSOL
			cudaFreeHost(host_reset);
#else
			cudaFreeHost(pred);
			cudaFreeHost(colsol);
#endif
			cudaFreeHost(v);
			cudaFreeHost(v2);
			cudaFreeHost(tt_jmin);
			cudaFreeHost(v_jmin);
			lapFree(min_private);
			cudaFreeHost(host_min_private);
			lapFree(colactive_private);
			lapFree(pred_private);
			lapFree(d_private);
			lapFree(colsol_private);
#ifdef LAP_CUDA_LOCAL_ROWSOL
			lapFree(rowsol_private);
#endif
			lapFree(v_private);
			lapFree(temp_storage_bytes);
			lapFree(temp_storage);
			omp_set_num_threads(old_threads);
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
