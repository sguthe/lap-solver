#pragma once

#include "../lap_solver.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef LAP_CUDA_OPENMP
#include <omp.h>
#endif

#include "lap_kernel.cuh"

namespace lap
{
	namespace cuda
	{
		template <class SC>
		class estimateEpsilon_struct
		{
		public:
			SC min;
			SC max;
			SC picked;
			int jmin;
			SC v_jmin;
		};

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			SC max;
			int jmin;
			int colsol;
		};

		template <class SC, class MS>
		void findMaximum(SC *v_private, MS *max_struct, cudaStream_t &stream, int min_count)
		{
			// normalize v
			if (min_count <= 32) findMaxSmall_kernel<<<1, 32, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else if (min_count <= 256) findMaxMedium_kernel<<<1, 256, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else findMaxLarge_kernel<<<1, 1024, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
		}

		template <class SC, class MS>
		SC mergeMaximum(MS *max_struct, int devices)
		{
			SC max_cost = max_struct[0].max;
			for (int tx = 1; tx < devices; tx++)
			{
				max_cost = std::max(max_cost, max_struct[tx].max);
			}
			return max_cost;
		}

		int getMinSize(int num_items)
		{
			if (num_items <= 1024) return (num_items + 31) >> 5;
			else if (num_items <= 65536) return (num_items + 255) >> 8;
			else return (num_items + 1023) >> 10;
		}

		template <class SC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC **v_private)
		{
			SC *mod_v;
			SC **mod_v_private;
			SC **min_cost_private;
			SC **max_cost_private;
			SC **picked_cost_private;
			int **jmin_private;
			int *perm;
			int **picked_private;
			int *picked;
			estimateEpsilon_struct<SC> *host_struct_private;
			unsigned int **semaphore_private;

			int devices = (int)iterator.ws.device.size();
#ifdef LAP_CUDA_OPENMP
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#endif

			cudaMallocHost(&mod_v, dim2 * sizeof(SC));
			lapAlloc(mod_v_private, devices, __FILE__, __LINE__);
			lapAlloc(min_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(max_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(perm, dim, __FILE__, __LINE__);
			lapAlloc(picked_private, devices, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			lapAlloc(semaphore_private, devices, __FILE__, __LINE__);
			cudaMallocHost(&host_struct_private, devices * sizeof(estimateEpsilon_struct<SC>));

			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (num_items + block_size.x - 1) / block_size.x;
				int count = getMinSize(num_items);
				cudaStream_t stream = iterator.ws.stream[t];
				cudaMalloc(&(mod_v_private[t]), num_items * sizeof(SC));
				cudaMalloc(&(picked_private[t]), num_items * sizeof(int));
				cudaMalloc(&(semaphore_private[t]), sizeof(unsigned int));
				cudaMalloc(&(min_cost_private[t]), count * sizeof(SC));
				cudaMalloc(&(max_cost_private[t]), count * sizeof(SC));
				cudaMalloc(&(picked_cost_private[t]), count * sizeof(SC));
				cudaMalloc(&(jmin_private[t]), count * sizeof(int));
				cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
				cudaMemsetAsync(semaphore_private[t], 0, sizeof(unsigned int), stream);
			}

			SC lower_bound = SC(0);
			SC greedy_bound = SC(0);
			SC upper_bound = SC(0);

			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				cudaStream_t stream = iterator.ws.stream[0];
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (dim2 + block_size.x - 1) / block_size.x;

				for (int i = 0; i < dim; i++)
				{
					auto *tt = iterator.getRow(0, i);

					if (num_items <= 1024) getMinMaxBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (num_items <= 65536) getMinMaxBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinMaxBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);

					checkCudaErrors(cudaStreamSynchronize(stream));

					SC min_cost = host_struct_private[0].min;
					int jmin = host_struct_private[0].jmin;

					if (i == 0) updateEstimatedVFirst_kernel<<<grid_size, block_size, 0, stream>>>(mod_v_private[0], tt, picked_private[0], min_cost, jmin, dim2);
					else if (i == 1) updateEstimatedVSecond_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], mod_v_private[0], tt, picked_private[0], min_cost, jmin, dim2);
					else updateEstimatedV_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], mod_v_private[0], tt, picked_private[0], min_cost, jmin, dim2);

					lower_bound += min_cost;
					upper_bound += host_struct_private[0].max;
					greedy_bound += host_struct_private[0].picked;
				}
				findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
				subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
			}
			else
			{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(t, i);

						if (num_items <= 1024) getMinMaxBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
						else if (num_items <= 65536) getMinMaxBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
						else getMinMaxBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC min_cost = host_struct_private[0].min;
						SC max_cost = host_struct_private[0].max;
						SC picked_cost = host_struct_private[0].picked;
						int jmin = host_struct_private[0].jmin;

						for (int tx = 1; tx < devices; tx++)
						{
							if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
							{
								picked_cost = host_struct_private[tx].picked;
								jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
							}
							min_cost = std::min(min_cost, host_struct_private[tx].min);
							max_cost = std::max(max_cost, host_struct_private[tx].max);
						}

						if (i == 0) updateEstimatedVFirst_kernel<<<grid_size, block_size, 0, stream>>>(mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else if (i == 1) updateEstimatedVSecond_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else updateEstimatedV_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);

						if (t == 0)
						{
							lower_bound += min_cost;
							upper_bound += max_cost;
							greedy_bound += picked_cost;
						}
					}
					findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					SC max_v = mergeMaximum<SC>(host_struct_private, devices);
					subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
				}
#else
				for (int i = 0; i < dim; i++)
				{
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto *tt = iterator.getRow(t, i);
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						if (num_items <= 1024) getMinMaxBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
						else if (num_items <= 65536) getMinMaxBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
						else getMinMaxBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					}

					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

					SC min_cost = host_struct_private[0].min;
					SC max_cost = host_struct_private[0].max;
					SC picked_cost = host_struct_private[0].picked;
					int jmin = host_struct_private[0].jmin;

					for (int tx = 1; tx < devices; tx++)
					{
						if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
						{
							picked_cost = host_struct_private[tx].picked;
							jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
						}
						min_cost = std::min(min_cost, host_struct_private[tx].min);
						max_cost = std::max(max_cost, host_struct_private[tx].max);
					}

					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto *tt = iterator.getRow(t, i);
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;

						if (i == 0) updateEstimatedVFirst_kernel<<<grid_size, block_size, 0, stream>>>(mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else if (i == 1) updateEstimatedVSecond_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
						else updateEstimatedV_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], mod_v_private[t], tt, picked_private[t], min_cost, jmin - iterator.ws.part[t].first, num_items);
					}

					lower_bound += min_cost;
					upper_bound += max_cost;
					greedy_bound += picked_cost;
				}
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
				}
				for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
				SC max_v = mergeMaximum<SC>(host_struct_private, devices);
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
				}
#endif
			}

			greedy_bound = std::min(greedy_bound, upper_bound);

			SC initial_gap = upper_bound - lower_bound;
			SC greedy_gap = greedy_bound - lower_bound;

#ifdef LAP_DEBUG
			lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap << std::endl;
			lapDebug << "  upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

			SC upper = std::numeric_limits<SC>::max();
			SC lower;

#ifdef LAP_CUDA_OPENMP
			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				cudaStream_t stream = iterator.ws.stream[0];
				cudaMemsetAsync(picked_private[0], 0, num_items * sizeof(int), stream);
			}
			else
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
				}
			}
#else
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				cudaStream_t stream = iterator.ws.stream[t];
				cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
			}
#endif

			lower_bound = SC(0);
			upper_bound = SC(0);

			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				cudaStream_t stream = iterator.ws.stream[0];
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (dim2 + block_size.x - 1) / block_size.x;

				int last_jmin = dim2;
				for (int i = dim - 1; i >= 0; --i)
				{
					auto *tt = iterator.getRow(0, i);

					if (num_items <= 1024) getMinSecondBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], last_jmin, std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (num_items <= 65536) getMinSecondBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], last_jmin, std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinSecondBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], last_jmin, std::numeric_limits<SC>::max(), i, num_items, dim2);
					checkCudaErrors(cudaStreamSynchronize(stream));

					SC min_cost = host_struct_private[0].min;
					SC second_cost = host_struct_private[0].max;
					SC picked_cost = host_struct_private[0].picked;
					int jmin = host_struct_private[0].jmin;
					SC v_jmin = host_struct_private[0].v_jmin;

					if (i == 0)
					{
						cudaMemsetAsync(&(picked_private[0][jmin]), 1, 1, stream);
					}
					else
					{
						last_jmin = jmin;
					}
					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
			}
			else
			{
				int last_jmin = dim2;
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int num_items = iterator.ws.part[t].second - start;
					cudaStream_t stream = iterator.ws.stream[t];
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;
					
					for (int i = dim - 1; i >= 0; --i)
					{
						auto *tt = iterator.getRow(t, i);
#pragma omp barrier
						if (num_items <= 1024) getMinSecondBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], last_jmin - start, std::numeric_limits<SC>::max(), i, num_items, dim2);
						else if (num_items <= 65536) getMinSecondBestMedium_kernel<<< (num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], last_jmin - start, std::numeric_limits<SC>::max(), i, num_items, dim2);
						else getMinSecondBestLarge_kernel<<< (num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[0]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], last_jmin - start, std::numeric_limits<SC>::max(), i, num_items, dim2);
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC min_cost = host_struct_private[0].min;
						SC second_cost = host_struct_private[0].max;
						SC picked_cost = host_struct_private[0].picked;
						int jmin = host_struct_private[0].jmin;
						SC v_jmin = host_struct_private[0].v_jmin;

						for (int tx = 1; tx < devices; tx++)
						{
							if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
							{
								picked_cost = host_struct_private[tx].picked;
								jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
								v_jmin = host_struct_private[tx].v_jmin;
							}
							if (host_struct_private[tx].min < min_cost)
							{
								second_cost = std::min(min_cost, host_struct_private[tx].max);
								min_cost = host_struct_private[tx].min;
							}
							else
							{
								second_cost = std::min(second_cost, host_struct_private[tx].min);
							}
						}

						if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
						{
							if (i == 0)
							{
								cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, 1, stream);
							}
							else
							{
								last_jmin = jmin;
							}
						}
						if (t == 0)
						{
							perm[i] = i;
							mod_v[i] = second_cost - min_cost;
							// need to use the same v values in total
							lower_bound += min_cost + v_jmin;
							upper_bound += picked_cost + v_jmin;
						}
					}
				}
#else
				for (int i = dim - 1; i >= 0; --i)
				{
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto *tt = iterator.getRow(t, i);
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						if (num_items <= 1024) getMinSecondBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], last_jmin - start, std::numeric_limits<SC>::max(), i, num_items, dim2);
						else if (num_items <= 65536) getMinSecondBestMedium_kernel<<< (num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], last_jmin - start, std::numeric_limits<SC>::max(), i, num_items, dim2);
						else getMinSecondBestLarge_kernel<<< (num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[0]), semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], last_jmin - start, std::numeric_limits<SC>::max(), i, num_items, dim2);
					}

					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

					SC min_cost = host_struct_private[0].min;
					SC second_cost = host_struct_private[0].max;
					SC picked_cost = host_struct_private[0].picked;
					int jmin = host_struct_private[0].jmin;
					SC v_jmin = host_struct_private[0].v_jmin;

					for (int tx = 1; tx < devices; tx++)
					{
						if ((host_struct_private[tx].picked < picked_cost) || ((host_struct_private[tx].picked == picked_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
						{
							picked_cost = host_struct_private[tx].picked;
							jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
							v_jmin = host_struct_private[tx].v_jmin;
						}
						if (host_struct_private[tx].min < min_cost)
						{
							second_cost = std::min(min_cost, host_struct_private[tx].max);
							min_cost = host_struct_private[tx].min;
						}
						else
						{
							second_cost = std::min(second_cost, host_struct_private[tx].min);
						}
					}

					if (i == 0)
					{
						for (int t = 0; t < devices; t++) if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
						{
							cudaSetDevice(iterator.ws.device[t]);
							cudaStream_t stream = iterator.ws.stream[t];
							cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, 1, stream);
						}
					}
					else
					{
						last_jmin = jmin;
					}

					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
#endif
			}
			upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

			if (initial_gap < SC(4) * greedy_gap)
			{
				// sort permutation by keys
				std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

				lower_bound = SC(0);
				upper_bound = SC(0);
				// greedy search
				if (devices == 1)
				{
					cudaSetDevice(iterator.ws.device[0]);
					cudaStream_t stream = iterator.ws.stream[0];
					int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;

					cudaMemsetAsync(picked_private[0], 0, dim2 * sizeof(int), stream);

					int last_jmin = dim2;

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(0, perm[i]);

						if (num_items <= 1024) getMinimalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], last_jmin, std::numeric_limits<SC>::max(), num_items, dim2);
						else if (num_items <= 65536) getMinimalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], last_jmin, std::numeric_limits<SC>::max(), num_items, dim2);
						else getMinimalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], last_jmin, std::numeric_limits<SC>::max(), num_items, dim2);

						checkCudaErrors(cudaStreamSynchronize(stream));

						SC min_cost = host_struct_private[0].picked;
						SC min_cost_real = host_struct_private[0].min;
						int jmin = host_struct_private[0].jmin;
						SC v_jmin = host_struct_private[0].v_jmin;

						upper_bound += min_cost + v_jmin;
						// need to use the same v values in total
						lower_bound += min_cost_real + v_jmin;

						if (i + 1 == dim)
						{
							cudaMemsetAsync(&(picked_private[0][jmin]), 1, sizeof(int), stream);
						}
						else
						{
							last_jmin = jmin;
						}

						picked[i] = jmin;
					}
				}
				else
				{
					int last_jmin = dim2;
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;

						cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);

						for (int i = 0; i < dim; i++)
						{
#pragma omp barrier
							auto *tt = iterator.getRow(t, perm[i]);

							if (num_items <= 1024) getMinimalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], last_jmin - iterator.ws.part[t].first, std::numeric_limits<SC>::max(), num_items, dim2);
							else if (num_items <= 65536) getMinimalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], last_jmin - iterator.ws.part[t].first, std::numeric_limits<SC>::max(), num_items, dim2);
							else getMinimalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], last_jmin - iterator.ws.part[t].first, std::numeric_limits<SC>::max(), num_items, dim2);

							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC min_cost = host_struct_private[0].picked;
							SC min_cost_real = host_struct_private[0].min;
							int jmin = host_struct_private[0].jmin;
							SC v_jmin = host_struct_private[0].v_jmin;
							for (int tx = 1; tx < devices; tx++)
							{
								if ((host_struct_private[tx].picked < min_cost) || ((host_struct_private[tx].picked == min_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
								{
									min_cost = host_struct_private[tx].picked;
									jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
									v_jmin = host_struct_private[tx].v_jmin;
								}
								min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
							}
							if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
							{
								if (i + 1 == dim)
								{
									cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, sizeof(int), stream);
								}
								else
								{
									last_jmin = jmin;
								}
							}
							if (t == 0)
							{
								upper_bound += min_cost + v_jmin;
								// need to use the same v values in total
								lower_bound += min_cost_real + v_jmin;
								picked[i] = jmin;
							}
						}
					}
#else
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];

						cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream);
					}
					for (int i = 0; i < dim; i++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (num_items + block_size.x - 1) / block_size.x;

							auto *tt = iterator.getRow(t, perm[i]);

							if (num_items <= 1024) getMinimalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], last_jmin - iterator.ws.part[t].first, std::numeric_limits<SC>::max(), num_items, dim2);
							else if (num_items <= 65536) getMinimalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], last_jmin - iterator.ws.part[t].first, std::numeric_limits<SC>::max(), num_items, dim2);
							else getMinimalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], last_jmin - iterator.ws.part[t].first, std::numeric_limits<SC>::max(), num_items, dim2);
						}

						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

						SC min_cost = host_struct_private[0].picked;
						SC min_cost_real = host_struct_private[0].min;
						int jmin = host_struct_private[0].jmin;
						SC v_jmin = host_struct_private[0].v_jmin;
						for (int tx = 1; tx < devices; tx++)
						{
							if ((host_struct_private[tx].picked < min_cost) || ((host_struct_private[tx].picked == min_cost) && (host_struct_private[tx].jmin + iterator.ws.part[tx].first < jmin)))
							{
								min_cost = host_struct_private[tx].picked;
								jmin = host_struct_private[tx].jmin + iterator.ws.part[tx].first;
								v_jmin = host_struct_private[tx].v_jmin;
							}
							min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
						}

						upper_bound += min_cost + v_jmin;
						// need to use the same v values in total
						lower_bound += min_cost_real + v_jmin;
						if (i + 1 == dim)
						{
							for (int t = 0; t < devices; t++)
							{
								if ((jmin >= iterator.ws.part[t].first) && (jmin < iterator.ws.part[t].second))
								{
									cudaSetDevice(iterator.ws.device[t]);
									cudaStream_t stream = iterator.ws.stream[t];

									cudaMemsetAsync(&(picked_private[t][jmin - iterator.ws.part[t].first]), 1, sizeof(int), stream);
								}
							}
						}
						else
						{
							last_jmin = jmin;
						}
						picked[i] = jmin;
					}
#endif
				}
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

				if (devices == 1)
				{
					cudaSetDevice(iterator.ws.device[0]);
					int start = iterator.ws.part[0].first;
					int end = iterator.ws.part[0].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[0];
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (size + block_size.x - 1) / block_size.x;
					for (int i = dim - 1; i >= 0; --i)
					{
						auto *tt = iterator.getRow(0, perm[i]);
						updateVSingle_kernel<<<grid_size, block_size, 0, stream>>>(tt, v_private[0], picked_private[0], picked[i], dim2);
					}
					findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
					subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						for (int i = dim - 1; i >= 0; --i)
						{
							auto *tt = iterator.getRow(t, perm[i]);
							if ((picked[i] >= start) && (picked[i] < end))
							{
								updateVMultiStart_kernel<<<1, 1, 0, stream>>>(tt, v_private[t], picked_private[t], &(mod_v[i]), picked[i] - start);
							}
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							updateVMulti_kernel<<<grid_size, block_size, 0, stream>>>(tt, v_private[t], picked_private[t], mod_v[i], size);
						}

						findMaximum(v_private[t], &(host_struct_private[t]), stream, size);
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC max_v = mergeMaximum<SC>(host_struct_private, devices);
						subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, size);
					}
#else
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						findMaximum(v_private[t], &(host_struct_private[t]), stream, size);
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
					SC max_v = mergeMaximum<SC>(host_struct_private, devices);
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;
						subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
					}
#endif
				}

				SC old_upper_bound = upper_bound;
				SC old_lower_bound = lower_bound;
				upper_bound = SC(0);
				lower_bound = SC(0);
				if (devices == 1)
				{
					cudaSetDevice(iterator.ws.device[0]);
					cudaStream_t stream = iterator.ws.stream[0];
					int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (num_items + block_size.x - 1) / block_size.x;

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(0, perm[i]);

						if (num_items <= 1024) getFinalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						else if (num_items <= 65536) getFinalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						else getFinalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[0]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);

						checkCudaErrors(cudaStreamSynchronize(stream));

						SC picked_cost = host_struct_private[0].picked;
						SC v_picked = host_struct_private[0].v_jmin;
						SC min_cost_real = host_struct_private[0].min;

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (num_items + block_size.x - 1) / block_size.x;

						for (int i = 0; i < dim; i++)
						{
#pragma omp barrier
							auto *tt = iterator.getRow(t, perm[i]);

							if (num_items <= 1024) getFinalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							else if (num_items <= 65536) getFinalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							else getFinalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);

							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC picked_cost = host_struct_private[0].picked;
							SC v_picked = host_struct_private[0].v_jmin;
							SC min_cost_real = host_struct_private[0].min;
							for (int tx = 1; tx < devices; tx++)
							{
								picked_cost = std::min(picked_cost, host_struct_private[tx].picked);
								v_picked = std::min(v_picked, host_struct_private[tx].v_jmin);
								min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
							}

							if (t == 0)
							{
								// need to use all picked v for the lower bound as well
								upper_bound += picked_cost;
								lower_bound += min_cost_real + v_picked;
							}
						}
					}
#else
					for (int i = 0; i < dim; i++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (num_items + block_size.x - 1) / block_size.x;

							auto *tt = iterator.getRow(t, perm[i]);

							if (num_items <= 1024) getFinalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							else if (num_items <= 65536) getFinalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							else getFinalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt, v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
						}

						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));

						SC picked_cost = host_struct_private[0].picked;
						SC v_picked = host_struct_private[0].v_jmin;
						SC min_cost_real = host_struct_private[0].min;
						for (int tx = 1; tx < devices; tx++)
						{
							picked_cost = std::min(picked_cost, host_struct_private[tx].picked);
							v_picked = std::min(v_picked, host_struct_private[tx].v_jmin);
							min_cost_real = std::min(min_cost_real, host_struct_private[tx].min);
						}

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
#endif
				}
				upper_bound = std::min(upper_bound, old_upper_bound);
				lower_bound = std::max(lower_bound, old_lower_bound);
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif
			}

			getUpperLower(upper, lower, greedy_gap, initial_gap, dim2);

			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(mod_v_private[t]);
				cudaFree(picked_private[t]);
				cudaFree(semaphore_private[t]);
				cudaFree(min_cost_private[t]);
				cudaFree(max_cost_private[t]);
				cudaFree(picked_cost_private[t]);
				cudaFree(jmin_private[t]);
			}

			cudaFreeHost(mod_v);
			lapFree(mod_v_private);
			lapFree(min_cost_private);
			lapFree(max_cost_private);
			lapFree(picked_cost_private);
			lapFree(jmin_private);
			lapFree(perm);
			lapFree(picked_private);
			lapFree(picked);
			lapFree(semaphore_private);
			cudaFreeHost(host_struct_private);


#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#endif
			return std::pair<SC, SC>((SC)upper, (SC)lower);
		}

		template <class SC, class TC, class CF, class I>
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
#ifdef LAP_DEBUG
			SC *v;
#endif
			SC *h_total_d;
			SC *h_total_eps;
			// for calculating h2
			TC *tt_jmin;
			SC *v_jmin;

			int devices = (int)iterator.ws.device.size();
#ifdef LAP_CUDA_OPENMP
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#else
			const TC **tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);
#endif

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

#ifdef LAP_CUDA_COMPARE_CPU
			SC *d_tmp;
			unsigned char *colactive_tmp;
			lapAlloc(d_tmp, dim2, __FILE__, __LINE__);
			lapAlloc(colactive_tmp, dim2, __FILE__, __LINE__);
#endif

			// used for copying
			min_struct<SC> *host_min_private;
#ifdef LAP_DEBUG
			cudaMallocHost(&v, dim2 * sizeof(SC));
#endif
			cudaMallocHost(&h_total_d, dim2 * sizeof(SC));
			cudaMallocHost(&h_total_eps, dim2 * sizeof(SC));
			cudaMallocHost(&host_min_private, devices * sizeof(min_struct<SC>));
			cudaMallocHost(&tt_jmin, sizeof(TC));
			cudaMallocHost(&v_jmin, sizeof(SC));

			SC **min_private;
			int **jmin_private;
			int **csol_private;
			char **colactive_private;
			int **pred_private;
			SC **d_private;
			int *pred;
			int *colsol;
			int **colsol_private;
			SC **v_private;
			SC **total_eps_private;
			SC **total_d_private;
			unsigned int **semaphore_private;
			// on device
			lapAlloc(min_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(csol_private, devices, __FILE__, __LINE__);
			lapAlloc(colactive_private, devices, __FILE__, __LINE__);
			lapAlloc(pred_private, devices, __FILE__, __LINE__);
			lapAlloc(d_private, devices, __FILE__, __LINE__);
			lapAlloc(colsol_private, devices, __FILE__, __LINE__);
			lapAlloc(v_private, devices, __FILE__, __LINE__);
			lapAlloc(total_d_private, devices, __FILE__, __LINE__);
			lapAlloc(total_eps_private, devices, __FILE__, __LINE__);
			lapAlloc(semaphore_private, devices, __FILE__, __LINE__);
			cudaMallocHost(&colsol, sizeof(int) * dim2);
			cudaMallocHost(&pred, sizeof(int) * dim2);

			for (int t = 0; t < devices; t++)
			{
				// single device code
				cudaSetDevice(iterator.ws.device[t]);
				int start = iterator.ws.part[t].first;
				int end = iterator.ws.part[t].second;
				int size = end - start;
				cudaStream_t stream = iterator.ws.stream[t];
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (size + block_size.x - 1) / block_size.x;
				int count = getMinSize(size);
				cudaMalloc(&(min_private[t]), sizeof(SC) * count);
				cudaMalloc(&(jmin_private[t]), sizeof(int) * count);
				cudaMalloc(&(csol_private[t]), sizeof(int) * count);
				cudaMalloc(&(colactive_private[t]), sizeof(char) * size);
				cudaMalloc(&(d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(v_private[t]), sizeof(SC) * size);
				cudaMalloc(&(total_d_private[t]), sizeof(SC) * size);
				cudaMalloc(&(total_eps_private[t]), sizeof(SC) * size);
				cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
				cudaMalloc(&(pred_private[t]), sizeof(int) * size);
				cudaMalloc(&(semaphore_private[t]), sizeof(int));
				if (!use_epsilon) cudaMemsetAsync(v_private[t], 0, sizeof(SC) * size, stream);
				cudaMemsetAsync(semaphore_private[t], 0, sizeof(unsigned int), stream);
			}

			SC epsilon_upper, epsilon_lower;

			if (use_epsilon)
			{
				std::pair<SC, SC> eps = estimateEpsilon(dim, dim2, iterator, v_private);
				epsilon_upper = eps.first;
				epsilon_lower = eps.second;
			}
			else
			{
				epsilon_upper = SC(0);
				epsilon_lower = SC(0);
			}


#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			SC epsilon = epsilon_upper;

			bool first = true;
			bool second = false;

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= SC(0))
			{
#ifdef LAP_DEBUG
				if (first)
				{
#ifdef LAP_CUDA_OPENMP
					if (devices == 1)
					{
						cudaSetDevice(iterator.ws.device[0]);
						int start = iterator.ws.part[0].first;
						int end = iterator.ws.part[0].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[0];
						cudaMemcpyAsync(&(v[start]), v_private[0], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						checkCudaErrors(cudaStreamSynchronize(stream));
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
							cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
							checkCudaErrors(cudaStreamSynchronize(stream));
						}
					}
#else
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#endif
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
				memset(colsol, -1, dim2 * sizeof(int));

#ifdef LAP_CUDA_OPENMP
				if (devices == 1)
				{
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
				}
				else
				{
#pragma omp parallel for
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * size, stream);
						cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * size, stream);
						cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
					}
				}
#else
				for (int t = 0; t < devices; t++)
				{
					// upload v to devices
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * size, stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
				}
#endif

				int jmin, colsol_old;
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

				int dim_limit = ((epsilon > SC(0)) && (first)) ? dim : dim2;

				bool require_colsol_copy = false;

				if (devices == 1)
				{
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[t];

					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (size + block_size.x - 1) / block_size.x;

					for (int f = 0; f < dim_limit; f++)
					{
						// start search and find minimum value
						if (require_colsol_copy)
						{
							if (f < dim)
							{
								auto tt = iterator.getRow(t, f);
								if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							}
							else
							{
								if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							}
							require_colsol_copy = false;
						}
						else
						{
							if (f < dim)
							{
								auto tt = iterator.getRow(t, f);
								if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							}
							else
							{
								if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
							}
						}
						// min is now set so we need to find the correspoding minima for free and taken columns
						checkCudaErrors(cudaStreamSynchronize(stream));
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
						jmin = host_min_private[t].jmin;
						colsol_old = host_min_private[t].colsol;

						// dijkstraCheck
						if (colsol_old < 0)
						{
							endofpath = jmin;
							unassignedfound = true;
						}
						else
						{
							unassignedfound = false;
						}

#ifdef LAP_CUDA_COMPARE_CPU
						{
							cudaMemcpyAsync(d_tmp, d_private[t], dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream);
							cudaMemcpyAsync(colactive_tmp, colactive_private[t], dim2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
							checkCudaErrors(cudaStreamSynchronize(stream));
							SC min_tmp = std::numeric_limits<SC>::max();
							int jmin_tmp = dim2;
							int colsol_old_tmp = 0;
							for (int j = 0; j < dim2; j++)
							{
								if (colactive_tmp[j] != 0)
								{
									if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
									{
										min_tmp = d_tmp[j];
										jmin_tmp = j;
										colsol_old_tmp = colsol[j];
									}
								}
							}
							if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
							{
								std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
							}
						}
#endif

						if (f >= dim) markedSkippedColumns_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, size);

						bool fast = unassignedfound;

						while (!unassignedfound)
						{
							int i = colsol_old;
							if (i < dim)
							{
								// get row
								auto tt = iterator.getRow(t, i);
								// continue search
								if (size <= 1024) continueSearchJMinMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
								else if (size <= 65536) continueSearchJMinMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
								else continueSearchJMinMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
							}
							else
							{
								// continue search
								if (size <= 1024) continueSearchJMinMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
								else if (size <= 65536) continueSearchJMinMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
								else continueSearchJMinMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size, dim2);
							}
							// min is now set so we need to find the correspoding minima for free and taken columns
							checkCudaErrors(cudaStreamSynchronize(stream));
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
							count++;

							min_n = host_min_private[t].min;
							jmin = host_min_private[t].jmin;
							colsol_old = host_min_private[t].colsol;

							min = std::max(min, min_n);

							// dijkstraCheck
							if (colsol_old < 0)
							{
								endofpath = jmin;
								unassignedfound = true;
							}
							else
							{
								unassignedfound = false;
							}

#ifdef LAP_CUDA_COMPARE_CPU
							{
								cudaMemcpyAsync(d_tmp, d_private[t], dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream);
								cudaMemcpyAsync(colactive_tmp, colactive_private[t], dim2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
								checkCudaErrors(cudaStreamSynchronize(stream));
								SC min_tmp = std::numeric_limits<SC>::max();
								int jmin_tmp = dim2;
								int colsol_old_tmp = 0;
								for (int j = 0; j < dim2; j++)
								{
									if (colactive_tmp[j] != 0)
									{
										if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
										{
											min_tmp = d_tmp[j];
											jmin_tmp = j;
											colsol_old_tmp = colsol[j];
										}
									}
								}
								if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
								{
									std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
								}
							}
#endif

							if (i >= dim) markedSkippedColumnsUpdate_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, size);
						}

						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							if (epsilon > SC(0))
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
						}
						else
						{
							// update column prices. can increase or decrease
							if (epsilon > SC(0))
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred, pred_private[t], size);
							}
							else
							{
								updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred, pred_private[t], size);
							}
							// reset row and column assignments along the alternating path.
							checkCudaErrors(cudaStreamSynchronize(stream));
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
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
							//cudaMemcpyAsync(colsol_private[t], colsol, dim2 * sizeof(int), cudaMemcpyHostToDevice, stream);
							require_colsol_copy = true;
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
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
					}

					if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, size);

					// download updated v
					cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_DEBUG
					cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#endif
					checkCudaErrors(cudaStreamSynchronize(stream));
				}
				else
				{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
					{
						int t = omp_get_thread_num();
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];

						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;

						for (int f = 0; f < dim_limit; f++)
						{
							// start search and find minimum value
							if (require_colsol_copy)
							{
								if (f < dim)
								{
									auto tt = iterator.getRow(t, f);
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							else
							{
								if (f < dim)
								{
									auto tt = iterator.getRow(t, f);
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							// min is now set so we need to find the correspoding minima for free and taken columns
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							if (t == 0)
							{
								require_colsol_copy = false;
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
								min = host_min_private[0].min;
								jmin = host_min_private[0].jmin;
								colsol_old = host_min_private[0].colsol;
								for (int t = 1; t < devices; t++)
								{
									if ((host_min_private[t].min < min) || ((host_min_private[t].min == min) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
									{
										min = host_min_private[t].min;
										jmin = host_min_private[t].jmin + iterator.ws.part[t].first;
										colsol_old = host_min_private[t].colsol;
									}
								}

								// dijkstraCheck
								if (colsol_old < 0)
								{
									endofpath = jmin;
									unassignedfound = true;
								}
								else
								{
									unassignedfound = false;
								}
							}
#pragma omp barrier
#ifdef LAP_CUDA_COMPARE_CPU
							{
								cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
								cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
								if (t == 0)
								{
									SC min_tmp = std::numeric_limits<SC>::max();
									int jmin_tmp = dim2;
									int colsol_old_tmp = 0;
									for (int j = 0; j < dim2; j++)
									{
										if (colactive_tmp[j] != 0)
										{
											if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
											{
												min_tmp = d_tmp[j];
												jmin_tmp = j;
												colsol_old_tmp = colsol[j];
											}
										}
									}
									if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
									{
										std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
									}
								}
							}
#endif
							// mark last column scanned
							if (f >= dim) markedSkippedColumns_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, size);

							bool fast = unassignedfound;

							while (!unassignedfound)
							{
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
								int i = colsol_old;
								if (i < dim)
								{
									// get row
									auto tt = iterator.getRow(t, i);
									// single device
									if ((jmin >= start) && (jmin < end))
									{
										setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, tt_jmin, &(tt[jmin - start]), v_jmin, &(v_private[t][jmin - start]));
										checkCudaErrors(cudaStreamSynchronize(stream));
									}
									// propagate h2
#pragma omp barrier
									// continue search
									if (size <= 1024) continueSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) continueSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else continueSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if ((jmin >= start) && (jmin < end))
									{
										setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, v_jmin, &(v_private[t][jmin - start]));
										checkCudaErrors(cudaStreamSynchronize(stream));
									}
									// propagate h2
#pragma omp barrier
									// continue search
									if (size <= 1024) continueSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) continueSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else continueSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
								// min is now set so we need to find the correspoding minima for free and taken columns
								checkCudaErrors(cudaStreamSynchronize(stream));
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
									count++;

									min_n = host_min_private[0].min;
									jmin = host_min_private[0].jmin;
									colsol_old = host_min_private[0].colsol;
									for (int t = 1; t < devices; t++)
									{
										if ((host_min_private[t].min < min_n) || ((host_min_private[t].min == min_n) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
										{
											min_n = host_min_private[t].min;
											jmin = host_min_private[t].jmin + iterator.ws.part[t].first;
											colsol_old = host_min_private[t].colsol;
										}
									}

									min = std::max(min, min_n);

									// dijkstraCheck
									if (colsol_old < 0)
									{
										endofpath = jmin;
										unassignedfound = true;
									}
									else
									{
										unassignedfound = false;
									}
								}
#pragma omp barrier
#ifdef LAP_CUDA_COMPARE_CPU
								{
									cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
									cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
									checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
									if (t == 0)
									{
										SC min_tmp = std::numeric_limits<SC>::max();
										int jmin_tmp = dim2;
										int colsol_old_tmp = 0;
										for (int j = 0; j < dim2; j++)
										{
											if (colactive_tmp[j] != 0)
											{
												if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
												{
													min_tmp = d_tmp[j];
													jmin_tmp = j;
													colsol_old_tmp = colsol[j];
												}
											}
										}
										if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
										{
											std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
										}
									}
								}
#endif
								// mark last column scanned
								if (i >= dim) markedSkippedColumnsUpdate_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, size);
							}

							// update column prices. can increase or decrease
							if (fast)
							{
								if ((endofpath >= start) && (endofpath < end))
								{
									colsol[endofpath] = f;
									rowsol[f] = endofpath;
									if (epsilon > SC(0))
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
								}
								else
								{
									if (epsilon > SC(0))
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
									}
								}
							}
							else
							{
								if (epsilon > SC(0))
								{
									updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred + start, pred_private[t], size);
								}
								else
								{
									updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred + start, pred_private[t], size);
								}
								// reset row and column assignments along the alternating path.
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
#ifdef LAP_ROWS_SCANNED
								if (t == 0) {
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
								if (t == 0) resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
								if (t == 0) require_colsol_copy = true;
#pragma omp barrier
							}
#ifndef LAP_QUIET
							if (t == 0)
							{
								int level;
								if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
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
						}

						if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, size);

						// download updated v
						cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_DEBUG
						cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#endif
						checkCudaErrors(cudaStreamSynchronize(stream));
					} // end of #pragma omp parallel
#else
					for (int f = 0; f < dim_limit; f++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (size + block_size.x - 1) / block_size.x;
							// start search and find minimum value
							if (require_colsol_copy)
							{
								if (f < dim)
								{
									auto tt = iterator.getRow(t, f);
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							else
							{
								if (f < dim)
								{
									auto tt = iterator.getRow(t, f);
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
								else
								{
									if (size <= 1024) initializeSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) initializeSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
									else initializeSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
						}
						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
						require_colsol_copy = false;
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
						min = host_min_private[0].min;
						jmin = host_min_private[0].jmin;
						colsol_old = host_min_private[0].colsol;
						for (int t = 1; t < devices; t++)
						{
							if ((host_min_private[t].min < min) || ((host_min_private[t].min == min) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
							{
								min = host_min_private[t].min;
								jmin = host_min_private[t].jmin;
								colsol_old = host_min_private[t].colsol;
							}
						}

						// dijkstraCheck
						if (colsol_old < 0)
						{
							endofpath = jmin;
							unassignedfound = true;
						}
						else
						{
							unassignedfound = false;
						}
						if (f >= dim)
						{
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								markedSkippedColumns_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, size);
							}
						}

#ifdef LAP_CUDA_COMPARE_CPU
						{
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
								cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
							}
							for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
							SC min_tmp = std::numeric_limits<SC>::max();
							int jmin_tmp = dim2;
							int colsol_old_tmp = 0;
							for (int j = 0; j < dim2; j++)
							{
								if (colactive_tmp[j] != 0)
								{
									if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
									{
										min_tmp = d_tmp[j];
										jmin_tmp = j;
										colsol_old_tmp = colsol[j];
									}
								}
							}
							if ((min_tmp != min) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
							{
								std::cout << "initializeSearch: " << min << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
							}
						}
#endif
						bool fast = unassignedfound;
						while (!unassignedfound)
						{
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
							int i = colsol_old;
							if (i < dim)
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									// get row
									tt[t] = iterator.getRow(t, i);
									// initialize Search
									if ((jmin >= start) && (jmin < end))
									{
										setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, tt_jmin, &(tt[jmin - start]), v_jmin, &(v_private[t][jmin - start]));
									}
								}
								// single device
								checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[iterator.ws.find(jmin)]));
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									// continue search
									if (size <= 1024) continueSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) continueSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else continueSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, (SC)tt_jmin[0], v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							else
							{
								{
									int t = iterator.ws.find(jmin);
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									cudaStream_t stream = iterator.ws.stream[t];
									setColInactive_kernel<<<1, 1, 0, stream>>>(colactive_private[t], jmin - start, v_jmin, &(v_private[t][jmin - start]));
									checkCudaErrors(cudaStreamSynchronize(stream));
								}
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									// continue search
									if (size <= 1024) continueSearchMinSmall_kernel<<<(size + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else if (size <= 65536) continueSearchMinMedium_kernel<<<(size + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
									else continueSearchMinLarge_kernel<<<(size + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin[0], min, std::numeric_limits<SC>::max(), size, dim2);
								}
							}
							for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
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
							count++;

							min_n = host_min_private[0].min;
							jmin = host_min_private[0].jmin;
							colsol_old = host_min_private[0].colsol;
							for (int t = 1; t < devices; t++)
							{
								if ((host_min_private[t].min < min_n) || ((host_min_private[t].min == min_n) && (colsol_old >= 0) && (host_min_private[t].colsol < 0)))
								{
									min_n = host_min_private[t].min;
									jmin = host_min_private[t].jmin;
									colsol_old = host_min_private[t].colsol;
								}
							}


							min = std::max(min, min_n);

							// dijkstraCheck
							if (colsol_old < 0)
							{
								endofpath = jmin;
								unassignedfound = true;
							}
							else
							{
								unassignedfound = false;
							}

							// mark last column scanned (single device)
							if (i >= dim)
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									markedSkippedColumnsUpdate_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, size);
								}
							}

#ifdef LAP_CUDA_COMPARE_CPU
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									cudaMemcpyAsync(&(d_tmp[start]), d_private[t], size * sizeof(SC), cudaMemcpyDeviceToHost, stream);
									cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
								}
								for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
								SC min_tmp = std::numeric_limits<SC>::max();
								int jmin_tmp = dim2;
								int colsol_old_tmp = 0;
								for (int j = 0; j < dim2; j++)
								{
									if (colactive_tmp[j] != 0)
									{
										if ((d_tmp[j] < min_tmp) || ((d_tmp[j] == min_tmp) && (colsol[j] < 0) && (colsol_old_tmp >= 0)))
										{
											min_tmp = d_tmp[j];
											jmin_tmp = j;
											colsol_old_tmp = colsol[j];
										}
									}
								}
								if ((min_tmp != min_n) || (jmin_tmp != jmin) || (colsol_old_tmp != colsol_old))
								{
									std::cout << "continueSearch: " << min_n << " " << jmin << " " << colsol_old << " vs. " << min_tmp << " " << jmin_tmp << " " << colsol_old_tmp << std::endl;
								}
							}
#endif
						}

						// update column prices. can increase or decrease
						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								if ((endofpath >= start) && (endofpath < end))
								{
									if (epsilon > SC(0))
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size, &(colsol_private[t][endofpath - start]), colsol[endofpath]);
									}
								}
								else
								{
									if (epsilon > SC(0))
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, size);
									}
									else
									{
										updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], size);
									}
								}
							}
						}
						else
						{
							for (int t = 0; t < devices; t++)
							{
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								int size = end - start;
								cudaStream_t stream = iterator.ws.stream[t];
								dim3 block_size, grid_size;
								block_size.x = 256;
								grid_size.x = (size + block_size.x - 1) / block_size.x;
								if (epsilon > SC(0))
								{
									updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred + start, pred_private[t], size);
								}
								else
								{
									updateColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred + start, pred_private[t], size);
								}
							}
							// reset row and column assignments along the alternating path.
							for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
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
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
							require_colsol_copy = true;
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, f + 1, dim_limit, " rows")) != 0)
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
					}

					// download updated v
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<grid_size, block_size, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, size);
						cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
						cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#ifdef LAP_DEBUG
						cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
#endif
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#endif
				}
#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
					if (devices == 1)
					{
						cudaSetDevice(iterator.ws.device[0]);
						int start = iterator.ws.part[0].first;
						int end = iterator.ws.part[0].second;
						int size = end - start;
						cudaStream_t stream = iterator.ws.stream[0];
						dim3 block_size, grid_size;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						findMaximum(v_private[0], &(host_min_private[0]), stream, dim2);
						subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
					}
					else
					{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel
						{
							int t = omp_get_thread_num();
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (size + block_size.x - 1) / block_size.x;
							findMaximum(v_private[t], &(host_min_private[t]), stream, size);
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC max_v = mergeMaximum<SC>(host_min_private, devices);
							subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, size);
						}
#else
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							findMaximum(v_private[t], &(host_min_private[t]), stream, size);
						}
						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
						SC max_v = mergeMaximum<SC>(host_min_private, devices);
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size;
							block_size.x = 256;
							grid_size.x = (num_items + block_size.x - 1) / block_size.x;
							subtractMaximum_kernel<<<grid_size, block_size, 0, stream>>>(v_private[t], max_v, num_items);
						}
#endif
					}
				}
#endif
				// get total_d and total_eps (total_eps was already fixed for the dim2 != dim_limit case
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel for reduction (+:total_d) reduction (+:total_eps)
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
				}
#else
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
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
			lapInfo << "row\tscanned\tlength" << std::endl;
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << f << "\t" << scancount[f] << "\t" << pathlength[f] << std::endl;
			}

			lapFree(scancount);
#endif

			for (int j = 0; j < dim2; j++) rowsol[colsol[j]] = j;

			// free CUDA memory
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(min_private[t]);
				cudaFree(jmin_private[t]);
				cudaFree(csol_private[t]);
				cudaFree(colactive_private[t]);
				cudaFree(d_private[t]);
				cudaFree(v_private[t]);
				cudaFree(total_d_private[t]);
				cudaFree(total_eps_private[t]);
				cudaFree(pred_private[t]);
				cudaFree(colsol_private[t]);
				cudaFree(semaphore_private[t]);
			}

			// free reserved memory.
#ifdef LAP_DEBUG
			cudaFreeHost(v);
#endif
			cudaFreeHost(colsol);
			cudaFreeHost(pred);
			cudaFreeHost(h_total_d);
			cudaFreeHost(h_total_eps);
			cudaFreeHost(tt_jmin);
			cudaFreeHost(v_jmin);
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(csol_private);
			cudaFreeHost(host_min_private);
			lapFree(colactive_private);
			lapFree(pred_private);
			lapFree(d_private);
			lapFree(colsol_private);
			lapFree(v_private);
			lapFree(total_d_private);
			lapFree(total_eps_private);
			lapFree(semaphore_private);
#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#else
			lapFree(tt);
#endif
#ifdef LAP_CUDA_COMPARE_CPU
			lapFree(d_tmp);
			lapFree(colactive_tmp);
#endif
			// set device back to first one
			cudaSetDevice(iterator.ws.device[0]);
		}

		template <class SC, class TC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			TC *row = new TC[dim];
			int *d_rowsol;
			TC *d_row;
			cudaMalloc(&d_rowsol, dim * sizeof(int));
			cudaMalloc(&d_row, dim * sizeof(TC));
			cudaMemcpyAsync(d_rowsol, rowsol, dim * sizeof(int), cudaMemcpyHostToDevice, stream);
			costfunc.getCost(d_row, stream, d_rowsol, dim);
			cudaMemcpyAsync(row, d_row, dim * sizeof(TC), cudaMemcpyDeviceToHost, stream);
			checkCudaErrors(cudaStreamSynchronize(stream));
			cudaFree(d_row);
			cudaFree(d_rowsol);
			for (int i = 0; i < dim; i++) my_cost += row[i];
			delete[] row;
			return my_cost;
		}

		// shortcut for square problems
		template <class SC, class TC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
		{
			lap::cuda::solve<SC, TC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
		}

		// shortcut for square problems
		template <class SC, class TC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			return lap::cuda::cost<SC, TC, CF>(dim, dim, costfunc, rowsol, stream);
		}
	}
}
