#pragma once

#include "../lap_solver.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "lap_kernel.cuh"

#include <fstream>

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
			SC v_jmin;
			int jmin;
		};

		template <class SC>
		class min_struct
		{
		public:
			SC min;
			SC max;
			int jmin;
			int colsol;
			int data_valid;
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
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC **v_private, int *perm)
		{
#ifdef LAP_DEBUG
			auto start_time = std::chrono::high_resolution_clock::now();
#endif
			SC *mod_v;
			SC **mod_v_private;
			SC **min_cost_private;
			SC **max_cost_private;
			SC **picked_cost_private;
			int **jmin_private;
			int **start_private;
			//int *perm;
			int **picked_private;
			int *picked;
			int *data_valid;
			estimateEpsilon_struct<SC> *host_struct_private;
			estimateEpsilon_struct<SC> **gpu_struct_private;
			unsigned int **semaphore_private;

			int devices = (int)iterator.ws.device.size();
			bool peerEnabled = iterator.ws.peerAccess();

			int max_threads = omp_get_max_threads();
			if (max_threads < devices) omp_set_num_threads(devices);

			decltype(iterator.getRow(0, 0, false)) *tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);

			checkCudaErrors(cudaMallocHost(&mod_v, dim2 * sizeof(SC)));
			lapAlloc(mod_v_private, devices, __FILE__, __LINE__);
			lapAlloc(min_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(max_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			//lapAlloc(perm, dim, __FILE__, __LINE__);
			lapAlloc(picked_private, devices, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			lapAlloc(semaphore_private, devices, __FILE__, __LINE__);
			lapAlloc(start_private, devices, __FILE__, __LINE__);
			lapAlloc(gpu_struct_private, devices, __FILE__, __LINE__);
			checkCudaErrors(cudaMallocHost(&host_struct_private, (dim * devices) * sizeof(estimateEpsilon_struct<SC>)));
			checkCudaErrors(cudaMallocHost(&data_valid, (dim * devices) * sizeof(int)));

			{
				int *host_start;
				checkCudaErrors(cudaMallocHost(&host_start, devices * sizeof(int)));
				for (int t = 0; t < devices; t++)
				{
					host_start[t] = iterator.ws.part[t].first;

					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int count = getMinSize(num_items);

					checkCudaErrors(cudaMalloc(&(mod_v_private[t]), num_items * sizeof(SC)));
					checkCudaErrors(cudaMalloc(&(picked_private[t]), num_items * sizeof(int)));
					checkCudaErrors(cudaMalloc(&(semaphore_private[t]), 2 * sizeof(unsigned int)));
					checkCudaErrors(cudaMalloc(&(min_cost_private[t]), count * sizeof(SC)));
					checkCudaErrors(cudaMalloc(&(max_cost_private[t]), count * sizeof(SC)));
					checkCudaErrors(cudaMalloc(&(picked_cost_private[t]), count * sizeof(SC)));
					checkCudaErrors(cudaMalloc(&(jmin_private[t]), count * sizeof(int)));
					checkCudaErrors(cudaMalloc(&(gpu_struct_private[t]), sizeof(estimateEpsilon_struct<SC>)));
					checkCudaErrors(cudaMalloc(&(start_private[t]), devices * sizeof(int)));
				}
				for (int t = 0; t < devices; t++)
				{
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];

					checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));
					checkCudaErrors(cudaMemsetAsync(semaphore_private[t], 0, 2 * sizeof(unsigned int), stream));
					checkCudaErrors(cudaMemcpyAsync(start_private[t], host_start, devices * sizeof(int), cudaMemcpyHostToDevice, stream));
				}
				checkCudaErrors(cudaFreeHost(host_start));
			}

			SC lower_bound = SC(0);
			SC greedy_bound = SC(0);
			SC upper_bound = SC(0);

			if (devices == 1)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
				cudaStream_t stream = iterator.ws.stream[0];
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;

				int bs, gs;
				if (num_items <= 1024)
				{
					bs = 32; gs = (num_items + 31) >> 5;
				}
				else if (num_items <= 65536)
				{
					bs = 256; gs = (num_items + 255) >> 8;
				}
				else
				{
					bs = 1024; gs = (num_items + 1023) >> 10;
				}

				for (int i = 0; i < dim; i++)
				{
					auto *tt = iterator.getRow(0, i, true);

					if (num_items <= 1024) getMinMaxBestSingleSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (num_items <= 65536) getMinMaxBestSingleMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinMaxBestSingleLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);

					if (i == 0) updateEstimatedVFirst_kernel<<<gs, bs, 0, stream>>>(mod_v_private[0], tt, picked_private[0], min_cost_private[0], jmin_private[0], dim2);
					else if (i == 1) updateEstimatedVSecond_kernel<<<gs, bs, 0, stream>>>(v_private[0], mod_v_private[0], tt, picked_private[0], min_cost_private[0], jmin_private[0], dim2);
					else updateEstimatedV_kernel<<<gs, bs, 0, stream>>>(v_private[0], mod_v_private[0], tt, picked_private[0], min_cost_private[0], jmin_private[0], dim2);
				}

				checkCudaErrors(cudaStreamSynchronize(stream));
				for (int i = 0; i < dim; i++)
				{
					lower_bound += host_struct_private[i].min;
					upper_bound += host_struct_private[i].max;
					greedy_bound += host_struct_private[i].picked;
				}
				findMaximum(v_private[0], gpu_struct_private[0], stream, dim2);
				subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[0], gpu_struct_private[0], dim2);
			}
			else
			{
				SC max_v;
				memset(data_valid, 0, dim * devices * sizeof(int));
#pragma omp parallel num_threads(devices) shared(max_v)
				{
					int t = omp_get_thread_num();
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int start = iterator.ws.part[t].first;
					int num_items = iterator.ws.part[t].second - start;
					cudaStream_t stream = iterator.ws.stream[t];

					int bs, gs;
					if (num_items <= 1024)
					{
						bs = 32; gs = (num_items + 31) >> 5;
					}
					else if (num_items <= 65536)
					{
						bs = 256; gs = (num_items + 255) >> 8;
					}
					else
					{
						bs = 1024; gs = (num_items + 1023) >> 10;
					}

					for (int i = 0; i < dim; i++)
					{
						tt[t] = iterator.getRow(t, i, true);

						if (num_items <= 1024) getMinMaxBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else if (num_items <= 65536) getMinMaxBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else getMinMaxBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
#pragma omp barrier
						if (devices > 32)
						{
							if (i == 0) updateEstimatedVFirstLarge_kernel<<<gs, bs, 0, stream>>>(mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else if (i == 1) updateEstimatedVSecondLarge_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else updateEstimatedVLarge_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
						}
						else
						{
							if (i == 0) updateEstimatedVFirst_kernel<<<gs, bs, 0, stream>>>(mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else if (i == 1) updateEstimatedVSecond_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else updateEstimatedV_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], tt[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
						}
					}
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					if (t == 0)
					{
						for (int i = 0; i < dim; i++)
						{
							SC t_min_cost = host_struct_private[i * devices].min;
							SC t_max_cost = host_struct_private[i * devices].max;
							SC t_picked_cost = host_struct_private[i * devices].picked;
							int t_jmin = host_struct_private[i * devices].jmin;

							// read additional values
							for (int ti = 1; ti < devices; ti++)
							{
								SC c_min_cost = host_struct_private[ti + i * devices].min;
								SC c_max_cost = host_struct_private[ti + i * devices].max;
								SC c_picked_cost = host_struct_private[ti + i * devices].picked;
								int c_jmin = host_struct_private[ti + i * devices].jmin;
								if (c_min_cost < t_min_cost) t_min_cost = c_min_cost;
								if (c_max_cost > t_max_cost) t_max_cost = c_max_cost;
								if ((c_picked_cost < t_picked_cost) || ((c_picked_cost == t_picked_cost) && (c_jmin < t_jmin)))
								{
									t_jmin = c_jmin;
									t_picked_cost = c_picked_cost;
								}
							}

							lower_bound += t_min_cost;
							upper_bound += t_max_cost;
							greedy_bound += t_picked_cost;
						}
					}
#pragma omp barrier
					findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					max_v = mergeMaximum<SC>(host_struct_private, devices);
					subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>> (v_private[t], max_v, num_items);
				}
			}

			greedy_bound = std::min(greedy_bound, upper_bound);

			SC initial_gap = upper_bound - lower_bound;
			SC greedy_gap = greedy_bound - lower_bound;
#ifdef TODO
			SC initial_greedy_gap = greedy_gap;
#endif

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
			{
				std::stringstream ss;
				ss << "upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif

			SC upper = std::numeric_limits<SC>::max();
			SC lower;

			lower_bound = SC(0);
			upper_bound = SC(0);

			if (devices == 1)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
				cudaStream_t stream = iterator.ws.stream[0];
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;

				checkCudaErrors(cudaMemsetAsync(picked_private[0], 0, num_items * sizeof(int), stream));

				for (int i = dim - 1; i >= 0; --i)
				{
					auto *tt = iterator.getRow(0, i, true);

					if (num_items <= 1024) getMinSecondBestSingleSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (num_items <= 65536) getMinSecondBestSingleMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinSecondBestSingleLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
				}
				checkCudaErrors(cudaStreamSynchronize(stream));

				for (int i = dim - 1; i >= 0; --i)
				{
					SC min_cost = host_struct_private[i].min;
					SC second_cost = host_struct_private[i].max;
					SC picked_cost = host_struct_private[i].picked;
					SC v_jmin = host_struct_private[i].v_jmin;

					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
			}
			else
			{
#pragma omp parallel num_threads(devices)
				{
					int t = omp_get_thread_num();
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int start = iterator.ws.part[t].first;
					int num_items = iterator.ws.part[t].second - start;
					cudaStream_t stream = iterator.ws.stream[t];

					checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));

					for (int i = 0; i < dim; i++) host_struct_private[i * devices + t].jmin = -1;
					for (int i = dim - 1; i >= 0; --i)
					{
#pragma omp barrier
						auto *tt = iterator.getRow(t, i, true);

						if (i == dim - 1)
						{
							if (num_items <= 1024) getMinSecondBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else if (num_items <= 65536) getMinSecondBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else getMinSecondBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						}
						else if (devices > 32)
						{
							if (num_items <= 1024) getMinSecondBestLargeSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							else if (num_items <= 65536) getMinSecondBestLargeMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							else getMinSecondBestLargeLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
						}
						else
						{
							if (num_items <= 1024) getMinSecondBestSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							else if (num_items <= 65536) getMinSecondBestMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							else getMinSecondBestLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
						}
					}
					checkCudaErrors(cudaStreamSynchronize(stream));
				}
				for (int i = dim - 1; i >= 0; --i)
				{
					SC min_cost = host_struct_private[i * devices].min;
					SC second_cost = host_struct_private[i * devices].max;
					SC picked_cost = host_struct_private[i * devices].picked;
					SC v_jmin = host_struct_private[i * devices].v_jmin;
					int jmin = host_struct_private[i * devices].jmin;

					// read additional values
					for (int ti = 1; ti < devices; ti++)
					{
						SC c_min_cost = host_struct_private[i * devices + ti].min;
						SC c_second_cost = host_struct_private[i * devices + ti].max;
						SC c_picked_cost = host_struct_private[i * devices + ti].picked;
						SC c_vjmin = host_struct_private[i * devices + ti].v_jmin;
						int c_jmin = host_struct_private[i * devices + ti].jmin;
						if (c_min_cost < min_cost)
						{
							if (min_cost < c_second_cost) second_cost = min_cost;
							else second_cost = c_second_cost;
							min_cost = c_min_cost;
						}
						else if (c_min_cost < second_cost) second_cost = c_min_cost;
						if ((c_picked_cost < picked_cost) || ((c_picked_cost == picked_cost) && (c_jmin < jmin)))
						{
							jmin = c_jmin;
							picked_cost = c_picked_cost;
							v_jmin = c_vjmin;
						}
					}

					perm[i] = i;
					mod_v[i] = second_cost - min_cost;
					// need to use the same v values in total
					lower_bound += min_cost + v_jmin;
					upper_bound += picked_cost + v_jmin;
				}
			}
			upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
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
					checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
					cudaStream_t stream = iterator.ws.stream[0];
					int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;

					checkCudaErrors(cudaMemcpyAsync(mod_v_private[0], v_private[0], dim2 * sizeof(SC), cudaMemcpyDeviceToDevice, stream));
					checkCudaErrors(cudaMemsetAsync(picked_private[0], 0, dim2 * sizeof(int), stream));

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(0, perm[i], true);

						if (num_items <= 1024) getMinimalCostSingleSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
						else if (num_items <= 65536) getMinimalCostSingleMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
						else getMinimalCostSingleLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt, v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
					}

					checkCudaErrors(cudaStreamSynchronize(stream));

					for (int i = 0; i < dim; i++)
					{
						SC min_cost = host_struct_private[i].picked;
						SC min_cost_real = host_struct_private[i].min;
						int jmin = host_struct_private[i].jmin;
						SC v_jmin = host_struct_private[i].v_jmin;

						upper_bound += min_cost + v_jmin;
						// need to use the same v values in total
						lower_bound += min_cost_real + v_jmin;

						picked[i] = jmin;
					}
				}
				else
				{
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						cudaSetDevice(iterator.ws.device[t]);
						int start = iterator.ws.part[t].first;
						int num_items = iterator.ws.part[t].second - start;
						cudaStream_t stream = iterator.ws.stream[t];

						checkCudaErrors(cudaMemcpyAsync(mod_v_private[t], v_private[t], num_items * sizeof(SC), cudaMemcpyDeviceToDevice, stream));
						checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));
						for (int i = 0; i < dim; i++) host_struct_private[i * devices + t].jmin = -1;

						for (int i = 0; i < dim; i++)
						{
#pragma omp barrier
							auto *tt = iterator.getRow(t, perm[i], true);

							if (i == 0)
							{
								if (num_items <= 1024) getMinimalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
								else if (num_items <= 65536) getMinimalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
								else getMinimalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
							}
							else if (devices > 32)
							{
								if (num_items <= 1024) getMinimalCostLargeSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
								else if (num_items <= 65536) getMinimalCostLargeMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
								else getMinimalCostLargeLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
							}
							else
							{
								if (num_items <= 1024) getMinimalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
								else if (num_items <= 65536) getMinimalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
								else getMinimalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt, v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
							}
						}
						checkCudaErrors(cudaStreamSynchronize(stream));
					}

					for (int i = 0; i < dim; i++)
					{
						SC t_min_cost = host_struct_private[i * devices].picked;
						SC t_min_cost_real = host_struct_private[i * devices].min;
						int t_jmin = host_struct_private[i * devices].jmin;
						SC t_vjmin = host_struct_private[i * devices].v_jmin;

						// read additional values
						for (int ti = 1; ti < devices; ti++)
						{
							SC c_min_cost = host_struct_private[i * devices + ti].picked;
							SC c_min_cost_real = host_struct_private[i * devices + ti].min;
							int c_jmin = host_struct_private[i * devices + ti].jmin;
							SC c_vjmin = host_struct_private[i * devices + ti].v_jmin;

							if ((c_min_cost < t_min_cost) || ((c_min_cost == t_min_cost) && (c_jmin < t_jmin)))
							{
								t_jmin = c_jmin;
								t_min_cost = c_min_cost;
								t_vjmin = c_vjmin;
							}
							if (c_min_cost_real < t_min_cost_real) t_min_cost_real = c_min_cost_real;
						}
						upper_bound += t_min_cost + t_vjmin;
						// need to use the same v values in total
						lower_bound += t_min_cost_real + t_vjmin;

						picked[i] = t_jmin;
					}
				}
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif

				if (devices == 1)
				{
					checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
					int start = iterator.ws.part[0].first;
					int end = iterator.ws.part[0].second;
					int num_items = end - start;
					cudaStream_t stream = iterator.ws.stream[0];

					for (int i = dim - 1; i >= 0; --i)
					{
						auto *tt = iterator.getRow(0, perm[i], true);
						updateVSingle_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(tt, v_private[0], picked_private[0], picked[i], dim2);
					}
					findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
					subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
				}
				else
				{
					for (int i = 0; i < dim; i++) host_struct_private[i].jmin = 0; 
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int num_items = end - start;
						cudaStream_t stream = iterator.ws.stream[t];

						checkCudaErrors(cudaMemsetAsync(&(gpu_struct_private[t]->jmin), 0, sizeof(int), stream));
						for (int i = dim - 1; i >= 0; --i)
						{
							tt[t] = iterator.getRow(t, perm[i], true);
							if (num_items <= 1024) updateVMultiSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
							else if (num_items <= 65536) updateVMulti_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
							else updateVMulti_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
#pragma omp barrier
						}
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC max_v = mergeMaximum<SC>(host_struct_private, devices);
						subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[t], max_v, num_items);
					}
				}

				SC old_upper_bound = upper_bound;
				SC old_lower_bound = lower_bound;
				upper_bound = SC(0);
				lower_bound = SC(0);
				if (devices == 1)
				{
					checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
					cudaStream_t stream = iterator.ws.stream[0];
					int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;

					for (int i = 0; i < dim; i++)
					{
						auto *tt = iterator.getRow(0, perm[i], true);

						if (num_items <= 1024) getFinalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						else if (num_items <= 65536) getFinalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						else getFinalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt, v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
					}

					checkCudaErrors(cudaStreamSynchronize(stream));

					for (int i = 0; i < dim; i++)
					{
						SC picked_cost = host_struct_private[i].picked;
						SC v_picked = host_struct_private[i].v_jmin;
						SC min_cost_real = host_struct_private[i].min;

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
				else
				{
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int num_items = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						for (int i = 0; i < dim; i++)
						{
#pragma omp barrier
							tt[t] = iterator.getRow(t, perm[i], true);

							if (num_items <= 1024) getFinalCostSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							else if (num_items <= 65536) getFinalCostMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							else getFinalCostLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
						}
						checkCudaErrors(cudaStreamSynchronize(stream));
					}

					for (int i = 0; i < dim; i++)
					{
						SC picked_cost = host_struct_private[i * devices].picked;
						SC v_picked = host_struct_private[i * devices].v_jmin;
						SC min_cost_real = host_struct_private[i * devices].min;
						// read additional values
						for (int ti = 1; ti < devices; ti++)
						{
							picked_cost = std::min(picked_cost, host_struct_private[i * devices + ti].picked);
							v_picked = std::min(v_picked, host_struct_private[i * devices + ti].v_jmin);
							min_cost_real = std::min(min_cost_real, host_struct_private[i * devices + ti].min);
						}

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
				upper_bound = std::min(upper_bound, old_upper_bound);
				lower_bound = std::max(lower_bound, old_lower_bound);
				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				double ratio = (double)greedy_gap / (double)initial_gap;
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << ratio;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif
#ifdef TODO
				double ratio2 = (double)greedy_gap / (double)initial_greedy_gap;
				if (ratio2 > 1.0e-09)
				{
					for (int i = 0; i < dim; i++)
					{
						v[i] = (SC)((double)v2[i] * ratio2 + (double)v[i] * (1.0 - ratio2));
					}
				}
#endif
			}

			getUpperLower(upper, lower, greedy_gap, initial_gap, dim2);

			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				checkCudaErrors(cudaFree(mod_v_private[t]));
				checkCudaErrors(cudaFree(picked_private[t]));
				checkCudaErrors(cudaFree(semaphore_private[t]));
				checkCudaErrors(cudaFree(min_cost_private[t]));
				checkCudaErrors(cudaFree(max_cost_private[t]));
				checkCudaErrors(cudaFree(picked_cost_private[t]));
				checkCudaErrors(cudaFree(jmin_private[t]));
				checkCudaErrors(cudaFree(gpu_struct_private[t]));
				checkCudaErrors(cudaFree(start_private[t]));
			}

			checkCudaErrors(cudaFreeHost(mod_v));
			lapFree(mod_v_private);
			lapFree(min_cost_private);
			lapFree(max_cost_private);
			lapFree(picked_cost_private);
			lapFree(jmin_private);
			//lapFree(perm);
			lapFree(picked_private);
			lapFree(picked);
			lapFree(semaphore_private);
			checkCudaErrors(cudaFreeHost(host_struct_private));
			lapFree(tt);
			lapFree(start_private);
			lapFree(gpu_struct_private);

			if (max_threads < devices) omp_set_num_threads(max_threads);
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
			SC *tt_jmin;
			SC *v_jmin;

			int devices = (int)iterator.ws.device.size();

			const TC **tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);

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
			min_struct<SC> **gpu_min_private;
#ifdef LAP_DEBUG
			checkCudaErrors(cudaMallocHost(&v, dim2 * sizeof(SC)));
#endif
			checkCudaErrors(cudaMallocHost(&h_total_d, dim2 * sizeof(SC)));
			checkCudaErrors(cudaMallocHost(&h_total_eps, dim2 * sizeof(SC)));
			checkCudaErrors(cudaMallocHost(&host_min_private, devices * sizeof(min_struct<SC>)));
			checkCudaErrors(cudaMallocHost(&tt_jmin, sizeof(SC)));
			checkCudaErrors(cudaMallocHost(&v_jmin, sizeof(SC)));

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
			int **start_private;
			unsigned int **semaphore_private;
			int *perm;
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
			lapAlloc(perm, dim, __FILE__, __LINE__);
			lapAlloc(start_private, devices, __FILE__, __LINE__);
			lapAlloc(gpu_min_private, devices, __FILE__, __LINE__);
			checkCudaErrors(cudaMallocHost(&colsol, sizeof(int) * dim2));
			checkCudaErrors(cudaMallocHost(&pred, sizeof(int) * dim2));

			int *host_start;
			checkCudaErrors(cudaMallocHost(&host_start, devices * sizeof(int)));
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				int count = getMinSize(num_items);

				host_start[t] = iterator.ws.part[t].first;
				checkCudaErrors(cudaMalloc(&(min_private[t]), sizeof(SC) * count));
				checkCudaErrors(cudaMalloc(&(jmin_private[t]), sizeof(int) * count));
				checkCudaErrors(cudaMalloc(&(csol_private[t]), sizeof(int) * count));
				checkCudaErrors(cudaMalloc(&(colactive_private[t]), sizeof(char) * num_items));
				checkCudaErrors(cudaMalloc(&(d_private[t]), sizeof(SC) * num_items));
				checkCudaErrors(cudaMalloc(&(v_private[t]), sizeof(SC) * num_items));
				checkCudaErrors(cudaMalloc(&(total_d_private[t]), sizeof(SC) * num_items));
				checkCudaErrors(cudaMalloc(&(total_eps_private[t]), sizeof(SC) * num_items));
				checkCudaErrors(cudaMalloc(&(colsol_private[t]), sizeof(int) * num_items));
				checkCudaErrors(cudaMalloc(&(pred_private[t]), sizeof(int) * num_items));
				checkCudaErrors(cudaMalloc(&(semaphore_private[t]), 2 * sizeof(int)));
				checkCudaErrors(cudaMalloc(&(start_private[t]), sizeof(int) * devices));
				checkCudaErrors(cudaMalloc(&(gpu_min_private[t]), sizeof(min_struct<SC>)));
			}
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				cudaStream_t stream = iterator.ws.stream[t];

				if (!use_epsilon) checkCudaErrors(cudaMemsetAsync(v_private[t], 0, sizeof(SC) * num_items, stream));
				checkCudaErrors(cudaMemsetAsync(semaphore_private[t], 0, 2 * sizeof(unsigned int), stream));
				checkCudaErrors(cudaMemcpyAsync(start_private[t], host_start, devices * sizeof(int), cudaMemcpyHostToDevice, stream));
			}

			SC epsilon_upper, epsilon_lower;

			if (use_epsilon)
			{
				std::pair<SC, SC> eps = estimateEpsilon(dim, dim2, iterator, v_private, perm);
				epsilon_upper = eps.first;
				epsilon_lower = eps.second;
			}
			else
			{
				epsilon_upper = SC(0);
				epsilon_lower = SC(0);
				for (int i = 0; i < dim; i++) perm[i] = i;
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
			bool peerEnabled = iterator.ws.peerAccess();

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= SC(0))
			{
#ifdef LAP_DEBUG
				if (first)
				{
					for (int t = 0; t < devices; t++)
					{
						checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int num_items = end - start;
						cudaStream_t stream = iterator.ws.stream[t];
						checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
					}
					for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
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

				for (int t = 0; t < devices; t++)
				{
					// upload v to devices
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					checkCudaErrors(cudaMemsetAsync(total_d_private[t], 0, sizeof(SC) * num_items, stream));
					checkCudaErrors(cudaMemsetAsync(total_eps_private[t], 0, sizeof(SC) * num_items, stream));
					checkCudaErrors(cudaMemsetAsync(colsol_private[t], -1, num_items * sizeof(int), stream));
				}

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
					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int num_items = end - start;
					cudaStream_t stream = iterator.ws.stream[t];

					for (int fc = 0; fc < dim_limit; fc++)
					{
						int f = (fc < dim) ? perm[fc] : fc;
						// start search and find minimum value
						if (require_colsol_copy)
						{
							if (f < dim)
							{
								auto tt = iterator.getRow(t, f, false);
								if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							else
							{
								if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							require_colsol_copy = false;
						}
						else
						{
							if (f < dim)
							{
								auto tt = iterator.getRow(t, f, false);
								if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							else
							{
								if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
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
							checkCudaErrors(cudaMemcpyAsync(d_tmp, d_private[t], dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream));
							checkCudaErrors(cudaMemcpyAsync(colactive_tmp, colactive_private[t], dim2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
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

						if (f >= dim) markedSkippedColumns_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, num_items);

						bool fast = unassignedfound;

						while (!unassignedfound)
						{
							int i = colsol_old;
							if (i < dim)
							{
								// get row
								auto tt = iterator.getRow(t, i, false);
								// continue search
								if (num_items <= 1024) continueSearchJMinMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (num_items <= 65536) continueSearchJMinMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else continueSearchJMinMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
							}
							else
							{
								// continue search
								if (num_items <= 1024) continueSearchJMinMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (num_items <= 65536) continueSearchJMinMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else continueSearchJMinMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
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
								checkCudaErrors(cudaMemcpyAsync(d_tmp, d_private[t], dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream));
								checkCudaErrors(cudaMemcpyAsync(colactive_tmp, colactive_private[t], dim2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
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

							if (i >= dim) markedSkippedColumnsUpdate_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, num_items);
						}

						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							if (epsilon > SC(0))
							{
								updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
							else
							{
								updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], num_items, &(colsol_private[t][endofpath]), colsol[endofpath]);
							}
						}
						else
						{
							// update column prices. can increase or decrease
							if (epsilon > SC(0))
							{
								updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred, pred_private[t], num_items);
							}
							else
							{
								updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred, pred_private[t], num_items);
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
							require_colsol_copy = true;
						}
#ifndef LAP_QUIET
						{
							int level;
							if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
							{
								long long hit, miss;
								iterator.getHitMiss(hit, miss);
								total_hit += hit;
								total_miss += miss;
								if ((hit != 0) || (miss != 0))
								{
									if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
									else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
								}
								old_complete = fc + 1;
							}
						}
#endif
					}

					if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, num_items);

					// download updated v
					checkCudaErrors(cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
					checkCudaErrors(cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#ifdef LAP_DEBUG
					checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#endif
					checkCudaErrors(cudaStreamSynchronize(stream));
				}
				else /* devices > 1*/
				{
					int triggered = -1;
					int start_t = -1;
#pragma omp parallel num_threads(devices) shared(triggered, start_t, unassignedfound, require_colsol_copy, min, jmin, colsol_old)
					{
						int t = omp_get_thread_num();
						checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
						int start = iterator.ws.part[t].first;
						int end = iterator.ws.part[t].second;
						int num_items = end - start;
						cudaStream_t stream = iterator.ws.stream[t];

						for (int fc = 0; fc < dim_limit; fc++)
						{
							int f = (fc < dim) ? perm[fc] : fc;
#pragma omp barrier
							// start search and find minimum value
							if (require_colsol_copy)
							{
								if (f < dim)
								{
									auto tt = iterator.getRow(t, f, false);
									if (peerEnabled)
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
									if (peerEnabled)
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
							}
							else
							{
								if (f < dim)
								{
									auto tt = iterator.getRow(t, f, false);
									if (peerEnabled)
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt, colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
									if (peerEnabled)
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (num_items <= 1024) initializeSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) initializeSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
							}
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							if (t == 0)
							{
								// Dijkstra search
								min = host_min_private[0].min;
								jmin = host_min_private[0].jmin;
								colsol_old = host_min_private[0].colsol;

								// read additional values
								for (int ti = 1; ti < devices; ti++)
								{
									SC c_min = host_min_private[ti].min;
									int c_jmin = host_min_private[ti].jmin + iterator.ws.part[ti].first;
									int c_colsol = host_min_private[ti].colsol;
									if ((c_min < min) || ((c_min == min) && (colsol_old >= 0) && (c_colsol < 0)))
									{
										min = c_min;
										jmin = c_jmin;
										colsol_old = c_colsol;
									}
								}

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
							if (f >= dim)
							{
								markedSkippedColumns_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], dim, num_items);
							}

#ifdef LAP_CUDA_COMPARE_CPU
							{
								checkCudaErrors(cudaMemcpyAsync(&(d_tmp[start]), d_private[t], num_items * sizeof(SC), cudaMemcpyDeviceToHost, stream));
								checkCudaErrors(cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], num_items * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
								checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
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
							bool fast = unassignedfound;
							while (!unassignedfound)
							{
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
								int i = colsol_old;
								if (i < dim)
								{
									if ((jmin >= start) && (jmin < end))
									{
										triggered = t;
										start_t = start;
										host_min_private[triggered].data_valid = 0;
									}
									// continue search
									if (peerEnabled)
									{
										// get row
										tt[t] = iterator.getRow(t, i, false);
#pragma omp barrier
										if (num_items <= 1024) continueSearchMinPeerSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin - start_t]), &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) continueSearchMinPeerMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin - start_t]), &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinPeerLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin - start_t]), &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
#pragma omp barrier
										// get row
										tt[t] = iterator.getRow(t, i, false);
										if (num_items <= 1024) continueSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) continueSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
									if ((jmin >= start) && (jmin < end))
									{
										triggered = t;
										start_t = start;
										host_min_private[triggered].data_valid = 0;
									}
#pragma omp barrier
									// continue search
									if (peerEnabled)
									{
										if (num_items <= 1024) continueSearchMinPeerSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) continueSearchMinPeerMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinPeerLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (num_items <= 1024) continueSearchMinSmall_kernel<<<(num_items + 31) >> 5, 32, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (num_items <= 65536) continueSearchMinMedium_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinLarge_kernel<<<(num_items + 1023) >> 10, 1024, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
								if (t == 0)
								{
									min_n = host_min_private[0].min;
									jmin = host_min_private[0].jmin;
									colsol_old = host_min_private[0].colsol;

									// read additional values
									for (int ti = 1; ti < devices; ti++)
									{
										SC c_min = host_min_private[ti].min;
										int c_jmin = host_min_private[ti].jmin + iterator.ws.part[ti].first;
										int c_colsol = host_min_private[ti].colsol;
										if ((c_min < min_n) || ((c_min == min_n) && (colsol_old >= 0) && (c_colsol < 0)))
										{
											min_n = c_min;
											jmin = c_jmin;
											colsol_old = c_colsol;
										}
									}

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
								// mark last column scanned (single device)
								if (i >= dim)
								{
									markedSkippedColumnsUpdate_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min_n, jmin - start, colsol_private[t], d_private[t], dim, num_items);
								}

#ifdef LAP_CUDA_COMPARE_CPU
								{
									checkCudaErrors(cudaMemcpyAsync(&(d_tmp[start]), d_private[t], num_items * sizeof(SC), cudaMemcpyDeviceToHost, stream));
									checkCudaErrors(cudaMemcpyAsync(&(colactive_tmp[start]), colactive_private[t], num_items * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
									checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
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
										updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items, &(colsol_private[t][endofpath - start]), f);
									}
									else
									{
										updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], num_items, &(colsol_private[t][endofpath - start]), f);
									}
								}
								else
								{
									if (epsilon > SC(0))
									{
										updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items);
									}
									else
									{
										updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], num_items);
									}
								}
							}
							else
							{
								if (epsilon > SC(0))
								{
									updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred + start, pred_private[t], num_items);
								}
								else
								{
									updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred + start, pred_private[t], num_items);
								}
								checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
#pragma omp barrier
								if (t == 0)
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
									resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
									require_colsol_copy = true;
								}
							}
#pragma omp barrier
#ifndef LAP_QUIET
							if (t == 0)
							{
								int level;
								if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
								{
									long long hit, miss;
									iterator.getHitMiss(hit, miss);
									total_hit += hit;
									total_miss += miss;
									if ((hit != 0) || (miss != 0))
									{
										if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
										else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (fc + 1 - old_complete) << " + " << fc + 1 - old_complete << ")" << std::endl;
									}
									old_complete = fc + 1;
								}
							}
#endif
						}

						// download updated v
						if (dim2 != dim_limit) updateUnassignedColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colsol_private[t], v_private[t], total_eps_private[t], epsilon, num_items);
						checkCudaErrors(cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
						checkCudaErrors(cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#ifdef LAP_DEBUG
						checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#endif
						checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
					}
				}
#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
					if (devices == 1)
					{
						checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
						int start = iterator.ws.part[0].first;
						int end = iterator.ws.part[0].second;
						int num_items = end - start;
						cudaStream_t stream = iterator.ws.stream[0];

						findMaximum(v_private[0], &(host_min_private[0]), stream, dim2);
						subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
					}
					else
					{
						for (int t = 0; t < devices; t++)
						{
							checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int num_items = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							findMaximum(v_private[t], &(host_min_private[t]), stream, num_items);
						}
						for (int t = 0; t < devices; t++) checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
						SC max_v = mergeMaximum<SC>(host_min_private, devices);
						for (int t = 0; t < devices; t++)
						{
							checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
							int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];

							subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[t], max_v, num_items);
						}
#endif
					}
				}
				// get total_d and total_eps (total_eps was already fixed for the dim2 != dim_limit case
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
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

			// free CUDA memory
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				checkCudaErrors(cudaFree(min_private[t]));
				checkCudaErrors(cudaFree(jmin_private[t]));
				checkCudaErrors(cudaFree(csol_private[t]));
				checkCudaErrors(cudaFree(colactive_private[t]));
				checkCudaErrors(cudaFree(d_private[t]));
				checkCudaErrors(cudaFree(v_private[t]));
				checkCudaErrors(cudaFree(total_d_private[t]));
				checkCudaErrors(cudaFree(total_eps_private[t]));
				checkCudaErrors(cudaFree(pred_private[t]));
				checkCudaErrors(cudaFree(colsol_private[t]));
				checkCudaErrors(cudaFree(semaphore_private[t]));
				checkCudaErrors(cudaFree(start_private[t]));
				checkCudaErrors(cudaFree(gpu_min_private[t]));
			}

			// free reserved memory.
#ifdef LAP_DEBUG
			checkCudaErrors(cudaFreeHost(v));
#endif
			checkCudaErrors(cudaFreeHost(colsol));
			checkCudaErrors(cudaFreeHost(pred));
			checkCudaErrors(cudaFreeHost(h_total_d));
			checkCudaErrors(cudaFreeHost(h_total_eps));
			checkCudaErrors(cudaFreeHost(tt_jmin));
			checkCudaErrors(cudaFreeHost(v_jmin));
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(csol_private);
			checkCudaErrors(cudaFreeHost(host_min_private));
			lapFree(colactive_private);
			lapFree(pred_private);
			lapFree(d_private);
			lapFree(colsol_private);
			lapFree(v_private);
			lapFree(total_d_private);
			lapFree(total_eps_private);
			lapFree(semaphore_private);
			lapFree(tt);
			lapFree(perm);
			lapFree(start_private);
			lapFree(gpu_min_private);

#ifdef LAP_CUDA_COMPARE_CPU
			lapFree(d_tmp);
			lapFree(colactive_tmp);
#endif
			// set device back to first one
			checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));
		}

		template <class SC, class TC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			TC *row = new TC[dim];
			int *d_rowsol;
			TC *d_row;
			checkCudaErrors(cudaMalloc(&d_rowsol, dim * sizeof(int)));
			checkCudaErrors(cudaMalloc(&d_row, dim * sizeof(TC)));
			checkCudaErrors(cudaMemcpyAsync(d_rowsol, rowsol, dim * sizeof(int), cudaMemcpyHostToDevice, stream));
			costfunc.getCost(d_row, stream, d_rowsol, dim);
			checkCudaErrors(cudaMemcpyAsync(row, d_row, dim * sizeof(TC), cudaMemcpyDeviceToHost, stream));
			checkCudaErrors(cudaStreamSynchronize(stream));
			checkCudaErrors(cudaFree(d_row));
			checkCudaErrors(cudaFree(d_rowsol));
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
