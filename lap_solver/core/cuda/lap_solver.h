#pragma once

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef LAP_CUDA_OPENMP
#include <omp.h>
#endif

#include "lap_kernel.cuh"

#include <fstream>

namespace lap
{
	namespace cuda
	{
		template <typename T>
		void allocPinned(T * &ptr, unsigned long long width, const char *file, const int line)
		{
			__checkCudaErrors(cudaMallocHost(&ptr, sizeof(T) * width), file, line);
#ifndef LAP_QUIET
			allocationLogger.alloc(1, ptr, width, file, line);
#endif
		}

		template <typename T>
		void freePinned(T *&ptr)
		{
			if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
			allocationLogger.free(1, ptr);
#endif
			checkCudaErrors(cudaFreeHost(ptr));
			ptr = (T *)NULL;
		}

		template <typename T>
		void allocDevice(T * &ptr, unsigned long long width, const char *file, const int line)
		{
			__checkCudaErrors(cudaMalloc(&ptr, sizeof(T) * width), file, line);
#ifndef LAP_QUIET
			allocationLogger.alloc(2, ptr, width, file, line);
#endif
		}

		template <typename T>
		void freeDevice(T *&ptr)
		{
			if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
			allocationLogger.free(2, ptr);
#endif
			checkCudaErrors(cudaFree(ptr));
			ptr = (T *)NULL;
		}

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
			if (min_count <= 32) findMaxSmall_kernel<<<1, 32, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else if (min_count <= 256) findMaxMedium_kernel<<<1, 256, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
			else findMaxLarge_kernel<<<1, 1024, 0, stream>>>(max_struct, v_private, std::numeric_limits<SC>::lowest(), min_count);
		}

		template <class SC, class MS>
		void findMaximum(SC *v_private, int *colsol_private, MS *max_struct, cudaStream_t &stream, int min_count)
		{
			if (min_count <= 32) findMaxSmall_kernel<<<1, 32, 0, stream>>>(max_struct, v_private, colsol_private, std::numeric_limits<SC>::lowest(), min_count);
			else if (min_count <= 256) findMaxMedium_kernel<<<1, 256, 0, stream>>>(max_struct, v_private, colsol_private, std::numeric_limits<SC>::lowest(), min_count);
			else findMaxLarge_kernel<<<1, 1024, 0, stream>>>(max_struct, v_private, colsol_private, std::numeric_limits<SC>::lowest(), min_count);
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

		int getBlockSize(int num_items)
		{
			if (num_items <= 1024) return 32;
			else if (num_items <= 65536) return 256;
			else return 1024;
		}

		template <class I>
		void selectDevice(int &start, int &num_items, cudaStream_t &stream, int &bs, int &gs, int t, I &iterator)
		{
			checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
			start = iterator.ws.part[t].first;
			num_items = iterator.ws.part[t].second - start;
			stream = iterator.ws.stream[t];

			bs = getBlockSize(num_items);
			gs = getMinSize(num_items);
		}

		template <class SC, class TC, class I>
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
			int **picked_private;
			int *picked;
			int *data_valid;
			estimateEpsilon_struct<SC> *host_struct_private;
			estimateEpsilon_struct<SC> **gpu_struct_private;
			unsigned int **semaphore_private;


#ifdef LAP_CUDA_OPENMP
			int devices = (int)iterator.ws.device.size();
			bool peerEnabled = iterator.ws.peerAccess();

			int max_threads = omp_get_max_threads();
			if (max_threads < devices) omp_set_num_threads(devices);
#else
			int devices = 1;
#endif

			decltype(iterator.getRow(0, 0, false)) *tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);

			lapAllocPinned(mod_v, dim2, __FILE__, __LINE__);
			lapAlloc(mod_v_private, devices, __FILE__, __LINE__);
			lapAlloc(min_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(max_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_cost_private, devices, __FILE__, __LINE__);
			lapAlloc(jmin_private, devices, __FILE__, __LINE__);
			lapAlloc(picked_private, devices, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			lapAlloc(semaphore_private, devices, __FILE__, __LINE__);
			lapAlloc(start_private, devices, __FILE__, __LINE__);
			lapAlloc(gpu_struct_private, devices, __FILE__, __LINE__);
			lapAllocPinned(host_struct_private, dim2 * devices, __FILE__, __LINE__);
			lapAllocPinned(data_valid, dim2 * devices, __FILE__, __LINE__);

			{
				int *host_start;
				lapAllocPinned(host_start, devices, __FILE__, __LINE__);
				for (int t = 0; t < devices; t++)
				{
					host_start[t] = iterator.ws.part[t].first;

					checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int count = getMinSize(num_items);

					lapAllocDevice(mod_v_private[t], num_items, __FILE__, __LINE__);
					lapAllocDevice(picked_private[t], num_items, __FILE__, __LINE__);
					lapAllocDevice(semaphore_private[t], 2, __FILE__, __LINE__);
					lapAllocDevice(min_cost_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(max_cost_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(picked_cost_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(jmin_private[t], count, __FILE__, __LINE__);
					lapAllocDevice(gpu_struct_private[t], 1, __FILE__, __LINE__);
					lapAllocDevice(start_private[t], devices, __FILE__, __LINE__);
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
				lapFreePinned(host_start);
			}

			SC lower_bound = SC(0);
			SC greedy_bound = SC(0);
			SC upper_bound = SC(0);

			if (devices == 1)
			{
				int start, num_items, bs, gs;
				cudaStream_t stream;
				selectDevice(start, num_items, stream, bs, gs, 0, iterator);

#ifdef LAP_CUDA_GRAPH
				cudaGraph_t graph;
				cudaGraphNode_t kernelNode[2];
				cudaGraphExec_t graphExec;
				cudaKernelNodeParams kernelNodeParams[2];
				memset(&(kernelNodeParams), 0, sizeof(kernelNodeParams));
#endif

				for (int i = 0; i < dim; i++)
				{
					tt[0] = iterator.getRow(0, i, true);
#ifdef LAP_CUDA_GRAPH
					SC limit_lowest = std::numeric_limits<SC>::lowest();
					SC limit_max = std::numeric_limits<SC>::max();
					auto ms = &(host_struct_private[i]);
					void* kernelArgs0[13] = { &ms, &(semaphore_private[0]), &(min_cost_private[0]), &(max_cost_private[0]), &(picked_cost_private[0]), &(jmin_private[0]), &(tt[0]), &(picked_private[0]), &limit_lowest, &limit_max, &i, &num_items, &dim2 };
					void* kernelArgs1[8] = { &i, &(v_private[0]), &(mod_v_private[0]), &(tt[0]), &(picked_private[0]), &(min_cost_private[0]), &(jmin_private[0]), &dim2 };

					if (i == 0)
					{
						checkCudaErrors(cudaGraphCreate(&graph, 0));
						if (bs == 32) kernelNodeParams[0].func = (void*)getMinMaxBestSingleSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
						else if (bs == 256) kernelNodeParams[0].func = (void*)getMinMaxBestSingleMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
						else kernelNodeParams[0].func = (void*)getMinMaxBestSingleLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
						kernelNodeParams[0].gridDim = gs;
						kernelNodeParams[0].blockDim = bs;
						kernelNodeParams[0].sharedMemBytes = 0;
						kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
						kernelNodeParams[0].extra = NULL;
						checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[0]), graph, 0, 0, &(kernelNodeParams[0])));
						kernelNodeParams[1].func = (void*)(updateEstimatedV_kernel<SC, TC>);
						kernelNodeParams[1].gridDim = gs;
						kernelNodeParams[1].blockDim = bs;
						kernelNodeParams[1].sharedMemBytes = 0;
						kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
						kernelNodeParams[1].extra = NULL;
						std::vector<cudaGraphNode_t> nodeDependencies;
						nodeDependencies.push_back(kernelNode[0]);
						checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[1]), graph, nodeDependencies.data(), nodeDependencies.size(), &(kernelNodeParams[1])));
						checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
					}
					else
					{
						kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
						kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
						checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode[0], &(kernelNodeParams[0])));
						checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode[1], &(kernelNodeParams[1])));
					}
					checkCudaErrors(cudaGraphLaunch(graphExec, stream));
#else
					if (bs == 32) getMinMaxBestSingleSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinMaxBestSingleMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinMaxBestSingleLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);

					if (i == 0) updateEstimatedVFirst_kernel<<<gs, bs, 0, stream>>>(mod_v_private[0], tt[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
					else if (i == 1) updateEstimatedVSecond_kernel<<<gs, bs, 0, stream>>>(v_private[0], mod_v_private[0], tt[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
					else updateEstimatedV_kernel<<<gs, bs, 0, stream>>>(v_private[0], mod_v_private[0], tt[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
#endif
				}

				for (int i = dim; i < dim2; i++)
				{
#ifdef LAP_CUDA_GRAPH
					SC limit_lowest = std::numeric_limits<SC>::lowest();
					SC limit_max = std::numeric_limits<SC>::max();
					auto ms = &(host_struct_private[i]);
					void* kernelArgs0[12] = { &ms, &(semaphore_private[0]), &(min_cost_private[0]), &(max_cost_private[0]), &(picked_cost_private[0]), &(jmin_private[0]), &(picked_private[0]), &limit_lowest, &limit_max, &i, &num_items, &dim2 };
					void* kernelArgs1[7] = { &i, &(v_private[0]), &(mod_v_private[0]), &(picked_private[0]), &(min_cost_private[0]), &(jmin_private[0]), &dim2 };

					if (i == dim)
					{
						if (dim > 0)
						{
							checkCudaErrors(cudaGraphExecDestroy(graphExec));
							checkCudaErrors(cudaGraphDestroy(graph));
						}
						checkCudaErrors(cudaGraphCreate(&graph, 0));
						if (bs == 32) kernelNodeParams[0].func = (void*)getMinMaxBestSingleSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
						else if (bs == 256) kernelNodeParams[0].func = (void*)getMinMaxBestSingleMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
						else kernelNodeParams[0].func = (void*)getMinMaxBestSingleLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
						kernelNodeParams[0].gridDim = gs;
						kernelNodeParams[0].blockDim = bs;
						kernelNodeParams[0].sharedMemBytes = 0;
						kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
						kernelNodeParams[0].extra = NULL;
						checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[0]), graph, 0, 0, &(kernelNodeParams[0])));
						kernelNodeParams[1].func = (void*)(updateEstimatedVVirtual_kernel<SC>);
						kernelNodeParams[1].gridDim = gs;
						kernelNodeParams[1].blockDim = bs;
						kernelNodeParams[1].sharedMemBytes = 0;
						kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
						kernelNodeParams[1].extra = NULL;
						std::vector<cudaGraphNode_t> nodeDependencies;
						nodeDependencies.push_back(kernelNode[0]);
						checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[1]), graph, nodeDependencies.data(), nodeDependencies.size(), &(kernelNodeParams[1])));
						checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
					}
					else
					{
						kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
						kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
						checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode[0], &(kernelNodeParams[0])));
						checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode[1], &(kernelNodeParams[1])));
					}
					checkCudaErrors(cudaGraphLaunch(graphExec, stream));
#else
					if (bs == 32) getMinMaxBestSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinMaxBestSingleMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinMaxBestSingleLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], picked_private[0], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, num_items, dim2);

					if (i == 0) updateEstimatedVFirst_kernel<<<gs, bs, 0, stream>>>(mod_v_private[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
					else if (i == 1) updateEstimatedVSecond_kernel<<<gs, bs, 0, stream>>>(v_private[0], mod_v_private[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
					else updateEstimatedV_kernel<<<gs, bs, 0, stream>>>(v_private[0], mod_v_private[0], picked_private[0], min_cost_private[0], jmin_private[0], dim2);
#endif
				}

				// no perf issue here
				checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
				checkCudaErrors(cudaGraphExecDestroy(graphExec));
				checkCudaErrors(cudaGraphDestroy(graph));
#endif
				for (int i = 0; i < dim2; i++)
				{
					lower_bound += host_struct_private[i].min;
					upper_bound += host_struct_private[i].max;
					greedy_bound += host_struct_private[i].picked;
				}
				findMaximum(v_private[0], gpu_struct_private[0], stream, dim2);
				subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[0], gpu_struct_private[0], dim2);
			}
#ifdef LAP_CUDA_OPENMP
			else
			{
				SC max_v;
				memset(data_valid, 0, dim * devices * sizeof(int));
#pragma omp parallel num_threads(devices) shared(max_v)
				{
					int t = omp_get_thread_num();
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, t, iterator);

#ifdef LAP_CUDA_GRAPH
					cudaGraph_t graph[2];
					cudaGraphNode_t kernelNode[2];
					cudaGraphExec_t graphExec[2];
					cudaKernelNodeParams kernelNodeParams[2];
					memset(&(kernelNodeParams), 0, sizeof(kernelNodeParams));
#endif

					for (int i = 0; i < dim; i++)
					{
						tt[t] = iterator.getRow(t, i, true);

#ifdef LAP_CUDA_GRAPH
						SC limit_lowest = std::numeric_limits<SC>::lowest();
						SC limit_max = std::numeric_limits<SC>::max();
						auto ms = &(host_struct_private[t + i * devices]);
						auto dv = &(data_valid[t + i * devices]);
						void* kernelArgs0[16] = { &ms, &(gpu_struct_private[t]), &(semaphore_private[t]), &dv, &(min_cost_private[t]), &(max_cost_private[t]), &(picked_cost_private[t]), &(jmin_private[t]), &(tt[t]), &(picked_private[t]), &limit_lowest, &limit_max, &i, &start, &num_items, &dim2 };

						if (i == 0)
						{
							checkCudaErrors(cudaGraphCreate(&graph[0], 0));
							if (bs == 32) kernelNodeParams[0].func = (void*)getMinMaxBestSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
							else if (bs == 256) kernelNodeParams[0].func = (void*)getMinMaxBestMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
							else kernelNodeParams[0].func = (void*)getMinMaxBestLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
							kernelNodeParams[0].gridDim = gs;
							kernelNodeParams[0].blockDim = bs;
							kernelNodeParams[0].sharedMemBytes = 0;
							kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
							kernelNodeParams[0].extra = NULL;
							checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[0]), graph[0], 0, 0, &(kernelNodeParams[0])));
							checkCudaErrors(cudaGraphInstantiate(&(graphExec[0]), graph[0], NULL, NULL, 0));
						}
						else
						{
							kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
							checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[0], kernelNode[0], &(kernelNodeParams[0])));
						}
						checkCudaErrors(cudaGraphLaunch(graphExec[0], stream));
#else
						if (bs == 32) getMinMaxBestSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else if (bs == 256) getMinMaxBestMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else getMinMaxBestLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
#endif
#pragma omp barrier
#ifdef LAP_CUDA_GRAPH
						auto ms2 = &(host_struct_private[i * devices]);
						void* kernelArgs1[13] = { &i, &(mod_v_private[t]), &(gpu_struct_private[t]), &(semaphore_private[t]), &(tt[t]), &(picked_private[t]), &dv, &ms2, &start, &num_items, &dim2, &limit_max, &devices };

						if (i == 0)
						{
							checkCudaErrors(cudaGraphCreate(&graph[1], 0));
							if (devices > 32) kernelNodeParams[1].func = (void*)(updateEstimatedVLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>);
							else kernelNodeParams[1].func = (void*)(updateEstimatedVSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>);
							kernelNodeParams[1].gridDim = gs;
							kernelNodeParams[1].blockDim = bs;
							kernelNodeParams[1].sharedMemBytes = 0;
							kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
							kernelNodeParams[1].extra = NULL;
							checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[1]), graph[1], NULL, NULL, &(kernelNodeParams[1])));
							checkCudaErrors(cudaGraphInstantiate(&graphExec[1], graph[1], NULL, NULL, 0));
						}
						else
						{
							kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
							checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[1], kernelNode[1], &(kernelNodeParams[1])));
						}
						checkCudaErrors(cudaGraphLaunch(graphExec[1], stream));
#else
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
#endif
					}

					for (int i = dim; i < dim2; i++)
					{
#ifdef LAP_CUDA_GRAPH
						SC limit_lowest = std::numeric_limits<SC>::lowest();
						SC limit_max = std::numeric_limits<SC>::max();
						auto ms = &(host_struct_private[t + i * devices]);
						auto dv = &(data_valid[t + i * devices]);
						void* kernelArgs0[15] = { &ms, &(gpu_struct_private[t]), &(semaphore_private[t]), &dv, &(min_cost_private[t]), &(max_cost_private[t]), &(picked_cost_private[t]), &(jmin_private[t]), &(picked_private[t]), &limit_lowest, &limit_max, &i, &start, &num_items, &dim2 };

						if (i == dim)
						{
							if (dim > 0)
							{
								checkCudaErrors(cudaGraphExecDestroy(graphExec[0]));
								checkCudaErrors(cudaGraphDestroy(graph[0]));
							}
							checkCudaErrors(cudaGraphCreate(&graph[0], 0));
							if (bs == 32) kernelNodeParams[0].func = (void*)getMinMaxBestSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
							else if (bs == 256) kernelNodeParams[0].func = (void*)getMinMaxBestMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
							else kernelNodeParams[0].func = (void*)getMinMaxBestLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
							kernelNodeParams[0].gridDim = gs;
							kernelNodeParams[0].blockDim = bs;
							kernelNodeParams[0].sharedMemBytes = 0;
							kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
							kernelNodeParams[0].extra = NULL;
							checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[0]), graph[0], 0, 0, &(kernelNodeParams[0])));
							checkCudaErrors(cudaGraphInstantiate(&(graphExec[0]), graph[0], NULL, NULL, 0));
						}
						else
						{
							kernelNodeParams[0].kernelParams = (void**)kernelArgs0;
							checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[0], kernelNode[0], &(kernelNodeParams[0])));
						}
						checkCudaErrors(cudaGraphLaunch(graphExec[0], stream));
#else
						if (bs == 32) getMinMaxBestSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else if (bs == 256) getMinMaxBestMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						else getMinMaxBestLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), gpu_struct_private[t], semaphore_private[t], &(data_valid[t + i * devices]), min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], picked_private[t], std::numeric_limits<SC>::lowest(), std::numeric_limits<SC>::max(), i, start, num_items, dim2);
#endif
#pragma omp barrier
#ifdef LAP_CUDA_GRAPH
						auto ms2 = &(host_struct_private[i * devices]);
						void* kernelArgs1[12] = { &i, &(mod_v_private[t]), &(gpu_struct_private[t]), &(semaphore_private[t]), &(picked_private[t]), &dv, &ms2, &start, &num_items, &dim2, &limit_max, &devices };

						if (i == dim)
						{
							if (dim > 0)
							{
								checkCudaErrors(cudaGraphExecDestroy(graphExec[1]));
								checkCudaErrors(cudaGraphDestroy(graph[1]));
							}
							checkCudaErrors(cudaGraphCreate(&graph[1], 0));
							if (devices > 32) kernelNodeParams[1].func = (void*)(updateEstimatedVLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>);
							else kernelNodeParams[1].func = (void*)(updateEstimatedVSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>);
							kernelNodeParams[1].gridDim = gs;
							kernelNodeParams[1].blockDim = bs;
							kernelNodeParams[1].sharedMemBytes = 0;
							kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
							kernelNodeParams[1].extra = NULL;
							checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[1]), graph[1], NULL, NULL, &(kernelNodeParams[1])));
							checkCudaErrors(cudaGraphInstantiate(&graphExec[1], graph[1], NULL, NULL, 0));
						}
						else
						{
							kernelNodeParams[1].kernelParams = (void**)kernelArgs1;
							checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[1], kernelNode[1], &(kernelNodeParams[1])));
						}
						checkCudaErrors(cudaGraphLaunch(graphExec[1], stream));
#else
						if (devices > 32)
						{
							if (i == 0) updateEstimatedVFirstLarge_kernel<<<gs, bs, 0, stream>>>(mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else if (i == 1) updateEstimatedVSecondLarge_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else updateEstimatedVLarge_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
						}
						else
						{
							if (i == 0) updateEstimatedVFirst_kernel<<<gs, bs, 0, stream>>>(mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else if (i == 1) updateEstimatedVSecond_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
							else updateEstimatedV_kernel<<<gs, bs, 0, stream>>>(v_private[t], mod_v_private[t], gpu_struct_private[t], semaphore_private[t], picked_private[t], &(data_valid[i * devices]), &(host_struct_private[i * devices]), start, num_items, dim2, std::numeric_limits<SC>::max(), devices);
						}
#endif
					}
					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
					checkCudaErrors(cudaGraphExecDestroy(graphExec[0]));
					checkCudaErrors(cudaGraphExecDestroy(graphExec[1]));
					checkCudaErrors(cudaGraphDestroy(graph[0]));
					checkCudaErrors(cudaGraphDestroy(graph[1]));
#endif
#pragma omp barrier
					if (t == 0)
					{
						for (int i = 0; i < dim2; i++)
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
					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
					max_v = mergeMaximum<SC>(host_struct_private, devices);
					subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[t], max_v, num_items);
				}
			}
#endif

			greedy_bound = std::min(greedy_bound, upper_bound);

			SC initial_gap = upper_bound - lower_bound;
			SC greedy_gap = greedy_bound - lower_bound;
			SC initial_greedy_gap = greedy_gap;

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
				int start, num_items, bs, gs;
				cudaStream_t stream;
				selectDevice(start, num_items, stream, bs, gs, 0, iterator);

				checkCudaErrors(cudaMemsetAsync(picked_private[0], 0, num_items * sizeof(int), stream));

#ifdef LAP_CUDA_GRAPH
				cudaGraph_t graph;
				cudaGraphNode_t kernelNode;
				cudaGraphExec_t graphExec;
				cudaKernelNodeParams kernelNodeParams;
				memset(&(kernelNodeParams), 0, sizeof(kernelNodeParams));
#endif
				for (int i = dim2 - 1; i >= dim; --i)
				{
#ifdef LAP_CUDA_GRAPH
					auto ms = &(host_struct_private[i]);
					SC limit_max = std::numeric_limits<SC>::max();
					void* kernelArgs[12] = { &ms, &(semaphore_private[0]), &(min_cost_private[0]), &(max_cost_private[0]), &(picked_cost_private[0]), &(jmin_private[0]), &(v_private[0]), &(picked_private[0]), &limit_max, &i, &num_items, &dim2 };

					if (i == dim2 - 1)
					{
						checkCudaErrors(cudaGraphCreate(&graph, 0));
						if (bs == 32) kernelNodeParams.func = getMinSecondBestSingleSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
						else if (bs == 256) kernelNodeParams.func = getMinSecondBestSingleMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
						else kernelNodeParams.func = getMinSecondBestSingleLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
						kernelNodeParams.gridDim = gs;
						kernelNodeParams.blockDim = bs;
						kernelNodeParams.sharedMemBytes = 0;
						kernelNodeParams.kernelParams = (void**)kernelArgs;
						kernelNodeParams.extra = NULL;
						checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode), graph, NULL, NULL, &(kernelNodeParams)));
						checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
					}
					else
					{
						kernelNodeParams.kernelParams = (void**)kernelArgs;
						checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &(kernelNodeParams)));
					}
					checkCudaErrors(cudaGraphLaunch(graphExec, stream));
#else
					if (bs == 32) getMinSecondBestSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinSecondBestSingleMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinSecondBestSingleLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
#endif
				}
				for (int i = dim - 1; i >= 0; --i)
				{
					tt[0] = iterator.getRow(0, i, true);

#ifdef LAP_CUDA_GRAPH
					auto ms = &(host_struct_private[i]);
					SC limit_max = std::numeric_limits<SC>::max();
					void* kernelArgs[13] = { &ms, &(semaphore_private[0]), &(min_cost_private[0]), &(max_cost_private[0]), &(picked_cost_private[0]), &(jmin_private[0]), &(tt[0]), &(v_private[0]), &(picked_private[0]), &limit_max, &i, &num_items, &dim2 };

					if (i == dim - 1)
					{
						if (dim != dim2)
						{
							checkCudaErrors(cudaGraphExecDestroy(graphExec));
							checkCudaErrors(cudaGraphDestroy(graph));
						}
						checkCudaErrors(cudaGraphCreate(&graph, 0));
						if (bs == 32) kernelNodeParams.func = getMinSecondBestSingleSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
						else if (bs == 256) kernelNodeParams.func = getMinSecondBestSingleMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
						else kernelNodeParams.func = getMinSecondBestSingleLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
						kernelNodeParams.gridDim = gs;
						kernelNodeParams.blockDim = bs;
						kernelNodeParams.sharedMemBytes = 0;
						kernelNodeParams.kernelParams = (void**)kernelArgs;
						kernelNodeParams.extra = NULL;
						checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode), graph, NULL, NULL, &(kernelNodeParams)));
						checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
					}
					else
					{
						kernelNodeParams.kernelParams = (void**)kernelArgs;
						checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &(kernelNodeParams)));
					}
					checkCudaErrors(cudaGraphLaunch(graphExec, stream));
#else
					if (bs == 32) getMinSecondBestSingleSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else if (bs == 256) getMinSecondBestSingleMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
					else getMinSecondBestSingleLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], max_cost_private[0], picked_cost_private[0], jmin_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), i, num_items, dim2);
#endif
				}
				// no perf issue here
				checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
				checkCudaErrors(cudaGraphExecDestroy(graphExec));
				checkCudaErrors(cudaGraphDestroy(graph));
#endif

				for (int i = 0; i < dim2; i++)
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
#ifdef LAP_CUDA_OPENMP
			else
			{
#pragma omp parallel num_threads(devices)
				{
					int t = omp_get_thread_num();
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, t, iterator);

					checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));

#ifdef LAP_CUDA_GRAPH
					cudaGraph_t graph;
					cudaGraphNode_t kernelNode;
					cudaGraphExec_t graphExec;
					cudaKernelNodeParams kernelNodeParams;
					memset(&(kernelNodeParams), 0, sizeof(kernelNodeParams));
					int graphState = 0;
#endif

					for (int i = 0; i < dim2; i++) host_struct_private[i * devices + t].jmin = -1;
					for (int i = dim2 - 1; i >= dim; --i)
					{
#pragma omp barrier
						if (i == dim2 - 1)
						{
							if (bs == 32) getMinSecondBestSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else if (bs == 256) getMinSecondBestMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else getMinSecondBestLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						}
						else
						{
#ifdef LAP_CUDA_GRAPH
							auto ms = &(host_struct_private[i * devices + t]);
							auto ms2 = &(host_struct_private[(i + 1) * devices]);
							SC limit_max = std::numeric_limits<SC>::max();
							void* kernelArgs[16] = { &ms, &(gpu_struct_private[t]), &(semaphore_private[t]), &(min_cost_private[t]), &(max_cost_private[t]), &(picked_cost_private[t]), &(jmin_private[t]), &(v_private[t]), &(picked_private[t]), &ms2, &limit_max, &i, &start, &num_items, &dim2, &devices };

							if (graphState == 0)
							{
								checkCudaErrors(cudaGraphCreate(&graph, 0));
								if (devices > 32)
								{
									if (bs == 32) kernelNodeParams.func = getMinSecondBestLargeSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
									else if (bs == 256) kernelNodeParams.func = getMinSecondBestLargeMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
									else kernelNodeParams.func = getMinSecondBestLargeLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
								}
								else
								{
									if (bs == 32) kernelNodeParams.func = getMinSecondBestSmallSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
									else if (bs == 256) kernelNodeParams.func = getMinSecondBestSmallMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
									else kernelNodeParams.func = getMinSecondBestSmallLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
								}
								kernelNodeParams.gridDim = gs;
								kernelNodeParams.blockDim = bs;
								kernelNodeParams.sharedMemBytes = 0;
								kernelNodeParams.kernelParams = (void**)kernelArgs;
								kernelNodeParams.extra = NULL;
								checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode), graph, NULL, NULL, &(kernelNodeParams)));
								checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
								graphState = 1;
							}
							else
							{
								kernelNodeParams.kernelParams = (void**)kernelArgs;
								checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &(kernelNodeParams)));
							}
							checkCudaErrors(cudaGraphLaunch(graphExec, stream));
#else
							if (devices > 32)
							{
								if (bs == 32) getMinSecondBestLargeSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestLargeMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestLargeLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
							else
							{
								if (bs == 32) getMinSecondBestSmallSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestSmallMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestSmallLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
#endif
						}
					}
					for (int i = dim - 1; i >= 0; --i)
					{
#pragma omp barrier
						tt[t] = iterator.getRow(t, i, true);

						if (i == dim2 - 1)
						{
							if (bs == 32) getMinSecondBestSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else if (bs == 256) getMinSecondBestMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
							else getMinSecondBestLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), i, start, num_items, dim2);
						}
						else
						{
#ifdef LAP_CUDA_GRAPH
							auto ms = &(host_struct_private[i * devices + t]);
							auto ms2 = &(host_struct_private[(i + 1) * devices]);
							SC limit_max = std::numeric_limits<SC>::max();
							void* kernelArgs[17] = { &ms, &(gpu_struct_private[t]), &(semaphore_private[t]), &(min_cost_private[t]), &(max_cost_private[t]), &(picked_cost_private[t]), &(jmin_private[t]), &(tt[t]), &(v_private[t]), &(picked_private[t]), &ms2, &limit_max, &i, &start, &num_items, &dim2, &devices };

							if (graphState < 2)
							{
								if (graphState == 1)
								{
									checkCudaErrors(cudaGraphExecDestroy(graphExec));
									checkCudaErrors(cudaGraphDestroy(graph));
								}
								checkCudaErrors(cudaGraphCreate(&graph, 0));
								if (devices > 32)
								{
									if (bs == 32) kernelNodeParams.func = getMinSecondBestLargeSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
									else if (bs == 256) kernelNodeParams.func = getMinSecondBestLargeMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
									else kernelNodeParams.func = getMinSecondBestLargeLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
								}
								else
								{
									if (bs == 32) kernelNodeParams.func = getMinSecondBestSmallSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
									else if (bs == 256) kernelNodeParams.func = getMinSecondBestSmallMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
									else kernelNodeParams.func = getMinSecondBestSmallLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
								}
								kernelNodeParams.gridDim = gs;
								kernelNodeParams.blockDim = bs;
								kernelNodeParams.sharedMemBytes = 0;
								kernelNodeParams.kernelParams = (void**)kernelArgs;
								kernelNodeParams.extra = NULL;
								checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode), graph, NULL, NULL, &(kernelNodeParams)));
								checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
								graphState = 2;
							}
							else
							{
								kernelNodeParams.kernelParams = (void**)kernelArgs;
								checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &(kernelNodeParams)));
							}
							checkCudaErrors(cudaGraphLaunch(graphExec, stream));
#else
							if (devices > 32)
							{
								if (bs == 32) getMinSecondBestLargeSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestLargeMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestLargeLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
							else
							{
								if (bs == 32) getMinSecondBestSmallSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else if (bs == 256) getMinSecondBestSmallMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
								else getMinSecondBestSmallLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], min_cost_private[t], max_cost_private[t], picked_cost_private[t], jmin_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i + 1) * devices]), std::numeric_limits<SC>::max(), i, start, num_items, dim2, devices);
							}
#endif
						}
					}
					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
					checkCudaErrors(cudaGraphExecDestroy(graphExec));
					checkCudaErrors(cudaGraphDestroy(graph));
#endif
				}
				for (int i = 0; i < dim2; i++)
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
#endif
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
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, 0, iterator);

					checkCudaErrors(cudaMemcpyAsync(mod_v_private[0], v_private[0], dim2 * sizeof(SC), cudaMemcpyDeviceToDevice, stream));
					checkCudaErrors(cudaMemsetAsync(picked_private[0], 0, dim2 * sizeof(int), stream));

#ifdef LAP_CUDA_GRAPH
					cudaGraph_t graph[2];
					cudaGraphNode_t kernelNode[2];
					cudaGraphExec_t graphExec[2];
					cudaKernelNodeParams kernelNodeParams[2];
					bool graphCreated[2] = { false, false };
					memset(&(kernelNodeParams), 0, sizeof(kernelNodeParams));
#endif
					for (int i = 0; i < dim2; i++)
					{
						if (perm[i] < dim)
						{
							tt[0] = iterator.getRow(0, perm[i], true);

#ifdef LAP_CUDA_GRAPH
							SC limit_max = std::numeric_limits<SC>::max();
							auto ms = &(host_struct_private[i]);
							void* kernelArgs[11] = { &ms, &(semaphore_private[0]), &(picked_cost_private[0]), &(jmin_private[0]), &(min_cost_private[0]), &(tt[0]), &(v_private[0]), &(picked_private[0]), &limit_max, &num_items, &dim2 };

							if (!graphCreated[0])
							{
								checkCudaErrors(cudaGraphCreate(&graph[0], 0));
								if (bs == 32) kernelNodeParams[0].func = (void*)getMinimalCostSingleSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
								else if (bs == 256) kernelNodeParams[0].func = (void*)getMinimalCostSingleMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
								else kernelNodeParams[0].func = (void*)getMinimalCostSingleLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
								kernelNodeParams[0].gridDim = gs;
								kernelNodeParams[0].blockDim = bs;
								kernelNodeParams[0].sharedMemBytes = 0;
								kernelNodeParams[0].kernelParams = (void**)kernelArgs;
								kernelNodeParams[0].extra = NULL;
								checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[0]), graph[0], 0, 0, &(kernelNodeParams[0])));
								checkCudaErrors(cudaGraphInstantiate(&(graphExec[0]), graph[0], NULL, NULL, 0));
								graphCreated[0] = true;
							}
							else
							{
								kernelNodeParams[0].kernelParams = (void**)kernelArgs;
								checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[0], kernelNode[0], &(kernelNodeParams[0])));
							}
							checkCudaErrors(cudaGraphLaunch(graphExec[0], stream));
#else
							if (bs == 32) getMinimalCostSingleSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else if (bs == 256) getMinimalCostSingleMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else getMinimalCostSingleLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], tt[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
#endif
						}
						else
						{
#ifdef LAP_CUDA_GRAPH
							SC limit_max = std::numeric_limits<SC>::max();
							auto ms = &(host_struct_private[i]);
							void* kernelArgs[10] = { &ms, &(semaphore_private[0]), &(picked_cost_private[0]), &(jmin_private[0]), &(min_cost_private[0]), &(v_private[0]), &(picked_private[0]), &limit_max, &num_items, &dim2 };

							if (!graphCreated[1])
							{
								checkCudaErrors(cudaGraphCreate(&graph[1], 0));
								if (bs == 32) kernelNodeParams[1].func = (void*)getMinimalCostSingleSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
								else if (bs == 256) kernelNodeParams[1].func = (void*)getMinimalCostSingleMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
								else kernelNodeParams[1].func = (void*)getMinimalCostSingleLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
								kernelNodeParams[1].gridDim = gs;
								kernelNodeParams[1].blockDim = bs;
								kernelNodeParams[1].sharedMemBytes = 0;
								kernelNodeParams[1].kernelParams = (void**)kernelArgs;
								kernelNodeParams[1].extra = NULL;
								checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[1]), graph[1], 0, 0, &(kernelNodeParams[1])));
								checkCudaErrors(cudaGraphInstantiate(&(graphExec[1]), graph[1], NULL, NULL, 0));
								graphCreated[1] = true;
							}
							else
							{
								kernelNodeParams[1].kernelParams = (void**)kernelArgs;
								checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[1], kernelNode[1], &(kernelNodeParams[1])));
							}
							checkCudaErrors(cudaGraphLaunch(graphExec[1], stream));
#else
							if (bs == 32) getMinimalCostSingleSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else if (bs == 256) getMinimalCostSingleMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
							else getMinimalCostSingleLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], picked_cost_private[0], jmin_private[0], min_cost_private[0], v_private[0], picked_private[0], std::numeric_limits<SC>::max(), num_items, dim2);
#endif
						}
					}

					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
					for (int i = 0; i < 2; i++) checkCudaErrors(cudaGraphExecDestroy(graphExec[i]));
					for (int i = 0; i < 2; i++) checkCudaErrors(cudaGraphDestroy(graph[i]));
#endif

					for (int i = 0; i < dim2; i++)
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
#ifdef LAP_CUDA_OPENMP
				else
				{
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						checkCudaErrors(cudaMemcpyAsync(mod_v_private[t], v_private[t], num_items * sizeof(SC), cudaMemcpyDeviceToDevice, stream));
						checkCudaErrors(cudaMemsetAsync(picked_private[t], 0, num_items * sizeof(int), stream));
						for (int i = 0; i < dim2; i++) host_struct_private[i * devices + t].jmin = -1;

#ifdef LAP_CUDA_GRAPH
						cudaGraph_t graph[2];
						cudaGraphNode_t kernelNode[2];
						cudaGraphExec_t graphExec[2];
						cudaKernelNodeParams kernelNodeParams[2];
						bool graphCreated[2] = { false, false };
						memset(&(kernelNodeParams), 0, sizeof(kernelNodeParams));
#endif
						for (int i = 0; i < dim2; i++)
						{
#pragma omp barrier
							if (perm[i] < dim)
							{
								tt[t] = iterator.getRow(t, perm[i], true);

								if (i == 0)
								{
									if (bs == 32) getMinimalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else if (bs == 256) getMinimalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else getMinimalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
								}
								else
								{
#ifdef LAP_CUDA_GRAPH
									SC limit_max = std::numeric_limits<SC>::max();
									auto ms = &(host_struct_private[i * devices + t]);
									auto ms2 = &(host_struct_private[(i - 1) * devices]);
									void* kernelArgs[15] = { &ms, &(gpu_struct_private[t]), &(semaphore_private[t]), &(picked_cost_private[t]), &(jmin_private[t]), &(min_cost_private[t]), &(tt[t]), &(v_private[t]), &(picked_private[t]), &ms2, &limit_max, &start, &num_items, &dim2, &devices };

									if (!graphCreated[0])
									{
										checkCudaErrors(cudaGraphCreate(&graph[0], 0));
										if (devices > 32)
										{
											if (bs == 32) kernelNodeParams[0].func = (void*)getMinimalCostLargeSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
											else if (bs == 256) kernelNodeParams[0].func = (void*)getMinimalCostLargeMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
											else kernelNodeParams[0].func = (void*)getMinimalCostLargeLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
										}
										else
										{
											if (bs == 32) kernelNodeParams[0].func = (void*)getMinimalCostSmallSmall_kernel<estimateEpsilon_struct<SC>, SC, TC>;
											else if (bs == 256) kernelNodeParams[0].func = (void*)getMinimalCostSmallMedium_kernel<estimateEpsilon_struct<SC>, SC, TC>;
											else kernelNodeParams[0].func = (void*)getMinimalCostSmallLarge_kernel<estimateEpsilon_struct<SC>, SC, TC>;
										}
										kernelNodeParams[0].gridDim = gs;
										kernelNodeParams[0].blockDim = bs;
										kernelNodeParams[0].sharedMemBytes = 0;
										kernelNodeParams[0].kernelParams = (void**)kernelArgs;
										kernelNodeParams[0].extra = NULL;
										checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[0]), graph[0], 0, 0, &(kernelNodeParams[0])));
										checkCudaErrors(cudaGraphInstantiate(&(graphExec[0]), graph[0], NULL, NULL, 0));
										graphCreated[0] = true;
									}
									else
									{
										kernelNodeParams[0].kernelParams = (void**)kernelArgs;
										checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[0], kernelNode[0], &(kernelNodeParams[0])));
									}
									checkCudaErrors(cudaGraphLaunch(graphExec[0], stream));
#else
									if (devices > 32)
									{
										if (bs == 32) getMinimalCostLargeSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostLargeMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostLargeLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
									else
									{
										if (bs == 32) getMinimalCostSmallSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostSmallMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostSmallLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], tt[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
#endif
								}
							}
							else
							{
#pragma omp barrier
								if (i == 0)
								{
									if (bs == 32) getMinimalCostSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else if (bs == 256) getMinimalCostMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
									else getMinimalCostLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], std::numeric_limits<SC>::max(), start, num_items, dim2);
								}
								else
								{
#ifdef LAP_CUDA_GRAPH
									SC limit_max = std::numeric_limits<SC>::max();
									auto ms = &(host_struct_private[i * devices + t]);
									auto ms2 = &(host_struct_private[(i - 1) * devices]);
									void* kernelArgs[14] = { &ms, &(gpu_struct_private[t]), &(semaphore_private[t]), &(picked_cost_private[t]), &(jmin_private[t]), &(min_cost_private[t]), &(v_private[t]), &(picked_private[t]), &ms2, &limit_max, &start, &num_items, &dim2, &devices };

									if (!graphCreated[1])
									{
										checkCudaErrors(cudaGraphCreate(&graph[1], 0));
										if (devices > 32)
										{
											if (bs == 32) kernelNodeParams[1].func = (void*)getMinimalCostLargeSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
											else if (bs == 256) kernelNodeParams[1].func = (void*)getMinimalCostLargeMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
											else kernelNodeParams[1].func = (void*)getMinimalCostLargeLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
										}
										else
										{
											if (bs == 32) kernelNodeParams[1].func = (void*)getMinimalCostSmallSmallVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
											else if (bs == 256) kernelNodeParams[1].func = (void*)getMinimalCostSmallMediumVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
											else kernelNodeParams[1].func = (void*)getMinimalCostSmallLargeVirtual_kernel<estimateEpsilon_struct<SC>, SC>;
										}
										kernelNodeParams[1].gridDim = gs;
										kernelNodeParams[1].blockDim = bs;
										kernelNodeParams[1].sharedMemBytes = 0;
										kernelNodeParams[1].kernelParams = (void**)kernelArgs;
										kernelNodeParams[1].extra = NULL;
										checkCudaErrors(cudaGraphAddKernelNode(&(kernelNode[1]), graph[1], 0, 0, &(kernelNodeParams[1])));
										checkCudaErrors(cudaGraphInstantiate(&(graphExec[1]), graph[1], NULL, NULL, 0));
										graphCreated[1] = true;
									}
									else
									{
										kernelNodeParams[1].kernelParams = (void**)kernelArgs;
										checkCudaErrors(cudaGraphExecKernelNodeSetParams(graphExec[1], kernelNode[1], &(kernelNodeParams[1])));
									}
									checkCudaErrors(cudaGraphLaunch(graphExec[1], stream));
#else
									if (devices > 32)
									{
										if (bs == 32) getMinimalCostLargeSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostLargeMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostLargeLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
									else
									{
										if (bs == 32) getMinimalCostSmallSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else if (bs == 256) getMinimalCostSmallMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
										else getMinimalCostSmallLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i * devices + t]), gpu_struct_private[t], semaphore_private[t], picked_cost_private[t], jmin_private[t], min_cost_private[t], v_private[t], picked_private[t], &(host_struct_private[(i - 1) * devices]), std::numeric_limits<SC>::max(), start, num_items, dim2, devices);
									}
#endif
								}
							}
						}
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
						for (int i = 0; i < 2; i++) checkCudaErrors(cudaGraphExecDestroy(graphExec[i]));
						for (int i = 0; i < 2; i++) checkCudaErrors(cudaGraphDestroy(graph[i]));
#endif
					}

					for (int i = 0; i < dim2; i++)
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
#endif
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
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, 0, iterator);

					for (int i = dim2 - 1; i >= 0; --i)
					{
						if (perm[i] < dim)
						{
							tt[0] = iterator.getRow(0, perm[i], true);
							if (bs == 32) updateVSingleSmall_kernel<<<gs, bs, 0, stream>>>(tt[0], v_private[0], picked_private[0], picked[i], dim2);
							else if (bs == 256) updateVSingle_kernel<<<gs, bs, 0, stream>>>(tt[0], v_private[0], picked_private[0], picked[i], dim2);
							else updateVSingle_kernel<<<gs, bs, 0, stream>>>(tt[0], v_private[0], picked_private[0], picked[i], dim2);
						}
						else
						{
							if (bs == 32) updateVSingleSmall_kernel<<<gs, bs, 0, stream>>>(v_private[0], picked_private[0], picked[i], dim2);
							else if (bs == 256) updateVSingle_kernel<<<gs, bs, 0, stream>>>(v_private[0], picked_private[0], picked[i], dim2);
							else updateVSingle_kernel<<<gs, bs, 0, stream>>>(v_private[0], picked_private[0], picked[i], dim2);
						}
					}
					findMaximum(v_private[0], &(host_struct_private[0]), stream, dim2);
					subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[0], &(host_struct_private[0]), dim2);
				}
#ifdef LAP_CUDA_OPENMP
				else
				{
					for (int i = 0; i < dim2; i++) host_struct_private[i].jmin = 0; 
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						checkCudaErrors(cudaMemsetAsync(&(gpu_struct_private[t]->jmin), 0, sizeof(int), stream));
						for (int i = dim2 - 1; i >= 0; --i)
						{
							if (perm[i] < dim)
							{
								tt[t] = iterator.getRow(t, perm[i], true);
								if (bs == 32) updateVMultiSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
								else if (bs == 256) updateVMulti_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
								else updateVMulti_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], tt[t], v_private[t], picked_private[t], picked[i] - start, num_items);
							}
							else
							{
								if (bs == 32) updateVMultiSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], v_private[t], picked_private[t], picked[i] - start, num_items);
								else if (bs == 256) updateVMulti_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], v_private[t], picked_private[t], picked[i] - start, num_items);
								else updateVMulti_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), gpu_struct_private[t], semaphore_private[t], v_private[t], picked_private[t], picked[i] - start, num_items);
							}
#pragma omp barrier
						}
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						findMaximum(v_private[t], &(host_struct_private[t]), stream, num_items);
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
						SC max_v = mergeMaximum<SC>(host_struct_private, devices);
						subtractMaximum_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(v_private[t], max_v, num_items);
					}
				}
#endif

				SC old_upper_bound = upper_bound;
				SC old_lower_bound = lower_bound;
				upper_bound = SC(0);
				lower_bound = SC(0);
				if (devices == 1)
				{
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, 0, iterator);

					for (int i = 0; i < dim2; i++)
					{
						if (perm[i] < dim)
						{
							tt[0] = iterator.getRow(0, perm[i], true);

							if (bs == 32) getFinalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else if (bs == 256) getFinalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else getFinalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], tt[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						}
						else
						{
							if (bs == 32) getFinalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else if (bs == 256) getFinalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
							else getFinalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[i]), semaphore_private[0], min_cost_private[0], picked_cost_private[0], max_cost_private[0], v_private[0], std::numeric_limits<SC>::max(), picked[i], num_items);
						}
					}

					// no perf issue here
					checkCudaErrors(cudaStreamSynchronize(stream));

					for (int i = 0; i < dim2; i++)
					{
						SC picked_cost = host_struct_private[i].picked;
						SC v_picked = host_struct_private[i].v_jmin;
						SC min_cost_real = host_struct_private[i].min;

						// need to use all picked v for the lower bound as well
						upper_bound += picked_cost;
						lower_bound += min_cost_real + v_picked;
					}
				}
#ifdef LAP_CUDA_OPENMP
				else
				{
#pragma omp parallel num_threads(devices)
					{
						int t = omp_get_thread_num();

						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						for (int i = 0; i < dim2; i++)
						{
#pragma omp barrier
							if (perm[i] < dim)
							{
								tt[t] = iterator.getRow(t, perm[i], true);

								if (bs == 32) getFinalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else if (bs == 256) getFinalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else getFinalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], tt[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							}
							else
							{
								if (bs == 32) getFinalCostSmall_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else if (bs == 256) getFinalCostMedium_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
								else getFinalCostLarge_kernel<<<gs, bs, 0, stream>>>(&(host_struct_private[t + i * devices]), semaphore_private[t], min_cost_private[t], picked_cost_private[t], max_cost_private[t], v_private[t], std::numeric_limits<SC>::max(), picked[i] - iterator.ws.part[t].first, num_items);
							}
						}
						// no perf issue here
						checkCudaErrors(cudaStreamSynchronize(stream));
					}

					for (int i = 0; i < dim2; i++)
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
#endif
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
				double ratio2 = (double)greedy_gap / (double)initial_greedy_gap;
				if (ratio2 > 1.0e-09)
				{
					if (devices == 1)
					{
						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, 0, iterator);

						interpolateV_kernel<<<gs, bs, 0, stream >>>(v_private[0], mod_v_private[0], ratio2, dim2);
					}
#ifdef LAP_CUDA_OPENMP
					else
					{
#pragma omp parallel num_threads(devices)
						{
							int t = omp_get_thread_num();

							int start, num_items, bs, gs;
							cudaStream_t stream;
							selectDevice(start, num_items, stream, bs, gs, t, iterator);

							interpolateV_kernel<<<gs, bs, 0, stream >>>(v_private[t], mod_v_private[t], ratio2, num_items);
						}
					}
#endif
				}
			}

			getUpperLower(upper, lower, greedy_gap, initial_gap, dim, dim2);

			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				lapFreeDevice(mod_v_private[t]);
				lapFreeDevice(picked_private[t]);
				lapFreeDevice(semaphore_private[t]);
				lapFreeDevice(min_cost_private[t]);
				lapFreeDevice(max_cost_private[t]);
				lapFreeDevice(picked_cost_private[t]);
				lapFreeDevice(jmin_private[t]);
				lapFreeDevice(gpu_struct_private[t]);
				lapFreeDevice(start_private[t]);
			}

			lapFreePinned(mod_v);
			lapFree(mod_v_private);
			lapFree(min_cost_private);
			lapFree(max_cost_private);
			lapFree(picked_cost_private);
			lapFree(jmin_private);
			lapFree(picked_private);
			lapFree(picked);
			lapFree(semaphore_private);
			lapFreePinned(host_struct_private);
			lapFree(tt);
			lapFree(start_private);
			lapFree(gpu_struct_private);
			lapFreePinned(data_valid);

#ifdef LAP_CUDA_OPENMP
			if (max_threads < devices) omp_set_num_threads(max_threads);
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

			int  endofpath = -1;
#ifdef LAP_DEBUG
			SC *v;
#endif
			SC *h_total_d;
			SC *h_total_eps;
			// for calculating h2
			SC *tt_jmin;
			SC *v_jmin;

			int devices = (int)iterator.ws.device.size();

#ifdef LAP_CUDA_OPENMP
			int old_max_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#endif

			const TC **tt;
			lapAlloc(tt, devices, __FILE__, __LINE__);

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<TC> eps_list;
#endif


			// used for copying
			min_struct<SC> *host_min_private;
			min_struct<SC> **gpu_min_private;
#ifdef LAP_DEBUG
			lapAllocPinned(v, dim2, __FILE__, __LINE__);
#endif
			lapAllocPinned(h_total_d, dim2, __FILE__, __LINE__);
			lapAllocPinned(h_total_eps, dim2, __FILE__, __LINE__);
			lapAllocPinned(host_min_private, devices, __FILE__, __LINE__);
			lapAllocPinned(tt_jmin, 1, __FILE__, __LINE__);
			lapAllocPinned(v_jmin, 1, __FILE__, __LINE__);

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
			lapAlloc(perm, dim2, __FILE__, __LINE__);
			lapAlloc(start_private, devices, __FILE__, __LINE__);
			lapAlloc(gpu_min_private, devices, __FILE__, __LINE__);
			lapAllocPinned(colsol, dim2, __FILE__, __LINE__);
			lapAllocPinned(pred, dim2, __FILE__, __LINE__);

			int *host_start;
			lapAllocPinned(host_start, devices, __FILE__, __LINE__);
			for (int t = 0; t < devices; t++)
			{
				checkCudaErrors(cudaSetDevice(iterator.ws.device[t]));
				int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
				int count = getMinSize(num_items);

				host_start[t] = iterator.ws.part[t].first;
				lapAllocDevice(min_private[t], count, __FILE__, __LINE__);
				lapAllocDevice(jmin_private[t], count, __FILE__, __LINE__);
				lapAllocDevice(csol_private[t], count, __FILE__, __LINE__);
				lapAllocDevice(colactive_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(d_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(v_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(total_d_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(total_eps_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(colsol_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(pred_private[t], num_items, __FILE__, __LINE__);
				lapAllocDevice(semaphore_private[t], 2, __FILE__, __LINE__);
				lapAllocDevice(start_private[t], devices, __FILE__, __LINE__);
				lapAllocDevice(gpu_min_private[t], 1, __FILE__, __LINE__);
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

			TC epsilon_upper, epsilon_lower;

			if (use_epsilon)
			{
				std::pair<SC, SC> eps = estimateEpsilon<SC, TC, I>(dim, dim2, iterator, v_private, perm);
				epsilon_upper = (TC)eps.first;
				epsilon_lower = (TC)eps.second;
			}
			else
			{
				epsilon_upper = TC(0);
				epsilon_lower = TC(0);
			}


#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			TC epsilon = epsilon_upper;

			bool first = true;
			bool second = false;
			bool reverse = true;
			bool peerEnabled = iterator.ws.peerAccess();

			if ((!use_epsilon) || (epsilon > SC(0)))
			{
				for (int i = 0; i < dim2; i++) perm[i] = i;
				reverse = false;
			}

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= TC(0))
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
				if (epsilon == TC(0)) epsilon = TC(-1.0);
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

#ifdef LAP_MINIMIZE_V
//				int dim_limit = ((reverse) || (epsilon < TC(0))) ? dim2 : dim;
				int dim_limit = dim2;
#else
				int dim_limit = dim2;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim_limit, " rows");
#endif
				long long count = 0ll;

				bool require_colsol_copy = false;

				if (devices == 1)
				{
					int t = 0;
					int start, num_items, bs, gs;
					cudaStream_t stream;
					selectDevice(start, num_items, stream, bs, gs, t, iterator);
					
#ifdef LAP_CUDA_GRAPH
					cudaGraph_t initializeSearchGraph[4], markedSkippedColumnsGraph, continueSearchGraph[2], markedSkippedColumnsUpdateGraph, updateColumnPricesGraph[4];
					cudaGraphNode_t initializeSearchNode[4], markedSkippedColumnsNode, continueSearchNode[2], markedSkippedColumnsUpdateNode, updateColumnPricesNode[4];
					cudaGraphExec_t initializeSearchExec[4], markedSkippedColumnsExec, continueSearchExec[2], markedSkippedColumnsUpdateExec, updateColumnPricesExec[4];
					cudaKernelNodeParams initializeSearchParams[4], markedSkippedColumnsParams, continueSearchParams[2], markedSkippedColumnsUpdateParams, updateColumnPricesParams[4];
					bool initializeSearchCreated[4] = { false, false, false, false };
					bool markedSkippedColumnsCreated = false;
					bool continueSearchCreated[2] = { false, false };
					bool markedSkippedColumnsUpdateCreated = false;
					bool updateColumnPricesCreated[4] = { false, false, false, false };
					memset(&(initializeSearchParams), 0, sizeof(initializeSearchParams));
#endif

					for (int fc = 0; fc < dim_limit; fc++)
					{
						// mark as incomplete
						host_min_private[t].min = std::numeric_limits<SC>::infinity();
						int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
						// start search and find minimum value
						if (require_colsol_copy)
						{
							if (f < dim)
							{
								tt[t] = iterator.getRow(t, f, false);
#ifdef LAP_CUDA_GRAPH
								auto ms = &(host_min_private[t]);
								SC limit_max = std::numeric_limits<SC>::max();
								void* kernelArgs[16] = { &ms, &(semaphore_private[t]), &(min_private[t]), &(jmin_private[t]), &(csol_private[t]), &(v_private[t]), &(d_private[t]), &(tt[t]), &(colactive_private[t]), &(colsol_private[t]), &(colsol), &(pred_private[t]), &f, &limit_max, &num_items, &dim2 };
								if (!initializeSearchCreated[0])
								{
									checkCudaErrors(cudaGraphCreate(&(initializeSearchGraph[0]), 0));
									if (bs == 32) initializeSearchParams[0].func = (void*)initializeSearchMinSmallCopy_kernel<min_struct<SC>, SC, TC>;
									else if (bs == 256) initializeSearchParams[0].func = (void*)initializeSearchMinMediumCopy_kernel<min_struct<SC>, SC, TC>;
									else initializeSearchParams[0].func = (void*)initializeSearchMinLargeCopy_kernel<min_struct<SC>, SC, TC>;
									initializeSearchParams[0].gridDim = dim3(gs, 1);
									initializeSearchParams[0].blockDim = dim3(bs, 1, 1);
									initializeSearchParams[0].sharedMemBytes = 0;
									initializeSearchParams[0].kernelParams = (void**)kernelArgs;
									initializeSearchParams[0].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(initializeSearchNode[0]), initializeSearchGraph[0], 0, 0, &(initializeSearchParams[0])));
									checkCudaErrors(cudaGraphInstantiate(&(initializeSearchExec[0]), initializeSearchGraph[0], NULL, NULL, 0));
									initializeSearchCreated[0] = true;
								}
								else
								{
									initializeSearchParams[0].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(initializeSearchExec[0], initializeSearchNode[0], &(initializeSearchParams[0])));
								}
								checkCudaErrors(cudaGraphLaunch(initializeSearchExec[0], stream));
#else
								if (bs == 32) initializeSearchMinSmallCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMediumCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLargeCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
#endif
							}
							else
							{
#ifdef LAP_CUDA_GRAPH
								auto ms = &(host_min_private[t]);
								SC limit_max = std::numeric_limits<SC>::max();
								void* kernelArgs[15] = { &ms, &(semaphore_private[t]), &(min_private[t]), &(jmin_private[t]), &(csol_private[t]), &(v_private[t]), &(d_private[t]), &(colactive_private[t]), &(colsol_private[t]), &(colsol), &(pred_private[t]), &f, &limit_max, &num_items, &dim2 };
								if (!initializeSearchCreated[1])
								{
									checkCudaErrors(cudaGraphCreate(&(initializeSearchGraph[1]), 0));
									if (bs == 32) initializeSearchParams[1].func = (void*)initializeSearchMinSmallVirtualCopy_kernel<min_struct<SC>, SC>;
									else if (bs == 256) initializeSearchParams[1].func = (void*)initializeSearchMinMediumVirtualCopy_kernel<min_struct<SC>, SC>;
									else initializeSearchParams[1].func = (void*)initializeSearchMinLargeVirtualCopy_kernel<min_struct<SC>, SC>;
									initializeSearchParams[1].gridDim = dim3(gs, 1);
									initializeSearchParams[1].blockDim = dim3(bs, 1, 1);
									initializeSearchParams[1].sharedMemBytes = 0;
									initializeSearchParams[1].kernelParams = (void**)kernelArgs;
									initializeSearchParams[1].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(initializeSearchNode[1]), initializeSearchGraph[1], 0, 0, &(initializeSearchParams[1])));
									checkCudaErrors(cudaGraphInstantiate(&(initializeSearchExec[1]), initializeSearchGraph[1], NULL, NULL, 0));
									initializeSearchCreated[1] = true;
								}
								else
								{
									initializeSearchParams[1].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(initializeSearchExec[1], initializeSearchNode[1], &(initializeSearchParams[1])));
								}
								checkCudaErrors(cudaGraphLaunch(initializeSearchExec[1], stream));
#else
								if (bs == 32) initializeSearchMinSmallVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMediumVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLargeVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
#endif
							}
							require_colsol_copy = false;
						}
						else
						{
							if (f < dim)
							{
								tt[t] = iterator.getRow(t, f, false);
#ifdef LAP_CUDA_GRAPH
								auto ms = &(host_min_private[t]);
								SC limit_max = std::numeric_limits<SC>::max();
								void* kernelArgs[15] = { &ms, &(semaphore_private[t]), &(min_private[t]), &(jmin_private[t]), &(csol_private[t]), &(v_private[t]), &(d_private[t]), &(tt[t]), &(colactive_private[t]), &(colsol_private[t]), &(pred_private[t]), &f, &limit_max, &num_items, &dim2 };
								if (!initializeSearchCreated[2])
								{
									checkCudaErrors(cudaGraphCreate(&(initializeSearchGraph[2]), 0));
									if (bs == 32) initializeSearchParams[2].func = (void*)initializeSearchMinSmall_kernel<min_struct<SC>, SC, TC>;
									else if (bs == 256) initializeSearchParams[2].func = (void*)initializeSearchMinMedium_kernel<min_struct<SC>, SC, TC>;
									else initializeSearchParams[2].func = (void*)initializeSearchMinLarge_kernel<min_struct<SC>, SC, TC>;
									initializeSearchParams[2].gridDim = dim3(gs, 1);
									initializeSearchParams[2].blockDim = dim3(bs, 1, 1);
									initializeSearchParams[2].sharedMemBytes = 0;
									initializeSearchParams[2].kernelParams = (void**)kernelArgs;
									initializeSearchParams[2].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(initializeSearchNode[2]), initializeSearchGraph[2], 0, 0, &(initializeSearchParams[2])));
									checkCudaErrors(cudaGraphInstantiate(&(initializeSearchExec[2]), initializeSearchGraph[2], NULL, NULL, 0));
									initializeSearchCreated[2] = true;
								}
								else
								{
									initializeSearchParams[2].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(initializeSearchExec[2], initializeSearchNode[2], &(initializeSearchParams[2])));
								}
								checkCudaErrors(cudaGraphLaunch(initializeSearchExec[2], stream));
#else
								if (bs == 32) initializeSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
#endif
							}
							else
							{
#ifdef LAP_CUDA_GRAPH
								auto ms = &(host_min_private[t]);
								SC limit_max = std::numeric_limits<SC>::max();
								void* kernelArgs[14] = { &ms, &(semaphore_private[t]), &(min_private[t]), &(jmin_private[t]), &(csol_private[t]), &(v_private[t]), &(d_private[t]), &(colactive_private[t]), &(colsol_private[t]), &(pred_private[t]), &f, &limit_max, &num_items, &dim2 };
								if (!initializeSearchCreated[3])
								{
									checkCudaErrors(cudaGraphCreate(&(initializeSearchGraph[3]), 0));
									if (bs == 32) initializeSearchParams[3].func = (void*)initializeSearchMinSmallVirtual_kernel<min_struct<SC>, SC>;
									else if (bs == 256) initializeSearchParams[3].func = (void*)initializeSearchMinMediumVirtual_kernel<min_struct<SC>, SC>;
									else initializeSearchParams[3].func = (void*)initializeSearchMinLargeVirtual_kernel<min_struct<SC>, SC>;
									initializeSearchParams[3].gridDim = dim3(gs, 1);
									initializeSearchParams[3].blockDim = dim3(bs, 1, 1);
									initializeSearchParams[3].sharedMemBytes = 0;
									initializeSearchParams[3].kernelParams = (void**)kernelArgs;
									initializeSearchParams[3].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(initializeSearchNode[3]), initializeSearchGraph[3], 0, 0, &(initializeSearchParams[3])));
									checkCudaErrors(cudaGraphInstantiate(&(initializeSearchExec[3]), initializeSearchGraph[3], NULL, NULL, 0));
									initializeSearchCreated[3] = true;
								}
								else
								{
									initializeSearchParams[3].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(initializeSearchExec[3], initializeSearchNode[3], &(initializeSearchParams[3])));
								}
								checkCudaErrors(cudaGraphLaunch(initializeSearchExec[3], stream));
#else
								if (bs == 32) initializeSearchMinSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) initializeSearchMinMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
								else initializeSearchMinLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
#endif
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

#ifdef LAP_CUDA_GRAPH
						{
							int jmin_start = jmin - start;
							void* kernelArgs[8] = { &(colactive_private[t]), &min, &jmin_start, &(colsol_private[t]), &(d_private[t]), &f, &dim, &num_items };
							if (!markedSkippedColumnsCreated)
							{
								checkCudaErrors(cudaGraphCreate(&(markedSkippedColumnsGraph), 0));
								markedSkippedColumnsParams.func = (void*)markedSkippedColumns_kernel<SC>;
								markedSkippedColumnsParams.gridDim = dim3(gs, 1);
								markedSkippedColumnsParams.blockDim = dim3(bs, 1, 1);
								markedSkippedColumnsParams.sharedMemBytes = 0;
								markedSkippedColumnsParams.kernelParams = (void**)kernelArgs;
								markedSkippedColumnsParams.extra = NULL;
								checkCudaErrors(cudaGraphAddKernelNode(&(markedSkippedColumnsNode), markedSkippedColumnsGraph, 0, 0, &(markedSkippedColumnsParams)));
								checkCudaErrors(cudaGraphInstantiate(&(markedSkippedColumnsExec), markedSkippedColumnsGraph, NULL, NULL, 0));
								markedSkippedColumnsCreated = true;
							}
							else
							{
								markedSkippedColumnsParams.kernelParams = (void**)kernelArgs;
								checkCudaErrors(cudaGraphExecKernelNodeSetParams(markedSkippedColumnsExec, markedSkippedColumnsNode, &(markedSkippedColumnsParams)));
							}
							checkCudaErrors(cudaGraphLaunch(markedSkippedColumnsExec, stream));
						}
#else
						markedSkippedColumns_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], f, dim, num_items);
#endif

						bool fast = unassignedfound;

						while (!unassignedfound)
						{
							int i = colsol_old;
							if (i < dim)
							{
								// get row
								tt[t] = iterator.getRow(t, i, false);
								// continue search
#ifdef LAP_CUDA_GRAPH
								auto ms = &(host_min_private[t]);
								SC limit_max = std::numeric_limits<SC>::max();
								void* kernelArgs[17] = { &ms, &(semaphore_private[t]), &(min_private[t]), &(jmin_private[t]), &(csol_private[t]), &(v_private[t]), &(d_private[t]), &(tt[t]), &(colactive_private[t]), &(colsol_private[t]), &(pred_private[t]), &i, &jmin, &min, &limit_max, &num_items, &dim2 };
								if (!continueSearchCreated[0])
								{
									checkCudaErrors(cudaGraphCreate(&(continueSearchGraph[0]), 0));
									if (bs == 32) continueSearchParams[0].func = (void*)continueSearchJMinMinSmall_kernel<min_struct<SC>, SC, TC>;
									else if (bs == 256) continueSearchParams[0].func = (void*)continueSearchJMinMinMedium_kernel<min_struct<SC>, SC, TC>;
									else continueSearchParams[0].func = (void*)continueSearchJMinMinLarge_kernel<min_struct<SC>, SC, TC>;
									continueSearchParams[0].gridDim = dim3(gs, 1);
									continueSearchParams[0].blockDim = dim3(bs, 1, 1);
									continueSearchParams[0].sharedMemBytes = 0;
									continueSearchParams[0].kernelParams = (void**)kernelArgs;
									continueSearchParams[0].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(continueSearchNode[0]), continueSearchGraph[0], 0, 0, &(continueSearchParams[0])));
									checkCudaErrors(cudaGraphInstantiate(&(continueSearchExec[0]), continueSearchGraph[0], NULL, NULL, 0));
									continueSearchCreated[0] = true;
								}
								else
								{
									continueSearchParams[0].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(continueSearchExec[0], continueSearchNode[0], &(continueSearchParams[0])));
								}
								checkCudaErrors(cudaGraphLaunch(continueSearchExec[0], stream));
#else
								if (bs == 32) continueSearchJMinMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) continueSearchJMinMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else continueSearchJMinMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
#endif
							}
							else
							{
								// continue search
#ifdef LAP_CUDA_GRAPH
								auto ms = &(host_min_private[t]);
								SC limit_max = std::numeric_limits<SC>::max();
								void* kernelArgs[16] = { &ms, &(semaphore_private[t]), &(min_private[t]), &(jmin_private[t]), &(csol_private[t]), &(v_private[t]), &(d_private[t]), &(colactive_private[t]), &(colsol_private[t]), &(pred_private[t]), &i, &jmin, &min, &limit_max, &num_items, &dim2 };
								if (!continueSearchCreated[1])
								{
									checkCudaErrors(cudaGraphCreate(&(continueSearchGraph[1]), 0));
									if (bs == 32) continueSearchParams[1].func = (void*)continueSearchJMinMinSmallVirtual_kernel<min_struct<SC>, SC>;
									else if (bs == 256) continueSearchParams[1].func = (void*)continueSearchJMinMinMediumVirtual_kernel<min_struct<SC>, SC>;
									else continueSearchParams[1].func = (void*)continueSearchJMinMinLargeVirtual_kernel<min_struct<SC>, SC>;
									continueSearchParams[1].gridDim = dim3(gs, 1);
									continueSearchParams[1].blockDim = dim3(bs, 1, 1);
									continueSearchParams[1].sharedMemBytes = 0;
									continueSearchParams[1].kernelParams = (void**)kernelArgs;
									continueSearchParams[1].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(continueSearchNode[1]), continueSearchGraph[1], 0, 0, &(continueSearchParams[1])));
									checkCudaErrors(cudaGraphInstantiate(&(continueSearchExec[1]), continueSearchGraph[1], NULL, NULL, 0));
									continueSearchCreated[1] = true;
								}
								else
								{
									continueSearchParams[1].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(continueSearchExec[1], continueSearchNode[1], &(continueSearchParams[1])));
								}
								checkCudaErrors(cudaGraphLaunch(continueSearchExec[1], stream));
#else
								if (bs == 32) continueSearchJMinMinSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else if (bs == 256) continueSearchJMinMinMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
								else continueSearchJMinMinLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), num_items, dim2);
#endif
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

#ifdef LAP_CUDA_GRAPH
							{
								int jmin_start = jmin - start;
								void* kernelArgs[8] = { &(colactive_private[t]), &min, &jmin_start, &(colsol_private[t]), &(d_private[t]), &f, &dim, &num_items };
								if (!markedSkippedColumnsUpdateCreated)
								{
									checkCudaErrors(cudaGraphCreate(&(markedSkippedColumnsUpdateGraph), 0));
									markedSkippedColumnsUpdateParams.func = (void*)markedSkippedColumnsUpdate_kernel<SC>;
									markedSkippedColumnsUpdateParams.gridDim = dim3(gs, 1);
									markedSkippedColumnsUpdateParams.blockDim = dim3(bs, 1, 1);
									markedSkippedColumnsUpdateParams.sharedMemBytes = 0;
									markedSkippedColumnsUpdateParams.kernelParams = (void**)kernelArgs;
									markedSkippedColumnsUpdateParams.extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(markedSkippedColumnsUpdateNode), markedSkippedColumnsUpdateGraph, 0, 0, &(markedSkippedColumnsUpdateParams)));
									checkCudaErrors(cudaGraphInstantiate(&(markedSkippedColumnsUpdateExec), markedSkippedColumnsUpdateGraph, NULL, NULL, 0));
									markedSkippedColumnsUpdateCreated = true;
								}
								else
								{
									markedSkippedColumnsUpdateParams.kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(markedSkippedColumnsUpdateExec, markedSkippedColumnsUpdateNode, &(markedSkippedColumnsUpdateParams)));
								}
								checkCudaErrors(cudaGraphLaunch(markedSkippedColumnsUpdateExec, stream));
							}
#else
							markedSkippedColumnsUpdate_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], i, dim, num_items);
#endif
						}

						if (fast)
						{
							colsol[endofpath] = f;
							rowsol[f] = endofpath;
							if (epsilon > TC(0))
							{
#ifdef LAP_CUDA_GRAPH
								auto colsol_priv = &(colsol_private[t][endofpath]);
								void* kernelArgs[10] = { &(colactive_private[t]), &min, &(v_private[t]), &(d_private[t]), &(total_d_private[t]), &(total_eps_private[t]), &epsilon, &num_items, &colsol_priv, &(colsol[endofpath]) };
								if (!updateColumnPricesCreated[0])
								{
									checkCudaErrors(cudaGraphCreate(&(updateColumnPricesGraph[0]), 0));
									updateColumnPricesParams[0].func = (void*)updateColumnPricesEpsilonFast_kernel<SC, TC>;
									updateColumnPricesParams[0].gridDim = dim3(gs, 1);
									updateColumnPricesParams[0].blockDim = dim3(bs, 1, 1);
									updateColumnPricesParams[0].sharedMemBytes = 0;
									updateColumnPricesParams[0].kernelParams = (void**)kernelArgs;
									updateColumnPricesParams[0].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(updateColumnPricesNode[0]), updateColumnPricesGraph[0], 0, 0, &(updateColumnPricesParams[0])));
									checkCudaErrors(cudaGraphInstantiate(&(updateColumnPricesExec[0]), updateColumnPricesGraph[0], NULL, NULL, 0));
									updateColumnPricesCreated[0] = true;
								}
								else
								{
									updateColumnPricesParams[0].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(updateColumnPricesExec[0], updateColumnPricesNode[0], &(updateColumnPricesParams[0])));
								}
								checkCudaErrors(cudaGraphLaunch(updateColumnPricesExec[0], stream));
#else
								updateColumnPricesEpsilonFast_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items, &(colsol_private[t][endofpath]), colsol[endofpath]);
#endif
							}
							else
							{
#ifdef LAP_CUDA_GRAPH
								auto colsol_priv = &(colsol_private[t][endofpath]);
								void* kernelArgs[7] = { &(colactive_private[t]), &min, &(v_private[t]), &(d_private[t]), &num_items, &colsol_priv, &(colsol[endofpath]) };
								if (!updateColumnPricesCreated[1])
								{
									checkCudaErrors(cudaGraphCreate(&(updateColumnPricesGraph[1]), 0));
									updateColumnPricesParams[1].func = (void*)updateColumnPricesFast_kernel<SC>;
									updateColumnPricesParams[1].gridDim = dim3(gs, 1);
									updateColumnPricesParams[1].blockDim = dim3(bs, 1, 1);
									updateColumnPricesParams[1].sharedMemBytes = 0;
									updateColumnPricesParams[1].kernelParams = (void**)kernelArgs;
									updateColumnPricesParams[1].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(updateColumnPricesNode[1]), updateColumnPricesGraph[1], 0, 0, &(updateColumnPricesParams[1])));
									checkCudaErrors(cudaGraphInstantiate(&(updateColumnPricesExec[1]), updateColumnPricesGraph[1], NULL, NULL, 0));
									updateColumnPricesCreated[1] = true;
								}
								else
								{
									updateColumnPricesParams[1].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(updateColumnPricesExec[1], updateColumnPricesNode[1], &(updateColumnPricesParams[1])));
								}
								checkCudaErrors(cudaGraphLaunch(updateColumnPricesExec[1], stream));
#else
								updateColumnPricesFast_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], num_items, &(colsol_private[t][endofpath]), colsol[endofpath]);
#endif
							}
						}
						else
						{
							// update column prices. can increase or decrease
							if (epsilon > TC(0))
							{
#ifdef LAP_CUDA_GRAPH
								void* kernelArgs[10] = { &(colactive_private[t]), &min, &(v_private[t]), &(d_private[t]), &(total_d_private[t]), &(total_eps_private[t]), &epsilon, &pred, &(pred_private[t]), &num_items };
								if (!updateColumnPricesCreated[2])
								{
									checkCudaErrors(cudaGraphCreate(&(updateColumnPricesGraph[2]), 0));
									updateColumnPricesParams[2].func = (void*)updateColumnPricesEpsilonCopy_kernel<SC, TC>;
									updateColumnPricesParams[2].gridDim = dim3(gs, 1);
									updateColumnPricesParams[2].blockDim = dim3(bs, 1, 1);
									updateColumnPricesParams[2].sharedMemBytes = 0;
									updateColumnPricesParams[2].kernelParams = (void**)kernelArgs;
									updateColumnPricesParams[2].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(updateColumnPricesNode[2]), updateColumnPricesGraph[2], 0, 0, &(updateColumnPricesParams[2])));
									checkCudaErrors(cudaGraphInstantiate(&(updateColumnPricesExec[2]), updateColumnPricesGraph[2], NULL, NULL, 0));
									updateColumnPricesCreated[2] = true;
								}
								else
								{
									updateColumnPricesParams[2].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(updateColumnPricesExec[2], updateColumnPricesNode[2], &(updateColumnPricesParams[2])));
								}
								checkCudaErrors(cudaGraphLaunch(updateColumnPricesExec[2], stream));
#else
								updateColumnPricesEpsilonCopy_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred, pred_private[t], num_items);
#endif
							}
							else
							{
#ifdef LAP_CUDA_GRAPH
								void* kernelArgs[7] = { &(colactive_private[t]), &min, &(v_private[t]), &(d_private[t]), &pred, &(pred_private[t]), &num_items };
								if (!updateColumnPricesCreated[3])
								{
									checkCudaErrors(cudaGraphCreate(&(updateColumnPricesGraph[3]), 0));
									updateColumnPricesParams[3].func = (void*)updateColumnPricesCopy_kernel<SC>;
									updateColumnPricesParams[3].gridDim = dim3(gs, 1);
									updateColumnPricesParams[3].blockDim = dim3(bs, 1, 1);
									updateColumnPricesParams[3].sharedMemBytes = 0;
									updateColumnPricesParams[3].kernelParams = (void**)kernelArgs;
									updateColumnPricesParams[3].extra = NULL;
									checkCudaErrors(cudaGraphAddKernelNode(&(updateColumnPricesNode[3]), updateColumnPricesGraph[3], 0, 0, &(updateColumnPricesParams[3])));
									checkCudaErrors(cudaGraphInstantiate(&(updateColumnPricesExec[3]), updateColumnPricesGraph[3], NULL, NULL, 0));
									updateColumnPricesCreated[3] = true;
							}
								else
								{
									updateColumnPricesParams[3].kernelParams = (void**)kernelArgs;
									checkCudaErrors(cudaGraphExecKernelNodeSetParams(updateColumnPricesExec[3], updateColumnPricesNode[3], &(updateColumnPricesParams[3])));
								}
								checkCudaErrors(cudaGraphLaunch(updateColumnPricesExec[3], stream));
#else
								updateColumnPricesCopy_kernel<<<gs, bs, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred, pred_private[t], num_items);
#endif
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

					// download updated v
					checkCudaErrors(cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
					checkCudaErrors(cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#ifdef LAP_DEBUG
					checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#endif
					checkCudaErrors(cudaStreamSynchronize(stream));
#ifdef LAP_CUDA_GRAPH
					for (int i = 0; i < 4; i++)
					{
						if (initializeSearchCreated[i])
						{
							checkCudaErrors(cudaGraphExecDestroy(initializeSearchExec[i]));
							checkCudaErrors(cudaGraphDestroy(initializeSearchGraph[i]));
							initializeSearchCreated[i] = false;
						}
					}
					if (markedSkippedColumnsCreated)
					{
						checkCudaErrors(cudaGraphExecDestroy(markedSkippedColumnsExec));
						checkCudaErrors(cudaGraphDestroy(markedSkippedColumnsGraph));
						markedSkippedColumnsCreated = false;
					}
					for (int i = 0; i < 2; i++)
					{
						if (continueSearchCreated[i])
						{
							checkCudaErrors(cudaGraphExecDestroy(continueSearchExec[i]));
							checkCudaErrors(cudaGraphDestroy(continueSearchGraph[i]));
							continueSearchCreated[i] = false;
						}
					}
					if (markedSkippedColumnsUpdateCreated)
					{
						checkCudaErrors(cudaGraphExecDestroy(markedSkippedColumnsUpdateExec));
						checkCudaErrors(cudaGraphDestroy(markedSkippedColumnsUpdateGraph));
						markedSkippedColumnsUpdateCreated = false;
					}
					for (int i = 0; i < 4; i++)
					{
						if (updateColumnPricesCreated[i])
						{
							checkCudaErrors(cudaGraphExecDestroy(updateColumnPricesExec[i]));
							checkCudaErrors(cudaGraphDestroy(updateColumnPricesGraph[i]));
							updateColumnPricesCreated[i] = false;
						}
					}
#endif
				}
#ifdef LAP_CUDA_OPENMP
				else /* devices > 1*/
				{
					int triggered = -1;
					int start_t = -1;
#pragma omp parallel num_threads(devices) shared(triggered, start_t, unassignedfound, require_colsol_copy, min, jmin, colsol_old)
					{
						int t = omp_get_thread_num();
						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, t, iterator);

						for (int fc = 0; fc < dim_limit; fc++)
						{
							int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
#pragma omp barrier
							// start search and find minimum value
							if (require_colsol_copy)
							{
								if (f < dim)
								{
									tt[t] = iterator.getRow(t, f, false);
									if (peerEnabled)
									{
										if (bs == 32) initializeSearchMinSmallCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (bs == 32) initializeSearchMinSmallCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
									if (peerEnabled)
									{
										if (bs == 32) initializeSearchMinSmallVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeVirtualCopy_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (bs == 32) initializeSearchMinSmallVirtualCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumVirtualCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeVirtualCopyRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], colsol + start, pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
							}
							else
							{
								if (f < dim)
								{
									tt[t] = iterator.getRow(t, f, false);
									if (peerEnabled)
									{
										if (bs == 32) initializeSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (bs == 32) initializeSearchMinSmallRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
									if (peerEnabled)
									{
										if (bs == 32) initializeSearchMinSmallVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeVirtual_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (bs == 32) initializeSearchMinSmallVirtualRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) initializeSearchMinMediumVirtualRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
										else initializeSearchMinLargeVirtualRemote_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], f, std::numeric_limits<SC>::max(), num_items, dim2);
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
							markedSkippedColumns_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], f, dim, num_items);

							bool fast = unassignedfound;
							while (!unassignedfound)
							{
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
								int i = colsol_old;

								if ((jmin >= start) && (jmin < start + num_items))
								{
									triggered = t;
									start_t = start;
									host_min_private[triggered].data_valid = 0;
								}

								if (i < dim)
								{
									// continue search
									if (peerEnabled)
									{
										// get row
										tt[t] = iterator.getRow(t, i, false);
#pragma omp barrier
										if (bs == 32) continueSearchMinPeerSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin - start_t]), &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) continueSearchMinPeerMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin - start_t]), &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinPeerLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(tt[triggered][jmin - start_t]), &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
#pragma omp barrier
										// get row
										tt[t] = iterator.getRow(t, i, false);
										if (bs == 32) continueSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) continueSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], tt[t], colactive_private[t], colsol_private[t], pred_private[t], i, tt_jmin, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
								}
								else
								{
#pragma omp barrier
									// continue search
									if (peerEnabled)
									{
										if (bs == 32) continueSearchMinPeerSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) continueSearchMinPeerMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinPeerLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), semaphore_private[t], min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, &(v_private[triggered][jmin - start_t]), jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
									}
									else
									{
										if (bs == 32) continueSearchMinSmall_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else if (bs == 256) continueSearchMinMedium_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
										else continueSearchMinLarge_kernel<<<gs, bs, 0, stream>>>(&(host_min_private[t]), gpu_min_private[t], semaphore_private[t], &(host_min_private[triggered].data_valid), min_private[t], jmin_private[t], csol_private[t], v_private[t], d_private[t], colactive_private[t], colsol_private[t], pred_private[t], i, v_jmin, jmin - start, min, std::numeric_limits<SC>::max(), num_items, dim2);
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
								// mark last column scanned
								markedSkippedColumnsUpdate_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, jmin - start, colsol_private[t], d_private[t], i, dim, num_items);

							}

							// update column prices. can increase or decrease
							if (fast)
							{
								if ((endofpath >= start) && (endofpath < start + num_items))
								{
									colsol[endofpath] = f;
									rowsol[f] = endofpath;
									if (epsilon > TC(0))
									{
										updateColumnPricesEpsilonFast_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items, &(colsol_private[t][endofpath - start]), f);
									}
									else
									{
										updateColumnPricesFast_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], num_items, &(colsol_private[t][endofpath - start]), f);
									}
								}
								else
								{
									if (epsilon > TC(0))
									{
										updateColumnPricesEpsilon_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, num_items);
									}
									else
									{
										updateColumnPrices_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], num_items);
									}
								}
							}
							else
							{
								if (epsilon > TC(0))
								{
									updateColumnPricesEpsilonCopy_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], total_d_private[t], total_eps_private[t], epsilon, pred + start, pred_private[t], num_items);
								}
								else
								{
									updateColumnPricesCopy_kernel<<<(num_items + 255) >> 8, 256, 0, stream>>>(colactive_private[t], min, v_private[t], d_private[t], pred + start, pred_private[t], num_items);
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
						checkCudaErrors(cudaMemcpyAsync(&(h_total_d[start]), total_d_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
						checkCudaErrors(cudaMemcpyAsync(&(h_total_eps[start]), total_eps_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#ifdef LAP_DEBUG
						checkCudaErrors(cudaMemcpyAsync(&(v[start]), v_private[t], sizeof(SC) * num_items, cudaMemcpyDeviceToHost, stream));
#endif
						checkCudaErrors(cudaStreamSynchronize(iterator.ws.stream[t]));
					}
				}
#endif
#ifdef LAP_MINIMIZE_V
				if (epsilon > TC(0))
				{
					if (devices == 1)
					{
						int start, num_items, bs, gs;
						cudaStream_t stream;
						selectDevice(start, num_items, stream, bs, gs, 0, iterator);

						if (dim_limit < dim2)
						{
							findMaximum(v_private[0], colsol_private[0], &(host_min_private[0]), stream, dim2);
							subtractMaximumLimited_kernel<<<gs, bs, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
						}
						else
						{
							findMaximum(v_private[0], &(host_min_private[0]), stream, dim2);
							subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[0], &(host_min_private[0]), dim2);
						}
					}
#ifdef LAP_CUDA_OPENMP
					else
					{
#pragma omp parallel num_threads(devices)
						{
							int t = omp_get_thread_num();
							int start, num_items, bs, gs;
							cudaStream_t stream;
							selectDevice(start, num_items, stream, bs, gs, t, iterator);

							if (dim_limit < dim2)
							{
								findMaximum(v_private[t], colsol_private[t], &(host_min_private[t]), stream, num_items);
							}
							else
							{
								findMaximum(v_private[t], &(host_min_private[t]), stream, num_items);
							}
							checkCudaErrors(cudaStreamSynchronize(stream));
#pragma omp barrier
							SC max_v = mergeMaximum<SC>(host_min_private, devices);

							if (dim_limit < dim2)
							{
								subtractMaximumLimited_kernel<<<gs, bs, 0, stream>>>(v_private[t], max_v, num_items);
							}
							else
							{
								subtractMaximum_kernel<<<gs, bs, 0, stream>>>(v_private[t], max_v, num_items);
							}
						}
					}
#endif
				}
#endif
				// get total_d and total_eps
				for (int i = 0; i < dim2; i++)
				{
					total_d += h_total_d[i];
					total_eps += h_total_eps[i];
				}

#ifdef LAP_DEBUG
				if (epsilon > TC(0))
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
				lapFreeDevice(min_private[t]);
				lapFreeDevice(jmin_private[t]);
				lapFreeDevice(csol_private[t]);
				lapFreeDevice(colactive_private[t]);
				lapFreeDevice(d_private[t]);
				lapFreeDevice(v_private[t]);
				lapFreeDevice(total_d_private[t]);
				lapFreeDevice(total_eps_private[t]);
				lapFreeDevice(pred_private[t]);
				lapFreeDevice(colsol_private[t]);
				lapFreeDevice(semaphore_private[t]);
				lapFreeDevice(start_private[t]);
				lapFreeDevice(gpu_min_private[t]);
			}

			// free reserved memory.
#ifdef LAP_DEBUG
			lapFreePinned(v);
#endif
			lapFreePinned(colsol);
			lapFreePinned(pred);
			lapFreePinned(h_total_d);
			lapFreePinned(h_total_eps);
			lapFreePinned(tt_jmin);
			lapFreePinned(v_jmin);
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(csol_private);
			lapFreePinned(host_min_private);
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
			lapFreePinned(host_start);

			// set device back to first one
			checkCudaErrors(cudaSetDevice(iterator.ws.device[0]));

#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_max_threads);
#endif
		}

		template <class SC, class TC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			TC *row = new TC[dim];
			int *d_rowsol;
			TC *d_row;
			lapAllocDevice(d_rowsol, dim, __FILE__, __LINE__);
			lapAllocDevice(d_row, dim, __FILE__, __LINE__);
			checkCudaErrors(cudaMemcpyAsync(d_rowsol, rowsol, dim * sizeof(int), cudaMemcpyHostToDevice, stream));
			costfunc.getCost(d_row, stream, d_rowsol, dim);
			checkCudaErrors(cudaMemcpyAsync(row, d_row, dim * sizeof(TC), cudaMemcpyDeviceToHost, stream));
			checkCudaErrors(cudaStreamSynchronize(stream));
			lapFreeDevice(d_row);
			lapFreeDevice(d_rowsol);
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
