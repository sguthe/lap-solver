#pragma once
#include <iostream>

#include "lap_kernel.cuh"
#include "lap_worksharing.h"

namespace lap
{
	namespace cuda
	{
		// Wrapper around simple CPU tiles cost function
		template <class TC, typename GETCOST>
		class CpuCostFunction : public lap::SimpleCostFunction<TC, GETCOST>
		{
		protected:
			bool sequential;
		public:
			CpuCostFunction(GETCOST& getcost, bool sequential = false) : lap::SimpleCostFunction<TC, GETCOST>(getcost), sequential(sequential) {}
			~CpuCostFunction() {}
		public:
			__forceinline bool isSequential() const { return sequential; }
		};

		// Wrapper around per-row cost funtion, e.g. CUDA, OpenCL or OpenMPI
		// Requires separate kernel to calculate costs for a given rowsol
		template <class TC, typename GETCOSTROW, typename GETCOST>
		class RowCostFunction
		{
		protected:
			GETCOSTROW getcostrow;
			GETCOST getcost;
			TC initialEpsilon;
			TC lowerEpsilon;
		public:
			RowCostFunction(GETCOSTROW &getcostrow, GETCOST &getcost) : getcostrow(getcostrow), getcost(getcost), initialEpsilon(0), lowerEpsilon(0) {}
			~RowCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline const TC getLowerEpsilon() const { return lowerEpsilon; }
			__forceinline void setLowerEpsilon(TC eps) { lowerEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end, int rows) const { getcostrow(row, t, stream, x, start, end, rows); }
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim) const { getcost(row, stream, rowsol, dim); }
		};

		// Wrapper around simple cost function
		template <class TC, typename GETCOST, typename STATE>
		class SimpleCostFunction
		{
		protected:
			GETCOST getcost;
			STATE *state;
			TC initialEpsilon;
			TC lowerEpsilon;

			int devices;

		public:
			SimpleCostFunction(GETCOST &getcost, STATE *state, int devices) : getcost(getcost), state(state), initialEpsilon(0), lowerEpsilon(0), devices(devices)
			{
			}
			~SimpleCostFunction()
			{
			}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline const TC getLowerEpsilon() const { return lowerEpsilon; }
			__forceinline void setLowerEpsilon(TC eps) { lowerEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end, int rows) const
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				grid_size.y = rows;
				getCostRow_kernel<<<grid_size, block_size, 0, stream>>>(row, getcost, state[t], x, start, end - start);
			}
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim) const
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (dim + block_size.x - 1) / block_size.x;
				getCost_kernel<<<grid_size, block_size, 0, stream>>>(row, getcost, state[0], rowsol, dim);
			}
		};

		// Costs stored in a host table. Used for conveniency only
		// This can be constructed using a CPU CostFunction.
		template <class TC>
		class CPUTableCost
		{
		protected:
			int x_size;
			int y_size;
			TC** cc;
			int* stride;
			Worksharing& ws;
			bool pinned;
		protected:
			template <class DirectCost>
			void initTable(DirectCost& cost)
			{
				int devices = (int)ws.device.size();
				lapAlloc(cc, devices, __FILE__, __LINE__);
				lapAlloc(stride, devices, __FILE__, __LINE__);
#ifdef LAP_OPENMP
				if (cost.isSequential())
				{
					// cost table needs to be initialized sequentially
#pragma omp parallel num_threads(devices)
					{
						const int t = omp_get_thread_num();
						stride[t] = ws.part[t].second - ws.part[t].first;
						if (pinned) lapAllocPinned(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						else lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						// first touch
						cc[t][0] = TC(0);
					}
					for (int x = 0; x < x_size; x++)
					{
						for (int t = 0; t < devices; t++)
						{
							cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
						}
					}
				}
				else
				{
					// create and initialize in parallel
#pragma omp parallel num_threads(devices)
					{
						const int t = omp_get_thread_num();
						stride[t] = ws.part[t].second - ws.part[t].first;
						if (pinned) lapAllocPinned(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						else lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						// first touch
						cc[t][0] = TC(0);
						for (int x = 0; x < x_size; x++)
						{
							cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
						}
					}
				}
#else
				for (int t = 0; t < devices; t++)
				{
					stride[t] = ws.part[t].second - ws.part[t].first;
					if (pinned) lapAllocPinned(cc[t], (long long)(stride[t])* (long long)x_size, __FILE__, __LINE__);
					else lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
					for (int x = 0; x < x_size; x++)
					{
						cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
					}
				}
#endif
			}

		public:
			template <class DirectCost> CPUTableCost(int x_size, int y_size, DirectCost& cost, Worksharing& ws, bool pinned) :
				x_size(x_size), y_size(y_size), ws(ws), pinned(pinned) {
				initTable(cost);
			}
			template <class DirectCost> CPUTableCost(int size, DirectCost& cost, Worksharing& ws, bool pinned) :
				x_size(size), y_size(size), ws(ws), pinned(pinned) {
				initTable(cost);
			}
			~CPUTableCost()
			{
				int devices = (int)ws.device.size();
				if (pinned) for (int t = 0; t < devices; t++) lapFreePinned(cc[t]);
				else for (int t = 0; t < devices; t++) lapFree(cc[t]);
				lapFree(cc);
				lapFree(stride);
			}
		public:
			__forceinline const TC* getRow(int t, int x) const { return cc[t] + (long long)x * (long long)stride[t]; }
			__forceinline const TC getCost(int x, int y) const
			{
				int t = 0;
				while (y >= ws.part[t].second) t++;
				long long off_y = y - ws.part[t].first;
				long long off_x = x;
				off_x *= stride[t];
				return cc[t][off_x + off_y];
			}
			__forceinline void getCostRow(TC* row, int t, cudaStream_t stream, int x, int start, int end, int rows) const
			{
				cudaMemcpyAsync(row, getRow(t, x), (end - start) * sizeof(TC) * rows, cudaMemcpyHostToDevice, stream);
			}
			__forceinline void getCost(TC* row, cudaStream_t stream, int* rowsol, int dim) const
			{
				std::cerr << "Function not supported, use lap::cost() on cpu cost matrix instead." << std::endl;
			}
		};

		// Costs stored in a device table. Used for conveniency only
		// This can be constructed using a host matrix or simple cost function.
		template <class TC>
		class GPUTableCost
		{
		protected:
			int x_size;
			int y_size;
			TC** cc;
			int* stride;
			Worksharing& ws;
		protected:
			template <class DirectCost>
			void initTable(DirectCost& cost)
			{
				int devices = (int)ws.device.size();
				lapAlloc(cc, devices, __FILE__, __LINE__);
				lapAlloc(stride, devices, __FILE__, __LINE__);
#ifdef LAP_OPENMP
				// create and initialize in parallel
#pragma omp parallel num_threads(devices)
				{
					const int t = omp_get_thread_num();
#else
				for (int t = 0; t < devices; t++)
				{
#endif
					cudaSetDevice(ws.device[t]);
					stride[t] = ws.part[t].second - ws.part[t].first;
					lapAllocDevice(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
					cost.getCostRow(cc[t], t, ws.stream[t], 0, ws.part[t].first, ws.part[t].second, x_size);
				}
			}

		public:
			template <class DirectCost> GPUTableCost(int x_size, int y_size, DirectCost& cost, Worksharing& ws) :
				x_size(x_size), y_size(y_size), ws(ws) {
				initTable(cost);
			}
			template <class DirectCost> GPUTableCost(int size, DirectCost& cost, Worksharing& ws) :
				x_size(size), y_size(size), ws(ws) {
				initTable(cost);
			}
			~GPUTableCost()
			{
				int devices = (int)ws.device.size();
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(ws.device[t]);
					lapFreeDevice(cc[t]);
				}
				lapFree(cc);
				lapFree(stride);
			}
		public:
			__forceinline const TC* getRow(int t, int i) { return cc[t] + (long long)i * (long long)stride[t]; }
		};
	}
}
