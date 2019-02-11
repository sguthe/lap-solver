#pragma once

#include "../lap_solver.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef LAP_CUDA_OPENMP
#include <omp.h>
#endif

namespace lap
{
	namespace cuda
	{
		__device__ __forceinline__ float atomicMinExt(float *addr, float value) {
			float old;
			old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
				__uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
			return old;
		}

		__device__ __forceinline__ float atomicMaxExt(float *addr, float value) {
			float old;
			old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
				__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
			return old;
		}

		__device__ __forceinline__ double atomicMinExt(double *addr, double value) {
			double old;
			old = (value >= 0) ? __longlong_as_double(atomicMin((long long *)addr, __double_as_longlong(value))) :
				__longlong_as_double(atomicMax((unsigned long long *)addr, (unsigned long long)__double_as_longlong(value)));
			return old;
		}

		__device__ __forceinline__ double atomicMaxExt(double *addr, double value) {
			double old;
			old = (value >= 0) ? __longlong_as_double(atomicMax((long long *)addr, __double_as_longlong(value))) :
				__longlong_as_double(atomicMin((unsigned long long *)addr, (unsigned long long)__double_as_longlong(value)));
			return old;
		}

		__device__ __forceinline__ int atomicMinExt(int *addr, int value) {
			return atomicMin(addr, value);
		}

		__device__ __forceinline__ int atomicMaxExt(int *addr, int value) {
			return atomicMax(addr, value);
		}

		__device__ __forceinline__ long long atomicMinExt(long long *addr, long long value) {
			return atomicMin(addr, value);
		}

		__device__ __forceinline__ long long atomicMaxExt(long long *addr, long long value) {
			return atomicMax(addr, value);
		}

		template <class TC>
		__global__ void initMinMax_kernel(TC *minmax, TC min, TC max, int size)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;

			if (x >= size) return;
			x += x;
			minmax[x++] = min;
			minmax[x] = max;
		}

		template <class TC, class TC_IN>
		__global__ void minMax_kernel(TC *minmax, TC_IN *in, int size)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;

			if (x >= size) return;
			TC v_min, v_max;
			v_min = v_max = in[x];
			x += blockDim.x * gridDim.x;
#pragma unroll 4
			while (x < size)
			{
				TC v = in[x];
				if (v < v_min) v_min = v;
				else if (v > v_max) v_max = v;
				x += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(minmax[0]), v_min);
			atomicMaxExt(&(minmax[1]), v_max);
		}

		template <class SC, class TC, class I>
		SC guessEpsilon(int x_size, int y_size, I& iterator)
		{
			SC epsilon(0);
			int devices = (int)iterator.ws.device.size();
#ifdef LAP_CUDA_OPENMP
			int old_threads = omp_get_max_threads();
			omp_set_num_threads(devices);
#endif
			TC* minmax_cost;
			TC** d_out;
			cudaMallocHost(&minmax_cost, 2 * x_size * devices * sizeof(TC));
			lapAlloc(d_out, devices, __FILE__, __LINE__);
			memset(minmax_cost, 0, 2 * sizeof(TC) * devices);
			memset(d_out, 0, sizeof(TC*) * devices);
			if (devices == 1)
			{
				cudaSetDevice(iterator.ws.device[0]);
				int num_items = iterator.ws.part[0].second - iterator.ws.part[0].first;
				cudaStream_t stream = iterator.ws.stream[0];
				cudaMalloc(&(d_out[0]), 2 * x_size * sizeof(TC));
				dim3 block_size, grid_size, grid_size_min;
				block_size.x = 256;
				grid_size.x = (x_size + block_size.x - 1) / block_size.x;
				grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[0]);
				initMinMax_kernel<<<grid_size, block_size, 0, stream>>>(d_out[0], std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), x_size);
				for (int x = x_size - 1; x >= 0; --x)
				{
					auto tt = iterator.getRow(0, x);
					minMax_kernel<<<grid_size_min, block_size, 0, stream>>>(d_out[0] + 2 * x, tt, num_items);
				}
				cudaMemcpyAsync(minmax_cost, d_out[0], 2 * x_size * sizeof(TC), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				for (int x = 0; x < x_size; x++)
				{
					SC min_cost, max_cost;
					min_cost = (SC)minmax_cost[2 * x];
					max_cost = (SC)minmax_cost[2 * x + 1];
					epsilon += max_cost - min_cost;
				}
			}
			else
			{
#ifdef LAP_CUDA_OPENMP
#pragma omp parallel for
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMalloc(&(d_out[t]), 2 * x_size * sizeof(TC));
					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (x_size + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t]);
					initMinMax_kernel<<<grid_size, block_size, 0, stream>>>(d_out[t], std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), x_size);
					for (int x = x_size - 1; x >= 0; --x)
					{
						auto tt = iterator.getRow(t, x);
						minMax_kernel<<<grid_size_min, block_size, 0, stream>>>(d_out[t] + 2 * x, tt, num_items);
					}
					cudaMemcpyAsync(&(minmax_cost[2 * t * x_size]), d_out[t], 2 * x_size * sizeof(TC), cudaMemcpyDeviceToHost, stream);
					cudaError_t err = cudaStreamSynchronize(stream);
					if (err != 0)
					{
						std::cout << err << std::endl;
					}
				}
#else
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMalloc(&(d_out[t]), 2 * x_size * sizeof(TC));
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (x_size + block_size.x - 1) / block_size.x;
					initMinMax_kernel<<<grid_size, block_size, 0, stream>>>(d_out[t], std::numeric_limits<TC>::max(), std::numeric_limits<TC>::lowest(), x_size);
				}
				for (int x = x_size - 1; x >= 0; --x)
				{
					for (int t = 0; t < devices; t++)
					{
						cudaSetDevice(iterator.ws.device[t]);
						int num_items = iterator.ws.part[t].second - iterator.ws.part[t].first;
						cudaStream_t stream = iterator.ws.stream[t];
						auto tt = iterator.getRow(t, x);
						dim3 block_size, grid_size_min;
						block_size.x = 256;
						grid_size_min.x = std::min(grid_size.x, iterator.ws.sm_count[t]);
						minMax_kernel<<<grid_size_min, block_size, 0, stream>>>(d_out[t] + 2 * x, tt, num_items);
					}
				}
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(&(minmax_cost[2 * t * x_size]), d_out[t], 2 * x_size * sizeof(TC), cudaMemcpyDeviceToHost, stream);
				}
				for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
#endif
				for (int x = 0; x < x_size; x++)
				{
					SC min_cost, max_cost;
					min_cost = (SC)minmax_cost[2 * x];
					max_cost = (SC)minmax_cost[2 * x + 1];
					for (int t = 1; t < devices; t++)
					{
						max_cost = std::max(max_cost, (SC)minmax_cost[2 * (t * x_size + x)]);
						min_cost = std::min(min_cost, (SC)minmax_cost[2 * (t * x_size + x) + 1]);
					}
					epsilon += max_cost - min_cost;
				}
			}
			cudaFreeHost(minmax_cost);
			lapFree(d_out);
#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#endif
			return epsilon / (SC(10) * SC(x_size));
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
		__global__ void initializeMin_kernel(min_struct<SC> *s, SC min, int dim2)
		{
			s->min = min;
			s->jmin_free = dim2;
			s->jmin_taken = dim2;
		}

		template <class SC, class TC>
		__global__ void initializeSearchMin_kernel(SC *min, SC *v, SC *d, TC *tt, char *colactive, int *pred, int f, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;
			SC v_min;
			colactive[j] = 1;
			pred[j] = f;
			d[j] = v_min = tt[j] - v[j];
			j += blockDim.x * gridDim.x;
#pragma unroll 4
			while (j < size)
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = tt[j] - v[j];
				if (v0 < v_min) v_min = v0;
				j += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(min[0]), v_min);
		}

		template <class SC, class TC>
		__global__ void continueSearchMin_kernel(SC *min, SC *v, SC *d, TC *tt, char *colactive, int *pred, int i, SC h2, SC max, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;
			SC v_min = max;

#pragma unroll 4
			while (j < size)
			{
				if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = tt[j] - v[j] - h2;
					if (v2 < h)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					if (h < v_min) v_min = h;
				}
				j += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(min[0]), v_min);
		}

		template <class SC>
		__global__ void initializeSearchMin_kernel(SC *min, SC *v, SC *d, char *colactive, int *pred, int f, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;
			SC v_min;
			colactive[j] = 1;
			pred[j] = f;
			d[j] = v_min = -v[j];
			j += blockDim.x * gridDim.x;
#pragma unroll 4
			while (j < size)
			{
				SC v0;
				colactive[j] = 1;
				pred[j] = f;
				d[j] = v0 = -v[j];
				if (v0 < v_min) v_min = v0;
				j += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(min[0]), v_min);
		}

		template <class SC>
		__global__ void continueSearchMin_kernel(SC *min, SC *v, SC *d, char *colactive, int *pred, int i, SC h2, SC max, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;
			SC v_min = max;

#pragma unroll 4
			while (j < size)
			{
				if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = -v[j] - h2;
					if (v2 < h)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					if (h < v_min) v_min = h;
				}
				j += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(min[0]), v_min);
		}

		template <class SC, class TC>
		__global__ void continueSearchJMinMin_kernel(SC *min_val, SC *v, SC *d, TC *tt, char *colactive, int *pred, int i, int jmin, SC min, SC max, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;
			SC v_min = max;
			SC h2 = tt[jmin] - v[jmin] - min;

#pragma unroll 4
			while (j < size)
			{
				if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = tt[j] - v[j] - h2;
					if (v2 < h)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					if (h < v_min) v_min = h;
				}
				j += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(min_val[0]), v_min);
		}

		template <class SC>
		__global__ void continueSearchJMinMin_kernel(SC *min_val, SC *v, SC *d, char *colactive, int *pred, int i, int jmin, SC min, SC max, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;

			if (j >= size) return;
			SC v_min = max;
			SC h2 = -v[jmin] - min;

#pragma unroll 4
			while (j < size)
			{
				if (colactive[j] != 0)
				{
					SC h = d[j];
					SC v2 = -v[j] - h2;
					if (v2 < h)
					{
						pred[j] = i;
						d[j] = h = v2;
					}
					if (h < v_min) v_min = h;
				}
				j += blockDim.x * gridDim.x;
			}

			atomicMinExt(&(min_val[0]), v_min);
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

			for (int t = 0; t < devices; t++)
			{
				// single device code
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
				cudaMalloc(&(colsol_private[t]), sizeof(int) * size);
				cudaMalloc(&(pred_private[t]), sizeof(int) * size);
#ifdef LAP_CUDA_LOCAL_ROWSOL
				cudaMalloc(&(rowsol_private[t]), sizeof(int) * dim2);
#endif
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

#ifdef LAP_CUDA_OPENMP
				if (devices == 1)
				{
					// upload v to devices
					int t = 0;
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
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
						cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
#ifdef LAP_CUDA_LOCAL_ROWSOL
						cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
						cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
#endif
					}
				}
#else
				for (int t = 0; t < devices; t++)
				{
					// upload v to devices
					cudaSetDevice(iterator.ws.device[t]);
					int size = iterator.ws.part[t].second - iterator.ws.part[t].first;
					int start = iterator.ws.part[t].first;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(v_private[t], &(v[start]), sizeof(SC) * size, cudaMemcpyHostToDevice, stream);
#ifdef LAP_CUDA_LOCAL_ROWSOL
					cudaMemsetAsync(rowsol_private[t], -1, dim2 * sizeof(int), stream);
					cudaMemsetAsync(colsol_private[t], -1, size * sizeof(int), stream);
#endif
				}
#endif

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

					dim3 block_size, grid_size, grid_size_min;
					block_size.x = 256;
					grid_size.x = (size + block_size.x - 1) / block_size.x;
					grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t]);

					for (int f = 0; f < dim2; f++)
					{
#ifndef LAP_CUDA_LOCAL_ROWSOL
						// upload colsol to devices
						cudaMemcpyAsync(colsol_private[t], &(colsol[start]), sizeof(int) * size, cudaMemcpyHostToDevice, stream);
#endif
						// initialize Search
						initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
						// start search and find minimum value
						if (f < dim)
							initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], pred_private[t], f, size);
						else
							initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], colactive_private[t], pred_private[t], f, size);
						// min is now set so we need to find the correspoding minima for free and taken columns
						findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
						cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
						cudaStreamSynchronize(stream);
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
#ifdef LAP_CUDA_LOCAL_ROWSOL
						setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#else
						setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif

						while (!unassignedfound)
						{
#ifdef LAP_CUDA_LOCAL_ROWSOL
							cudaStreamSynchronize(stream);
#endif
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
#ifdef LAP_CUDA_LOCAL_ROWSOL
							int i = host_min_private[0].colsol_take;
#else
							int i = colsol[jmin];
#endif
							// initialize Search
							initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
							if (f < dim)
							{
								// get row
								auto tt = iterator.getRow(t, i);
								// continue search
								continueSearchJMinMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], tt, colactive_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size);
							}
							else
							{
								// continue search
								continueSearchJMinMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], colactive_private[t], pred_private[t], i, jmin, min, std::numeric_limits<SC>::max(), size);
							}
							// min is now set so we need to find the correspoding minima for free and taken columns
							findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
							cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
							cudaStreamSynchronize(stream);
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
#ifdef LAP_CUDA_LOCAL_ROWSOL
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#else
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
						}

						// update column prices. can increase or decrease
						// need to copy pred from GPUs here
#ifndef LAP_CUDA_LOCAL_ROWSOL
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
#ifndef LAP_CUDA_LOCAL_ROWSOL
						cudaStreamSynchronize(stream);
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
#ifdef LAP_CUDA_LOCAL_ROWSOL
#endif
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
					cudaStreamSynchronize(stream);
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

						dim3 block_size, grid_size, grid_size_min;
						block_size.x = 256;
						grid_size.x = (size + block_size.x - 1) / block_size.x;
						grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t]);

						for (int f = 0; f < dim2; f++)
						{
#ifndef LAP_CUDA_LOCAL_ROWSOL
							// upload colsol to devices
							cudaMemcpyAsync(colsol_private[t], &(colsol[start]), sizeof(int) * size, cudaMemcpyHostToDevice, stream);
#endif
							// initialize Search
							initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
							// start search and find minimum value
							if (f < dim)
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], pred_private[t], f, size);
							else
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], colactive_private[t], pred_private[t], f, size);
							// min is now set so we need to find the correspoding minima for free and taken columns
							findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
							cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
							cudaStreamSynchronize(stream);
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

							// mark last column scanned (single device)
							if ((jmin >= start) && (jmin < end))
							{
#ifdef LAP_CUDA_LOCAL_ROWSOL
								setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#else
								setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
							}
							while (!unassignedfound)
							{
								// update 'distances' between freerow and all unscanned columns, via next scanned column.
#ifdef LAP_CUDA_LOCAL_ROWSOL
								if ((jmin >= start) && (jmin < end))
								{
									cudaStreamSynchronize(stream);
								}
#pragma omp barrier
								int i = host_min_private[iterator.ws.find(jmin)].colsol_take;
#else
								int i = colsol[jmin];
#endif
								// initialize Search
								initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
								if (f < dim)
								{
									// get row
									auto tt = iterator.getRow(t, i);
									// single device
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(tt_jmin, &(tt[jmin - start]), sizeof(TC), cudaMemcpyDeviceToHost, stream);
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
										cudaStreamSynchronize(stream);
										h2 = tt_jmin[0] - v_jmin[0] - min;
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], tt, colactive_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size);
								}
								else
								{
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
										cudaStreamSynchronize(stream);
										h2 = -v_jmin[0] - min;
									}
									// propagate h2
#pragma omp barrier
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], colactive_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size);
								}
								// min is now set so we need to find the correspoding minima for free and taken columns
								findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
								cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
								cudaStreamSynchronize(stream);
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
								if ((jmin >= start) && (jmin < end))
								{
#ifdef LAP_CUDA_LOCAL_ROWSOL
									setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#else
									setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
								}
#pragma omp barrier
							}

							// update column prices. can increase or decrease
							// need to copy pred from GPUs here
#ifndef LAP_CUDA_LOCAL_ROWSOL
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
#ifndef LAP_CUDA_LOCAL_ROWSOL
							cudaStreamSynchronize(stream);
#pragma omp barrier
#endif
							// reset row and column assignments along the alternating path.
#ifdef LAP_CUDA_LOCAL_ROWSOL
							{
								if ((endofpath >= start) && (endofpath < end))
								{
									resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], host_reset, start, end, endofpath, f);
									cudaStreamSynchronize(stream);
								}
#pragma omp barrier
								while (host_reset->i != -1)
								{
									resetRowColumnAssignmentContinue_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], &(host_reset[0]), &(host_reset[1]), start, end, f);
									cudaStreamSynchronize(stream);
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
						cudaStreamSynchronize(stream);
					} // end of #pragma omp parallel
#else
					for (int f = 0; f < dim2; f++)
					{
						for (int t = 0; t < devices; t++)
						{
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							int end = iterator.ws.part[t].second;
							int size = end - start;
							cudaStream_t stream = iterator.ws.stream[t];
							dim3 block_size, grid_size, grid_size_min;
							block_size.x = 256;
							grid_size.x = (size + block_size.x - 1) / block_size.x;
							grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t]);
#ifndef LAP_CUDA_LOCAL_ROWSOL
							// upload colsol to devices
							cudaMemcpyAsync(colsol_private[t], &(colsol[start]), sizeof(int) * size, cudaMemcpyHostToDevice, stream);
#endif
							// initialize Search
							initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
							// start search and find minimum value
							if (f < dim)
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], iterator.getRow(t, f), colactive_private[t], pred_private[t], f, size);
							else
								initializeSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], colactive_private[t], pred_private[t], f, size);
							// min is now set so we need to find the correspoding minima for free and taken columns
							findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
							cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
						}
						for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
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

						// mark last column scanned (single device)
						{
							int t = iterator.ws.find(jmin);
							cudaSetDevice(iterator.ws.device[t]);
							int start = iterator.ws.part[t].first;
							cudaStream_t stream = iterator.ws.stream[t];
#ifdef LAP_CUDA_LOCAL_ROWSOL
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#else
							setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
						}

						while (!unassignedfound)
						{
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
#ifdef LAP_CUDA_LOCAL_ROWSOL
							cudaStreamSynchronize(iterator.ws.stream[iterator.ws.find(jmin)]);
							int i = host_min_private[iterator.ws.find(jmin)].colsol_take;
#else
							int i = colsol[jmin];
#endif
							if (f < dim)
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
									initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
									if ((jmin >= start) && (jmin < end))
									{
										cudaMemcpyAsync(tt_jmin, &(tt[t][jmin - start]), sizeof(TC), cudaMemcpyDeviceToHost, stream);
										cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
									}
								}
								// single device
								cudaStreamSynchronize(iterator.ws.stream[iterator.ws.find(jmin)]);
								h2 = tt_jmin[0] - v_jmin[0] - min;
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size, grid_size_min;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t]);
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], tt[t], colactive_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size);
								}
							}
							else
							{
								{
									int t = iterator.ws.find(jmin);
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									cudaStream_t stream = iterator.ws.stream[t];
									cudaMemcpyAsync(v_jmin, &(v_private[t][jmin - start]), sizeof(SC), cudaMemcpyDeviceToHost, stream);
									cudaStreamSynchronize(stream);
								}
								h2 = -v_jmin[0] - min;
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									int size = end - start;
									cudaStream_t stream = iterator.ws.stream[t];
									dim3 block_size, grid_size, grid_size_min;
									block_size.x = 256;
									grid_size.x = (size + block_size.x - 1) / block_size.x;
									grid_size_min.x = std::min(grid_size.x, (unsigned int)iterator.ws.sm_count[t]);
									// initialize Search
									initializeMin_kernel<<<1, 1, 0, stream>>>(min_private[t], std::numeric_limits<SC>::max(), dim2);
									// continue search
									continueSearchMin_kernel<<<grid_size_min, block_size, 0, stream>>>(&(min_private[t]->min), v_private[t], d_private[t], colactive_private[t], pred_private[t], i, h2, std::numeric_limits<SC>::max(), size);
								}
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
								// min is now set so we need to find the correspoding minima for free and taken columns
								findMin_kernel<<<grid_size, block_size, 0, stream>>>(min_private[t], d_private[t], colsol_private[t], start, end);
								cudaMemcpyAsync(&(host_min_private[t]), min_private[t], sizeof(min_struct<SC>), cudaMemcpyDeviceToHost, stream);
							}
							for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
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
							// mark last column scanned (single device)
							{
								int t = iterator.ws.find(jmin);
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								cudaStream_t stream = iterator.ws.stream[t];
#ifdef LAP_CUDA_LOCAL_ROWSOL
								setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], &(host_min_private[t]), colsol_private[t], std::numeric_limits<SC>::max(), jmin - start);
#else
								setColInactive_kernel<<<1, 1, 0, iterator.ws.stream[t]>>>(colactive_private[t], d_private[t], d2_private[t], std::numeric_limits<SC>::max(), jmin - start);
#endif
							}
						}

						// update column prices. can increase or decrease
						// need to copy pred from GPUs here
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
#ifndef LAP_CUDA_LOCAL_ROWSOL
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
						}
#ifndef LAP_CUDA_LOCAL_ROWSOL
						for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
#endif
						// reset row and column assignments along the alternating path.
#ifdef LAP_CUDA_LOCAL_ROWSOL
						{
							{
								int t = iterator.ws.find(endofpath);
								cudaSetDevice(iterator.ws.device[t]);
								int start = iterator.ws.part[t].first;
								int end = iterator.ws.part[t].second;
								cudaStream_t stream = iterator.ws.stream[t];
								resetRowColumnAssignment_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], host_reset, start, end, endofpath, f);
								cudaStreamSynchronize(stream);
							}
							while (host_reset->i != -1)
							{
								for (int t = 0; t < devices; t++)
								{
									cudaSetDevice(iterator.ws.device[t]);
									int start = iterator.ws.part[t].first;
									int end = iterator.ws.part[t].second;
									cudaStream_t stream = iterator.ws.stream[t];
									resetRowColumnAssignmentContinue_kernel<<<1, 1, 0, stream>>>(pred_private[t], colsol_private[t], rowsol_private[t], &(host_reset[0]), &(host_reset[1]), start, end, f);
								}
								for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
								std::swap(host_reset[0], host_reset[1]);
							}
						}
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
						cudaMemcpyAsync(&(v2[start]), v_private[t], sizeof(SC) * size, cudaMemcpyDeviceToHost, stream);
					}
					for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
#endif
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
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(iterator.ws.device[t]);
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int size = end - start;
					cudaStream_t stream = iterator.ws.stream[t];
					cudaMemcpyAsync(&(colsol[start]), colsol_private[t], sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
				}
				for (int t = 0; t < devices; t++) cudaStreamSynchronize(iterator.ws.stream[t]);
				for (int j = 0; j < dim2; j++) rowsol[colsol[j]] = j;
				cudaFreeHost(colsol);
			}
#endif
			// free CUDA memory
			for (int t = 0; t < devices; t++)
			{
				cudaSetDevice(iterator.ws.device[t]);
				cudaFree(min_private[t]);
				cudaFree(colactive_private[t]);
				cudaFree(d_private[t]);
				cudaFree(v_private[t]);
				if (temp_storage[t] != 0) cudaFree(temp_storage[t]);
				cudaFree(pred_private[t]);
				cudaFree(colsol_private[t]);
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
#ifdef LAP_CUDA_OPENMP
			omp_set_num_threads(old_threads);
#else
			lapFree(tt);
#endif
			// set device back to first one
			cudaSetDevice(iterator.ws.device[0]);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			SC my_cost(0);
			SC *row = new SC[dim2];
			int *d_rowsol;
			SC *d_row;
			cudaMalloc(&d_rowsol, dim2 * sizeof(int));
			cudaMalloc(&d_row, dim2 * sizeof(SC));
			cudaMemcpyAsync(d_rowsol, rowsol, dim2 * sizeof(int), cudaMemcpyHostToDevice, stream);
			costfunc.getCost(d_row, 0, d_rowsol, dim2);
			cudaMemcpyAsync(row, d_row, dim2 * sizeof(SC), cudaMemcpyDeviceToHost, stream);
			cudaFree(d_row);
			cudaFree(d_rowsol);
			cudaStreamSynchronize(stream);
			for (int i = 0; i < dim2; i++) my_cost += row[i];
			delete[] row;
			return my_cost;
		}

		// shortcut for square problems
		template <class SC, class TC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol)
		{
			lap::cuda::solve<SC, TC>(dim, dim, costfunc, iterator, rowsol);
		}

		// shortcut for square problems
		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol, cudaStream_t stream)
		{
			return lap::cuda::cost<SC, CF>(dim, dim, costfunc, rowsol, stream);
		}
	}
}
