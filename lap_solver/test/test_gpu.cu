#ifdef _OPENMP
#  define LAP_OPENMP
#endif
#define LAP_QUIET
//#define LAP_DEBUG
#include "../lap.h"

#include <random>
#include <string>
#include "test_options.h"

#include <cuda.h>

template <class C> void testGPUGeometricCached(long long max_tab, long long max_cached, bool omp, bool epsilon, bool disjoint, std::string name_C);

int main(int argc, char* argv[])
{
	Options opt;
	int r = opt.parseOptions(argc, argv);
	if (r != 0) return r;

	if (opt.use_double)
	{
		// note: cost function type is always float here and random is not supported, also tables are not tested here
		if (opt.use_single)
		{
			if (opt.run_geometric_cached) testGPUGeometricCached<double>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, false, false, std::string("double"));
			if (opt.run_geometric_disjoint_cached) testGPUGeometricCached<double>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, false, true, std::string("double"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_geometric_cached) testGPUGeometricCached<double>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, true, false, std::string("double"));
			if (opt.run_geometric_disjoint_cached) testGPUGeometricCached<double>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, true, true, std::string("double"));
		}
	}
	if (opt.use_float)
	{
		if (opt.use_single)
		{
			if (opt.run_geometric_cached) testGPUGeometricCached<float>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, false, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGPUGeometricCached<float>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, false, true, std::string("float"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_geometric_cached) testGPUGeometricCached<float>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, true, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGPUGeometricCached<float>(opt.lap_max_tab, opt.lap_max_cached, opt.use_omp, true, true, std::string("float"));
		}
	}

	return 0;
}

__global__
void getCostRow_kernel(float *cost, float *tab_s, float *tab_t, int x, int start, int end, int N)
{
	int y = start + threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= end) return;

	float d0 = tab_s[x] - tab_t[y];
	float d1 = tab_s[x + N] - tab_t[y + N];
	cost[threadIdx.x + blockIdx.x * blockDim.x] = d0 * d0 + d1 * d1;
}

__global__
void getCost_kernel(float *cost, float *tab_s, float *tab_t, int *rowsol, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = rowsol[x];

	float d0 = tab_s[x] - tab_t[y];
	float d1 = tab_s[x + N] - tab_t[y + N];
	cost[x] = d0 * d0 + d1 * d1;
}

template <class C>
void testGPUGeometricCached(long long max_tab, long long max_cached, bool omp, bool epsilon, bool disjoint, std::string name_C)
{
	for (long long NN = (max_tab * max_tab << 1); NN <= max_cached * max_cached; NN <<= 1)
	{
		int N = (int)floor(sqrt((double)NN));
		int entries = std::min(N, (int)((max_tab * max_tab) / N));

		std::cout << "Geometric";
		if (disjoint) std::cout << " Disjoint";
		std::cout << " R^2<" << name_C << "> " << N << "x" << N << " (" << entries << ")";
		if (omp) std::cout << " multithreaded";
		if (epsilon) std::cout << " with epsilon scaling";
		std::cout << std::endl;

		auto start_time = std::chrono::high_resolution_clock::now();

		std::uniform_real_distribution<float> distribution(0.0, 1.0);
		std::mt19937_64 generator(1234);

		float *tab_s = new float[2 * N];
		float *tab_t = new float[2 * N];
		for (int i = 0; i < N; i++)
		{
			tab_s[i] = distribution(generator);
			tab_t[i] = distribution(generator);
			tab_s[i + N] = distribution(generator);
			tab_t[i + N] = distribution(generator);
		}

		// order of coordinates is different, first all x then all y
		if (disjoint)
		{
			for (int i = 0; i < N; i++)
			{
				if ((i << 1) < N)
				{
					tab_t[i] += 1.0f;
				}
				else
				{
					tab_s[i] += 1.0f;
					tab_s[i + N] += 1.0f;
					tab_t[i + N] += 1.0f;
				}
			}
		}

		// enabled function
		int num_threads;

#ifdef LAP_OPENMP
		if (omp) num_threads = omp_get_max_threads();
		else num_threads = 1;
#else
		num_threads = 1;
#endif

		bool *enabled = new bool[num_threads];
		int *device = new int[num_threads];
		void **d_tab_s = new void*[num_threads];
		void **d_tab_t = new void*[num_threads];
		void **d_row = new void*[num_threads];

		int device_count;
		cudaDeviceProp deviceProp;
		cudaGetDeviceCount(&device_count);

		int num_enabled = 0;

		for (int i = 0; i < num_threads; i++)
		{
			enabled[i] = false;
			device[i] = -1;
			d_tab_s[i] = 0;
			d_tab_t[i] = 0;
			d_row[i] = 0;
		}

		for (int current_device = 0; (current_device < device_count) && (num_enabled < num_threads); current_device++)
		{
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
			if (deviceProp.computeMode != cudaComputeModeProhibited)
			{
				enabled[num_enabled] = true;
				device[num_enabled] = current_device;
				num_enabled++;
			}
		}

		if (num_enabled == 0)
		{
			std::cout << "No suitable CUDA device found." << std::endl;
			exit(-1);
		}

		for (int i = 0; i < num_enabled; i++)
		{
			cudaSetDevice(device[i]);
			cudaMalloc(&(d_tab_s[i]), 2 * N * sizeof(float));
			cudaMalloc(&(d_tab_t[i]), 2 * N * sizeof(float));
			cudaMalloc(&(d_row[i]), N * sizeof(float));
			cudaMemcpy(d_tab_s[i], tab_s, 2 * N * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_tab_t[i], tab_t, 2 * N * sizeof(float), cudaMemcpyHostToDevice);
		}

		int *rowsol = new int[N];

		if (omp)
		{
#ifdef LAP_OPENMP
			// set devices
#pragma omp parallel
			{
				int t = omp_get_thread_num();
				if (enabled[t]) cudaSetDevice(device[t]);
			}

			auto get_enabled = [enabled](int t) -> bool
			{
				return enabled[t];
			};

			// cost function
			auto get_cost_row = [d_tab_s, d_tab_t, d_row, N](float *row, int x, int start, int end)
			{
				int t = omp_get_thread_num();
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_kernel<<<grid_size, block_size>>>((float *)d_row[t], (float *)d_tab_s[t], (float *)d_tab_t[t], x, start, end, N);
				cudaMemcpy(row, d_row[t], (end - start) * sizeof(float), cudaMemcpyDeviceToHost);
			};

			lap::omp::RowCostFunction<float, decltype(get_enabled), decltype(get_cost_row)> costFunction(get_enabled, get_cost_row, 256);
			lap::omp::Worksharing ws(N, costFunction.getMultiple());

			if (4 * entries < N)
			{
				lap::omp::CachingIterator<C, float, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction, ws);
//				lap::omp::CachingIterator<C, float, decltype(costFunction), lap::omp::CacheSLRU> iterator(N, N, entries, costFunction);
				if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<float>(N, N, iterator));

				lap::displayTime(start_time, "setup complete", std::cout);
				lap::omp::solve<C>(N, costFunction, iterator, rowsol);
			}
			else
			{
				lap::omp::CachingIterator<C, float, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction, ws);
//				lap::omp::CachingIterator<C, float, decltype(costFunction), lap::omp::CacheLFU> iterator(N, N, entries, costFunction);
				if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<float>(N, N, iterator));

				lap::displayTime(start_time, "setup complete", std::cout);
				lap::omp::solve<C>(N, costFunction, iterator, rowsol);
			}
#endif
		}
		else
		{
			// cost function
			auto get_cost_row = [d_tab_s, d_tab_t, d_row, N](float *row, int x, int start, int end)
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_kernel<<<grid_size, block_size>>>((float *)d_row[0], (float *)d_tab_s[0], (float *)d_tab_t[0], x, start, end, N);
				cudaMemcpy(row, d_row[0], (end - start) * sizeof(float), cudaMemcpyDeviceToHost);
			};

			lap::RowCostFunction<float, decltype(get_cost_row)> costFunction(get_cost_row);

			if (4 * entries < N)
			{
				lap::CachingIterator<C, float, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction);
				if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<float>(N, N, iterator));

				lap::displayTime(start_time, "setup complete", std::cout);
				lap::solve<C>(N, costFunction, iterator, rowsol);
			}
			else
			{
				lap::CachingIterator<C, float, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction);
				if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<float>(N, N, iterator));

				lap::displayTime(start_time, "setup complete", std::cout);
				lap::solve<C>(N, costFunction, iterator, rowsol);
			}
		}

		{
			// set device back yo 0
			if (num_enabled > 1) cudaSetDevice(device[0]);
			std::stringstream ss;
			float my_cost = 0.0f;
			float *row = new float[N];
			// calculate costs directly
			{
				void *d_rowsol;
				cudaMalloc(&d_rowsol, N * sizeof(int));
				cudaMemcpy(d_rowsol, rowsol, N * sizeof(int), cudaMemcpyHostToDevice);
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (N + block_size.x - 1) / block_size.x;
				getCost_kernel<<<grid_size, block_size>>>((float *)d_row[0], (float *)d_tab_s[0], (float *)d_tab_t[0], (int *)d_rowsol, N);
				cudaMemcpy(row, d_row[0], N * sizeof(float), cudaMemcpyDeviceToHost);
				cudaFree(d_rowsol);
			}
			for (int i = 0; i < N; i++) my_cost += row[i];
			delete[] row;
			ss << "cost = " << my_cost;// lap::cost<C, float>(N, costFunction, rowsol);
			lap::displayTime(start_time, ss.str().c_str(), std::cout);
		}

		for (int i = 0; i < num_enabled; i++)
		{
			cudaSetDevice(device[i]);
			cudaFree(d_tab_s[i]);
			cudaFree(d_tab_t[i]);
			cudaFree(d_row[i]);
		}

		delete[] rowsol;
		delete[] tab_s;
		delete[] tab_t;
		delete[] enabled;
		delete[] device;
		delete[] d_tab_s;
		delete[] d_tab_t;
		delete[] d_row;
	}
}
