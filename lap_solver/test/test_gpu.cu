#define LAP_CUDA
#define LAP_QUIET
//#define LAP_DISPLAY_EVALUATED
//#define LAP_DEBUG
//#define LAP_NO_MEM_DEBUG
//#define LAP_ROWS_SCANNED
#include "../lap.h"

#include <random>
#include <string>
#include "test_options.h"

#include <cuda.h>

template <class C> void testGeometricCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool epsilon, bool disjoint, std::string name_C);

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
			if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, false, std::string("double"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, true, std::string("double"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, false, std::string("double"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, true, std::string("double"));
		}
	}
	if (opt.use_float)
	{
		if (opt.use_single)
		{
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, true, std::string("float"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, true, std::string("float"));
		}
	}

	return 0;
}

template <class C>
__global__
void getCostRow_kernel(C *cost, C *tab_s, C *tab_t, int x, int start, int end, int N)
{
	int y = start + threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= end) return;

	C d0 = tab_s[x] - tab_t[y];
	C d1 = tab_s[x + N] - tab_t[y + N];
	cost[threadIdx.x + blockIdx.x * blockDim.x] = d0 * d0 + d1 * d1;
}

template <class C>
__global__
void getCost_kernel(C *cost, C *tab_s, C *tab_t, int *rowsol, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = rowsol[x];

	float d0 = tab_s[x] - tab_t[y];
	float d1 = tab_s[x + N] - tab_t[y + N];
	cost[x] = d0 * d0 + d1 * d1;
}

template <class C>
void testGeometricCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool epsilon, bool disjoint, std::string name_C)
{
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Geometric";
			if (disjoint) std::cout << " Disjoint";
			std::cout << " R^2<" << name_C << "> " << N << "x" << N << " (" << (double)max_memory / 1073741824.0 << "GB / GPU)";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			C *tab_s = new C[2 * N];
			C *tab_t = new C[2 * N];
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
						tab_t[i] += C(1.0);
					}
					else
					{
						tab_s[i] += C(1.0);
						tab_s[i + N] += C(1.0);
						tab_t[i + N] += C(1.0);
					}
				}
			}

			// enabled function
			lap::cuda::Worksharing ws(N, 256);
			int num_enabled = (int)ws.device.size();

			C **d_tab_s = new C*[num_enabled];
			C **d_tab_t = new C*[num_enabled];

			for (int i = 0; i < num_enabled; i++)
			{
				d_tab_s[i] = 0;
				d_tab_t[i] = 0;
			}

			for (int i = 0; i < num_enabled; i++)
			{
				cudaSetDevice(ws.device[i]);
				cudaMalloc(&(d_tab_s[i]), 2 * N * sizeof(C));
				cudaMalloc(&(d_tab_t[i]), 2 * N * sizeof(C));
				cudaMemcpy(d_tab_s[i], tab_s, 2 * N * sizeof(C), cudaMemcpyHostToDevice);
				cudaMemcpy(d_tab_t[i], tab_t, 2 * N * sizeof(C), cudaMemcpyHostToDevice);
			}

			int *rowsol = new int[N];

			// cost function
			auto get_cost_row = [d_tab_s, d_tab_t, N](C *d_row, int t, int x, int start, int end)
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_kernel<<<grid_size, block_size>>>(d_row, d_tab_s[t], d_tab_t[t], x, start, end, N);
			};

			lap::cuda::RowCostFunction<C, decltype(get_cost_row)> costFunction(get_cost_row);

			// different cache size, so always use SLRU
			lap::cuda::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, max_memory / sizeof(C), costFunction, ws);
			lap::displayTime(start_time, "setup complete", std::cout);
			if (epsilon) costFunction.setInitialEpsilon(lap::cuda::guessEpsilon<C>(N, N, iterator));

			lap::cuda::solve<C, C>(N, costFunction, iterator, rowsol);

			{
				// set device back to 0
				cudaSetDevice(ws.device[0]);
				std::stringstream ss;
				C my_cost(0);
				C *row = new C[N];
				// calculate costs directly
				{
					int *d_rowsol;
					C *d_row;
					cudaMalloc(&d_rowsol, N * sizeof(int));
					cudaMalloc(&d_row, N * sizeof(C));
					cudaMemcpy(d_rowsol, rowsol, N * sizeof(int), cudaMemcpyHostToDevice);
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (N + block_size.x - 1) / block_size.x;
					getCost_kernel<<<grid_size, block_size>>>(d_row, d_tab_s[0], d_tab_t[0], d_rowsol, N);
					cudaMemcpy(row, d_row, N * sizeof(C), cudaMemcpyDeviceToHost);
					cudaFree(d_row);
					cudaFree(d_rowsol);
				}
				for (int i = 0; i < N; i++) my_cost += row[i];
				delete[] row;
				ss << "cost = " << my_cost;// lap::cost<C, float>(N, costFunction, rowsol);
				lap::displayTime(start_time, ss.str().c_str(), std::cout);
			}

			for (int i = 0; i < num_enabled; i++)
			{
				cudaSetDevice(ws.device[i]);
				cudaFree(d_tab_s[i]);
				cudaFree(d_tab_t[i]);
			}

			delete[] rowsol;
			delete[] tab_s;
			delete[] tab_t;
			delete[] d_tab_s;
			delete[] d_tab_t;
		}
	}
}
