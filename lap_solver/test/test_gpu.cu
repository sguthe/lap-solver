#define LAP_CUDA
#define LAP_QUIET
//#define LAP_DISPLAY_EVALUATED
//#define LAP_DEBUG
//#define LAP_NO_MEM_DEBUG
//#define LAP_ROWS_SCANNED
// these two don't work together at the moment
#ifndef LAP_ROWS_SCANNED
# define LAP_CUDA_LOCAL_ROWSOL
#endif
// should only be enabled for testing purposes
//#define LAP_CUDA_ALLOW_WDDM
// enable one thread per GPU
#define LAP_CUDA_OPENMP

#include "../lap.h"

#include <random>
#include <string>
#include "test_options.h"
#include "image.h"

#include <cuda.h>
#include <cuda_runtime.h>

template <class C> void testRandom(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testSanity(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testSanityCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testGeometric(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, bool disjoint, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testGeometricCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool epsilon, bool disjoint, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testRandomLowRank(long long min_tab, long long max_tab, long long max_memory, long long min_rank, long long max_rank, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testRandomLowRankCached(long long min_cached, long long max_cached, long long max_memory, long long min_rank, long long max_rank, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testImages(std::vector<std::string> &images, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);
template <class C> void testInteger(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent);


int main(int argc, char* argv[])
{
	Options opt;
	int r = opt.parseOptions(argc, argv);
	if (r != 0) return r;

	if (opt.use_double)
	{
		if (opt.use_single)
		{
			if (opt.run_sanity) testSanity<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_sanity_cached) testSanityCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, true, std::string("double"), opt.devices, opt.silent);
			if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_memory, opt.runs, false, std::string("double"), opt.devices, opt.silent);
		}
		if (opt.use_epsilon)
		{
			if (opt.run_sanity) testSanity<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_sanity_cached) testSanityCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, true, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, false, std::string("double"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, true, std::string("double"), opt.devices, opt.silent);
			if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_memory, opt.runs, true, std::string("double"), opt.devices, opt.silent);
		}
	}
	if (opt.use_float)
	{
		if (opt.use_single)
		{
			if (opt.run_sanity) testSanity<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_sanity_cached) testSanityCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, false, true, std::string("float"), opt.devices, opt.silent);
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_memory, opt.runs, false, std::string("float"), opt.devices, opt.silent);
		}
		if (opt.use_epsilon)
		{
			if (opt.run_sanity) testSanity<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_sanity_cached) testSanityCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.lap_min_rank, opt.lap_max_rank, opt.runs, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, true, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, false, std::string("float"), opt.devices, opt.silent);
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_min_cached, opt.lap_max_cached, opt.lap_max_memory, opt.runs, true, true, std::string("float"), opt.devices, opt.silent);
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_memory, opt.runs, true, std::string("float"), opt.devices, opt.silent);
		}
	}
	if (opt.run_integer)
	{
		if (opt.use_double)
		{
			if (opt.use_single) testInteger<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("double"), opt.devices, opt.silent);
			if (opt.use_epsilon) testInteger<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("double"), opt.devices, opt.silent);
		}
		if (opt.use_float)
		{
			if (opt.use_single) testInteger<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("float"), opt.devices, opt.silent);
			if (opt.use_epsilon) testInteger<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("float"), opt.devices, opt.silent);
		}
		if (opt.use_single) testInteger<long long>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, false, std::string("long long"), opt.devices, opt.silent);
		if (opt.use_epsilon) testInteger<long long>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_max_memory, opt.runs, true, std::string("long long"), opt.devices, opt.silent);
	}

	return 0;
}

template <class SC, class TC, class RCF, class CF, class TP>
void solveCachingCUDA(TP &start_time, int N1, int N2, RCF &get_cost_row, CF &get_cost, lap::cuda::Worksharing &ws, long long max_memory, int *rowsol, bool epsilon)
{
	lap::cuda::RowCostFunction<TC, decltype(get_cost_row), decltype(get_cost)> costFunction(get_cost_row, get_cost);

	// different cache size, so always use SLRU
	lap::cuda::CachingIterator<SC, TC, decltype(costFunction), lap::CacheSLRU> iterator(N1, N2, max_memory / sizeof(TC), costFunction, ws);
	lap::displayTime(start_time, "setup complete", std::cout);
	if (epsilon) costFunction.setInitialEpsilon(lap::cuda::guessEpsilon<SC, TC>(N1, N2, iterator));

	lap::cuda::solve<SC, TC>(N1, N2, costFunction, iterator, rowsol);

	std::stringstream ss;
	ss << "cost = " << lap::cuda::cost<SC>(N1, N2, costFunction, rowsol, ws.stream[0]);
	lap::displayTime(start_time, ss.str().c_str(), std::cout);
}

template <class SC, class TC, class CF, class TP>
void solveTableCUDA(TP &start_time, int N1, int N2, CF &get_cost_cpu, lap::cuda::Worksharing &ws, long long max_memory, int *rowsol, bool epsilon)
{
	lap::SimpleCostFunction<int, decltype(get_cost_cpu)> cpuCostFunction(get_cost_cpu);
	lap::TableCost<int> costMatrix(N1, N2, cpuCostFunction);

	// cost function (copy data from table)
	auto get_cost_row = [&costMatrix](int *d_row, int t, cudaStream_t stream, int x, int start, int end)
	{
		cudaMemcpyAsync(d_row, costMatrix.getRow(x) + start, (end - start) * sizeof(int), cudaMemcpyHostToDevice, stream);
	};

	// cost function
	auto get_cost = [](TC *d_row, cudaStream_t stream, int *d_rowsol, int N)
	{
		std::cerr << "Function not supported, use lap::cost() on cpu cost matrix instead." << std::endl;
	};

	lap::cuda::RowCostFunction<TC, decltype(get_cost_row), decltype(get_cost)> costFunction(get_cost_row, get_cost);

	// different cache size, so always use SLRU
	lap::cuda::CachingIterator<SC, TC, decltype(costFunction), lap::CacheSLRU> iterator(N1, N2, max_memory / sizeof(TC), costFunction, ws);
	lap::displayTime(start_time, "setup complete", std::cout);
	if (epsilon) costFunction.setInitialEpsilon((int)lap::cuda::guessEpsilon<SC, TC>(N1, N2, iterator));

	lap::cuda::solve<SC, TC>(N1, N2, costFunction, iterator, rowsol);

	std::stringstream ss;
	ss << "cost = " << lap::cost<SC>(N1, N2, costMatrix, rowsol);
	lap::displayTime(start_time, ss.str().c_str(), std::cout);
}


template <class C>
__global__
void getCostRow_geometric_kernel(C *cost, C *tab_s, C *tab_t, int x, int start, int end, int N)
{
	int y = start + threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= end) return;

	C d0 = tab_s[x] - tab_t[y];
	C d1 = tab_s[x + N] - tab_t[y + N];
	cost[threadIdx.x + blockIdx.x * blockDim.x] = d0 * d0 + d1 * d1;
}

template <class C>
__global__
void getCost_geometric_kernel(C *cost, C *tab_s, C *tab_t, int *rowsol, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = rowsol[x];

	float d0 = tab_s[x] - tab_t[y];
	float d1 = tab_s[x + N] - tab_t[y + N];
	cost[x] = d0 * d0 + d1 * d1;
}

template <class C>
void testGeometricCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool epsilon, bool disjoint, std::string name_C, std::vector<int> &devs, bool silent)
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
			lap::cuda::Worksharing ws(N, 256, devs, silent);
			int num_enabled = (int)ws.device.size();

			int step = (int)N / (int)std::min((long long)N, (long long)((num_enabled * max_memory) / (sizeof(C) * N)));

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
			auto get_cost_row = [&d_tab_s, &d_tab_t, &N](C *d_row, int t, cudaStream_t stream, int x, int start, int end)
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_geometric_kernel<<<grid_size, block_size, 0, stream>>>(d_row, d_tab_s[t], d_tab_t[t], x, start, end, N);
			};

			// cost function
			auto get_cost = [&d_tab_s, &d_tab_t](C *d_row, cudaStream_t stream, int *d_rowsol, int N)
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (N + block_size.x - 1) / block_size.x;
				getCost_geometric_kernel<<<grid_size, block_size, 0, stream>>>(d_row, d_tab_s[0], d_tab_t[0], d_rowsol, N);
			};

			solveCachingCUDA<C, C>(start_time, N, N, get_cost_row, get_cost, ws, max_memory, rowsol, epsilon);

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

template <class C>
__global__
void getCostRow_sanity_kernel(C *cost, C *vec, int x, int start, int end, int N)
{
	int y = start + threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= end) return;

	C r = vec[x] + vec[y + N];
	if (x != y) r += C(0.1);

	cost[threadIdx.x + blockIdx.x * blockDim.x] = r;
}

template <class C>
__global__
void getCost_sanity_kernel(C *cost, C *vec, int *rowsol, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = rowsol[x];

	C r = vec[x] + vec[y + N];
	if (x != y) r += C(0.1);

	cost[x] = r;
}

template <class C>
__global__
void getGTCost_sanity_kernel(C *cost, C *vec, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = x;

	C r = vec[x] + vec[y + N];
	if (x != y) r += C(0.1);

	cost[x] = r;
}

template <class C>
void testSanityCached(long long min_cached, long long max_cached, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Sanity<" << name_C << "> " << N << "x" << N << " (" << (double)max_memory / 1073741824.0 << "GB / GPU)";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			C *vec = new C[N << 1];

			for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);

			// enabled function
			lap::cuda::Worksharing ws(N, 256, devs, silent);
			int num_enabled = (int)ws.device.size();

			int step = (int)N / (int)std::min((long long)N, (long long)((num_enabled * max_memory) / (sizeof(C) * N)));

			C **d_vec = new C*[num_enabled];

			for (int i = 0; i < num_enabled; i++)
			{
				d_vec[i] = 0;
			}

			for (int i = 0; i < num_enabled; i++)
			{
				cudaSetDevice(ws.device[i]);
				cudaMalloc(&(d_vec[i]), 2 * N * sizeof(C));
				cudaMemcpy(d_vec[i], vec, 2 * N * sizeof(C), cudaMemcpyHostToDevice);
			}

			int *rowsol = new int[N];

			// cost function
			auto get_cost_row = [&d_vec, &N](C *d_row, int t, cudaStream_t stream, int x, int start, int end)
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
				getCostRow_sanity_kernel<<<grid_size, block_size, 0, stream>>>(d_row, d_vec[t], x, start, end, N);
			};

			// cost function
			auto get_cost = [&d_vec](C *d_row, cudaStream_t stream, int *d_rowsol, int N)
			{
				dim3 block_size, grid_size;
				block_size.x = 256;
				grid_size.x = (N + block_size.x - 1) / block_size.x;
				getCost_sanity_kernel<<<grid_size, block_size, 0, stream>>>(d_row, d_vec[0], d_rowsol, N);
			};

			solveCachingCUDA<C, C>(start_time, N, N, get_cost_row, get_cost, ws, max_memory, rowsol, epsilon);

			bool passed = true;
			for (long long i = 0; (passed) && (i < N); i++)
			{
				passed &= (rowsol[i] == i);
			}
			std::stringstream ss;
			if (passed) ss << "test passed: ";
			else ss << "test failed: ";
			{
				// set device back to 0
				cudaSetDevice(ws.device[0]);
				C my_cost(0);
				C *row = new C[N];
				// calculate costs directly
				{
					C *d_row;
					cudaMalloc(&d_row, N * sizeof(C));
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (N + block_size.x - 1) / block_size.x;
					getGTCost_sanity_kernel<<<grid_size, block_size>>>(d_row, d_vec[0], N);
					cudaMemcpy(row, d_row, N * sizeof(C), cudaMemcpyDeviceToHost);
					cudaFree(d_row);
				}
				for (int i = 0; i < N; i++) my_cost += row[i];
				delete[] row;
				ss << "ground truth cost = " << my_cost;
			}
			lap::displayTime(start_time, ss.str().c_str(), std::cout);

			for (int i = 0; i < num_enabled; i++)
			{
				cudaSetDevice(ws.device[i]);
				cudaFree(d_vec[i]);
			}

			delete[] rowsol;
			delete[] vec;
			delete[] d_vec;
		}
	}
}

template <class C>
__global__
void getCostRow_lowRank_kernel(C *cost, C *vec, int rank, int x, int start, int end, int N)
{
	int y = start + threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= end) return;

	C sum(0);
	for (long long k = 0; k < rank; k++)
	{
		sum += vec[k * N + x] * vec[k * N + y];
	}
	sum /= C(rank);

	cost[threadIdx.x + blockIdx.x * blockDim.x] = sum;
}

template <class C>
__global__
void getCost_lowRank_kernel(C *cost, C *vec, int rank, int *rowsol, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = rowsol[x];

	C sum(0);
	for (long long k = 0; k < rank; k++)
	{
		sum += vec[k * N + x] * vec[k * N + y];
	}
	sum /= C(rank);

	cost[x] = sum;
}

template <class C>
void testRandomLowRankCached(long long min_cached, long long max_cached, long long max_memory, long long min_rank, long long max_rank, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	for (long long rank = min_rank; rank <= max_rank; rank <<= 1)
	{
		for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));
				int entries = (int)std::min((long long)N, (long long)(max_memory / (sizeof(C) * N)));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " (" << entries << ") rank = " << rank;
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(0.0, 1.0);
				std::mt19937_64 generator(1234);

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
				}

				// enabled function
				lap::cuda::Worksharing ws(N, 256, devs, silent);
				int num_enabled = (int)ws.device.size();

				int step = (int)N / (int)std::min((long long)N, (long long)((num_enabled * max_memory) / (sizeof(C) * N)));

				C **d_vec = new C*[num_enabled];

				for (int i = 0; i < num_enabled; i++)
				{
					d_vec[i] = 0;
				}

				for (int i = 0; i < num_enabled; i++)
				{
					cudaSetDevice(ws.device[i]);
					cudaMalloc(&(d_vec[i]), 2 * N * sizeof(C));
					cudaMemcpy(d_vec[i], vec, 2 * N * sizeof(C), cudaMemcpyHostToDevice);
				}

				int *rowsol = new int[N];

				// cost function
				auto get_cost_row = [&d_vec, &N, &rank](C *d_row, int t, cudaStream_t stream, int x, int start, int end)
				{
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
					getCostRow_lowRank_kernel<<<grid_size, block_size, 0, stream>>>(d_row, d_vec[t], (int)rank, x, start, end, N);
				};

				// cost function
				auto get_cost = [&d_vec, &rank](C *d_row, cudaStream_t stream, int *d_rowsol, int N)
				{
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (N + block_size.x - 1) / block_size.x;
					getCost_lowRank_kernel<<<grid_size, block_size, 0, stream>>>(d_row, d_vec[0], (int)rank, d_rowsol, N);
				};

				solveCachingCUDA<C, C>(start_time, N, N, get_cost_row, get_cost, ws, max_memory, rowsol, epsilon);

				for (int i = 0; i < num_enabled; i++)
				{
					cudaSetDevice(ws.device[i]);
					cudaFree(d_vec[i]);
				}

				delete[] rowsol;
				delete[] vec;
				delete[] d_vec;
			}
		}
	}
}

template <class C>
void testInteger(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	// random costs (directly supply cost matrix)
	for (int range = 0; range < 3; range++)
	{
		for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));

				std::cout << "Integer";
				std::cout << "<" << name_C << " ";
				if (range == 0) std::cout << "1/10n";
				else if (range == 1) std::cout << "n";
				else std::cout << "10n";
				std::cout << "> " << N << "x" << N << " table";
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				int n;
				if (range == 0) n = N / 10;
				else if (range == 1) n = N;
				else n = 10 * N;
				std::uniform_int_distribution<int> distribution(0, n);
				std::mt19937_64 generator(1234);

				auto start_time = std::chrono::high_resolution_clock::now();

				auto get_cost = [&distribution, &generator](int x, int y) -> int
				{
					return distribution(generator);
				};

				lap::cuda::Worksharing ws(N, 256, devs, silent);

				int *rowsol = new int[N];

				solveTableCUDA<C, int>(start_time, N, N, get_cost, ws, max_memory, rowsol, epsilon);

				delete[] rowsol;
			}
		}
	}
}

template <class C> void testRandom(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Random";
			std::cout << "<" << name_C << "> " << N << "x" << N << " table";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			auto start_time = std::chrono::high_resolution_clock::now();

			int *rowsol = new int[N];

			auto get_cost = [&distribution, &generator](int x, int y) -> C
			{
				return distribution(generator);
			};

			lap::cuda::Worksharing ws(N, 256, devs, silent);

			solveTableCUDA<C, int>(start_time, N, N, get_cost, ws, max_memory, rowsol, epsilon);

			delete[] rowsol;
		}
	}
}

template <class C> void testSanity(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Sanity";
			std::cout << "<" << name_C << "> " << N << "x" << N << " table";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			auto start_time = std::chrono::high_resolution_clock::now();

			int *rowsol = new int[N];

			C *vec = new C[N << 1];

			for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);

			// cost functions
			auto get_cost = [&vec, &N](int x, int y) -> C
			{
				C r = vec[x] + vec[y + N];
				if (x == y) return r;
				else return r + C(0.1);
			};


			lap::cuda::Worksharing ws(N, 256, devs, silent);

			solveTableCUDA<C, int>(start_time, N, N, get_cost, ws, max_memory, rowsol, epsilon);

			bool passed = true;
			for (long long i = 0; (passed) && (i < N); i++)
			{
				passed &= (rowsol[i] == i);
			}
			std::stringstream ss;
			if (passed) ss << "test passed: ";
			else ss << "test failed: ";
			C real_cost(0);
			for (int i = 0; i < N; i++) real_cost += get_cost(i, i);
			ss << "ground truth cost = " << real_cost;
			lap::displayTime(start_time, ss.str().c_str(), std::cout);

			delete[] vec;
			delete[] rowsol;
		}
	}
}

template <class C> void testGeometric(long long min_tab, long long max_tab, long long max_memory, int runs, bool epsilon, bool disjoint, std::string name_C, std::vector<int> &devs, bool silent)
{
	// geometric costs in R^2 (supply function for calculating cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Geometric";
			if (disjoint) std::cout << " Disjoint";
			std::cout << " R^2<" << name_C << "> " << N << "x" << N << " table";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			C *tab_s = new C[2 * N];
			C *tab_t = new C[2 * N];

			for (int i = 0; i < 2 * N; i++)
			{
				tab_s[i] = distribution(generator);
				tab_t[i] = distribution(generator);
			}

			if (disjoint)
			{
				for (int i = 0; i < 2 * N; i += 2)
				{
					if (i < N)
					{
						tab_t[i] += C(1);
					}
					else
					{
						tab_s[i] += C(1);
						tab_s[i + 1] += C(1);
						tab_t[i + 1] += C(1);
					}
				}
			}

			// cost function
			auto get_cost = [&tab_s, &tab_t](int x, int y) -> C
			{
				int xx = x + x;
				int yy = y + y;
				C d0 = tab_s[xx] - tab_t[yy];
				C d1 = tab_s[xx + 1] - tab_t[yy + 1];
				return d0 * d0 + d1 * d1;
			};

			int *rowsol = new int[N];

			lap::cuda::Worksharing ws(N, 256, devs, silent);

			solveTableCUDA<C, int>(start_time, N, N, get_cost, ws, max_memory, rowsol, epsilon);

			delete[] tab_s;
			delete[] tab_t;
			delete[] rowsol;
		}
	}
}

template <class C> void testRandomLowRank(long long min_tab, long long max_tab, long long max_memory, long long min_rank, long long max_rank, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	// random costs (directly supply cost matrix)
	for (long long rank = min_rank; rank <= max_rank; rank <<= 1)
	{
		for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " table rank = " << rank;
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(0.0, 1.0);
				std::mt19937_64 generator(1234);

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				C max_vec;
				C min_vec;
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
					max_vec = vec[i * N];
					for (long long j = 1; j < N; j++) max_vec = std::max(max_vec, vec[i * N + j]);
					min_vec = vec[i * N];
					for (long long j = 1; j < N; j++) min_vec = std::min(min_vec, vec[i * N + j]);
				}

				// cost function
				auto get_cost = [&vec, &N, &rank, &max_vec](int x, int y) -> C
				{
					C sum(0);
					for (long long k = 0; k < rank; k++)
					{
						sum += vec[k * N + x] * vec[k * N + y];
					}
					return sum / C(rank);
				};

				int *rowsol = new int[N];

				lap::cuda::Worksharing ws(N, 256, devs, silent);

				solveTableCUDA<C, int>(start_time, N, N, get_cost, ws, max_memory, rowsol, epsilon);

				delete[] vec;
				delete[] rowsol;
			}
		}
	}
}

template <class C>
__device__ __forceinline__ C getCost_image(unsigned char *c00, unsigned char *c01, unsigned char *c02, int w0, int h0, int mval0,
	unsigned char *c10, unsigned char *c11, unsigned char *c12, int w1, int h1, int mval1, int x, int y)
{
	C r = C(c00[x]) / C(mval0) - C(c10[y]) / C(mval1);
	C g = C(c01[x]) / C(mval0) - C(c11[y]) / C(mval1);
	C b = C(c02[x]) / C(mval0) - C(c12[y]) / C(mval1);
	C u = C(x % w0) / C(w0 - 1) - C(y % w1) / C(w1 - 1);
	C v = C(x / w0) / C(h0 - 1) - C(y / w1) / C(h1 - 1);
	return r * r + g * g + b * b + u * u + v * v;
}

template <class C>
__global__
void getCostRow_image_kernel(C *row, unsigned char *c00, unsigned char *c01, unsigned char *c02, int w0, int h0, int mval0, 
	unsigned char *c10, unsigned char *c11, unsigned char *c12, int w1, int h1, int mval1, int x, int start, int end)
{
	int y = start + threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= end) return;

	row[threadIdx.x + blockIdx.x * blockDim.x] = getCost_image<C>(c00, c01, c02, w0, h0, mval0, c10, c11, c12, w1, h1, mval1, x, y);
}

template <class C>
__global__
void getCost_image_kernel(C *cost, unsigned char *c00, unsigned char *c01, unsigned char *c02, int w0, int h0, int mval0,
	unsigned char *c10, unsigned char *c11, unsigned char *c12, int w1, int h1, int mval1, int *rowsol, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= N) return;
	int y = rowsol[x];

	cost[x] = getCost_image<C>(c00, c01, c02, w0, h0, mval0, c10, c11, c12, w1, h1, mval1, x, y);
}

template <class C> void testImages(std::vector<std::string> &images, long long max_memory, int runs, bool epsilon, std::string name_C, std::vector<int> &devs, bool silent)
{
	std::cout << "Comparing images ";
	if (epsilon) std::cout << " with epsilon scaling";
	std::cout << std::endl;
	for (unsigned int a = 0; a < images.size() - 1; a++)
	{
		for (unsigned int b = a + 1; b < images.size(); b++)
		{
			PPMImage img_a(images[a]);
			PPMImage img_b(images[b]);
			std::cout << "Comparing image \"" << images[a] << "\" (" << img_a.width << "x" << img_a.height << ") with image \"" << images[b] << "\" (" << img_b.width << "x" << img_b.height << ")." << std::endl;
			for (int r = 0; r < runs; r++)
			{
				auto start_time = std::chrono::high_resolution_clock::now();

				int N1 = std::min(img_a.width * img_a.height, img_b.width * img_b.height);
				int N2 = std::max(img_a.width * img_a.height, img_b.width * img_b.height);

				lap::cuda::Worksharing ws(N2, 256, devs, silent);
				int num_devices = (int)ws.device.size();
				// make sure img[0] is at most as large as img[1]
				PPMImage *img[2];
				img[0] = new PPMImage[num_devices];
				img[1] = new PPMImage[num_devices];
				// rearrange data for GPU
				int size_a = img_a.width * img_a.height;
				unsigned char *buf_a = new unsigned char[size_a * 3];
				int size_b = img_b.width * img_b.height;
				unsigned char *buf_b = new unsigned char[size_b * 3];
				for (int y = 0; y < img_a.height; y++)
				{
					for (int x = 0; x < img_a.width; x++)
					{
						int off = x + y * img_a.width;
						buf_a[off] = img_a.raw[off * 3];
						buf_a[off + size_a] = img_a.raw[off * 3 + 1];
						buf_a[off + 2 * size_a] = img_a.raw[off * 3 + 2];
					}
				}
				for (int y = 0; y < img_b.height; y++)
				{
					for (int x = 0; x < img_b.width; x++)
					{
						int off = x + y * img_b.width;
						buf_b[off] = img_b.raw[off * 3];
						buf_b[off + size_b] = img_b.raw[off * 3 + 1];
						buf_b[off + 2 * size_b] = img_b.raw[off * 3 + 2];
					}
				}
				for (int t = 0; t < num_devices; t++)
				{
					cudaSetDevice(ws.device[t]);
					if (img_a.width * img_a.height < img_b.width * img_b.height)
					{
						img[0][t].width = img_a.width;
						img[0][t].height = img_a.height;
						img[0][t].max_val = img_a.max_val;
						cudaMalloc(&img[0][t].raw, img[0][t].width * img[0][t].height * 3);
						cudaMemcpyAsync(img[0][t].raw, buf_a, img[0][t].width * img[0][t].height * 3, cudaMemcpyHostToDevice);
						img[1][t].width = img_b.width;
						img[1][t].height = img_b.height;
						img[1][t].max_val = img_b.max_val;
						cudaMalloc(&img[1][t].raw, img[1][t].width * img[1][t].height * 3);
						cudaMemcpyAsync(img[1][t].raw, buf_b, img[1][t].width * img[1][t].height * 3, cudaMemcpyHostToDevice);
					}
					else
					{
						img[0][t].width = img_b.width;
						img[0][t].height = img_b.height;
						img[0][t].max_val = img_b.max_val;
						cudaMalloc(&img[0][t].raw, img[0][t].width * img[0][t].height * 3);
						cudaMemcpyAsync(img[0][t].raw, buf_b, img[0][t].width * img[0][t].height * 3, cudaMemcpyHostToDevice);
						img[1][t].width = img_a.width;
						img[1][t].height = img_a.height;
						img[1][t].max_val = img_a.max_val;
						cudaMalloc(&img[1][t].raw, img[1][t].width * img[1][t].height * 3);
						cudaMemcpyAsync(img[1][t].raw, buf_b, img[1][t].width * img[1][t].height * 3, cudaMemcpyHostToDevice);
					}
				}
				delete[] buf_a;
				delete[] buf_b;

				// cost function
				auto get_cost_row = [&img](C *d_row, int t, cudaStream_t stream, int x, int start, int end)
				{
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = ((end - start) + block_size.x - 1) / block_size.x;
					int size_0 = img[0][t].width * img[0][t].height;
					int size_1 = img[1][t].width * img[1][t].height;
					getCostRow_image_kernel<<<grid_size, block_size, 0, stream>>>(d_row,
						img[0][t].raw, img[0][t].raw + size_0, img[0][t].raw + 2 * size_0, img[0][t].width, img[0][t].height, img[0][t].max_val,
						img[1][t].raw, img[1][t].raw + size_1, img[1][t].raw + 2 * size_1, img[1][t].width, img[1][t].height, img[1][t].max_val, x, start, end);
				};

				// cost function
				auto get_cost = [&img](C *d_row, cudaStream_t stream, int *d_rowsol, int N2)
				{
					dim3 block_size, grid_size;
					block_size.x = 256;
					grid_size.x = (N2 + block_size.x - 1) / block_size.x;
					int size_0 = img[0][0].width * img[0][0].height;
					int size_1 = img[1][0].width * img[1][0].height;
					getCost_image_kernel<<<grid_size, block_size, 0, stream>>>(d_row,
						img[0][0].raw, img[0][0].raw + size_0, img[0][0].raw + 2 * size_0, img[0][0].width, img[0][0].height, img[0][0].max_val,
						img[1][0].raw, img[1][0].raw + size_1, img[1][0].raw + 2 * size_1, img[1][0].width, img[1][0].height, img[1][0].max_val, d_rowsol, N2);
				};

				int *rowsol = new int[N2];

				solveCachingCUDA<C, C>(start_time, N1, N2, get_cost_row, get_cost, ws, max_memory, rowsol, epsilon);

				for (int t = 0; t < num_devices; t++)
				{
					cudaFree(img[0][t].raw); img[0][t].raw = 0;
					cudaFree(img[1][t].raw); img[1][t].raw = 0;
				}
				delete[] rowsol;
				delete[] img[0];
				delete[] img[1];
			}
		}
	}
}
