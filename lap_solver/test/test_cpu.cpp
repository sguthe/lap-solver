#ifdef _OPENMP
#  define LAP_OPENMP
#endif
#define LAP_QUIET
//#define LAP_DEBUG
//#define LAP_NO_MEM_DEBUG
#include "../lap.h"

#include <random>
#include <string>
#include <fstream>
#include "test_options.h"

template <class CF> void testRandom(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, bool sanity, std::string name_C);
template <class CF> void testRandomCached(long long max_tab, long long min_cached, long long max_cached, int runs, bool omp, bool epsilon, bool sanity, std::string name_C);
template <class CF> void testGeometric(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C);
template <class CF> void testGeometricCached(long long max_tab, long long min_cached, long long max_cached, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C);
template <class CF> void testRandomLowRank(long long min_tab, long long max_tab, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C);
template <class CF> void testRandomLowRankCached(long long max_tab, long long min_cached, long long max_cached, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C);
template <class CF> void testImages(std::vector<std::string> &images, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C);

int main(int argc, char* argv[])
{
	Options opt;
	int r = opt.parseOptions(argc, argv);
	if (r != 0) return r;

	if (opt.use_double)
	{
			if (opt.use_single)
			{
				if (opt.run_sanity) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, true, std::string("double"));
				if (opt.run_sanity_cached) testRandomCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, false, true, std::string("double"));
				if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, false, std::string("double"));
				if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("double"));
				if (opt.run_random_low_rank_cached) testRandomLowRankCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("double"));
				if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, false, std::string("double"));
				if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, true, std::string("double"));
				if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, false, false, std::string("double"));
				if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, false, true, std::string("double"));
				if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("double"));
			}
			if (opt.use_epsilon)
			{
				if (opt.run_sanity) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, true, std::string("double"));
				if (opt.run_sanity_cached) testRandomCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, true, true, std::string("double"));
				if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, false, std::string("double"));
				if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("double"));
				if (opt.run_random_low_rank_cached) testRandomLowRankCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("double"));
				if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, false, std::string("double"));
				if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, true, std::string("double"));
				if (opt.run_geometric_cached) testGeometricCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, true, false, std::string("double"));
				if (opt.run_geometric_disjoint_cached) testGeometricCached<double>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, true, true, std::string("double"));
				if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("double"));
			}
		}
	if (opt.use_float)
	{
		if (opt.use_single)
		{
			if (opt.run_sanity) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, true, std::string("float"));
			if (opt.run_sanity_cached) testRandomCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, false, true, std::string("float"));
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, false, std::string("float"));
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, false, std::string("float"));
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, false, std::string("float"));
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, false, true, std::string("float"));
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, false, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, false, true, std::string("float"));
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, false, std::string("float"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_sanity) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, true, std::string("float"));
			if (opt.run_sanity_cached) testRandomCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, true, true, std::string("float"));
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, false, std::string("float"));
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_random_low_rank_cached) testRandomLowRankCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, true, std::string("float"));
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, false, std::string("float"));
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, true, true, std::string("float"));
			if (opt.run_geometric_cached) testGeometricCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, true, false, std::string("float"));
			if (opt.run_geometric_disjoint_cached) testGeometricCached<float>(opt.lap_max_tab, opt.lap_min_cached, opt.lap_max_cached, opt.runs, opt.use_omp, true, true, std::string("float"));
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, true, std::string("float"));
		}
	}

	return 0;
}

template <class C>
void testRandom(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, bool sanity, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			if (sanity) std::cout << "Sanity";
			else std::cout << "Random";
			std::cout << "<" << name_C << "> " << N << "x" << N;
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			C *tab = new C[NN];
			if (sanity)
			{
				C *vec = new C[N << 1];
				for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);
				if (omp)
				{
#ifdef LAP_OPENMP
#pragma omp parallel for
					for (long long i = 0; i < N; i++)
					{
						long long ii = i * N;
						long long j;
						for (j = 0; j < i; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
						tab[ii + i] = vec[i] + vec[i + N];
						for (j = i + 1; j < N; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
					}
#endif
				}
				else
				{
					for (long long i = 0; i < N; i++)
					{
						long long ii = i * N;
						long long j;
						for (j = 0; j < i; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
						tab[ii + i] = vec[i] + vec[i + N];
						for (j = i + 1; j < N; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
					}
				}
				delete[] vec;
			}
			else
			{
				for (long long i = 0; i < NN; i++) tab[i] = distribution(generator);
			}

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				lap::omp::Worksharing ws(N, 8);
				lap::omp::TableCost<C> costMatrix(N, N, tab, ws);
				lap::omp::DirectIterator<C, C, lap::omp::TableCost<C>> iterator(N, N, costMatrix, ws);
				if (epsilon) costMatrix.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator));

				lap::displayTime(start_time, "setup complete", std::cout);

				lap::omp::solve<C>(N, costMatrix, iterator, rowsol);

				{
					std::stringstream ss;
					ss << "cost = " << lap::omp::cost<C>(N, costMatrix, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
#endif
			}
			else
			{
				lap::TableCost<C> costMatrix(N, N, tab);
				lap::DirectIterator<C, C, lap::TableCost<C>> iterator(N, N, costMatrix);
				if (epsilon) costMatrix.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator));

				lap::displayTime(start_time, "setup complete", std::cout);

				lap::solve<C>(N, costMatrix, iterator, rowsol);

				{
					std::stringstream ss;
					ss << "cost = " << lap::cost<C>(N, costMatrix, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
			}

			if (sanity)
			{
				bool passed = true;
				for (long long i = 0; (passed) && (i < N); i++)
				{
					passed &= (rowsol[i] == i);
				}
				std::stringstream ss;
				if (passed) ss << "test passed: ";
				else ss << "test failed: ";
				C real_cost(0);
				for (long long i = 0; i < N; i++) real_cost += tab[i + i * N];
				ss << "ground truth cost = " << real_cost;
				lap::displayTime(start_time, ss.str().c_str(), std::cout);
			}
			delete[] rowsol;
			delete[] tab;
		}
	}
}

template <class C>
void testRandomCached(long long max_tab, long long min_cached, long long max_cached, int runs, bool omp, bool epsilon, bool sanity, std::string name_C)
{
	if (!sanity)
	{
		std::cout << "Random cached not supported." << std::endl;
		return;
	}
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));
			int entries = std::min(N, (int)((max_tab * max_tab) / N));

			if (sanity) std::cout << "Sanity";
			else std::cout << "Random";
			std::cout << "<" << name_C << "> " << N << "x" << N;
			if (omp) std::cout << " multithreaded";
			if (epsilon) std::cout << " with epsilon scaling";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
			std::mt19937_64 generator(1234);

			C *vec = new C[N << 1];
			if (sanity)
			{
				for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);
			}

			// cost function
			auto get_cost = [vec, N](int x, int y) -> C
			{
				C r = vec[x] + vec[y + N];
				if (x == y) return r;
				else return r + C(0.1);
			};

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				lap::omp::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
				lap::omp::Worksharing ws(N, costFunction.getMultiple());

				if (4 * entries < N)
				{
					lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction, ws);
					if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::omp::solve<C>(N, costFunction, iterator, rowsol);
				}
				else
				{
					lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction, ws);
					if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::omp::solve<C>(N, costFunction, iterator, rowsol);
				}

				{
					std::stringstream ss;
					ss << "cost = " << lap::omp::cost<C>(N, costFunction, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
#endif
			}
			else
			{
				lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);

				if (4 * entries < N)
				{
					lap::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction);
					if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::solve<C>(N, costFunction, iterator, rowsol);
				}
				else
				{
					lap::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction);
					if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::solve<C>(N, costFunction, iterator, rowsol);
				}

				{
					std::stringstream ss;
					ss << "cost = " << lap::cost<C>(N, costFunction, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
			}

			if (sanity)
			{
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
			}
			delete[] rowsol;
			delete[] vec;
		}
	}
}

template <class C>
void testRandomLowRank(long long min_tab, long long max_tab, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (long long rank = min_rank; (rank <= max_rank) && (rank * rank <= NN); rank <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " rank = " << rank;
				if (omp) std::cout << " multithreaded";
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(-1.0, 1.0);
				std::mt19937_64 generator(1234);

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
				}

				C *tab = new C[NN];
				if (omp)
				{
#pragma omp parallel for
					for (long long i = 0; i < N; i++)
					{
						for (long long j = 0; j < N; j++)
						{
							C sum(0);
							for (long long k = 0; k < rank; k++)
							{
								sum += vec[k * N + i] * vec[k * N + j];
							}
							tab[i * N + j] = (sum / C(rank)) + C(1);
						}
					}
				}
				else
				{
					for (long long i = 0; i < N; i++)
					{
						for (long long j = 0; j < N; j++)
						{
							C sum(0);
							for (long long k = 0; k < rank; k++)
							{
								sum += vec[k * N + i] * vec[k * N + j];
							}
							tab[i * N + j] = (sum / C(rank)) + C(1);
						}
					}
				}

				delete[] vec;

				int *rowsol = new int[N];

				if (omp)
				{
#ifdef LAP_OPENMP
					lap::omp::Worksharing ws(N, 8);
					lap::omp::TableCost<C> costMatrix(N, N, tab, ws);
					lap::omp::DirectIterator<C, C, lap::omp::TableCost<C>> iterator(N, N, costMatrix, ws);
					if (epsilon) costMatrix.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator));

					lap::displayTime(start_time, "setup complete", std::cout);

					lap::omp::solve<C>(N, costMatrix, iterator, rowsol);

					{
						std::stringstream ss;
						ss << "cost = " << lap::omp::cost<C>(N, costMatrix, rowsol);
						lap::displayTime(start_time, ss.str().c_str(), std::cout);
					}
#endif
				}
				else
				{
					lap::TableCost<C> costMatrix(N, N, tab);
					lap::DirectIterator<C, C, lap::TableCost<C>> iterator(N, N, costMatrix);
					if (epsilon) costMatrix.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator));

					lap::displayTime(start_time, "setup complete", std::cout);

					lap::solve<C>(N, costMatrix, iterator, rowsol);

					{
						std::stringstream ss;
						ss << "cost = " << lap::cost<C>(N, costMatrix, rowsol);
						lap::displayTime(start_time, ss.str().c_str(), std::cout);
					}
				}

				delete[] rowsol;
				delete[] tab;
			}
		}
	}
}

template <class C>
void testRandomLowRankCached(long long max_tab, long long min_cached, long long max_cached, long long min_rank, long long max_rank, int runs, bool omp, bool epsilon, std::string name_C)
{
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (long long rank = min_rank; (rank <= max_rank) && (rank * rank <= NN); rank <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));
				int entries = std::min(N, (int)((max_tab * max_tab) / N));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " rank = " << rank;
				if (omp) std::cout << " multithreaded";
				if (epsilon) std::cout << " with epsilon scaling";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(-1.0, 1.0);
				std::mt19937_64 generator(1234);

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
				}

				// cost function
				auto get_cost = [vec, N, rank](int x, int y) -> C
				{
					C sum(0);
					for (long long k = 0; k < rank; k++)
					{
						sum += vec[k * N + x] * vec[k * N + y];
					}
					return sum / C(rank) + C(1);
				};

				int *rowsol = new int[N];

				if (omp)
				{
#ifdef LAP_OPENMP
					lap::omp::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
					lap::omp::Worksharing ws(N, costFunction.getMultiple());

					if (4 * entries < N)
					{
						lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction, ws);
						if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
						lap::displayTime(start_time, "setup complete", std::cout);
						lap::omp::solve<C>(N, costFunction, iterator, rowsol);
					}
					else
					{
						lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction, ws);
						if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
						lap::displayTime(start_time, "setup complete", std::cout);
						lap::omp::solve<C>(N, costFunction, iterator, rowsol);
					}

					{
						std::stringstream ss;
						ss << "cost = " << lap::omp::cost<C>(N, costFunction, rowsol);
						lap::displayTime(start_time, ss.str().c_str(), std::cout);
					}
#endif
				}
				else
				{
					lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);

					if (4 * entries < N)
					{
						lap::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction);
						if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
						lap::displayTime(start_time, "setup complete", std::cout);
						lap::solve<C>(N, costFunction, iterator, rowsol);
					}
					else
					{
						lap::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction);
						if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
						lap::displayTime(start_time, "setup complete", std::cout);
						lap::solve<C>(N, costFunction, iterator, rowsol);
					}

					{
						std::stringstream ss;
						ss << "cost = " << lap::cost<C>(N, costFunction, rowsol);
						lap::displayTime(start_time, ss.str().c_str(), std::cout);
					}
				}

				delete[] rowsol;
				delete[] vec;
			}
		}
	}
}

template <class C>
void testGeometric(long long min_tab, long long max_tab, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C)
{
	// geometric costs in R^2 (supply function for calculating cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Geometric";
			if (disjoint) std::cout << " Disjoint";
			std::cout << " R^2<" << name_C << "> " << N << "x" << N;
			if (omp) std::cout << " multithreaded";
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
			auto get_cost = [tab_s, tab_t](int x, int y) -> C
			{
				int xx = x + x;
				int yy = y + y;
				C d0 = tab_s[xx] - tab_t[yy];
				C d1 = tab_s[xx + 1] - tab_t[yy + 1];
				return d0 * d0 + d1 * d1;
			};

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				lap::omp::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
				lap::omp::Worksharing ws(N, costFunction.getMultiple());
				lap::omp::TableCost<C> costMatrix(N, costFunction, ws);
				lap::omp::DirectIterator<C, C, decltype(costMatrix)> iterator(N, N, costMatrix, ws);
				if (epsilon) costMatrix.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator));

#endif
				delete[] tab_s;
				delete[] tab_t;
#ifdef LAP_OPENMP
				lap::displayTime(start_time, "setup complete", std::cout);

				lap::omp::solve<C>(N, costMatrix, iterator, rowsol);

				{
					std::stringstream ss;
					ss << "cost = " << lap::omp::cost<C>(N, costMatrix, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
#endif
			}
			else
			{
				lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
				lap::TableCost<C> costMatrix(N, costFunction);
				lap::DirectIterator<C, C, decltype(costMatrix)> iterator(N, N, costMatrix);
				if (epsilon) costMatrix.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator));

				delete[] tab_s;
				delete[] tab_t;

				lap::displayTime(start_time, "setup complete", std::cout);

				lap::solve<C>(N, costMatrix, iterator, rowsol);

				{
					std::stringstream ss;
					ss << "cost = " << lap::cost<C>(N, costMatrix, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
			}

			delete[] rowsol;
		}
	}
}

template <class C> 
void testGeometricCached(long long max_tab, long long min_cached, long long max_cached, int runs, bool omp, bool epsilon, bool disjoint, std::string name_C)
{
	for (long long NN = min_cached * min_cached; NN <= max_cached * max_cached; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
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
			auto get_cost = [tab_s, tab_t](int x, int y) -> C
			{
				int xx = x + x;
				int yy = y + y;
				C d0 = tab_s[xx] - tab_t[yy];
				C d1 = tab_s[xx + 1] - tab_t[yy + 1];
				return d0 * d0 + d1 * d1;
			};

			int *rowsol = new int[N];

			if (omp)
			{
#ifdef LAP_OPENMP
				lap::omp::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
				lap::omp::Worksharing ws(N, costFunction.getMultiple());

				if (4 * entries < N)
				{
					lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction, ws);
					if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::omp::solve<C>(N, costFunction, iterator, rowsol);
				}
				else
				{
					lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction, ws);
					if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::omp::solve<C>(N, costFunction, iterator, rowsol);
				}

				{
					std::stringstream ss;
					ss << "cost = " << lap::omp::cost<C>(N, costFunction, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
#endif
			}
			else
			{
				lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);

				if (4 * entries < N)
				{
					lap::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N, N, entries, costFunction);
					if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::solve<C>(N, costFunction, iterator, rowsol);
				}
				else
				{
					lap::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N, N, entries, costFunction);
					if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N, N, iterator, (int)(N / entries)));
					lap::displayTime(start_time, "setup complete", std::cout);
					lap::solve<C>(N, costFunction, iterator, rowsol);
				}

				{
					std::stringstream ss;
					ss << "cost = " << lap::cost<C>(N, costFunction, rowsol);
					lap::displayTime(start_time, ss.str().c_str(), std::cout);
				}
			}

			delete[] rowsol;
			delete[] tab_s;
			delete[] tab_t;
		}
	}
}

class PPMImage
{
public:
	int width, height, max_val;
	unsigned char *raw;
public:
	PPMImage(std::string &fname) : width(0), height(0), max_val(0), raw(0)
	{
		std::ifstream in(fname.c_str(), std::ios::in | std::ios::binary);
		if (in.is_open()) {
			std::string line;

			std::getline(in, line);
			if (line != "P6") {
				std::cout << "\"" << fname << "\" in not a ppm P6 file." << std::endl;
				exit(-1);
			}
			do std::getline(in, line); while (line[0] == '#');
			{
				std::stringstream sline(line);
				sline >> width;
				sline >> height;
			}
			std::getline(in, line);
			{
				std::stringstream sline(line);
				sline >> max_val;
			}

			raw = new unsigned char[width * height * 3];

			in.read((char *)raw, width * height * 3);
		}
		else
		{
			std::cout << "Can't open \"" << fname << "\"" << std::endl;
			exit(-1);
		}
		in.close();
	}
	~PPMImage() { delete[] raw; }
};

template <class C> void testImages(std::vector<std::string> &images, long long max_tab, int runs, bool omp, bool epsilon, std::string name_C)
{
	std::cout << "Comparing images ";
	if (omp) std::cout << " multithreaded";
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

				// make sure img[0] is at most as large as img[1]
				PPMImage *img[2];
				if (img_a.width * img_a.height < img_b.width * img_b.height)
				{
					img[0] = &img_a;
					img[1] = &img_b;
				}
				else
				{
					img[0] = &img_b;
					img[1] = &img_a;
				}
				int N1 = img[0]->width * img[0]->height;
				int N2 = img[1]->width * img[1]->height;

				// cost function
				auto get_cost = [&img](int x, int y) -> C
				{
					C r = C(img[0]->raw[3 * x]) / C(img[0]->max_val) - C(img[1]->raw[3 * y]) / C(img[1]->max_val);
					C g = C(img[0]->raw[3 * x + 1]) / C(img[0]->max_val) - C(img[1]->raw[3 * y + 1]) / C(img[1]->max_val);
					C b = C(img[0]->raw[3 * x + 2]) / C(img[0]->max_val) - C(img[1]->raw[3 * y + 2]) / C(img[1]->max_val);
					C u = C(x % img[0]->width) / C(img[0]->width - 1) - C(y % img[1]->width) / C(img[1]->width - 1);
					C v = C(x / img[0]->width) / C(img[0]->height - 1) - C(y / img[1]->width) / C(img[1]->height - 1);
					return r * r + g * g + b * b + u * u + v * v;
				};

				int *rowsol = new int[N2];

				long long entries = (max_tab * max_tab) / N2;


				if (omp)
				{
#ifdef LAP_OPENMP
					lap::omp::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
					lap::omp::Worksharing ws(N2, costFunction.getMultiple());

					if (N1 <= entries)
					{
						std::cout << "using table with " << N1 << " rows." << std::endl;

						lap::omp::TableCost<C> costMatrix(N1, N2, costFunction, ws);
						lap::omp::DirectIterator<C, C, decltype(costMatrix)> iterator(N1, N2, costMatrix, ws);
						if (epsilon) costMatrix.setInitialEpsilon(lap::omp::guessEpsilon<C>(N1, N2, iterator, std::max(1, (int)(N1 / entries))));

						lap::displayTime(start_time, "setup complete", std::cout);

						lap::omp::solve<C>(N1, N2, costMatrix, iterator, rowsol);

						{
							std::stringstream ss;
							ss << "cost = " << lap::omp::cost<C>(N1, N2, costMatrix, rowsol);
							lap::displayTime(start_time, ss.str().c_str(), std::cout);
						}
					}
					else
					{
						std::cout << "using caching with " << entries << "/" << N1 << " entries." << std::endl;

						if (4 * entries < N1)
						{
							lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N1, N2, (int)entries, costFunction, ws);
							if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N1, N2, iterator, std::max(1, (int)(N1 / entries))));
							lap::displayTime(start_time, "setup complete", std::cout);
							lap::omp::solve<C>(N1, N2, costFunction, iterator, rowsol);

							{
								std::stringstream ss;
								ss << "cost = " << lap::omp::cost<C>(N1, N2, costFunction, rowsol);
								lap::displayTime(start_time, ss.str().c_str(), std::cout);
							}
						}
						else
						{
							lap::omp::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N1, N2, (int)entries, costFunction, ws);
							if (epsilon) costFunction.setInitialEpsilon(lap::omp::guessEpsilon<C>(N1, N2, iterator, std::max(1, (int)(N1 / entries))));
							lap::displayTime(start_time, "setup complete", std::cout);
							lap::omp::solve<C>(N1, N2, costFunction, iterator, rowsol);

							{
								std::stringstream ss;
								ss << "cost = " << lap::omp::cost<C>(N1, N2, costFunction, rowsol);
								lap::displayTime(start_time, ss.str().c_str(), std::cout);
							}
						}
					}
#endif
				}
				else
				{
					lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);

					if (N1 <= entries)
					{
						std::cout << "using table with " << N1 << " rows." << std::endl;

						lap::TableCost<C> costMatrix(N1, N2, costFunction);
						lap::DirectIterator<C, C, decltype(costMatrix)> iterator(N1, N2, costMatrix);
						if (epsilon) costMatrix.setInitialEpsilon(lap::guessEpsilon<C>(N1, N2, iterator, std::max(1, (int)(N1 / entries))));

						lap::displayTime(start_time, "setup complete", std::cout);

						lap::solve<C>(N1, N2, costMatrix, iterator, rowsol);

						{
							std::stringstream ss;
							ss << "cost = " << lap::cost<C>(N1, N2, costMatrix, rowsol);
							lap::displayTime(start_time, ss.str().c_str(), std::cout);
						}
					}
					else
					{
						std::cout << "using caching with " << entries << "/" << N1 << " entries." << std::endl;

						if (4 * entries < N1)
						{
							lap::CachingIterator<C, C, decltype(costFunction), lap::CacheSLRU> iterator(N1, N2, (int)entries, costFunction);
							if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N1, N2, iterator, std::max(1, (int)(N1 / entries))));
							lap::displayTime(start_time, "setup complete", std::cout);
							lap::solve<C>(N1, N2, costFunction, iterator, rowsol);

							{
								std::stringstream ss;
								ss << "cost = " << lap::cost<C>(N1, N2, costFunction, rowsol);
								lap::displayTime(start_time, ss.str().c_str(), std::cout);
							}
						}
						else
						{
							lap::CachingIterator<C, C, decltype(costFunction), lap::CacheLFU> iterator(N1, N2, (int)entries, costFunction);
							if (epsilon) costFunction.setInitialEpsilon(lap::guessEpsilon<C>(N1, N2, iterator, std::max(1, (int)(N1 / entries))));
							lap::displayTime(start_time, "setup complete", std::cout);
							lap::solve<C>(N1, N2, costFunction, iterator, rowsol);

							{
								std::stringstream ss;
								ss << "cost = " << lap::cost<C>(N1, N2, costFunction, rowsol);
								lap::displayTime(start_time, ss.str().c_str(), std::cout);
							}
						}
					}
				}
			}
		}
	}
}
