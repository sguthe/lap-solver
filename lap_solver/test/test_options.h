#pragma once

#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>

class Options
{
public:
	long long lap_min_tab;
	long long lap_max_tab;
	long long lap_min_cached;
	long long lap_max_cached;

	bool use_double;
	bool use_float;
	bool use_single;
	bool use_epsilon;
	bool use_omp;

	bool run_random;
	bool run_geometric;
	bool run_geometric_cached;
	bool run_geometric_disjoint;
	bool run_geometric_disjoint_cached;
	std::vector<std::string> images;

	int runs;
public:
	Options()
	{
		lap_min_tab = lap_max_tab = lap_min_cached = lap_max_cached = 0ll;
		use_double = use_float = use_single = use_epsilon = use_omp = false;
		run_random = run_geometric = run_geometric_cached = run_geometric_disjoint = run_geometric_disjoint_cached = false;
		runs = 1;
	}
public:
	void setDefaultSize()
	{
		lap_min_tab = 1000ll;
		lap_max_tab = 64000ll;
		lap_min_cached = 64000ll;
		lap_max_cached = 256000ll;
	}
	void setDefault()
	{
		setDefaultSize();
		use_double = true;
		use_float = false;
		use_single = true;
		use_epsilon = true;
		use_omp = true;
		run_random = true;
		run_geometric = true;
		run_geometric_cached = true;
		run_geometric_disjoint = true;
		run_geometric_disjoint_cached = true;
	}

	int parseOptions(int argc, char* argv[])
	{
		if (argc == 1)
		{
			setDefault();
		}
		for (int i = 1; i < argc; i++)
		{
			if (!strcmp(argv[i], "-default"))
			{
				setDefault();
			}
			else if (!strcmp(argv[i], "-default_size"))
			{
				setDefaultSize();
			}
			else if (!strcmp(argv[i], "-table_min"))
			{
				lap_min_tab = atoll(argv[++i]);
			}
			else if (!strcmp(argv[i], "-table_max"))
			{
				lap_max_tab = atoll(argv[++i]);
			}
			else if (!strcmp(argv[i], "-cached_min"))
			{
				lap_min_cached = atoll(argv[++i]);
			}
			else if (!strcmp(argv[i], "-cached_max"))
			{
				lap_max_cached = atoll(argv[++i]);
			}
			else if (!strcmp(argv[i], "-double"))
			{
				use_double = true;
			}
			else if (!strcmp(argv[i], "-float"))
			{
				use_float = true;
			}
			else if (!strcmp(argv[i], "-single"))
			{
				use_single = true;
			}
			else if (!strcmp(argv[i], "-epsilon"))
			{
				use_epsilon = true;
			}
			else if (!strcmp(argv[i], "-random"))
			{
				run_random = true;
			}
			else if (!strcmp(argv[i], "-geometric"))
			{
				run_geometric = true;
			}
			else if (!strcmp(argv[i], "-geometric_cached"))
			{
				run_geometric_cached = true;
			}
			else if (!strcmp(argv[i], "-geometric_disjoint"))
			{
				run_geometric_disjoint = true;
			}
			else if (!strcmp(argv[i], "-geometric_disjoint_cached"))
			{
				run_geometric_disjoint_cached = true;
			}
			else if (!strcmp(argv[i], "-omp"))
			{
#ifdef _OPENMP
				use_omp = true;
#else
				std::cout << "OpenMP not enabled." << std::endl;
#endif
			}
			else if (!strcmp(argv[i], "-img"))
			{
				images.push_back(argv[++i]);
			}
			else if (!strcmp(argv[i], "-runs"))
			{
				runs = atoi(argv[++i]);
			}
			else
			{
				std::cout << "Unkown option: " << argv[i] << std::endl;
				return -1;
			}
		}
		return 0;
	}
};
