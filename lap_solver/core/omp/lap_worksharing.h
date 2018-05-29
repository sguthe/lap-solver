#pragma once
#include <utility>

namespace lap
{
	namespace omp
	{
		class Worksharing
		{
		public:
			std::pair<int, int> *part;
		public:
			Worksharing(int size, int multiple)
			{
				int max_threads = omp_get_max_threads();
				lapAlloc(part, max_threads, __FILE__, __LINE__);
				for (int p = 0; p < max_threads; p++)
				{
					long long x0 = (long long)p * (long long)size;
					x0 /= max_threads;
					x0 /= multiple;
					part[p].first = (int)(multiple * x0);
					if (p + 1 != max_threads)
					{
						long long x1 = (long long)(p + 1) * (long long)size;
						x1 /= max_threads;
						x1 /= multiple;
						part[p].second = (int)(multiple * x1);
					}
					else part[p].second = size;
				}
			}
			~Worksharing() { if (part != 0) lapFree(part); }
		};
	}
}
