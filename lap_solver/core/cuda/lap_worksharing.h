#pragma once
#include <utility>
#include <vector>
#include <cuda.h>

namespace lap
{
	namespace cuda
	{
		class Worksharing
		{
		public:
			std::pair<int, int> *part;
			std::vector<int> device;
		public:
			Worksharing(int size, int multiple)
			{
				int max_devices = (size + multiple - 1) / multiple;
				int device_count;
				cudaDeviceProp deviceProp;
				cudaGetDeviceCount(&device_count);

				for (int current_device = 0; ((current_device < device_count) && (device.size() < max_devices)); current_device++)
				{
					cudaGetDeviceProperties(&deviceProp, current_device);

					// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
					if (deviceProp.computeMode != cudaComputeModeProhibited)
					{
						device.push_back(current_device);
					}
				}

				if (device.size() == 0)
				{
					std::cout << "No suitable CUDA device found." << std::endl;
					exit(-1);
				}

				int max_threads = (int)device.size();
				lapAlloc(part, max_threads, __FILE__, __LINE__);
				for (int p = 0; p < max_threads; p++)
				{
					long long x0 = (long long)p * (long long)size;
					x0 += (max_threads * multiple) >> 1;
					x0 /= max_threads * multiple;
					part[p].first = (int)(multiple * x0);
					if (p + 1 != max_threads)
					{
						long long x1 = ((long long)p + 1ll) * (long long)size;
						x1 += (max_threads * multiple) >> 1;
						x1 /= max_threads * multiple;
						part[p].second = (int)(multiple * x1);
					}
					else part[p].second = size;
				}
			}
			~Worksharing() { if (part != 0) lapFree(part); }
		};
	}
}
