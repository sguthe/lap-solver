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
			std::vector<cudaStream_t> stream;
		public:
			Worksharing(int size, int multiple)
			{
				int max_devices = (size + multiple - 1) / multiple;
				int device_count;
				cudaDeviceProp deviceProp;
				cudaGetDeviceCount(&device_count);

				for (int current_device = 0; ((current_device < device_count) && ((int)device.size() < max_devices)); current_device++)
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

				int devices = (int)device.size();
				lapAlloc(part, devices, __FILE__, __LINE__);
				for (int p = 0; p < devices; p++)
				{
					long long x0 = (long long)p * (long long)size;
					x0 += (devices * multiple) >> 1;
					x0 /= devices * multiple;
					part[p].first = (int)(multiple * x0);
					if (p + 1 != devices)
					{
						long long x1 = ((long long)p + 1ll) * (long long)size;
						x1 += (devices * multiple) >> 1;
						x1 /= devices * multiple;
						part[p].second = (int)(multiple * x1);
					}
					else part[p].second = size;
				}
				stream.resize(devices);
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(device[t]);
					cudaStreamCreate(&stream[t]);
				}
			}
			~Worksharing()
			{
				if (part != 0) lapFree(part);
				int devices = (int)device.size();
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(device[t]);
					cudaStreamDestroy(stream[t]);
				}
			}
			int find(int x)
			{
				int devices = (int)device.size();
				for (int t = 0; t < devices; t++)
				{
					if ((x >= part[t].first) && (x < part[t].second)) return t;
				}
				return -1;
			}
		};
	}
}
