#pragma once
#include <utility>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

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
			std::vector<cudaEvent_t> event;
			std::vector<int> sm_count;
			std::vector<int> threads_per_sm;
		public:
			Worksharing(int size, int multiple, std::vector<int> &devs, bool silent)
			{
				int max_devices = (size + multiple - 1) / multiple;
				int device_count;

				cudaDeviceProp deviceProp;

				if (devs.empty())
				{
					cudaGetDeviceCount(&device_count);

#ifndef LAP_CUDA_ALLOW_WDDM
					bool allow_wddm = false;
					bool done_searching = false;

					while (!done_searching)
					{
						for (int current_device = 0; ((current_device < device_count) && ((int)device.size() < max_devices)); current_device++)
						{
							cudaGetDeviceProperties(&deviceProp, current_device);

							// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
							if (deviceProp.computeMode != cudaComputeModeProhibited)
							{
								if ((allow_wddm) || (deviceProp.tccDriver))
								{
									if (!silent) lapInfo << "Adding device " << current_device << ": " << deviceProp.name << std::endl;
									device.push_back(current_device);
									sm_count.push_back(deviceProp.multiProcessorCount);
									threads_per_sm.push_back(deviceProp.maxThreadsPerMultiProcessor);
								}
							}
						}
						if (device.empty())
						{
							if (allow_wddm) done_searching = true;
							else allow_wddm = true;
						}
						else
						{
							done_searching = true;
						}
					}
#else
					for (int current_device = 0; ((current_device < device_count) && ((int)device.size() < max_devices)); current_device++)
					{
						cudaGetDeviceProperties(&deviceProp, current_device);

						// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
						if (deviceProp.computeMode != cudaComputeModeProhibited)
						{
							if (!silent) lapInfo << "Adding device " << current_device << ": " << deviceProp.name << std::endl;
							device.push_back(current_device);
							sm_count.push_back(deviceProp.multiProcessorCount);
							threads_per_sm.push_back(deviceProp.maxThreadsPerMultiProcessor);
						}
					}
#endif
				}
				else
				{
					device_count = (int)devs.size();
					for (int i = 0; ((i < device_count) && ((int)device.size() < max_devices)); i++)
					{
						int current_device = devs[i];
						cudaGetDeviceProperties(&deviceProp, current_device);

						// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
						if (deviceProp.computeMode != cudaComputeModeProhibited)
						{
							if (!silent) lapInfo << "Adding device " << current_device << ": " << deviceProp.name << std::endl;
							device.push_back(current_device);
							sm_count.push_back(deviceProp.multiProcessorCount);
							threads_per_sm.push_back(deviceProp.maxThreadsPerMultiProcessor);
						}
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
				event.resize(devices);
				for (int t = 0; t < devices; t++)
				{
					cudaSetDevice(device[t]);
					cudaStreamCreate(&stream[t]);
					cudaEventCreateWithFlags(&event[t], cudaEventDisableTiming);
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
					cudaEventDestroy(event[t]);
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
