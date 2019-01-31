# Name the project you can choose any name you want here
PROJECT(lap) 

# Check requirements
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
FIND_PACKAGE(CUDA)

# set the compiler flags
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_C_COMPILER gcc)

# Release Mode
SET(CUDA_PROPAGATE_HOST_FLAGS "OFF")
SET(CMAKE_CXX_FLAGS "-Wall -O3 -Wfatal-errors -fstrict-aliasing -m64 -std=c++11 -fopenmp -flto -march=native")
SET(CUDA_NVCC_FLAGS "-arch=sm_52 --ptxas-options=-v -Xcompiler -Wall -O3 -std=c++11")

# Define where your executables should be put
SET(EXECUTABLE_OUTPUT_PATH ${MyProject_BINARY_DIR})

# Tell your source files here 
SET(test_cpu_SRCS
	../test/test_cpu.cpp
)
SET(test_gpu_SRCS
        ../test/test_gpu.cu
)

# Include directories
include_directories(../)
include_directories(../cub-1.8.0/)

# this command creates your executable from the files specified in MyProject_SRCS
ADD_EXECUTABLE(test_cpu ${test_cpu_SRCS})
CUDA_ADD_EXECUTABLE(test_gpu ${test_gpu_SRCS})

