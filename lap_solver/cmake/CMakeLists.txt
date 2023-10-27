# Check requirements
cmake_minimum_required(VERSION 3.18)

# Name the project you can choose any name you want here
project(lap)

include(CheckLanguage)
check_language(CUDA)
find_package(OpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif (CMAKE_CUDA_COMPILER)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-O3;-Wfatal-errors;-fstrict-aliasing;-m64;-qopenmp-link=static;-ipo;-marchcore-avx-i>")

  if (CMAKE_CUDA_COMPILER)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_52;--ptxas-options=-v;SHELL:-Xcompiler -Wall;-O3;SHELL:-Xcompiler -fopenmp;SHELL:-Xcompiler -ipo;SHELL:-Xcompiler -marchcore-avx-i;--expt-extended-lambda>")
  endif (CMAKE_CUDA_COMPILER)
endif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-O3;-Wfatal-errors;-fstrict-aliasing;-m64;-flto;-march=native;-mfpmath=sse>")

  if (CMAKE_CUDA_COMPILER)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_52;--ptxas-options=-v;SHELL:-Xcompiler -Wall;-O3;SHELL:-Xcompiler -fopenmp;SHELL:-Xcompiler -march=native;SHELL:-Xcompiler -mfpmath=sse;--expt-extended-lambda>")
  endif (CMAKE_CUDA_COMPILER)
endif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

# Define where your executables should be put
set(EXECUTABLE_OUTPUT_PATH .)

# Tell your source files here
set(test_cpu_SRCS
	  ../test/test_cpu.cpp
)

if (CMAKE_CUDA_COMPILER)
  set(test_gpu_SRCS
      ../test/test_gpu.cu
)
endif (CMAKE_CUDA_COMPILER)

# Include directories
include_directories(../)

add_executable(test_cpu ${test_cpu_SRCS})
target_link_libraries(test_cpu PUBLIC OpenMP::OpenMP_CXX)

# second build for counting evaluations
add_executable(test_cpu_evaluated ${test_cpu_SRCS})
target_compile_definitions(test_cpu_evaluated PUBLIC LAP_DISPLAY_EVALUATED)
target_link_libraries(test_cpu_evaluated PUBLIC OpenMP::OpenMP_CXX)

if (CMAKE_CUDA_COMPILER)
  add_executable(test_gpu ${test_gpu_SRCS})
  target_link_libraries(test_gpu PUBLIC OpenMP::OpenMP_CXX)

  # second build for counting evaluations
  add_executable(test_gpu_evaluated ${test_gpu_SRCS})
  target_compile_definitions(test_gpu_evaluated PUBLIC LAP_DISPLAY_EVALUATED)
  target_link_libraries(test_gpu_evaluated PUBLIC OpenMP::OpenMP_CXX)
endif (CMAKE_CUDA_COMPILER)
