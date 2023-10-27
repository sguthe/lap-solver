# Check requirements
cmake_minimum_required(VERSION 3.24)

# Name the project you can choose any name you want here
project(lap)

include(CheckLanguage)
check_language(CUDA)
find_package(OpenMP)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES auto)
  endif()
endif (CMAKE_CUDA_COMPILER)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-O3;-Wfatal-errors;-fstrict-aliasing;-m64;-qopenmp-link=static;-ipo;-marchcore-avx-i>")

  if (CMAKE_CUDA_COMPILER)
    if (OpenMP_CXX_FOUND)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;SHELL:-Xcompiler -Wall;-O3;SHELL:-Xcompiler -fopenmp;SHELL:-Xcompiler -ipo;SHELL:-Xcompiler -marchcore-avx-i;--expt-extended-lambda>")
    else (OpenMP_CXX_FOUND)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;SHELL:-Xcompiler -Wall;-O3;SHELL:-Xcompiler -ipo;SHELL:-Xcompiler -marchcore-avx-i;--expt-extended-lambda>")
    endif (OpenMP_CXX_FOUND)
  endif (CMAKE_CUDA_COMPILER)
endif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-O3;-Wfatal-errors;-fstrict-aliasing;-m64;-flto;-march=native;-mfpmath=sse>")

  if (CMAKE_CUDA_COMPILER)
    if (OpenMP_CXX_FOUND)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;SHELL:-Xcompiler -Wall;-O3;SHELL:-Xcompiler -fopenmp;SHELL:-Xcompiler -march=native;SHELL:-Xcompiler -mfpmath=sse;--expt-extended-lambda>")
    else (OpenMP_CXX_FOUND)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;SHELL:-Xcompiler -Wall;-O3;SHELL:-Xcompiler -march=native;SHELL:-Xcompiler -mfpmath=sse;--expt-extended-lambda>")
    endif (OpenMP_CXX_FOUND)
  endif (CMAKE_CUDA_COMPILER)
endif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_GENERATOR_PLATFORM x64)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Wall;/O2>")

  if (CMAKE_CUDA_COMPILER)
    if (OpenMP_CXX_FOUND)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;SHELL:-Xcompiler /Wall;-O3;SHELL:-Xcompiler /openmp;--expt-extended-lambda>")
    else (OpenMP_CXX_FOUND)
      add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;SHELL:-Xcompiler /Wall;-O3;--expt-extended-lambda>")
    endif (OpenMP_CXX_FOUND)
  endif (CMAKE_CUDA_COMPILER)
endif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

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
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_cpu PUBLIC OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

# second build for counting evaluations
add_executable(test_cpu_evaluated ${test_cpu_SRCS})
target_compile_definitions(test_cpu_evaluated PUBLIC LAP_DISPLAY_EVALUATED)
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_cpu_evaluated PUBLIC OpenMP::OpenMP_CXX)
endif (OpenMP_CXX_FOUND)

if (CMAKE_CUDA_COMPILER)
  add_executable(test_gpu ${test_gpu_SRCS})
  if (OpenMP_CXX_FOUND)
    target_link_libraries(test_gpu PUBLIC OpenMP::OpenMP_CXX)
  endif (OpenMP_CXX_FOUND)

  # second build for counting evaluations
  add_executable(test_gpu_evaluated ${test_gpu_SRCS})
  target_compile_definitions(test_gpu_evaluated PUBLIC LAP_DISPLAY_EVALUATED)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(test_gpu_evaluated PUBLIC OpenMP::OpenMP_CXX)
  endif (OpenMP_CXX_FOUND)
endif (CMAKE_CUDA_COMPILER)

set (passRegex "test passed")
set (failRegex "test failed")

add_test(NAME sanity_sequential_table_single_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -single)
add_test(NAME sanity_sequential_table_epsilon_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -epsilon)
add_test(NAME sanity_sequential_table_epsilon_16k COMMAND test_cpu -table_min 16000 -table_max 16000 -sanity -double -epsilon)
add_test(NAME sanity_sequential_caching_epsilon_1k COMMAND test_cpu -memory 1048576 -cached_min 1000 -cached_max 1000 -sanity_cached -double -epsilon)
add_test(NAME sanity_sequential_caching_epsilon_16k COMMAND test_cpu -memory 1048576 -cached_min 16000 -cached_max 16000 -sanity_cached -double -epsilon)
if (OpenMP_CXX_FOUND)
  add_test(NAME sanity_parallel_table_single_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -single -omp)
  add_test(NAME sanity_parallel_table_epsilon_1k COMMAND test_cpu -table_min 1000 -table_max 1000 -sanity -double -epsilon -omp)
  add_test(NAME sanity_parallel_table_epsilon_16k COMMAND test_cpu -table_min 16000 -table_max 16000 -sanity -double -epsilon -omp)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # disable these for now as they seem to be broken
  else (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_test(NAME sanity_parallel_caching_epsilon_1k COMMAND test_cpu -memory 1048576 -cached_min 1000 -cached_max 1000 -sanity_cached -double -epsilon -omp)
    add_test(NAME sanity_parallel_caching_epsilon_16k COMMAND test_cpu -memory 1048576 -cached_min 16000 -cached_max 16000 -sanity_cached -double -epsilon -omp)
  endif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif (OpenMP_CXX_FOUND)

set_property(TEST sanity_sequential_table_single_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_table_single_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_table_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_table_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_table_epsilon_16k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_table_epsilon_16k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_caching_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_caching_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
set_property(TEST sanity_sequential_caching_epsilon_16k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
set_property(TEST sanity_sequential_caching_epsilon_16k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
if (OpenMP_CXX_FOUND)
  set_property(TEST sanity_parallel_table_single_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_table_single_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  set_property(TEST sanity_parallel_table_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_table_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  set_property(TEST sanity_parallel_table_epsilon_16k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
  set_property(TEST sanity_parallel_table_epsilon_16k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # disable these for now as they seem to be broken
  else (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_property(TEST sanity_parallel_caching_epsilon_1k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
    set_property(TEST sanity_parallel_caching_epsilon_1k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
    set_property(TEST sanity_parallel_caching_epsilon_16k PROPERTY PASS_REGULAR_EXPRESSION "${passRegex}")
    set_property(TEST sanity_parallel_caching_epsilon_16k PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
  endif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif (OpenMP_CXX_FOUND)
