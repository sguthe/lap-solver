#
# 
# Name the project you can choose any name you want here
PROJECT(lap) 


# Check requirements
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

# set the compiler flags
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_C_COMPILER gcc)

# Debug Mode
#SET(CMAKE_CXX_FLAGS "-fPIC -g -D DEBUG -Wall -Wfatal-errors -fno-strict-aliasing -m64")
# Release Mode
SET(CMAKE_CXX_FLAGS "-g -Wall -O3 -Wfatal-errors -fstrict-aliasing -m64 -std=c++11 -fopenmp -ftree-vectorize -flto")
#SET(CMAKE_CXX_FLAGS "-pg -Wall -O2 -Wfatal-errors -fstrict-aliasing -m64 -std=c++11 -fopenmp")


# Add needed packages (e.g. libraries)
#FIND_PACKAGE(CUDA REQUIRED)


# Define where your executables should be put
set(EXECUTABLE_OUTPUT_PATH ${MyProject_BINARY_DIR})


# Tellll your source files here 
SET(test_cpu_SRCS
	test/test_cpu.cpp
)


# this command creates your executable from the files specified in MyProject_SRCS
ADD_EXECUTABLE(test_cpu ${test_cpu_SRCS})

