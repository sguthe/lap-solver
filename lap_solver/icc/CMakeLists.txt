# Name the project you can choose any name you want here
PROJECT(lap) 

# Check requirements
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

# set the compiler flags
SET(CMAKE_CXX_COMPILER icpc)
SET(CMAKE_C_COMPILER icc)

# Release Mode
SET(CMAKE_CXX_FLAGS "-Wall -O3 -Wfatal-errors -fstrict-aliasing -m64 -std=c++11 -fopenmp -qopenmp-link=static -ipo")

# Define where your executables should be put
SET(EXECUTABLE_OUTPUT_PATH ${MyProject_BINARY_DIR})

# Tell your source files here 
SET(test_cpu_SRCS
	../test/test_cpu.cpp
)

# Include directories
include_directories(../)

# this command creates your executable from the files specified in MyProject_SRCS
ADD_EXECUTABLE(test_cpu ${test_cpu_SRCS})

