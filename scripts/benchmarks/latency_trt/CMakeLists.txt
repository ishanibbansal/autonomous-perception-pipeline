cmake_minimum_required(VERSION 3.15)
project(benchmark_latency CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Executable target
add_executable(benchmark_latency benchmark_latency.cpp)

# Include paths for CUDA and TensorRT
target_include_directories(benchmark_latency PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  /usr/include/x86_64-linux-gnu
  /usr/local/cuda/include
)

# Search for libnvinfer_plugin
find_library(NVINFER_PLUGIN_LIB
  NAMES nvinfer_plugin libnvinfer_plugin.so.11 libnvinfer_plugin.so
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
)

# Link libraries
target_link_libraries(benchmark_latency PRIVATE
  nvinfer
  ${NVINFER_PLUGIN_LIB}
  CUDA::cudart
)
