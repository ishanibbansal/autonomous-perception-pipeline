cmake_minimum_required(VERSION 3.15)
project(student_bev_inference)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS 2 Packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Add Executable
add_executable(bev_node src/student_bev_node.cpp)

# Include directories (Standard CMake bypass)
target_include_directories(bev_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)

# Link TensorRT, CUDA, OpenCV, and ROS 2
target_link_libraries(bev_node
  nvinfer
  CUDA::cudart
  ${OpenCV_LIBS}
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  cv_bridge::cv_bridge
)

install(TARGETS bev_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()