cmake_minimum_required(VERSION 2.8)

project(CppDNN)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
add_executable(${PROJECT_NAME} "main.cpp")
