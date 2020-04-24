cmake_minimum_required(VERSION 2.8)

project(EigenDNN)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)
#add_subdirectory(example/simple_main)
