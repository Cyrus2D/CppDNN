cmake_minimum_required(VERSION 2.8)

project(EigenDNN)
add_executable(${PROJECT_NAME} "main.cpp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
