cmake_minimum_required(VERSION 3.16)

project(EigenDNN)

# Configurações do compilador C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall -Wextra")

# Encontrar dependências
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Incluir diretórios de header
include_directories(src)

# Adicionar subdiretórios
add_subdirectory(src)
add_subdirectory(example/simple_main)

# Exemplo de bioinformática
add_executable(bioinformatics_example example/bioinformatics_main.cpp)
target_link_libraries(bioinformatics_example Eigen3::Eigen)
target_include_directories(bioinformatics_example PRIVATE src)

# Configurações de instalação
install(TARGETS bioinformatics_example DESTINATION bin)

# Configurações de debug/release
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DDEBUG")
    message(STATUS "Debug mode enabled")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
    message(STATUS "Release mode enabled")
endif()

# Informações do projeto
message(STATUS "=== Sistema Integrado de Bioinformática ===")
message(STATUS "Projeto: ${PROJECT_NAME}")
message(STATUS "Versão: 2.0 - IA Melhorada")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===========================================")
