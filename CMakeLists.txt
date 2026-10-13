cmake_minimum_required(VERSION 3.20)
project(lost_paradise_native CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra)

enable_testing()

add_subdirectory(src/ps4elf)
add_subdirectory(src/formats)
add_subdirectory(tools/inspect)
add_subdirectory(tools/par)
add_subdirectory(tools/gmd)
add_subdirectory(tools/gmt)
add_subdirectory(src/host)
add_subdirectory(tests)

find_package(glslang REQUIRED)
add_library(gcn STATIC src/gcn/decoder.cpp src/gcn/analysis.cpp src/gcn/translator.cpp src/gcn/spirv.cpp)
target_include_directories(gcn PUBLIC src)
target_link_libraries(gcn PRIVATE glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)
add_executable(lpn-gcn tools/gcn/main.cpp)
target_link_libraries(lpn-gcn PRIVATE gcn)
