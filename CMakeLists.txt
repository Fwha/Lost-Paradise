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
add_subdirectory(tests)
