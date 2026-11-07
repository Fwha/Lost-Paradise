# PS4 .pkg extraction (pkg.h): the launcher's Tools panel, and lpn-pkg on the command line.
find_package(ZLIB REQUIRED)
add_library(pkg STATIC pkg.cpp crypto.cpp)
target_include_directories(pkg PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(pkg PRIVATE ZLIB::ZLIB)

add_executable(lpn-pkg main.cpp)
target_link_libraries(lpn-pkg PRIVATE pkg)
