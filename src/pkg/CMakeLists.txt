# PS4 .pkg extraction (pkg.h) and putting games, updates and DLC in the game folder (install.h): the launcher's Tools panel, and lpn-pkg on the command line.
find_package(ZLIB REQUIRED)
add_library(pkg STATIC pkg.cpp crypto.cpp install.cpp)
target_include_directories(pkg PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(pkg PRIVATE lpn-common ZLIB::ZLIB)

add_executable(lpn-pkg main.cpp)
target_link_libraries(lpn-pkg PRIVATE pkg)
