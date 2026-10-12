add_library(ps4elf STATIC elf.cpp dynlib.cpp ehframe.cpp nid.cpp)
target_include_directories(ps4elf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
