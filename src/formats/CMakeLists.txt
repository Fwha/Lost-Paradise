add_library(formats STATIC gmd.cpp gmt.cpp par.cpp skeleton.cpp sllz.cpp)
target_include_directories(formats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
