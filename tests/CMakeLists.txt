add_executable(formats_test formats_test.cpp)
target_link_libraries(formats_test PRIVATE formats)
add_test(NAME formats_test COMMAND formats_test)
