add_executable(formats_test formats_test.cpp)
target_link_libraries(formats_test PRIVATE formats)
add_test(NAME formats_test COMMAND formats_test)

add_executable(gcn_test gcn_test.cpp)
target_link_libraries(gcn_test PRIVATE gcn)
add_test(NAME gcn_test COMMAND gcn_test)
