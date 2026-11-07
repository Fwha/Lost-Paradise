add_executable(formats_test formats_test.cpp)
target_link_libraries(formats_test PRIVATE formats)
add_test(NAME formats_test COMMAND formats_test)

add_executable(gcn_test gcn_test.cpp)
target_link_libraries(gcn_test PRIVATE gcn)
add_test(NAME gcn_test COMMAND gcn_test)

add_executable(game_dir_test game_dir_test.cpp)
target_link_libraries(game_dir_test PRIVATE lpn-common)
add_test(NAME game_dir_test COMMAND game_dir_test)
