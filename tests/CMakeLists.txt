add_executable(formats_test formats_test.cpp)
target_link_libraries(formats_test PRIVATE formats)
add_test(NAME formats_test COMMAND formats_test)

add_executable(gcn_test gcn_test.cpp)
target_link_libraries(gcn_test PRIVATE gcn)
add_test(NAME gcn_test COMMAND gcn_test)

add_executable(game_dir_test game_dir_test.cpp)
target_link_libraries(game_dir_test PRIVATE lpn-common)
add_test(NAME game_dir_test COMMAND game_dir_test)

add_executable(pkg_test pkg_test.cpp)
target_link_libraries(pkg_test PRIVATE pkg)
add_test(NAME pkg_test COMMAND pkg_test)

# lpn-pkg on a fake PKG that make_fake_pkg.py makes (it needs Python's cryptography package).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import cryptography" RESULT_VARIABLE no_crypto
                    OUTPUT_QUIET ERROR_QUIET)
    if(no_crypto EQUAL 0)
        add_test(NAME pkg_extract_test
                 COMMAND "${CMAKE_COMMAND}" -DPYTHON=${Python3_EXECUTABLE} -DLPN_PKG=$<TARGET_FILE:lpn-pkg>
                         -DSRC=${PROJECT_SOURCE_DIR} -DWORK=${CMAKE_CURRENT_BINARY_DIR}/pkg_extract_test
                         -P "${CMAKE_CURRENT_SOURCE_DIR}/pkg_extract_test.cmake")
    else()
        message(STATUS "Python cryptography not found: no pkg_extract_test")
    endif()
endif()
