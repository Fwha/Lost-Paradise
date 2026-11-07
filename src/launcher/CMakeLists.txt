# lpn-launcher: Play and Settings in front of lpn-host (scripts/make-appimage.sh packs both).
find_package(SDL2 QUIET)
if(NOT SDL2_FOUND)
    message(STATUS "SDL2 not found: no lpn-launcher")
    return()
endif()
add_executable(lpn-launcher
    main.cpp
    ../host/menu_font.cpp
    ../host/stb_impl.cpp
)
target_link_libraries(lpn-launcher PRIVATE lpn-common SDL2::SDL2)
target_compile_definitions(lpn-launcher PRIVATE LPN_HAVE_SDL2)
target_include_directories(lpn-launcher SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/external/stb")
