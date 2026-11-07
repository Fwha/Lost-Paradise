# lpn-launcher: Play and Settings in front of lpn-host (scripts/make-appimage.sh packs both).
find_package(SDL2 QUIET)
if(NOT SDL2_FOUND)
    message(STATUS "SDL2 not found: no lpn-launcher")
    return()
endif()
add_executable(lpn-launcher
    main.cpp
    updater.cpp
    ../host/menu_font.cpp
    ../host/pad_glyphs.cpp
    ../host/stb_impl.cpp
)
target_link_libraries(lpn-launcher PRIVATE lpn-common SDL2::SDL2)
target_compile_definitions(lpn-launcher PRIVATE LPN_HAVE_SDL2)
target_include_directories(lpn-launcher SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/external/stb"
                           "${PROJECT_SOURCE_DIR}/external/nanosvg")

# The version Check for updates compares with the latest release's tag: the release workflow
# passes the tag, else git describe.
if(NOT LPN_VERSION)
    execute_process(COMMAND git describe --tags --always --dirty WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    OUTPUT_VARIABLE LPN_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT LPN_VERSION)
    set(LPN_VERSION dev)
endif()
message(STATUS "lpn-launcher version ${LPN_VERSION}")
target_compile_definitions(lpn-launcher PRIVATE LPN_VERSION="${LPN_VERSION}")
