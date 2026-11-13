# lpn-launcher: Play, Settings and Tools (.pkg unpacking) in front of lpn-host (scripts/make-appimage.sh packs both).
find_package(SDL2 QUIET)
if(NOT SDL2_FOUND)
    message(STATUS "SDL2 not found: no lpn-launcher")
    return()
endif()
add_executable(lpn-launcher
    main.cpp
    settings_dialog.cpp
    updater.cpp
    ../host/menu_font.cpp
    ../host/pad_glyphs.cpp
    ../host/stb_impl.cpp
)
target_link_libraries(lpn-launcher PRIVATE lpn-common pkg SDL2::SDL2)
target_compile_definitions(lpn-launcher PRIVATE LPN_HAVE_SDL2)
if(WIN32)
    # "Lost Paradise.exe": a windowed program (no console), with its icon and manifest.
    target_sources(lpn-launcher PRIVATE "${PROJECT_SOURCE_DIR}/packaging/windows/launcher.rc")
    set_target_properties(lpn-launcher PROPERTIES WIN32_EXECUTABLE ON OUTPUT_NAME "Lost Paradise")
    target_compile_definitions(lpn-launcher PRIVATE SDL_MAIN_HANDLED)
    target_sources(lpn-launcher PRIVATE settings_dialog_win32.cpp)
    target_link_libraries(lpn-launcher PRIVATE comdlg32 shell32 ole32 comctl32)
else()
    # GTK 3 is opened at run time (settings_dialog_gtk.cpp): nothing to build or link against.
    target_sources(lpn-launcher PRIVATE settings_dialog_gtk.cpp)
    target_link_libraries(lpn-launcher PRIVATE ${CMAKE_DL_LIBS})
endif()
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
