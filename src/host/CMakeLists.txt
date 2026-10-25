enable_language(ASM)

add_executable(lpn-host
    main.cpp
    module.cpp
    memory.cpp
    linker.cpp
    hle.cpp
    hle_kernel.cpp
    hle_memory.cpp
    hle_pthread.cpp
    hle_file.cpp
    hle_videoout.cpp
    hle_gnm.cpp
    hle_gnm_packets.cpp
    hle_services.cpp
    equeue.cpp
    pm4.cpp
    profile.cpp
    coverage.cpp
    combo_trace.cpp
    gpu/frame_dump.cpp
    gpu/formats.cpp
    gpu/renderer.cpp
    gpu/tiling.cpp
    gpu/vk.cpp
    window.cpp
    save_menu.cpp
    display_menu.cpp
    display_settings.cpp
    tls.cpp
    crash.cpp
    trampoline.S
)
find_package(ZLIB REQUIRED)
find_package(Vulkan REQUIRED)
target_link_libraries(lpn-host PRIVATE ps4elf gcn ZLIB::ZLIB Vulkan::Vulkan)
# The guest TLS slot must sit at a fixed offset from %fs, which the initial-exec model gives.
target_compile_options(lpn-host PRIVATE -ftls-model=initial-exec)
# Vulkan structs are initialised as {sType} and the rest zeroed.
target_compile_options(lpn-host PRIVATE -Wno-missing-field-initializers)

# The game window is optional so the host still builds and runs headless.
find_package(SDL2 QUIET)
if(SDL2_FOUND)
    target_link_libraries(lpn-host PRIVATE SDL2::SDL2)
    target_compile_definitions(lpn-host PRIVATE LPN_HAVE_SDL2)
else()
    message(STATUS "SDL2 not found: lpn-host will run without a window")
endif()
