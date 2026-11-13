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
    mod_files.cpp
    hle_videoout.cpp
    hle_gnm.cpp
    hle_gnm_packets.cpp
    hle_services.cpp
    equeue.cpp
    pm4.cpp
    profile.cpp
    coverage.cpp
    combo_trace.cpp
    input_history.cpp
    title_menu.cpp
    pc_settings.cpp
    pause_menu.cpp
    hooks.cpp
    draw_sites.cpp
    gpu/frame_dump.cpp
    gpu/dlss.cpp
    gpu/fsr.cpp
    gpu/upscale.cpp
    gpu/formats.cpp
    gpu/renderer.cpp
    gpu/tiling.cpp
    gpu/vk.cpp
    window.cpp
    save_menu.cpp
    pad_glyphs.cpp
    button_icon_files.cpp
    menu_font.cpp
    stb_impl.cpp
    display_menu.cpp
    display_settings.cpp
    tls.cpp
    mman_win.cpp
    $<$<BOOL:${WIN32}>:${PROJECT_SOURCE_DIR}/packaging/windows/host.rc>
    crash.cpp
    trampoline.S
)
find_package(ZLIB REQUIRED)
find_package(Vulkan REQUIRED)
target_link_libraries(lpn-host PRIVATE lpn-common ps4elf gcn formats ZLIB::ZLIB Vulkan::Vulkan)
# stb_truetype and stb_image (public domain, external/stb) for the menu font and save icons.
target_include_directories(lpn-host SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/external/stb")
# nanosvg (zlib, external/nanosvg) draws the controller button icons (pad_glyphs.cpp).
target_include_directories(lpn-host SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/external/nanosvg")
if(WIN32)
    # The game runs on the main thread: a big stack, all committed, since PS4 code touches
    # stack pages out of order (no stack probes) and Windows only grows the stack in sequence.
    target_link_options(lpn-host PRIVATE -Xlinker --stack -Xlinker 0x2000000,0x2000000)
    # Keeps Windows' own allocations low, out of the guest range (8 GiB to 1 TiB, mman_win.cpp).
    target_link_options(lpn-host PRIVATE -Wl,--disable-high-entropy-va)
    target_link_libraries(lpn-host PRIVATE winmm)
else()
    # The guest TLS slot must sit at a fixed offset from %fs, which the initial-exec model gives.
    target_compile_options(lpn-host PRIVATE -ftls-model=initial-exec)
endif()
# Vulkan structs are initialised as {sType} and the rest zeroed.
target_compile_options(lpn-host PRIVATE -Wno-missing-field-initializers)

# The game window is optional so the host still builds and runs headless.
find_package(SDL2 QUIET)
if(SDL2_FOUND)
    target_link_libraries(lpn-host PRIVATE SDL2::SDL2)
    target_compile_definitions(lpn-host PRIVATE LPN_HAVE_SDL2)
    if(WIN32)
        target_compile_definitions(lpn-host PRIVATE SDL_MAIN_HANDLED)
    endif()
else()
    message(STATUS "SDL2 not found: lpn-host will run without a window")
endif()

# NVIDIA DLSS: scripts/fetch-dlss.sh puts the SDK in third_party/dlss. Without it the host builds
# without DLSS. The DLSS runtime goes beside lpn-host, where NGX looks for it.
set(LPN_DLSS_SDK "${PROJECT_SOURCE_DIR}/third_party/dlss" CACHE PATH "NVIDIA DLSS SDK")
# Linux only for now: the Windows SDK libraries are built for MSVC.
if(NOT WIN32 AND EXISTS "${LPN_DLSS_SDK}/include/nvsdk_ngx_vk.h")
    message(STATUS "DLSS SDK: ${LPN_DLSS_SDK}")
    target_include_directories(lpn-host SYSTEM PRIVATE "${LPN_DLSS_SDK}/include")
    target_link_libraries(lpn-host PRIVATE "${LPN_DLSS_SDK}/lib/Linux_x86_64/libnvsdk_ngx.a" ${CMAKE_DL_LIBS})
    target_compile_definitions(lpn-host PRIVATE LPN_HAVE_DLSS)
    file(GLOB LPN_DLSS_RUNTIME "${LPN_DLSS_SDK}/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*")
    add_custom_command(TARGET lpn-host POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LPN_DLSS_RUNTIME} $<TARGET_FILE_DIR:lpn-host>)
else()
    message(STATUS "DLSS SDK not found: lpn-host builds without DLSS (scripts/fetch-dlss.sh)")
endif()

# AMD FSR 3.1 (vendor-neutral temporal upscaling, gpu/fsr.cpp): scripts/fetch-fsr.sh puts
# FSR-Vulkan (MIT) in third_party/fsr-vulkan. Only its FSR 3.1 upscaler is built, into lpn-host.
set(LPN_FSR_DIR "${PROJECT_SOURCE_DIR}/third_party/fsr-vulkan" CACHE PATH "FSR-Vulkan source tree")
if(EXISTS "${LPN_FSR_DIR}/include/ffx_vk_portable.h")
    message(STATUS "FSR-Vulkan: ${LPN_FSR_DIR}")
    enable_language(C) # its C API is C, and its targets ask for C11
    set(FFX_VK_PORTABLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FFX_VK_PORTABLE_BUILD_PROBE OFF CACHE BOOL "" FORCE)
    set(FFX_VK_PORTABLE_BUILD_FSR3_3_1_5_SCAFFOLD OFF CACHE BOOL "" FORCE)
    set(FFX_VK_PORTABLE_BUILD_FSR4_V07_VULKAN OFF CACHE BOOL "" FORCE)
    set(FFX_VK_PORTABLE_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory("${LPN_FSR_DIR}" "${CMAKE_BINARY_DIR}/fsr-vulkan" EXCLUDE_FROM_ALL SYSTEM)
    # Its sources are AMD's and FSR-Vulkan's: built without this project's warnings.
    foreach(t ffx_vulkan_portable ffx_vulkan_fsr3_host_1_1_4 ffx_vulkan_fsr3_vk_backend_1_1_4)
        if(TARGET ${t})
            target_compile_options(${t} PRIVATE -w)
        endif()
    endforeach()
    target_link_libraries(lpn-host PRIVATE ffx-vulkan::portable)
    target_compile_definitions(lpn-host PRIVATE LPN_HAVE_FSR)
else()
    message(STATUS "FSR-Vulkan not found: lpn-host builds without FSR (scripts/fetch-fsr.sh)")
endif()
