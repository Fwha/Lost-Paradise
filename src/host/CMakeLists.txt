enable_language(ASM)

add_executable(lpn-host
    main.cpp
    module.cpp
    linker.cpp
    hle.cpp
    hle_kernel.cpp
    hle_memory.cpp
    hle_pthread.cpp
    hle_file.cpp
    tls.cpp
    crash.cpp
    trampoline.S
)
target_link_libraries(lpn-host PRIVATE ps4elf)
# The guest TLS slot must sit at a fixed offset from %fs, which the initial-exec model gives.
target_compile_options(lpn-host PRIVATE -ftls-model=initial-exec)
