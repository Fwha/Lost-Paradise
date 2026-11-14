# Shared by lpn-host and lpn-launcher: the settings file and where the game is.
add_library(lpn-common STATIC settings_file.cpp game_dir.cpp compat.cpp mods.cpp x86_emu.cpp)
target_include_directories(lpn-common PUBLIC "${PROJECT_SOURCE_DIR}/src")
