// Force-included into AMD's FidelityFX SDK 1.1.4 sources when they are built with MinGW (GCC on
// Windows): they were written for MSVC's CRT, which declares these through headers MinGW's does
// not pull in the same way. (FSR-Vulkan's own shim covers non-Windows builds only.)
#pragma once

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <new>

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif

inline int lpnFfxSwprintfS(wchar_t* buffer, std::size_t size, const wchar_t* format, ...) {
    va_list args;
    va_start(args, format);
    const int n = std::vswprintf(buffer, size, format, args);
    va_end(args);
    return n;
}
#ifndef swprintf_s
#define swprintf_s lpnFfxSwprintfS
#endif
