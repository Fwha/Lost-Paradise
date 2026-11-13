#include "host/gpu/upscale.h"

#include "host/gpu/dlss.h"
#include "host/gpu/fsr.h"

#include <algorithm>

namespace host::render::upscale {

bool builtIn(Upscaler u) {
    switch (u) {
    case Upscaler::Dlss: return dlss::builtIn();
    case Upscaler::Fsr: return fsr::builtIn();
    default: return false;
    }
}

bool usable(Upscaler u) {
    switch (u) {
    case Upscaler::Dlss: return dlss::usable();
    case Upscaler::Fsr: return fsr::usable();
    default: return false;
    }
}

void jitter(uint64_t n, uint32_t renderW, uint32_t outW, float* x, float* y) {
    const float scale = renderW ? std::max(1.0f, float(outW) / float(renderW)) : 1.0f;
    const uint64_t phases = std::max<uint64_t>(8, uint64_t(8.0f * scale * scale + 0.5f));
    auto halton = [](uint64_t i, uint32_t base) {
        float f = 1, r = 0;
        for (; i; i /= base) f /= float(base), r += f * float(i % base);
        return r;
    };
    const uint64_t i = n % phases + 1; // index 0 of the sequence is 0, 0
    *x = halton(i, 2) - 0.5f;
    *y = halton(i, 3) - 0.5f;
}

} // namespace host::render::upscale
