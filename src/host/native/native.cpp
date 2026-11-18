#include "host/native/native.h"

namespace host::native {

Pass* aaPass(); // aa.cpp

namespace {

Pass* const* all() {
    static Pass* const list[] = {aaPass()};
    return list;
}
constexpr size_t kPasses = 1;

} // namespace

std::span<Pass* const> passes() { return {all(), kPasses}; }

Claim claim(const std::string& psName) {
    for (Pass* p : passes())
        if (const int role = p->claims(psName)) return {p, role};
    return {};
}

// gl_VertexIndex 0, 1, 2: a triangle over the whole viewport.
const char* const kFullscreenVs = R"(#version 450
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

} // namespace host::native
