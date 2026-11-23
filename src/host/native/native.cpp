#include "host/native/native.h"

namespace host::native {

Pass* aaPass();     // aa.cpp
Pass* shadowPass(); // shadow.cpp
Pass* prepassPass(); // prepass.cpp
Pass* materialPass(); // material.cpp
Pass* characterPass(); // character.cpp
Pass* particlePass(); // particle.cpp

namespace {

Pass* const* all() {
    static Pass* const list[] = {aaPass(), shadowPass(), prepassPass(), materialPass(), characterPass(), particlePass()};
    return list;
}
constexpr size_t kPasses = 6;

} // namespace

std::span<Pass* const> passes() { return {all(), kPasses}; }

void swizzle(const Target& t, int32_t out[4]) {
    // As gcn/translator.cpp's export: by components (0 to 4) and COMP_SWAP.
    static const char* const kSwizzle[5][4] = {{"xyzw", "xyzw", "xyzw", "xyzw"},
                                               {"xxxx", "yyyy", "zzzz", "wwww"},
                                               {"xyzw", "xwzw", "yxzw", "wxzw"},
                                               {"xyzw", "xywz", "zyxw", "wyxz"},
                                               {"xyzw", "zyxw", "wzyx", "wxyz"}};
    const char* s = kSwizzle[t.comps > 4 ? 4 : t.comps][t.swap & 3];
    for (int i = 0; i < 4; ++i) out[i] = s[i] == 'w' ? 3 : s[i] - 'x';
}

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
