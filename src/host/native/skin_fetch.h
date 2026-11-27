#pragma once

namespace host::native {

// Character and prepass fetches share VGPR-bit decoding and palette addressing.
// PREPASS_SKIN uses a compact seven-vec4 constant buffer.
inline constexpr const char* kSkinFetch = R"(uint word(uint b, uint i) {
    switch (b) {
    case 0u: return v0.w[i];
    case 1u: return v1.w[i];
    case 2u: return v2.w[i];
    case 3u: return v3.w[i];
    case 4u: return v4.w[i];
    default: return v5.w[i];
    }
}
uint conv(uint raw, uint bits, uint nfmt) {
    int sx = bitfieldExtract(int(raw),0,int(bits));
    if (nfmt == 4u) return raw;
    if (nfmt == 5u) return uint(sx);
    if (nfmt == 7u) return floatBitsToUint(unpackHalf2x16(raw).x);
    if (nfmt == 0u)
        return floatBitsToUint(float(raw) / float((1u << bits) - 1u));
    return floatBitsToUint(max(float(sx) / float((1u << (bits-1u)) - 1u),-1.0));
}
// This returns VGPR bits. In particular the bone indices are never numerically
// converted to float and back. The host permits only exact uint/sint decoding.
uvec4 fetchBits(uint b) {
#ifdef PREPASS_SKIN
    uvec4 s = cb.c[int(b)];
#else
    uvec4 s = cb.c[449 + int(b)];
#endif
#ifdef PREPASS_SKIN
    if (uint(gl_VertexIndex) >= s.w) return uvec4(0u);
#endif
    uint i = s.x + uint(gl_VertexIndex) * s.y;
    uint dfmt = s.z & 15u, nfmt = (s.z >> 4) & 7u;
    uvec4 e = uvec4(0u,0u,0u,floatBitsToUint(1.0));
    if (dfmt == 4u || dfmt == 11u || dfmt == 13u || dfmt == 14u) {
        uint n = dfmt == 4u ? 1u : dfmt == 11u ? 2u : dfmt == 13u ? 3u : 4u;
        for (uint c = 0u; c < n; ++c) e[c] = word(b,i+c);
    } else if (dfmt == 2u || dfmt == 5u || dfmt == 12u) {
        uint n = dfmt == 2u ? 1u : dfmt == 5u ? 2u : 4u;
        for (uint c = 0u; c < n; ++c)
            e[c] = conv((word(b,i+c/2u) >> (16u*(c&1u))) & 0xffffu,16u,nfmt);
    } else {
        uint n = dfmt == 1u ? 1u : dfmt == 3u ? 2u : 4u;
        uint x = word(b,i);
        for (uint c = 0u; c < n; ++c)
            e[c] = conv((x >> (8u*c)) & 0xffu,8u,nfmt);
    }
    return e;
}
vec4 fetch(uint b) { return uintBitsToFloat(fetchBits(b)); }
vec4 boneRow(uint index, uint row) {
#ifdef PREPASS_SKIN
    uint i = cb.c[6].x + (48u*index + 16u*row)/4u;
#else
    uint i = cb.c[456].x + (48u*index + 16u*row)/4u;
#endif
    return uintBitsToFloat(uvec4(palette.w[i],palette.w[i+1u],
                                palette.w[i+2u],palette.w[i+3u]));
}
)";

} // namespace host::native
