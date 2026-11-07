// SHA-256 (FIPS 180-4), AES-128 (FIPS 197), XTS (IEEE 1619) and RSA by Montgomery multiplication.
#include "pkg/crypto.h"

#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define PKG_AESNI 1
#endif

namespace pkg::crypto {

// SHA-256

namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

uint32_t ror(uint32_t x, int n) { return x >> n | x << (32 - n); }

struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64];
    size_t used = 0;
    uint64_t total = 0;

    void block(const uint8_t* p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(p[4 * i]) << 24 | p[4 * i + 1] << 16 | p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ w[i - 15] >> 3;
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ w[i - 2] >> 10;
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    void update(const void* data, size_t size) {
        const auto* p = static_cast<const uint8_t*>(data);
        total += size;
        while (size) {
            const size_t n = std::min(size, 64 - used);
            std::memcpy(buf + used, p, n);
            used += n, p += n, size -= n;
            if (used == 64) block(buf), used = 0;
        }
    }
    Digest finish() {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (used != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = uint8_t(bits >> (56 - 8 * i));
        update(len, 8);
        Digest out;
        for (int i = 0; i < 32; ++i) out[i] = uint8_t(h[i / 4] >> (24 - 8 * (i % 4)));
        return out;
    }
};

} // namespace

Digest sha256(const void* data, size_t size) {
    Sha256 s;
    s.update(data, size);
    return s.finish();
}

Digest hmacSha256(const void* key, size_t keySize, const void* data, size_t size) {
    uint8_t k[64] = {};
    if (keySize > 64) {
        const Digest d = sha256(key, keySize);
        std::memcpy(k, d.data(), 32);
    } else {
        std::memcpy(k, key, keySize);
    }
    uint8_t pad[64];
    for (int i = 0; i < 64; ++i) pad[i] = k[i] ^ 0x36;
    Sha256 inner;
    inner.update(pad, 64);
    inner.update(data, size);
    const Digest ih = inner.finish();
    for (int i = 0; i < 64; ++i) pad[i] = k[i] ^ 0x5c;
    Sha256 outer;
    outer.update(pad, 64);
    outer.update(ih.data(), 32);
    return outer.finish();
}

// AES-128

namespace {

constexpr uint8_t kSbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

struct InvSbox {
    uint8_t t[256];
    constexpr InvSbox() : t() {
        for (int i = 0; i < 256; ++i) t[kSbox[i]] = uint8_t(i);
    }
};
constexpr InvSbox kInv;

uint8_t xtime(uint8_t x) { return uint8_t(x << 1 ^ (x & 0x80 ? 0x1b : 0)); }
uint8_t mul(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    for (; b; b >>= 1, a = xtime(a))
        if (b & 1) r ^= a;
    return r;
}

#ifdef PKG_AESNI
const bool kCpuNi = __builtin_cpu_supports("aes") && __builtin_cpu_supports("sse2");
bool gUseNi = kCpuNi;
bool hasAesNi() { return gUseNi; }

__attribute__((target("aes,sse2"))) void niInverseKeys(const uint8_t enc[11][16], uint8_t dec[11][16]) {
    _mm_store_si128(reinterpret_cast<__m128i*>(dec[0]), _mm_load_si128(reinterpret_cast<const __m128i*>(enc[10])));
    for (int i = 1; i < 10; ++i)
        _mm_store_si128(reinterpret_cast<__m128i*>(dec[i]),
                        _mm_aesimc_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(enc[10 - i]))));
    _mm_store_si128(reinterpret_cast<__m128i*>(dec[10]), _mm_load_si128(reinterpret_cast<const __m128i*>(enc[0])));
}

__attribute__((target("aes,sse2"))) __m128i niEncrypt(const uint8_t k[11][16], __m128i b) {
    b = _mm_xor_si128(b, _mm_load_si128(reinterpret_cast<const __m128i*>(k[0])));
    for (int i = 1; i < 10; ++i) b = _mm_aesenc_si128(b, _mm_load_si128(reinterpret_cast<const __m128i*>(k[i])));
    return _mm_aesenclast_si128(b, _mm_load_si128(reinterpret_cast<const __m128i*>(k[10])));
}

__attribute__((target("aes,sse2"))) __m128i niDecrypt(const uint8_t k[11][16], __m128i b) {
    b = _mm_xor_si128(b, _mm_load_si128(reinterpret_cast<const __m128i*>(k[0])));
    for (int i = 1; i < 10; ++i) b = _mm_aesdec_si128(b, _mm_load_si128(reinterpret_cast<const __m128i*>(k[i])));
    return _mm_aesdeclast_si128(b, _mm_load_si128(reinterpret_cast<const __m128i*>(k[10])));
}
#endif

// The tweak times x in GF(2^128), little-endian.
void gfDouble(uint8_t t[16]) {
    uint8_t carry = 0;
    for (int i = 0; i < 16; ++i) {
        const uint8_t next = t[i] >> 7;
        t[i] = uint8_t(t[i] << 1 | carry);
        carry = next;
    }
    if (carry) t[0] ^= 0x87;
}

} // namespace

bool useAesNi(bool on) {
#ifdef PKG_AESNI
    const bool was = gUseNi;
    gUseNi = on && kCpuNi;
    return was;
#else
    return on && false;
#endif
}

Aes128::Aes128(const uint8_t key[16]) {
    std::memcpy(enc_[0], key, 16);
    uint8_t rcon = 1;
    for (int r = 1; r <= 10; ++r) {
        const uint8_t* p = enc_[r - 1];
        uint8_t* k = enc_[r];
        uint8_t t[4] = {kSbox[p[13]], kSbox[p[14]], kSbox[p[15]], kSbox[p[12]]};
        t[0] ^= rcon;
        rcon = xtime(rcon);
        for (int i = 0; i < 4; ++i) k[i] = p[i] ^ t[i];
        for (int i = 4; i < 16; ++i) k[i] = p[i] ^ k[i - 4];
    }
#ifdef PKG_AESNI
    if (kCpuNi) niInverseKeys(enc_, dec_);
#endif
}

void Aes128::encrypt(const uint8_t in[16], uint8_t out[16]) const {
#ifdef PKG_AESNI
    if (hasAesNi()) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                         niEncrypt(enc_, _mm_loadu_si128(reinterpret_cast<const __m128i*>(in))));
        return;
    }
#endif
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i] ^ enc_[0][i];
    for (int r = 1; r <= 10; ++r) {
        uint8_t t[16];
        // SubBytes and ShiftRows (the state is column-major: byte i is row i % 4, column i / 4).
        for (int i = 0; i < 16; ++i) t[i] = kSbox[s[(i + 4 * (i % 4)) % 16]];
        if (r < 10)
            for (int c = 0; c < 4; ++c) {
                uint8_t* col = t + 4 * c;
                const uint8_t a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
                col[0] = uint8_t(xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3);
                col[1] = uint8_t(a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3);
                col[2] = uint8_t(a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3);
                col[3] = uint8_t(xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3));
            }
        for (int i = 0; i < 16; ++i) s[i] = t[i] ^ enc_[r][i];
    }
    std::memcpy(out, s, 16);
}

void Aes128::decrypt(const uint8_t in[16], uint8_t out[16]) const {
#ifdef PKG_AESNI
    if (hasAesNi()) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                         niDecrypt(dec_, _mm_loadu_si128(reinterpret_cast<const __m128i*>(in))));
        return;
    }
#endif
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = in[i] ^ enc_[10][i];
    for (int r = 9; r >= 0; --r) {
        uint8_t t[16];
        // InvShiftRows and InvSubBytes.
        for (int i = 0; i < 16; ++i) t[(i + 4 * (i % 4)) % 16] = kInv.t[s[i]];
        for (int i = 0; i < 16; ++i) t[i] ^= enc_[r][i];
        if (r > 0)
            for (int c = 0; c < 4; ++c) {
                uint8_t* col = t + 4 * c;
                const uint8_t a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
                col[0] = mul(a0, 14) ^ mul(a1, 11) ^ mul(a2, 13) ^ mul(a3, 9);
                col[1] = mul(a0, 9) ^ mul(a1, 14) ^ mul(a2, 11) ^ mul(a3, 13);
                col[2] = mul(a0, 13) ^ mul(a1, 9) ^ mul(a2, 14) ^ mul(a3, 11);
                col[3] = mul(a0, 11) ^ mul(a1, 13) ^ mul(a2, 9) ^ mul(a3, 14);
            }
        std::memcpy(s, t, 16);
    }
    std::memcpy(out, s, 16);
}

void Aes128::xtsDecrypt(const Aes128& tweak, uint64_t sector, uint8_t* data, size_t size) const {
    uint8_t t[16] = {};
    for (int i = 0; i < 8; ++i) t[i] = uint8_t(sector >> (8 * i));
    tweak.encrypt(t, t);
    for (size_t off = 0; off + 16 <= size; off += 16) {
        uint8_t* b = data + off;
        for (int i = 0; i < 16; ++i) b[i] ^= t[i];
        decrypt(b, b);
        for (int i = 0; i < 16; ++i) b[i] ^= t[i];
        gfDouble(t);
    }
}

void aesCbcDecrypt(const uint8_t key[16], const uint8_t iv[16], uint8_t* data, size_t size) {
    const Aes128 aes(key);
    uint8_t prev[16];
    std::memcpy(prev, iv, 16);
    for (size_t off = 0; off + 16 <= size; off += 16) {
        uint8_t c[16];
        std::memcpy(c, data + off, 16);
        aes.decrypt(c, data + off);
        for (int i = 0; i < 16; ++i) data[off + i] ^= prev[i];
        std::memcpy(prev, c, 16);
    }
}

// RSA-2048: 64 32-bit limbs, least significant first, Montgomery form with R = 2^2048.

namespace {

constexpr int kLimbs = 64;
using Num = std::array<uint32_t, kLimbs>;

Num fromBytes(const uint8_t* b) {
    Num n{};
    for (int i = 0; i < 256; ++i) n[(255 - i) / 4] |= uint32_t(b[i]) << (8 * ((255 - i) % 4));
    return n;
}

bool geq(const Num& a, const Num& b) {
    for (int i = kLimbs - 1; i >= 0; --i)
        if (a[i] != b[i]) return a[i] > b[i];
    return true;
}

void sub(Num& a, const Num& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < kLimbs; ++i) {
        const uint64_t d = uint64_t(a[i]) - b[i] - borrow;
        a[i] = uint32_t(d);
        borrow = d >> 63;
    }
}

struct Mont {
    Num n, r2;
    uint32_t n0; // -n^-1 mod 2^32

    explicit Mont(const Num& mod) : n(mod) {
        uint32_t inv = 1; // Newton's iteration for n[0]^-1 mod 2^32
        for (int i = 0; i < 5; ++i) inv *= 2 - n[0] * inv;
        n0 = 0u - inv;
        // R^2 mod n = 2^4096 mod n, by doubling 1.
        Num x{};
        x[0] = 1;
        for (int i = 0; i < 4096; ++i) {
            const uint32_t top = x[kLimbs - 1] >> 31;
            for (int j = kLimbs - 1; j > 0; --j) x[j] = x[j] << 1 | x[j - 1] >> 31;
            x[0] <<= 1;
            if (top || geq(x, n)) sub(x, n);
        }
        r2 = x;
    }

    // a * b / R mod n (CIOS).
    Num mul(const Num& a, const Num& b) const {
        uint32_t t[kLimbs + 2] = {};
        for (int i = 0; i < kLimbs; ++i) {
            uint64_t c = 0;
            for (int j = 0; j < kLimbs; ++j) {
                const uint64_t s = uint64_t(a[j]) * b[i] + t[j] + c;
                t[j] = uint32_t(s), c = s >> 32;
            }
            uint64_t s = uint64_t(t[kLimbs]) + c;
            t[kLimbs] = uint32_t(s), t[kLimbs + 1] = uint32_t(s >> 32);
            const uint32_t m = t[0] * n0;
            c = (uint64_t(m) * n[0] + t[0]) >> 32;
            for (int j = 1; j < kLimbs; ++j) {
                const uint64_t s2 = uint64_t(m) * n[j] + t[j] + c;
                t[j - 1] = uint32_t(s2), c = s2 >> 32;
            }
            s = uint64_t(t[kLimbs]) + c;
            t[kLimbs - 1] = uint32_t(s);
            t[kLimbs] = t[kLimbs + 1] + uint32_t(s >> 32);
        }
        Num r;
        std::memcpy(r.data(), t, sizeof r);
        if (t[kLimbs] || geq(r, n)) sub(r, n);
        return r;
    }

    Num pow(const Num& base, const uint8_t* e, size_t eSize) const {
        Num one{};
        one[0] = 1;
        Num acc = mul(one, r2), b = mul(base, r2);
        for (size_t i = 0; i < eSize; ++i)
            for (int bit = 7; bit >= 0; --bit) {
                acc = mul(acc, acc);
                if (e[i] >> bit & 1) acc = mul(acc, b);
            }
        return mul(acc, one);
    }
};

std::array<uint8_t, 256> toBytes(const Num& n) {
    std::array<uint8_t, 256> b;
    for (int i = 0; i < 256; ++i) b[i] = uint8_t(n[(255 - i) / 4] >> (8 * ((255 - i) % 4)));
    return b;
}

} // namespace

std::array<uint8_t, 256> rsa2048Power(const uint8_t c[256], const uint8_t n[256], const uint8_t* e, size_t eSize) {
    const Mont m(fromBytes(n));
    Num base = fromBytes(c);
    if (geq(base, m.n)) sub(base, m.n);
    return toBytes(m.pow(base, e, eSize));
}

std::vector<uint8_t> rsa2048Decrypt(const uint8_t c[256], const uint8_t n[256], const uint8_t d[256]) {
    const std::array<uint8_t, 256> em = rsa2048Power(c, n, d, 256);
    // 00 02, at least 8 non-zero padding bytes, 00, the message.
    if (em[0] != 0 || em[1] != 2) return {};
    size_t i = 2;
    while (i < 256 && em[i] != 0) ++i;
    if (i < 10 || i == 256) return {};
    return {em.begin() + i + 1, em.end()};
}

} // namespace pkg::crypto
