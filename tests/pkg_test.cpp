// pkg/crypto against published test vectors, with and without AES-NI.
#include "pkg/crypto.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace pkg::crypto;

namespace {

int failures = 0;

std::string hex(const uint8_t* p, size_t n) {
    std::string s;
    char b[3];
    for (size_t i = 0; i < n; ++i) std::snprintf(b, sizeof b, "%02x", p[i]), s += b;
    return s;
}

void bytes(const char* h, uint8_t* out) {
    for (size_t i = 0; h[2 * i]; ++i) std::sscanf(h + 2 * i, "%2hhx", &out[i]);
}

void check(const char* what, const std::string& got, const char* want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what, got.c_str(), want);
    ++failures;
}

void aes() {
    // FIPS 197 appendix C.1.
    uint8_t key[16], pt[16], ct[16], out[16];
    bytes("000102030405060708090a0b0c0d0e0f", key);
    bytes("00112233445566778899aabbccddeeff", pt);
    bytes("69c4e0d86a7b0430d8cdb78070b4c55a", ct);
    const Aes128 a(key);
    a.encrypt(pt, out);
    check("aes encrypt", hex(out, 16), "69c4e0d86a7b0430d8cdb78070b4c55a");
    a.decrypt(ct, out);
    check("aes decrypt", hex(out, 16), "00112233445566778899aabbccddeeff");
    // IEEE 1619 XTS-AES-128 vector 2: keys of 0x11 and 0x22, data unit 0x3333333333, 32 bytes of 0x44.
    uint8_t k1[16], k2[16], d[32];
    std::memset(k1, 0x11, 16), std::memset(k2, 0x22, 16);
    bytes("c454185e6a16936e39334038acef838bfb186fff7480adc4289382ecd6d394f0", d);
    Aes128(k1).xtsDecrypt(Aes128(k2), 0x3333333333ull, d, sizeof d);
    check("xts decrypt", hex(d, 32), std::string(64, '4').c_str());
    // CBC: NIST SP 800-38A F.2.2, first block.
    bytes("2b7e151628aed2a6abf7158809cf4f3c", key);
    uint8_t iv[16], blk[16];
    bytes("000102030405060708090a0b0c0d0e0f", iv);
    bytes("7649abac8119b246cee98e9b12e9197d", blk);
    aesCbcDecrypt(key, iv, blk, 16);
    check("aes cbc decrypt", hex(blk, 16), "6bc1bee22e409f96e93d7e117393172a");
}

} // namespace

int main() {
    const Digest abc = sha256("abc", 3);
    check("sha256 abc", hex(abc.data(), 32), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string longer(1000, 'a');
    check("sha256 1000 a", hex(sha256(longer.data(), longer.size()).data(), 32),
          "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    // RFC 4231 test case 2.
    const Digest mac = hmacSha256("Jefe", 4, "what do ya want for nothing?", 28);
    check("hmac", hex(mac.data(), 32), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    aes();
    useAesNi(false);
    aes();
    useAesNi(true);
    if (failures) return 1;
    std::printf("pkg_test: ok\n");
    return 0;
}
