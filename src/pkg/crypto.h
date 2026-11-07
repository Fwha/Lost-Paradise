// The cryptography a fake PKG needs, written here so the extractor has no library to bring:
// SHA-256 and HMAC-SHA256, AES-128 (CBC for the IMAGE_KEY, XTS for the PFS image) and RSA-2048
// decryption with PKCS#1 v1.5 padding.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pkg::crypto {

using Digest = std::array<uint8_t, 32>;

Digest sha256(const void* data, size_t size);
Digest hmacSha256(const void* key, size_t keySize, const void* data, size_t size);

// AES-128 with its round keys; uses AES-NI when the CPU has it.
class Aes128 {
public:
    explicit Aes128(const uint8_t key[16]);
    void encrypt(const uint8_t in[16], uint8_t out[16]) const;
    void decrypt(const uint8_t in[16], uint8_t out[16]) const;
    // XTS decryption of one sector in place (this AES is the data key, tweak the tweak key),
    // the tweak being the sector number, little-endian.
    void xtsDecrypt(const Aes128& tweak, uint64_t sector, uint8_t* data, size_t size) const;

private:
    alignas(16) uint8_t enc_[11][16];
    alignas(16) uint8_t dec_[11][16]; // for AES-NI: the equivalent inverse cipher's keys
};

// For tests: turns AES-NI off (or back on, where the CPU has it); returns whether it was on.
bool useAesNi(bool on);

void aesCbcDecrypt(const uint8_t key[16], const uint8_t iv[16], uint8_t* data, size_t size);

// c^d mod n for a 2048-bit big-endian modulus and exponent, then the PKCS#1 v1.5 (type 2) padding
// taken off. Empty when the padding is wrong (the wrong key, or not a fake PKG).
std::vector<uint8_t> rsa2048Decrypt(const uint8_t c[256], const uint8_t n[256], const uint8_t d[256]);

// c^e mod n (rsa2048Decrypt without the padding step).
std::array<uint8_t, 256> rsa2048Power(const uint8_t c[256], const uint8_t n[256], const uint8_t* e, size_t eSize);

} // namespace pkg::crypto
