// The names of the game's shaders. The game loads its shaders from four archives in
// data/shader (sh_devil_ps4.par, mars_shader_ps4.par and the particle sets sh_ptc_ps4.par and
// sh_ptc_all_ps4.par): each .pso/.vso/.cso/.fxo entry holds one or more GCN programs, each
// beginning with `s_mov_b32 vcc_hi, <literal>` (ff 03 eb be) and followed by its `OrbShdr`
// footer. A program is named by a key over its code up to and including that footer, which the
// game copies to GPU memory unchanged. The native renderer finds the game's passes by these names.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace host::native {

// Indexes the archives under <gameDir>/data/shader on a thread of its own.
void indexShaderNames(const std::string& gameDir);

// The key of a program from its first bytes: `n` bytes at `code`, which hold its footer
// marker if the program is short enough (keyLength says how many bytes the key needs).
uint64_t shaderKey(const uint8_t* code, size_t n);
// How many bytes of `code` (n of them read) the key covers, 0 when the footer is not among them.
size_t keyLength(const uint8_t* code, size_t n);
constexpr size_t kMaxShaderBytes = 64 * 1024;

// Whether the index is built (until then every name is unknown).
bool shaderNamesReady();

// The archive entry that holds a program (e.g. "ps_aa_filter2.pso", "+N" when N more entries
// hold the same code), or null while the index is being built or when no entry does.
const std::string* shaderName(uint64_t key);

// The name of the program at a guest address, read from guest memory with `read`; "?" when
// unknown. Cached by address and by its first 64 bytes, so a call per draw is cheap.
using ReadGuest = bool (*)(uint64_t addr, void* dst, size_t size);
const std::string& shaderNameAt(uint64_t addr, ReadGuest read);

} // namespace host::native
