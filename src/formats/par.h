// PARC (.par) archives: the container for nearly all Lost Paradise game data.
//
// All header fields are big-endian.
//   0x00 "PARC"
//   0x04 u8 platform (2), u8 endianness (1 = big), u16 pad
//   0x08 u32 version (0x00020001)
//   0x0C u32 data size (0)
//   0x10 u32 folder count, u32 folder table offset
//   0x18 u32 file count,   u32 file table offset
//   0x20 names, 64 bytes each: all folders, then all files. Folder 0 is the root ".".
//
// Folder entry (32 bytes): child folder count, first child folder, file count, first file,
//   attributes, 3 x u32 reserved.
// File entry (32 bytes): flags (bit 31 = SLLZ-compressed), size, stored size, offset (low 32),
//   attributes, offset (high 32), u64 timestamp.
//
// Archives nest: a file inside a .par is often itself an uncompressed .par.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace formats {

struct ParFolder {
    std::string name;
    uint32_t folderCount = 0;
    uint32_t firstFolder = 0;
    uint32_t fileCount = 0;
    uint32_t firstFile = 0;
    uint32_t attributes = 0;
};

struct ParFile {
    std::string name;
    std::string path;        // relative to the archive root, '/'-separated
    bool compressed = false;
    uint32_t size = 0;       // decompressed size
    uint32_t storedSize = 0; // bytes in the archive
    uint64_t offset = 0;
    uint32_t attributes = 0;
    uint64_t timestamp = 0;
};

class ParArchive {
public:
    // The archive keeps a view of `data`, which must outlive it.
    static std::optional<ParArchive> parse(std::span<const uint8_t> data, std::string* error);
    static bool isPar(std::span<const uint8_t> data);

    const std::vector<ParFolder>& folders() const { return folders_; }
    const std::vector<ParFile>& files() const { return files_; }

    // The stored bytes of a file (still compressed if file.compressed).
    std::span<const uint8_t> raw(const ParFile& file) const;
    // The file's contents, decompressed if needed.
    bool read(const ParFile& file, std::vector<uint8_t>& out, std::string* error) const;

private:
    std::span<const uint8_t> data_;
    std::vector<ParFolder> folders_;
    std::vector<ParFile> files_;
};

} // namespace formats
