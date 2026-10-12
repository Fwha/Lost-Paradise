#include "formats/par.h"

#include "formats/sllz.h"

#include <cstring>
#include <functional>

namespace formats {

namespace {

constexpr size_t kNameSize = 64;
constexpr size_t kEntrySize = 32;
constexpr uint32_t kCompressedFlag = 0x80000000;

uint32_t be32(std::span<const uint8_t> d, size_t off) {
    return uint32_t(d[off]) << 24 | uint32_t(d[off + 1]) << 16 | uint32_t(d[off + 2]) << 8 |
           d[off + 3];
}

} // namespace

bool ParArchive::isPar(std::span<const uint8_t> data) {
    return data.size() >= 0x20 && std::memcmp(data.data(), "PARC", 4) == 0;
}

std::optional<ParArchive> ParArchive::parse(std::span<const uint8_t> data, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<ParArchive> {
        if (error) *error = msg;
        return std::nullopt;
    };
    if (!isPar(data)) return fail("not a PARC archive");
    if (data[5] != 1) return fail("little-endian PARC archives are not supported yet");

    const uint32_t folderCount = be32(data, 0x10);
    const uint32_t folderTable = be32(data, 0x14);
    const uint32_t fileCount = be32(data, 0x18);
    const uint32_t fileTable = be32(data, 0x1C);
    const uint64_t namesEnd = 0x20 + (uint64_t(folderCount) + fileCount) * kNameSize;
    if (namesEnd > data.size() || folderTable + uint64_t(folderCount) * kEntrySize > data.size() ||
        fileTable + uint64_t(fileCount) * kEntrySize > data.size())
        return fail("PARC tables run past the end of the archive");

    auto name = [&](uint64_t index) {
        const char* p = reinterpret_cast<const char*>(data.data() + 0x20 + index * kNameSize);
        return std::string(p, strnlen(p, kNameSize));
    };

    ParArchive ar;
    ar.data_ = data;
    for (uint32_t i = 0; i < folderCount; ++i) {
        const size_t e = folderTable + size_t(i) * kEntrySize;
        ar.folders_.push_back({name(i), be32(data, e), be32(data, e + 4), be32(data, e + 8),
                               be32(data, e + 12), be32(data, e + 16)});
    }
    for (uint32_t i = 0; i < fileCount; ++i) {
        const size_t e = fileTable + size_t(i) * kEntrySize;
        ParFile f;
        f.name = name(uint64_t(folderCount) + i);
        f.path = f.name;
        f.compressed = be32(data, e) & kCompressedFlag;
        f.size = be32(data, e + 4);
        f.storedSize = be32(data, e + 8);
        f.offset = be32(data, e + 12) | uint64_t(be32(data, e + 20)) << 32;
        f.attributes = be32(data, e + 16);
        f.timestamp = uint64_t(be32(data, e + 24)) << 32 | be32(data, e + 28);
        if (f.offset + f.storedSize > data.size())
            return fail("file '" + f.name + "' runs past the end of the archive");
        ar.files_.push_back(std::move(f));
    }

    // Build paths by walking the folder tree from the root.
    std::vector<bool> visited(folderCount, false);
    std::function<bool(uint32_t, const std::string&)> walk = [&](uint32_t idx,
                                                                 const std::string& prefix) {
        if (idx >= folderCount || visited[idx]) return false;
        visited[idx] = true;
        const ParFolder& d = ar.folders_[idx];
        if (uint64_t(d.firstFile) + d.fileCount > fileCount ||
            uint64_t(d.firstFolder) + d.folderCount > folderCount)
            return false;
        for (uint32_t i = 0; i < d.fileCount; ++i)
            ar.files_[d.firstFile + i].path = prefix + ar.files_[d.firstFile + i].name;
        for (uint32_t i = 0; i < d.folderCount; ++i) {
            const uint32_t child = d.firstFolder + i;
            if (!walk(child, prefix + ar.folders_[child].name + "/")) return false;
        }
        return true;
    };
    if (folderCount > 0 && !walk(0, "")) return fail("PARC folder tree is malformed");
    return ar;
}

std::span<const uint8_t> ParArchive::raw(const ParFile& file) const {
    return data_.subspan(file.offset, file.storedSize);
}

bool ParArchive::read(const ParFile& file, std::vector<uint8_t>& out, std::string* error) const {
    const auto bytes = raw(file);
    if (!file.compressed) {
        out.assign(bytes.begin(), bytes.end());
        return true;
    }
    if (!sllzDecompress(bytes, out, error)) return false;
    if (out.size() != file.size) {
        if (error) *error = "decompressed size mismatch for '" + file.name + "'";
        return false;
    }
    return true;
}

} // namespace formats
