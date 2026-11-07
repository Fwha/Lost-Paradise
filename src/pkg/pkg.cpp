// The PKG and PFS layouts are described at psdevwiki.com (PS4 PKG files, PFS); the fake-PKG
// key steps follow what the homebrew fpkg tools do.
#include "common/compat.h"
#include "pkg/pkg.h"

#include "pkg/crypto.h"
#include "pkg/keys.h"

#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace pkg {

namespace {

uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | p[1] << 16 | p[2] << 8 | p[3]; }
uint64_t be64(const uint8_t* p) { return uint64_t(be32(p)) << 32 | be32(p + 4); }
uint16_t le16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }
uint32_t le32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }
uint64_t le64(const uint8_t* p) { return le32(p) | uint64_t(le32(p + 4)) << 32; }

struct Error {
    std::string what;
};

class File {
public:
    explicit File(const std::string& path) : fd_(open(path.c_str(), O_RDONLY | O_CLOEXEC | LPN_O_BINARY)) {
        if (fd_ < 0) throw Error{"cannot open " + path + ": " + std::strerror(errno)};
    }
    ~File() { close(fd_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    void read(uint64_t off, void* dst, size_t n) const {
        auto* p = static_cast<uint8_t*>(dst);
        while (n) {
            const ssize_t got = pread(fd_, p, n, off_t(off));
            if (got <= 0) throw Error{"the PKG is cut short (read past its end)"};
            p += got, off += uint64_t(got), n -= size_t(got);
        }
    }
    std::vector<uint8_t> read(uint64_t off, size_t n) const {
        std::vector<uint8_t> v(n);
        read(off, v.data(), n);
        return v;
    }

private:
    int fd_;
};

// Something to read a PFS image or a file in one from.
struct Source {
    virtual ~Source() = default;
    virtual void read(uint64_t off, void* dst, size_t n) = 0;
};

// The PKG's PFS image, decrypted with XTS (0x1000-byte sectors) from the first block on.
class ImageSource : public Source {
public:
    ImageSource(const File& f, uint64_t base) : f_(f), base_(base) {}
    void setKeys(const uint8_t data[16], const uint8_t tweak[16], uint64_t startSector) {
        data_ = std::make_unique<crypto::Aes128>(data);
        tweak_ = std::make_unique<crypto::Aes128>(tweak);
        start_ = startSector;
    }
    void read(uint64_t off, void* dst, size_t n) override {
        if (!data_) return f_.read(base_ + off, dst, n);
        auto* out = static_cast<uint8_t*>(dst);
        constexpr uint64_t kSector = 0x1000;
        uint8_t sec[kSector];
        while (n) {
            const uint64_t s = off / kSector, in = off % kSector;
            const size_t take = size_t(std::min<uint64_t>(n, kSector - in));
            f_.read(base_ + s * kSector, sec, kSector);
            if (s >= start_) data_->xtsDecrypt(*tweak_, s, sec, kSector);
            std::memcpy(out, sec + in, take);
            out += take, off += take, n -= take;
        }
    }

private:
    const File& f_;
    uint64_t base_;
    std::unique_ptr<crypto::Aes128> data_, tweak_;
    uint64_t start_ = 0;
};

// A file in a PFS image: blocks of blockSize, one run or a list.
class FileSource : public Source {
public:
    FileSource(Source& img, uint32_t blockSize, uint64_t start, std::vector<uint32_t> blocks)
        : img_(img), bs_(blockSize), start_(start), blocks_(std::move(blocks)) {}
    void read(uint64_t off, void* dst, size_t n) override {
        if (blocks_.empty()) return img_.read(start_ * bs_ + off, dst, n);
        auto* out = static_cast<uint8_t*>(dst);
        while (n) {
            const uint64_t b = off / bs_, in = off % bs_;
            if (b >= blocks_.size()) throw Error{"a file in the PFS image runs past its blocks"};
            const size_t take = size_t(std::min<uint64_t>(n, bs_ - in));
            img_.read(uint64_t(blocks_[b]) * bs_ + in, out, take);
            out += take, off += take, n -= take;
        }
    }

private:
    Source& img_;
    uint64_t bs_, start_;
    std::vector<uint32_t> blocks_;
};

// A PFSC (compressed) file read as the data it holds: a header, then the offset of each block;
// a block whose stored size is the block size is stored as it is, a smaller one is deflated
// (after a 2-byte zlib header).
class PfscSource : public Source {
public:
    explicit PfscSource(std::unique_ptr<Source> src) : src_(std::move(src)) {
        uint8_t h[0x30];
        src_->read(0, h, sizeof h);
        if (le32(h) != 0x43534650) throw Error{"the PFS image's compressed file has no PFSC header"};
        bs_ = le32(h + 0xC);
        const uint64_t offsets = le64(h + 0x18);
        length_ = le64(h + 0x28);
        if (bs_ == 0 || bs_ > (1u << 24) || bs_ != le64(h + 0x10)) throw Error{"the PFSC header's block size is wrong"};
        const uint64_t blocks = (length_ + bs_ - 1) / bs_;
        if (blocks > (1ull << 26)) throw Error{"the PFSC header is corrupt"};
        std::vector<uint8_t> raw((blocks + 1) * 8);
        src_->read(offsets, raw.data(), raw.size());
        map_.resize(blocks + 1);
        for (size_t i = 0; i <= blocks; ++i) map_[i] = le64(&raw[i * 8]);
        cache_.resize(bs_);
        packed_.resize(bs_);
    }
    uint64_t length() const { return length_; }
    void read(uint64_t off, void* dst, size_t n) override {
        auto* out = static_cast<uint8_t*>(dst);
        while (n) {
            const uint64_t b = off / bs_, in = off % bs_;
            load(b);
            const size_t take = size_t(std::min<uint64_t>(n, bs_ - in));
            std::memcpy(out, cache_.data() + in, take);
            out += take, off += take, n -= take;
        }
    }

private:
    void load(uint64_t b) {
        if (b == cached_) return;
        if (b + 1 >= map_.size()) throw Error{"read past the end of the PFSC data"};
        const uint64_t at = map_[b], size = map_[b + 1] - map_[b];
        cached_ = ~0ull;
        if (map_[b + 1] < map_[b]) throw Error{"the PFSC block table is corrupt"};
        if (size == bs_) {
            src_->read(at, cache_.data(), bs_);
        } else if (size > bs_) {
            std::fill(cache_.begin(), cache_.end(), 0);
        } else {
            if (size < 2) throw Error{"a PFSC block is too small"};
            src_->read(at, packed_.data(), size_t(size));
            z_stream z{};
            if (inflateInit2(&z, -15) != Z_OK) throw Error{"zlib failed to start"};
            z.next_in = packed_.data() + 2;
            z.avail_in = uInt(size - 2);
            z.next_out = cache_.data();
            z.avail_out = uInt(bs_);
            const int r = inflate(&z, Z_FINISH);
            const size_t got = bs_ - z.avail_out;
            inflateEnd(&z);
            if (r != Z_STREAM_END && r != Z_BUF_ERROR && r != Z_OK) throw Error{"a PFSC block does not decompress"};
            std::fill(cache_.begin() + long(got), cache_.end(), 0);
        }
        cached_ = b;
    }

    std::unique_ptr<Source> src_;
    uint64_t bs_ = 0, length_ = 0, cached_ = ~0ull;
    std::vector<uint64_t> map_;
    std::vector<uint8_t> cache_, packed_;
};

// A PFS image: header, inode blocks, then the folders' and files' blocks. Inode 0 is the super
// root, whose uroot folder holds the files.
class Pfs {
public:
    struct Node {
        std::string name;
        uint32_t ino = 0;
        bool dir = false;
        std::vector<Node> children;
    };
    struct Inode {
        uint16_t mode = 0;
        uint32_t flags = 0;
        uint64_t size = 0, sizeCompressed = 0;
        uint32_t blocks = 0;
        int32_t db[12] = {}, ib[5] = {};
    };

    explicit Pfs(Source& src) : src_(src) {
        uint8_t h[0x400];
        src_.read(0, h, sizeof h);
        readHeader(h);
    }

    // For the outer image, which has to be read before its keys are known.
    static void header(const uint8_t* h, uint16_t& mode, uint32_t& blockSize) {
        if (le64(h) != 1 || le64(h + 8) != 20130315) throw Error{"no PFS image in the PKG (bad PFS header)"};
        mode = le16(h + 0x1C);
        blockSize = le32(h + 0x20);
    }

    uint32_t blockSize() const { return bs_; }
    bool signedImage() const { return mode_ & 1; }
    const Inode& inode(uint32_t i) const {
        if (i >= inodes_.size()) throw Error{"the PFS image names an inode it does not have"};
        return inodes_[i];
    }

    Node uroot() {
        Node root = loadDir(0, "", 0);
        for (Node& n : root.children)
            if (n.dir && n.name == "uroot") return std::move(n);
        throw Error{"the PFS image has no uroot folder"};
    }

    // A source for a file's stored bytes (PFSC for a compressed file).
    std::unique_ptr<Source> open(uint32_t ino) {
        const Inode& in = inode(ino);
        std::vector<uint32_t> list;
        if (in.blocks > 1 && in.db[1] != -1 && signedImage()) list = blockList(in);
        return std::make_unique<FileSource>(src_, bs_, uint64_t(uint32_t(in.db[0])), std::move(list));
    }
    // A file's length once decompressed.
    static uint64_t length(const Inode& in) { return in.flags & 1 ? std::max(in.size, in.sizeCompressed) : in.size; }

private:
    void readHeader(const uint8_t* h) {
        header(h, mode_, bs_);
        if (bs_ < 0x1000 || bs_ > (1u << 24)) throw Error{"the PFS block size is wrong"};
        const uint64_t count = le64(h + 0x30), blocks = le64(h + 0x40);
        if (count > 50'000'000 || blocks > 1'000'000) throw Error{"the PFS header is corrupt"};
        const size_t isz = signedImage() ? 0x2C8 : 0xA8;
        const size_t perBlock = bs_ / isz;
        std::vector<uint8_t> b(bs_);
        for (uint64_t i = 0; i < blocks && inodes_.size() < count; ++i) {
            src_.read(uint64_t(bs_) * (1 + i), b.data(), bs_);
            for (size_t j = 0; j < perBlock && inodes_.size() < count; ++j) inodes_.push_back(parse(&b[j * isz]));
        }
    }

    Inode parse(const uint8_t* p) const {
        Inode in;
        in.mode = le16(p);
        in.flags = le32(p + 4);
        in.size = le64(p + 8);
        in.sizeCompressed = le64(p + 0x10);
        in.blocks = le32(p + 0x60);
        for (int i = 0; i < 17; ++i) {
            const size_t at = signedImage() ? 0x64 + 36 * i + 32 : 0x64 + 4 * i;
            (i < 12 ? in.db[i] : in.ib[i - 12]) = int32_t(le32(p + at));
        }
        return in;
    }

    uint32_t readBlockPtr(uint64_t block, uint64_t index) {
        uint8_t v[4];
        src_.read(block * bs_ + index * 36 + 32, v, 4);
        return le32(v);
    }

    // A signed image's file blocks: 12 direct, then single and double indirect blocks of
    // (signature, block) pairs.
    std::vector<uint32_t> blockList(const Inode& in) {
        const uint64_t per = bs_ / 36;
        std::vector<uint32_t> list;
        list.reserve(in.blocks);
        for (int i = 0; i < 12 && list.size() < in.blocks; ++i) list.push_back(uint32_t(in.db[i]));
        for (uint64_t i = 0; i < per && list.size() < in.blocks; ++i) list.push_back(readBlockPtr(uint32_t(in.ib[0]), i));
        for (uint64_t j = 0; list.size() < in.blocks; ++j) {
            if (j >= per) throw Error{"a file in the PFS image is too big for its block table"};
            const uint32_t ind = readBlockPtr(uint32_t(in.ib[1]), j);
            for (uint64_t i = 0; i < per && list.size() < in.blocks; ++i) list.push_back(readBlockPtr(ind, i));
        }
        return list;
    }

    Node loadDir(uint32_t ino, const std::string& name, int depth) {
        if (depth > 64) throw Error{"the PFS image's folders nest too deep"};
        Node d{name, ino, true, {}};
        const Inode& in = inode(ino);
        if (in.blocks < 1 || in.blocks > 1'000'000 || in.db[0] < 1) throw Error{"a folder's inode in the PFS image is corrupt"};
        std::vector<uint8_t> b(bs_);
        for (uint32_t k = 0; k < in.blocks; ++k) {
            src_.read((uint64_t(uint32_t(in.db[0])) + k) * bs_, b.data(), bs_);
            for (size_t at = 0; at + 16 <= bs_;) {
                const uint32_t child = le32(&b[at]), type = le32(&b[at + 4]), len = le32(&b[at + 8]), ent = le32(&b[at + 12]);
                if (ent == 0) break;
                if (ent < 16 || at + ent > bs_ || len > ent - 16) throw Error{"a folder in the PFS image is corrupt"};
                std::string n(reinterpret_cast<const char*>(&b[at + 16]), len);
                n.resize(std::strlen(n.c_str()));
                if (type == 2 || type == 3) d.children.push_back({n, child, type == 3, {}});
                at += ent;
            }
        }
        for (Node& c : d.children)
            if (c.dir) c = loadDir(c.ino, c.name, depth + 1);
        return d;
    }

    Source& src_;
    uint16_t mode_ = 0;
    uint32_t bs_ = 0;
    std::vector<Inode> inodes_;
};

// A name from the PKG or its image as a relative path: no empty, "." or ".." parts.
bool safeName(const std::string& s) {
    if (s.empty() || s.front() == '/') return false;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t slash = std::min(s.find('/', at), s.size());
        const std::string part = s.substr(at, slash - at);
        if (part.empty() || part == "." || part == "..") return false;
        at = slash + 1;
    }
    return true;
}

// Names of the entries with fixed IDs (psdevwiki: PKG files, entries).
std::string entryName(uint32_t id) {
    static const std::map<uint32_t, const char*> names = {
        {0x0400, "license.dat"}, {0x0401, "license.info"}, {0x0402, "nptitle.dat"}, {0x0403, "npbind.dat"},
        {0x0404, "selfinfo.dat"}, {0x0406, "imageinfo.dat"}, {0x0407, "target-deltainfo.dat"},
        {0x0408, "origin-deltainfo.dat"}, {0x0409, "psreserved.dat"}, {0x1000, "param.sfo"},
        {0x1001, "playgo-chunk.dat"}, {0x1002, "playgo-chunk.sha"}, {0x1003, "playgo-manifest.xml"},
        {0x1004, "pronunciation.xml"}, {0x1005, "pronunciation.sig"}, {0x1006, "pic1.png"},
        {0x1007, "pubtoolinfo.dat"}, {0x1008, "app/playgo-chunk.dat"}, {0x1009, "app/playgo-chunk.sha"},
        {0x100A, "app/playgo-manifest.xml"}, {0x100B, "shareparam.json"}, {0x100C, "shareoverlayimage.png"},
        {0x100D, "save_data.png"}, {0x100E, "shareprivacyguardimage.png"}, {0x1200, "icon0.png"},
        {0x1220, "pic0.png"}, {0x1240, "snd0.at9"}, {0x1260, "changeinfo/changeinfo.xml"},
        {0x1280, "icon0.dds"}, {0x12A0, "pic0.dds"}, {0x12C0, "pic1.dds"},
    };
    if (auto it = names.find(id); it != names.end()) return it->second;
    char buf[48];
    auto num = [&](const char* fmt, unsigned n) { return std::snprintf(buf, sizeof buf, fmt, n), std::string(buf); };
    if (id >= 0x1201 && id <= 0x121F) return num("icon0_%02u.png", id - 0x1201);
    if (id >= 0x1241 && id <= 0x125F) return num("pic1_%02u.png", id - 0x1241);
    if (id >= 0x1261 && id <= 0x127F) return num("changeinfo/changeinfo_%02u.xml", id - 0x1261);
    if (id >= 0x1281 && id <= 0x129F) return num("icon0_%02u.dds", id - 0x1281);
    if (id >= 0x12C1 && id <= 0x12DF) return num("pic1_%02u.dds", id - 0x12C1);
    if (id >= 0x1400 && id <= 0x1463) return num("trophy/trophy%02u.trp", id - 0x1400);
    if (id >= 0x1600 && id <= 0x1609) return num("keymap_rp/%03u.png", id - 0x1600 + 1);
    if (id >= 0x1610 && id <= 0x17F9 && (id - 0x1610) % 16 < 10) {
        std::snprintf(buf, sizeof buf, "keymap_rp/%02u/%03u.png", (id - 0x1610) / 16, (id - 0x1610) % 16 + 1);
        return buf;
    }
    return {};
}

// A string from a param.sfo.
std::string sfoString(const std::vector<uint8_t>& d, const char* want) {
    if (d.size() < 20 || std::memcmp(d.data(), "\0PSF", 4) != 0) return {};
    const uint32_t keys = le32(&d[8]), data = le32(&d[12]), n = le32(&d[16]);
    for (uint32_t i = 0; i < n && 20 + 16 * size_t(i + 1) <= d.size(); ++i) {
        const uint8_t* e = &d[20 + 16 * i];
        const size_t key = size_t(keys) + le16(e), len = le32(e + 4), at = size_t(data) + le32(e + 12);
        if (key >= d.size() || at + len > d.size()) continue;
        if (std::strncmp(reinterpret_cast<const char*>(&d[key]), want, d.size() - key) != 0) continue;
        std::string s(reinterpret_cast<const char*>(&d[at]), len);
        s.resize(std::strlen(s.c_str()));
        return s;
    }
    return {};
}

struct Entry {
    uint32_t id, nameOffset, flags1, flags2, offset, size;
    uint8_t raw[32];
    bool encrypted() const { return flags1 & 0x80000000u; }
    unsigned keyIndex() const { return (flags2 & 0xF000) >> 12; }
};

struct Package {
    File file;
    Info info;
    std::vector<Entry> entries;
    std::vector<uint8_t> names; // ENTRY_NAMES
    std::vector<uint8_t> dk3;   // entry key 3, when the fake keys open it
    uint64_t pfsFlags = 0, pfsOffset = 0, pfsSize = 0;

    explicit Package(const std::string& path) : file(path) {
        uint8_t h[0x1000];
        file.read(0, h, sizeof h);
        if (be32(h) != 0x7F434E54) throw Error{"not a PS4 PKG (no CNT header)"};
        const uint32_t count = be32(h + 0x10), table = be32(h + 0x18);
        info.contentId.assign(reinterpret_cast<const char*>(h + 0x40), 0x24);
        info.contentId.resize(std::strlen(info.contentId.c_str()));
        if (info.contentId.size() >= 16) info.titleId = info.contentId.substr(7, 9);
        pfsFlags = be64(h + 0x408);
        pfsOffset = be64(h + 0x410);
        pfsSize = be64(h + 0x418);
        info.imageSize = pfsSize;
        if (count > 10000) throw Error{"the PKG's entry table is corrupt"};
        const std::vector<uint8_t> t = file.read(table, size_t(count) * 32);
        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t* p = &t[i * 32];
            Entry e{be32(p), be32(p + 4), be32(p + 8), be32(p + 12), be32(p + 16), be32(p + 20), {}};
            std::memcpy(e.raw, p, 32);
            entries.push_back(e);
        }
        for (const Entry& e : entries) {
            if (e.id == 0x200) names = file.read(e.offset, e.size);
            if (e.id == 0x1000 && e.size < (1u << 20)) {
                const std::vector<uint8_t> sfo = file.read(e.offset, e.size);
                info.category = sfoString(sfo, "CATEGORY");
                info.title = sfoString(sfo, "TITLE");
                if (const std::string id = sfoString(sfo, "TITLE_ID"); id.size() == 9) info.titleId = id;
            }
            // ENTRY_KEYS: a seed digest, 7 digests, then 7 RSA-encrypted keys.
            if (e.id == 0x10 && e.size >= 32 * 8 + 256 * 4) {
                const std::vector<uint8_t> k = file.read(uint64_t(e.offset) + 32 * 8 + 256 * 3, 256);
                dk3 = crypto::rsa2048Decrypt(k.data(), keys::kPkgDerivedKey3KeysetModulus,
                                             keys::kPkgDerivedKey3KeysetPrivateExponent);
            }
        }
        info.fake = dk3.size() == 32;
    }

    const Entry* find(uint32_t id) const {
        for (const Entry& e : entries)
            if (e.id == id) return &e;
        return nullptr;
    }

    // An entry's AES key and IV: SHA-256 of its table row and the derived key.
    void decryptEntry(const Entry& e, std::vector<uint8_t>& data) const {
        std::vector<uint8_t> in(e.raw, e.raw + 32);
        in.insert(in.end(), dk3.begin(), dk3.end());
        const crypto::Digest ivKey = crypto::sha256(in.data(), in.size());
        crypto::aesCbcDecrypt(ivKey.data() + 16, ivKey.data(), data.data(), data.size() & ~size_t(15));
    }

    // EKPFS, the PFS image's key: the IMAGE_KEY entry, AES- then RSA-decrypted.
    std::vector<uint8_t> ekpfs() const {
        const Entry* e = find(0x20);
        if (!info.fake || !e || e->size < 256) return {};
        std::vector<uint8_t> k = file.read(e->offset, 256);
        decryptEntry(*e, k);
        return crypto::rsa2048Decrypt(k.data(), keys::kFakeKeysetModulus, keys::kFakeKeysetPrivateExponent);
    }

    std::string entryPath(const Entry& e) const {
        if (e.nameOffset && e.nameOffset < names.size()) {
            std::string n(reinterpret_cast<const char*>(&names[e.nameOffset]), names.size() - e.nameOffset);
            n.resize(std::strlen(n.c_str()));
            return n;
        }
        return entryName(e.id);
    }
};

class Writer {
public:
    explicit Writer(const fs::path& p) : f_(std::fopen(p.string().c_str(), "wb")) {
        if (!f_) throw Error{"cannot write " + p.string() + ": " + std::strerror(errno)};
    }
    ~Writer() {
        if (f_) std::fclose(f_);
    }
    void write(const void* d, size_t n) {
        if (std::fwrite(d, 1, n, f_) != n) throw Error{std::string("cannot write a file: ") + std::strerror(errno)};
    }
    void close() {
        const int r = std::fclose(f_);
        f_ = nullptr;
        if (r != 0) throw Error{std::string("cannot write a file: ") + std::strerror(errno)};
    }

private:
    std::FILE* f_;
};

struct Cancelled {};

uint64_t totalSize(Pfs& pfs, const Pfs::Node& d) {
    uint64_t n = 0;
    for (const Pfs::Node& c : d.children) n += c.dir ? totalSize(pfs, c) : Pfs::length(pfs.inode(c.ino));
    return n;
}

void extractDir(Pfs& pfs, const Pfs::Node& d, const fs::path& out, const std::string& rel, Progress* pr,
                std::vector<uint8_t>& buf) {
    fs::create_directories(out);
    for (const Pfs::Node& c : d.children) {
        if (!safeName(c.name) || c.name.find('/') != std::string::npos) continue;
        const std::string relPath = rel.empty() ? c.name : rel + "/" + c.name;
        if (c.dir) {
            extractDir(pfs, c, out / c.name, relPath, pr, buf);
            continue;
        }
        if (pr) {
            if (pr->cancel) throw Cancelled{};
            std::lock_guard lock(pr->mutex);
            pr->file = relPath;
        }
        const Pfs::Inode& in = pfs.inode(c.ino);
        std::unique_ptr<Source> src = pfs.open(c.ino);
        uint64_t left = in.size;
        if (in.flags & 1) {
            auto z = std::make_unique<PfscSource>(std::move(src));
            left = std::min(Pfs::length(in), z->length());
            src = std::move(z);
        }
        Writer w(out / c.name);
        for (uint64_t at = 0; left;) {
            if (pr && pr->cancel) throw Cancelled{};
            const size_t n = size_t(std::min<uint64_t>(left, buf.size()));
            src->read(at, buf.data(), n);
            w.write(buf.data(), n);
            at += n, left -= n;
            if (pr) pr->done += n;
        }
        w.close();
    }
}

} // namespace

bool readInfo(const std::string& path, Info& info, std::string& err) {
    try {
        info = Package(path).info;
        return true;
    } catch (const Error& e) {
        err = e.what;
        return false;
    }
}

std::string folderName(const Info& info) {
    std::string id = info.titleId.empty() ? info.contentId : info.titleId;
    if (info.category.rfind("gp", 0) == 0) id += "-UPDATE";
    else if (info.category == "ac") id += "-DLC";
    return id;
}

bool extract(const std::string& path, const std::string& outDir, Progress* pr, std::string& err) {
    const fs::path out(outDir), part(outDir + ".part");
    std::error_code ec;
    if (fs::exists(out, ec)) {
        err = out.filename().string() + " is already in the game folder: move it away first";
        return false;
    }
    try {
        Package p(path);
        // The PFS image: its header is not encrypted, the rest is when its mode says so. A PKG
        // without one (DLC that is only a licence) has just its sce_sys entries.
        std::unique_ptr<ImageSource> imgP;
        std::unique_ptr<Pfs> outerP, inner;
        std::unique_ptr<Source> innerSrc;
        Pfs* game = nullptr;
        Pfs::Node uroot;
        if (p.pfsSize) {
            imgP = std::make_unique<ImageSource>(p.file, p.pfsOffset);
            ImageSource& img = *imgP;
            uint8_t h[0x400];
            img.read(0, h, sizeof h);
            uint16_t mode;
            uint32_t bs;
            Pfs::header(h, mode, bs);
            if (mode & 4) {
                if (!p.info.fake)
                    throw Error{"this is a retail PKG: its files are encrypted with keys only a PS4 has. "
                                "Use a fake PKG (fpkg) or a dump of the game folder"};
                std::vector<uint8_t> ek = p.ekpfs();
                if (ek.size() != 32) throw Error{"the PKG's image key does not open with the fake keys"};
                const uint8_t* seed = h + 0x370;
                // Newer images (pfs flags bit 61) key with HMAC(EKPFS, seed) in place of EKPFS.
                if (p.pfsFlags & 0x2000000000000000ull) {
                    const crypto::Digest k = crypto::hmacSha256(ek.data(), ek.size(), seed, 16);
                    ek.assign(k.begin(), k.end());
                }
                uint8_t msg[20] = {1, 0, 0, 0};
                std::memcpy(msg + 4, seed, 16);
                const crypto::Digest key = crypto::hmacSha256(ek.data(), ek.size(), msg, sizeof msg);
                img.setKeys(key.data() + 16, key.data(), bs / 0x1000);
            }
            outerP = std::make_unique<Pfs>(img);
            Pfs& outer = *outerP;
            uroot = outer.uroot();
            // The game's files: the compressed inner image pfs_image.dat when there is one, else
            // the outer image's own files.
            for (const Pfs::Node& c : uroot.children)
                if (!c.dir && c.name == "pfs_image.dat") {
                    innerSrc = outer.open(c.ino);
                    if (outer.inode(c.ino).flags & 1) innerSrc = std::make_unique<PfscSource>(std::move(innerSrc));
                }
            game = &outer;
            if (innerSrc) {
                inner = std::make_unique<Pfs>(*innerSrc);
                uroot = inner->uroot();
                game = inner.get();
            }
        }
        if (pr) pr->total = game ? totalSize(*game, uroot) : 0, pr->done = 0;

        fs::remove_all(part, ec); // a folder left by an earlier extraction that stopped
        std::vector<uint8_t> buf(1 << 20);
        try {
            if (game) extractDir(*game, uroot, part, "", pr, buf);
            fs::create_directories(part);
            // The entries' sce_sys files (param.sfo, icon0.png, ...), but not the keys and digests.
            for (const Entry& e : p.entries) {
                if (e.id < 0x400) continue;
                const std::string name = p.entryPath(e);
                if (name.empty() || !safeName(name)) continue;
                std::vector<uint8_t> d = p.file.read(e.offset, e.size);
                if (e.encrypted()) {
                    if (e.keyIndex() != 3 || !p.info.fake) continue;
                    p.decryptEntry(e, d);
                }
                const fs::path to = part / "sce_sys" / name;
                fs::create_directories(to.parent_path());
                Writer w(to);
                w.write(d.data(), d.size());
                w.close();
            }
            fs::rename(part, out);
        } catch (...) {
            fs::remove_all(part, ec);
            throw;
        }
        return true;
    } catch (const Error& e) {
        err = e.what;
    } catch (const Cancelled&) {
        err = "cancelled";
    } catch (const fs::filesystem_error& e) {
        err = e.what();
    } catch (const std::bad_alloc&) {
        err = "the PKG is corrupt (out of memory reading it)";
    }
    return false;
}

} // namespace pkg
