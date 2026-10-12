#include "formats/gmt.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace formats {

namespace {

enum HeaderTable {
    kAnimations, kGraphs, kGraphData, kStrings, kBoneGroups, kCurveGroups, kCurves, kKeyData,
    kHeaderTables
};
constexpr size_t kHeaderStart = 0x30;
constexpr size_t kAnimationSize = 0x40;
constexpr size_t kNameSize = 0x20;
constexpr size_t kCurveSize = 0x10;

// Curve formats seen in Lost Paradise. Keys are big-endian.
enum CurveFormat : uint32_t {
    kRotQuatShort = 2,  // int16 x4 (xyzw) / 16384
    kLocAxisFloat = 4,  // float, one axis chosen by the channel's high half (1, 2, 4)
    kLocVec3Float = 6,  // float x3
    kRotXShort = 19,    // int16 x2 (x, w) / 16384
    kRotYShort = 20,    // int16 x2 (y, w) / 16384
    kRotZShort = 21,    // int16 x2 (z, w) / 16384
};

class Reader {
public:
    explicit Reader(std::span<const uint8_t> d) : d_(d) {}
    bool has(uint64_t off, uint64_t size) const { return off + size <= d_.size(); }
    uint32_t u32(size_t o) const {
        return uint32_t(d_[o]) << 24 | uint32_t(d_[o + 1]) << 16 | uint32_t(d_[o + 2]) << 8 | d_[o + 3];
    }
    uint16_t u16(size_t o) const { return uint16_t(d_[o] << 8 | d_[o + 1]); }
    int16_t i16(size_t o) const { return int16_t(u16(o)); }
    float f32(size_t o) const {
        const uint32_t v = u32(o);
        float f;
        std::memcpy(&f, &v, 4);
        return f;
    }
    std::string name(size_t o) const {
        const char* p = reinterpret_cast<const char*>(d_.data() + o + 2);
        return std::string(p, strnlen(p, kNameSize - 2));
    }

private:
    std::span<const uint8_t> d_;
};

template <typename T>
size_t keyIndex(const GmtTrack<T>& t, float frame, float* blend) {
    *blend = 0;
    if (frame <= t.frames.front()) return 0;
    if (frame >= t.frames.back()) return t.frames.size() - 1;
    const auto it = std::upper_bound(t.frames.begin(), t.frames.end(), frame);
    const size_t i = size_t(it - t.frames.begin()) - 1;
    *blend = (frame - t.frames[i]) / float(t.frames[i + 1] - t.frames[i]);
    return i;
}

} // namespace

const GmtBone* GmtAnimation::find(const std::string& boneName) const {
    for (const auto& b : bones)
        if (b.name == boneName) return &b;
    return nullptr;
}

bool GmtFile::isGmt(std::span<const uint8_t> data) {
    return data.size() >= kHeaderStart + kHeaderTables * 8 && std::memcmp(data.data(), "GSGT", 4) == 0;
}

std::optional<GmtFile> GmtFile::parse(std::span<const uint8_t> data, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::optional<GmtFile> {
        if (error) *error = msg;
        return std::nullopt;
    };
    if (!isGmt(data)) return fail("not a GSGT animation");
    if (data[4] != 2 || data[5] != 1) return fail("only big-endian GSGT files are supported");
    Reader r(data);

    // Header tables are (count, offset) pairs.
    uint32_t count[kHeaderTables], offset[kHeaderTables];
    for (int i = 0; i < kHeaderTables; ++i) {
        count[i] = r.u32(kHeaderStart + i * 8);
        offset[i] = r.u32(kHeaderStart + i * 8 + 4);
    }
    if (!r.has(offset[kAnimations], uint64_t(count[kAnimations]) * kAnimationSize) ||
        !r.has(offset[kGraphs], uint64_t(count[kGraphs]) * 4) ||
        !r.has(offset[kStrings], uint64_t(count[kStrings]) * kNameSize) ||
        !r.has(offset[kBoneGroups], uint64_t(count[kBoneGroups]) * 4) ||
        !r.has(offset[kCurveGroups], uint64_t(count[kCurveGroups]) * 4) ||
        !r.has(offset[kCurves], uint64_t(count[kCurves]) * kCurveSize))
        return fail("GSGT tables run past the end of the file");

    // Graphs: u16 key count, then that many u16 frame numbers.
    auto graph = [&](uint32_t index, std::vector<uint16_t>& frames) {
        if (index >= count[kGraphs]) return false;
        const uint32_t o = r.u32(offset[kGraphs] + index * 4);
        if (!r.has(o, 2)) return false;
        const uint16_t n = r.u16(o);
        if (n == 0 || !r.has(o + 2, uint64_t(n) * 2)) return false;
        frames.resize(n);
        for (uint16_t i = 0; i < n; ++i) frames[i] = r.u16(o + 2 + i * 2);
        return std::is_sorted(frames.begin(), frames.end());
    };

    GmtFile file;
    for (uint32_t a = 0; a < count[kAnimations]; ++a) {
        // start, end, index, frame rate, name, bone group, first curve group, curve group count, ...
        const size_t e = offset[kAnimations] + a * kAnimationSize;
        GmtAnimation anim;
        anim.startFrame = r.u32(e);
        anim.endFrame = r.u32(e + 4);
        anim.frameRate = r.f32(e + 12);
        const uint32_t nameIndex = r.u32(e + 16);
        const uint32_t boneGroup = r.u32(e + 20);
        const uint32_t firstCurveGroup = r.u32(e + 24);
        if (nameIndex < count[kStrings]) anim.name = r.name(offset[kStrings] + nameIndex * kNameSize);
        if (boneGroup >= count[kBoneGroups]) return fail("bad bone group");

        // Bone group: first bone name (string index) and bone count. One curve group per bone.
        const uint16_t firstName = r.u16(offset[kBoneGroups] + boneGroup * 4);
        const uint16_t boneCount = r.u16(offset[kBoneGroups] + boneGroup * 4 + 2);
        if (uint64_t(firstName) + boneCount > count[kStrings] ||
            uint64_t(firstCurveGroup) + boneCount > count[kCurveGroups])
            return fail("bone group out of range");

        for (uint32_t b = 0; b < boneCount; ++b) {
            GmtBone bone;
            bone.name = r.name(offset[kStrings] + (firstName + b) * kNameSize);
            const size_t cg = offset[kCurveGroups] + (firstCurveGroup + b) * 4;
            const uint16_t firstCurve = r.u16(cg);
            const uint16_t curveCount = r.u16(cg + 2);
            if (uint64_t(firstCurve) + curveCount > count[kCurves]) return fail("curve group out of range");

            for (uint32_t c = firstCurve; c < uint32_t(firstCurve) + curveCount; ++c) {
                const size_t ce = offset[kCurves] + c * kCurveSize;
                const uint32_t graphIndex = r.u32(ce);
                const uint32_t keys = r.u32(ce + 4);
                const uint32_t format = r.u32(ce + 8);
                const uint32_t channel = r.u32(ce + 12);
                std::vector<uint16_t> frames;
                if (!graph(graphIndex, frames)) return fail("bad keyframe graph in " + bone.name);
                const size_t n = frames.size();

                auto need = [&](size_t keySize) { return r.has(keys, n * keySize); };
                switch (format) {
                case kRotQuatShort:
                    if (!need(8)) return fail("rotation keys run past the file");
                    bone.rotation.frames = frames;
                    for (size_t k = 0; k < n; ++k) {
                        Quat q;
                        for (int i = 0; i < 4; ++i) q[i] = r.i16(keys + k * 8 + i * 2) / 16384.0f;
                        bone.rotation.values.push_back(q);
                    }
                    break;
                case kRotXShort:
                case kRotYShort:
                case kRotZShort:
                    if (!need(4)) return fail("rotation keys run past the file");
                    bone.rotation.frames = frames;
                    for (size_t k = 0; k < n; ++k) {
                        Quat q{0, 0, 0, r.i16(keys + k * 4 + 2) / 16384.0f};
                        q[format - kRotXShort] = r.i16(keys + k * 4) / 16384.0f;
                        bone.rotation.values.push_back(q);
                    }
                    break;
                case kLocVec3Float:
                    if (!need(12)) return fail("location keys run past the file");
                    bone.location.frames = frames;
                    for (size_t k = 0; k < n; ++k)
                        bone.location.values.push_back(
                            {r.f32(keys + k * 12), r.f32(keys + k * 12 + 4), r.f32(keys + k * 12 + 8)});
                    break;
                case kLocAxisFloat: {
                    const uint32_t mask = channel >> 16;
                    const int axis = mask == 1 ? 0 : mask == 2 ? 1 : mask == 4 ? 2 : -1;
                    if (axis < 0) return fail("unknown location axis in " + bone.name);
                    if (!need(4)) return fail("location keys run past the file");
                    auto& t = bone.locationAxis[axis];
                    t.frames = frames;
                    for (size_t k = 0; k < n; ++k) t.values.push_back(r.f32(keys + k * 4));
                    break;
                }
                default:
                    if (std::find(file.skipped_.begin(), file.skipped_.end(), format) == file.skipped_.end())
                        file.skipped_.push_back(format);
                    break;
                }
            }
            anim.bones.push_back(std::move(bone));
        }
        file.animations_.push_back(std::move(anim));
    }
    return file;
}

Vec3 sampleTrack(const GmtTrack<Vec3>& t, float frame) {
    float u;
    const size_t i = keyIndex(t, frame, &u);
    if (u == 0) return t.values[i];
    Vec3 v;
    for (int k = 0; k < 3; ++k) v[k] = t.values[i][k] * (1 - u) + t.values[i + 1][k] * u;
    return v;
}

Quat sampleTrack(const GmtTrack<Quat>& t, float frame) {
    float u;
    const size_t i = keyIndex(t, frame, &u);
    const Quat& a = t.values[i];
    const Quat& b = t.values[std::min(i + 1, t.values.size() - 1)];
    const float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const float sign = dot < 0 ? -1.0f : 1.0f;
    Quat q;
    float len = 0;
    for (int k = 0; k < 4; ++k) {
        q[k] = a[k] * (1 - u) + sign * b[k] * u;
        len += q[k] * q[k];
    }
    len = std::sqrt(len);
    if (len > 0)
        for (auto& c : q) c /= len;
    return q;
}

float sampleTrack(const GmtTrack<float>& t, float frame) {
    float u;
    const size_t i = keyIndex(t, frame, &u);
    return u == 0 ? t.values[i] : t.values[i] * (1 - u) + t.values[i + 1] * u;
}

} // namespace formats
