// GMT (GSGT) skeletal animations.
//
// Big-endian. A file holds animations; each animation has per-bone curves of location and
// rotation keys sampled on integer frames. See docs/formats.md for the layout.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace formats {

using Vec3 = std::array<float, 3>;
using Quat = std::array<float, 4>; // xyzw

template <typename T>
struct GmtTrack {
    std::vector<uint16_t> frames;
    std::vector<T> values;
    bool empty() const { return frames.empty(); }
};

struct GmtBone {
    std::string name;
    GmtTrack<Vec3> location;
    GmtTrack<Quat> rotation;
    // Single-axis location curves (format 4): axis 0..2, overriding one component.
    std::array<GmtTrack<float>, 3> locationAxis;
};

struct GmtAnimation {
    std::string name;
    float frameRate = 30.0f;
    uint32_t startFrame = 0;
    uint32_t endFrame = 0;
    std::vector<GmtBone> bones;

    const GmtBone* find(const std::string& boneName) const;
};

class GmtFile {
public:
    static std::optional<GmtFile> parse(std::span<const uint8_t> data, std::string* error);
    static bool isGmt(std::span<const uint8_t> data);

    const std::vector<GmtAnimation>& animations() const { return animations_; }
    // Curve formats that were present but not decoded (face/hand patterns etc.).
    const std::vector<uint32_t>& skippedFormats() const { return skipped_; }

private:
    std::vector<GmtAnimation> animations_;
    std::vector<uint32_t> skipped_;
};

// Sample a track at `frame`, interpolating between keys (nlerp for rotations).
// The track must not be empty.
Vec3 sampleTrack(const GmtTrack<Vec3>& t, float frame);
Quat sampleTrack(const GmtTrack<Quat>& t, float frame);
float sampleTrack(const GmtTrack<float>& t, float frame);

} // namespace formats
