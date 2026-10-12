// Posing and skinning GMD models, optionally driven by a GMT animation.
#pragma once

#include "formats/gmd.h"
#include "formats/gmt.h"

#include <vector>

namespace formats {

GmdMatrix matIdentity();
GmdMatrix matMultiply(const GmdMatrix& a, const GmdMatrix& b);
GmdMatrix matFromTrs(const Vec3& t, const Quat& r, const Vec3& s);
Vec3 matTransformPoint(const GmdMatrix& m, const Vec3& p);

// World matrix of every node. With an animation, bones it names take their location and
// rotation from it at `frame`; everything else keeps the model's rest pose.
std::vector<GmdMatrix> poseNodes(const GmdModel& model, const GmtAnimation* anim, float frame);

// Skinned vertex positions of one mesh for the given node world matrices. Meshes without
// a matrix list are moved rigidly by their node (an assumption: every mesh in the models
// checked so far is skinned).
std::vector<Vec3> skinMesh(const GmdModel& model, const GmdMesh& mesh,
                           const std::vector<GmdMatrix>& world);

} // namespace formats
