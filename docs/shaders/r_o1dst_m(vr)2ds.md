# r_b1dst_a(vr)2d with r_o1dst_m(vr)2ds: a reading

Codex's semantic reading of the material pass's largest pair (567 draws a frame), from the
disassembly of the vertex shader (program 0 of r_b1dst_a(vr)2d.fxo) and pixel shader (program 1
of r_o1dst_m(vr)2ds.fxo). Not yet checked against the game: the native port is checked with
LPN_NATIVE_COMPARE.

```glsl
// Notation and scope
//
// All offsets below are bytes, including descriptor-table offsets.
// Descriptor names refer to their entry-time SGPR bindings, before reuse.
//
// VS:
//   VSDirect0 = entry s[4:7].
//   VSDirect1 = entry s[8:11].
//   VSV[k]    = V# loaded from entry s[12:13] + k.
//
// PS:
//   PSFlags   = entry s[0:3].
//   PSView    = entry s[4:7].
//   PSV[k]    = V# loaded from entry s[12:13] + k.
//   T[k]      = T# loaded from entry s[8:9] + k.
//   S[k]      = S# loaded from entry s[10:11] + k.
//
// F(buffer, offset) reads one float32.
// U(buffer, offset) reads one uint32.
// F2/F3/F4 read consecutive float32 components at +0,+4,+8,+12.
// Rows below are vec4s; row.xyz excludes the fourth component.
// B(hex) means uintBitsToFloat(hex), preserving the exact literal bits.
// sat(x) = clamp(x, 0, 1), componentwise.
// sq(x) = x*x, componentwise.
// has(flags, mask) = (flags & mask) != 0.
//
// sample(T[k], S[j], uv) denotes the original image_sample operation:
// implicit LOD, filtering/addressing/format/channel swizzle from the descriptors.
// Channel names below refer to image_sample results after descriptor swizzling.
//
// Names such as camera, light and fog describe the observed calculation.
// Exact engine field names and downstream interpretation of mrt1 are not
// recoverable from these two programs.
//
// Fetch-shader code and PS input-enable registers were not supplied.
// VS inputs are therefore specified by post-fetch VGPRs.
// PS rasterXY means entry vec2(v2,v3); rasterThird means entry v4.
// These normally correspond to raster-position XY and Z under the expected
// PS input configuration; verify that configuration when binding a replacement.


// va.s: constants
//
// VSV[0x00]:
//   +0x00 float2 primaryUVOffset
//   +0x08 float2 secondaryUVOffset
//   +0x18 float  normalFlattenScale
//   +0x1c float  vertexAlphaScale
//
// VSV[0x10]:
//   +0x160 float3 lightingPositionOffset
//   +0x170 float3 objectTranslationOffset
//
// VSV[0x20]:
//   +0x00 vec4 objectRowX
//   +0x10 vec4 objectRowY
//   +0x20 vec4 objectRowZ
//
// VSV[0x30]:
//   +0x00 vec4 projectionRowX
//   +0x10 vec4 projectionRowY
//   +0x20 vec4 projectionRowZ
//   +0x30 vec4 projectionRowW
//   +0x40 vec4 viewRowX
//   +0x50 vec4 viewRowY
//   +0x60 vec4 viewRowZ
//   +0x220 float3 cameraPosition
//
// VSV[0x40]:
//   +0x00 uint vertexFlags
//
// VSDirect0:
//   +0x40 float3 fogExtinction
//   +0x50 float3 fogPhaseQuadraticRGB
//   +0x5c float  fogPhaseDenominatorBase
//   +0x60 float3 fogPhaseInversePowerRGB
//   +0x6c float  fogPhaseDenominatorCosineScale
//   +0x70 float3 fogScatterTint
//   +0x7c float  fogPhaseOffsetAndScatterGain
//
// VSDirect1:
//   +0x80 float3 fogDirection
//
// Loaded but unused components:
//   VSV[0x10] +0x17c, +0x16c;
//   VSV[0x30] +0x44c is NOT loaded: the camera load is +0x220..+0x22c,
//   with +0x22c unused;
//   viewRowX/Y/Z.w;
//   VSDirect0 +0x4c;
//   VSDirect1 +0x8c.
//
// Vertex flag meanings:
//   bit 17, 0x00020000: evaluate vertex atmospheric attenuation/scattering.
//   bit 18, 0x00040000: suppress exported scattering RGB;
//                      attenuation remains active.
//   bit 19, 0x00080000: add lightingPositionOffset to param7 position;
//                      also participates in camera-subtraction selection.
//   bit 20, 0x00100000: force camera subtraction even when bit 19 is set.
//
// Camera subtraction is skipped ONLY when bit19 && !bit20.


// va.s: post-fetch inputs
//
// vec3 localPosition       = vec3(v4,v5,v6);
// vec4 encodedNormal       = vec4(v8,v9,v10,v11);
// vec4 encodedTangent      = vec4(v12,v13,v14,v15);
// vec4 vertexColor         = vec4(v16,v17,v18,v19);
// vec2 primaryUV           = vec2(v20,v21);
// vec2 secondaryUV         = vec2(v24,v25);
// vec4 encodedBakedLight   = vec4(v28,v29,v30,v31);
//
// Other fetched components, if present, are unused here.


// va.s: position arithmetic
//
// Preserve these instruction boundaries, operand order and FP32 behavior.
// mul32/add32/sub32 are original single-precision operations.
// mad32/mac32 represent the original v_mad_f32/v_mac_f32 semantics;
// do not replace the sequence with a reassociated matrix expression or
// assume a fused GLSL fma is equivalent.

uint vertexFlags = U(VSV[0x40], 0x00);

vec4 objectRowX = F4(VSV[0x20], 0x00);
vec4 objectRowY = F4(VSV[0x20], 0x10);
vec4 objectRowZ = F4(VSV[0x20], 0x20);
vec3 translationOffset = F3(VSV[0x10], 0x170);

float worldX = add32(translationOffset.x, objectRowX.w);
float worldY = add32(objectRowY.w, translationOffset.y);
float worldZ = add32(objectRowZ.w, translationOffset.z);

worldZ = mad32(localPosition.x, objectRowZ.x, worldZ);
worldX = mac32(objectRowX.x, localPosition.x, worldX);
worldY = mac32(objectRowY.x, localPosition.x, worldY);

worldX = mad32(objectRowX.y, localPosition.y, worldX);
worldY = mad32(objectRowY.y, localPosition.y, worldY);
worldZ = mac32(objectRowZ.y, localPosition.y, worldZ);

worldZ = mad32(objectRowZ.z, localPosition.z, worldZ);
worldX = mac32(objectRowX.z, localPosition.z, worldX);
worldY = mac32(objectRowY.z, localPosition.z, worldY);

vec3 worldPosition = vec3(worldX, worldY, worldZ);
vec3 clipInputPosition = worldPosition;

if (!(has(vertexFlags, 0x00080000) &&
      !has(vertexFlags, 0x00100000))) {
    vec3 camera = F3(VSV[0x30], 0x220);
    clipInputPosition.x = sub32(worldX, camera.x);
    clipInputPosition.y = sub32(worldY, camera.y);
    clipInputPosition.z = sub32(worldZ, camera.z);
}

// This offset does not affect clipInputPosition.
vec3 lightingPosition = worldPosition;
if (has(vertexFlags, 0x00080000)) {
    vec3 offset = F3(VSV[0x10], 0x160);
    lightingPosition.x = add32(offset.x, worldX);
    lightingPosition.y = add32(offset.y, worldY);
    lightingPosition.z = add32(offset.z, worldZ);
}

vec4 viewRowX = F4(VSV[0x30], 0x40);
vec4 viewRowY = F4(VSV[0x30], 0x50);
vec4 viewRowZ = F4(VSV[0x30], 0x60);

float viewX = mul32(viewRowX.x, clipInputPosition.x);
viewX = mac32(viewRowX.y, clipInputPosition.y, viewX);
viewX = mac32(viewRowX.z, clipInputPosition.z, viewX);

float viewY = mul32(viewRowY.x, clipInputPosition.x);
viewY = mac32(viewRowY.y, clipInputPosition.y, viewY);
viewY = mad32(viewRowY.z, clipInputPosition.z, viewY);

float viewZ = mul32(viewRowZ.x, clipInputPosition.x);
viewZ = mac32(viewRowZ.y, clipInputPosition.y, viewZ);
viewZ = mad32(viewRowZ.z, clipInputPosition.z, viewZ);

vec4 projectionRowW = F4(VSV[0x30], 0x30);
float clipW = projectionRowW.w;
clipW = mac32(projectionRowW.x, viewX, clipW);
clipW = mac32(projectionRowW.y, viewY, clipW);
clipW = mad32(projectionRowW.z, viewZ, clipW);

vec4 projectionRowX = F4(VSV[0x30], 0x00);
vec4 projectionRowY = F4(VSV[0x30], 0x10);
vec4 projectionRowZ = F4(VSV[0x30], 0x20);

float clipX = mad32(projectionRowX.x, viewX, projectionRowX.w);
float clipY = mad32(projectionRowY.x, viewX, projectionRowY.w);
float clipZ = mac32(projectionRowZ.x, viewX, projectionRowZ.w);

clipX = mac32(projectionRowX.y, viewY, clipX);
clipY = mac32(projectionRowY.y, viewY, clipY);
clipZ = mac32(projectionRowZ.y, viewY, clipZ);

clipX = mac32(projectionRowX.z, viewZ, clipX);
clipY = mac32(projectionRowY.z, viewZ, clipY);
clipZ = mac32(projectionRowZ.z, viewZ, clipZ);

pos0 = vec4(clipX, clipY, clipZ, clipW);

// pos0 is the shader export before any API-specific clip-depth conversion.


// va.s: remaining calculations

vec3 fogTransmission = vec3(1);
vec3 fogScatter = vec3(0);

if (has(vertexFlags, 0x00020000)) {
    vec3 extinction = F3(VSDirect0, 0x40);

    fogTransmission =
        exp2(extinction * clipW * B(0x3fb8aa3b));

    vec3 cameraToVertex =
        lightingPosition - F3(VSV[0x30], 0x220);

    float cosine =
        -dot(F3(VSDirect1, 0x80),
             cameraToVertex * inversesqrt(dot(cameraToVertex,
                                             cameraToVertex)));

    float inverseRoot =
        inversesqrt(F(VSDirect0, 0x5c) +
                    F(VSDirect0, 0x6c) * cosine);

    float inversePower = inverseRoot * inverseRoot * inverseRoot;
    float common = F(VSDirect0, 0x7c);

    vec3 phase =
        F3(VSDirect0, 0x50) * (cosine*cosine + common) +
        F3(VSDirect0, 0x60) * inversePower;

    fogScatter =
        (vec3(common) - fogTransmission) *
        F3(VSDirect0, 0x70) * phase;
}

if (has(vertexFlags, 0x00040000))
    fogScatter = vec3(0);

// Decode factor is exactly B(0x40008102), not exactly 2.
vec3 objectNormal =
    encodedNormal.xyz * B(0x40008102) - vec3(1);

vec3 objectTangent =
    encodedTangent.xyz * B(0x40008102) - vec3(1);

float tangentHandedness =
    encodedTangent.w * B(0x40008102) - 1;

mat3 objectLinear = rows(
    objectRowX.xyz,
    objectRowY.xyz,
    objectRowZ.xyz
);

// These are neither normalized nor inverse-transpose transformed in the VS.
vec3 transformedNormal = objectLinear * objectNormal;
vec3 transformedTangent = objectLinear * objectTangent;

float normalFlatten =
    F(VSV[0x00], 0x18) *
    (1 - sat(encodedNormal.w * B(0x3f808102)));

vec3 bakedLight =
    sq(0.25 * encodedBakedLight.xyz /
       (1 - encodedBakedLight.w));

param0 = vec4(
    primaryUV   + F2(VSV[0x00], 0x00),
    secondaryUV + F2(VSV[0x00], 0x08)
);

param1 = vec4(
    vertexColor.rgb * fogTransmission,
    vertexColor.a * F(VSV[0x00], 0x1c)
);

param2 = vec4(fogScatter, normalFlatten);
param3 = vec4(transformedNormal, 0);
param4 = vec4(bakedLight, 0);
param5 = vec4(fogTransmission, 0);
param6 = vec4(transformedTangent, tangentHandedness);
param7 = vec4(lightingPosition, 0);


// VS-to-PS interface
//
// param0.xy: primary UV; base, normal and specular-material textures.
// param0.zw: secondary UV; two detail/modulation textures.
// param1.rgb: vertex RGB already attenuated by vertex fog;
//             modulates direct diffuse and specular terms.
// param1.a: exported but never read by m1.s.
// param2.rgb: atmospheric scattering, optionally added after tone mapping.
// param2.w: flattens normal XY and blends texture modulation toward identity.
// param3.xyz: transformed geometric normal for the tangent frame.
// param3.w: zero; unused.
// param4.xyz: baked RGB lighting, added to PS point-light accumulation.
// param4.w: zero; unused.
// param5.xyz: vertex fog transmission, applied to baked/point lighting.
// param5.w: zero; unused.
// param6.xyz: transformed tangent.
// param6.w: tangent-frame handedness.
// param7.xyz: lighting position for point lights, view direction and fog.
// param7.w: zero; unused.
//
// PS uses v_interp_p1/p2 with entry v0/v1 for all accessed varyings.


// m1.s: constants
//
// PSFlags:
//   +0x00 uint pixelFlags
//
// PSView:
//   +0x218 float2 rasterToUV
//   +0x220 float3 cameraPosition
//   +0x22c uint outputAlphaFlags
//   +0x230 float3 finalBlendColor
//   +0x23c float  finalBlendWeight
//
// PSV[0x00]:
//   +0x00c uint screenMaskFlags
//   +0x02c float depthReconstructAdd
//   +0x03c float depthReconstructNumerator
//   +0x04c float diffuseScreenBlendBase
//   +0x2b0 float depthEdgeRadius
//   +0x2b4 float depthEdgeMinimum
//   +0x2b8 float depthEdgeThresholdScale
//   +0x2bc float normalEdgeRadius
//   +0x2c0 float normalEdgeMinimum
//   +0x2c4 float normalEdgeThresholdScale
//   +0x2c8 float facingMinimum
//   +0x2cc float facingWidthFar
//   +0x2d0 float facingThresholdFar
//   +0x2d4 float maskDistanceStart
//   +0x2d8 float maskDistanceNear
//   +0x2dc float maskDistanceEnd
//   +0x2e0 float maskMinimumRadius
//   +0x2e4 float maskDistanceFadeEnd
//   +0x2e8 float maskDistanceFadeAmount
//   +0x2ec float maskOutputThreshold
//   +0x2f0 float facingDistanceFar
//   +0x2f4 float facingDistanceNear
//   +0x2f8 float facingWidthNear
//   +0x2fc float facingThresholdNear
//
// PSV[0x10]:
//   +0x000 float exposure
//   +0x008 float screenLightingDivisor
//   +0x010 float3 distanceFogColor
//   +0x028 float distanceFogScale
//   +0x02c float distanceFogBias
//   +0x030 float3 outputRGBScale
//   +0x03c float outputAlphaScale
//   +0x090 float shadowRScale
//   +0x094 float shadowRBias
//   +0x098 float shadowGScale
//   +0x09c float shadowGBias
//   +0x0b0 float shadowTintScale
//   +0x0b4 float shadowTintBias
//   +0x0b8 float shadowTintGain0
//   +0x0bc float shadowTintGain1
//   +0x100 float3 ambientRGB
//
// PSV[0x30]:
//   +0x000 float3 directionalColor0
//   +0x010 float3 directionalColor1
//   +0x020 vec4 pointColorAndInnerDistance[0]
//   +0x030 vec4 pointColorAndInnerDistance[1]
//   +0x040 vec4 pointColorAndInnerDistance[2]
//   +0x050 vec4 pointColorAndInnerDistance[3]
//   +0x070 float3 shadowTintBase
//   +0x080 float3 directionalVector0
//   +0x090 float3 directionalVector1
//   +0x0a0 vec4 pointPositionAndFalloff[0]
//   +0x0b0 vec4 pointPositionAndFalloff[1]
//   +0x0c0 vec4 pointPositionAndFalloff[2]
//   +0x0d0 vec4 pointPositionAndFalloff[3]
//   +0x0f0 float secondarySpecularScale
//
// PSV[0x50]:
//   +0x01c float alphaFactor0
//
// PSV[0x60]:
//   +0x000 float3 diffuseMultiplier
//   +0x00c float alphaFactor1
//   +0x010 float3 specularMultiplier
//   +0x01c float ambientTextureGain
//
// PSV[0x70]:
//   +0x040 float2 specularTextureUVScale
//   +0x048 float2 normalTextureUVScale
//   +0x050 float2 specularDetailUVScale
//   +0x090 float3 diffuseColor
//   +0x09c float alphaFactor2
//   +0x0a0 float3 specularColor
//   +0x0ac float specularExponent
//   +0x0b0 float3 ambientTextureTint
//
// Unused components within vector loads:
//   PSV[0x10] +0x004,+0x00c,+0x01c,+0x020,+0x024,+0x10c;
//   PSV[0x30] +0x00c,+0x01c,+0x07c,+0x08c,+0x09c;
//   PSV[0x70] +0x0bc.
//
// PSV[0x10] +0x090..+0x09c are loaded and applied twice.


// m1.s: texture bindings
//
// Meaning                  T# offset   S# offset   Channels   Coordinates
//
// Tangent normal XY        0x0a0       0x050       RG         uv0 * normalTextureUVScale
// Screen light/shadow      0x1a0       0x0d0       RGA        rasterUV; pixelFlags bit3
// Diffuse detail           0x020       0x010       RGB        uv1
// Screen lighting          0x1c0       0x0e0       RGBA       rasterUV
// Base color/alpha         0x000       0x000       RGBA       uv0
// Specular material        0x060       0x030       RGB        uv0 * specularTextureUVScale
// Specular detail          0x080       0x040       RGB        uv1 * specularDetailUVScale
// Screen depth             0x100       0x080       R          rasterUV, plus optional offsets
// Screen normal            0x1e0       0x0f0       RGB        rasterUV and two offsets
//
// The screen-normal offset samples occur whenever bit25 is clear,
// even if all screenMaskFlags feature bits are clear.
// Additional screen-depth offset samples require screenMaskFlags bit3.
// Sampled screen shadow B is not requested; screen A becomes shadowCoverage.
// The screen-normal samples are separate from the material normal map.


// m1.s: branch flags
//
// pixelFlags = U(PSFlags,0):
//   bit 2,  0x00000004: add param2.rgb after tone mapping;
//                      disable the PS distance-fog blend.
//   bit 3,  0x00000008: sample screen shadow/light RGA;
//                      apply its R/G remap twice;
//                      modify shadow tint;
//                      bypass final outputRGBScale/outputAlphaScale.
//   bit 16, 0x00010000: enable point light 0.
//   bit 17, 0x00020000: enable point light 1.
//   bit 18, 0x00040000: enable point light 2.
//   bit 19, 0x00080000: enable point light 3.
//   bit 22, 0x00400000: enable PS distance fog when bit2 is clear.
//   bit 23, 0x00800000: apply final color blend and scale mrt1 R/A.
//   bit 25, 0x02000000: skip screen-mask calculation; mrt1 R starts at zero.
//
// screenMaskFlags = U(PSV[0x00],0x00c):
//   bit 1, 0x2: enable view-facing mask with normal-response gate.
//   bit 2, 0x4: enable screen-normal edge mask.
//   bit 3, 0x8: enable screen-depth edge mask.
//
// outputAlphaFlags = U(PSView,0x22c):
//   bit 0, 0x1: replace mrt0 A with scaled rasterThird;
//               mrt1 A still uses the material alpha.
//
// Other conditional branches are per-pixel tests:
//   point light: execute attenuation only when saturated N·L > 0;
//   normal/depth mask: threshold comparisons described below;
//   facing mask: optional threshold gate;
//   final screen mask: discard values below maskOutputThreshold.
// No discard/kill instruction occurs.


// m1.s: interpolated inputs and normal reconstruction

uint pixelFlags = U(PSFlags, 0x00);

vec2 uv0 = attr0.xy;
vec2 uv1 = attr0.zw;
vec2 rasterToUV = F2(PSView, 0x218);
vec2 rasterUV = rasterXY * rasterToUV;

vec3 position = attr7.xyz;
vec3 geometricNormal = attr3.xyz;
vec3 tangent = attr6.xyz;

// Cross-product order is normal × tangent.
vec3 bitangent = cross(geometricNormal, tangent) * attr6.w;

float flatten = attr2.w;

vec2 sampledNormalXY =
    sample(T[0x0a0], S[0x050],
           uv0 * F2(PSV[0x70], 0x048)).rg;

// Equivalent to remapping mix(sampledNormalXY, 0.5, flatten) to [-1,1].
vec2 normalXY = (2*sampledNormalXY - vec2(1)) * (1-flatten);
float normalZ = sqrt(sat(1-dot(normalXY, normalXY)));

vec3 N = normalize(
    tangent * normalXY.x +
    bitangent * normalXY.y +
    geometricNormal * normalZ
);


// m1.s: shadow/light screen factors

float shadowR = 1;
float shadowG = 1;
float shadowCoverage = 1;

if (has(pixelFlags, 0x00000008)) {
    vec3 rga =
        sample(T[0x1a0], S[0x0d0], rasterUV).rga;

    shadowR = rga.x;
    shadowG = rga.y;
    shadowCoverage = rga.z;

    // Both repetitions are present in the program.
    for (int repeat = 0; repeat < 2; ++repeat) {
        shadowR =
            F(PSV[0x10],0x090)*shadowR + F(PSV[0x10],0x094);
        shadowG =
            shadowG*F(PSV[0x10],0x098) + F(PSV[0x10],0x09c);
    }
}

// No saturation on these remaps or tint calculations.
vec3 shadowTintBase = F3(PSV[0x30], 0x070);
vec3 directionalShadowTint =
    shadowTintBase +
    (vec3(1)-shadowTintBase) * (shadowR*shadowG);

vec3 indirectShadowTint = directionalShadowTint;

if (has(pixelFlags, 0x00000008)) {
    indirectShadowTint =
        (F(PSV[0x10],0x0b8) + F(PSV[0x10],0x0bc)) *
        (directionalShadowTint * F(PSV[0x10],0x0b0) +
         vec3(F(PSV[0x10],0x0b4)));
}


// m1.s: point lighting

vec3 pointLighting = vec3(0);

for (int i = 0; i < 4; ++i) {
    if (has(pixelFlags, 0x00010000u << i)) {
        vec4 positionFalloff =
            F4(PSV[0x30], 0x0a0 + 0x10*i);
        vec4 colorInnerDistance =
            F4(PSV[0x30], 0x020 + 0x10*i);

        vec3 lightToSurface = position - positionFalloff.xyz;
        float distanceSquared = dot(lightToSurface, lightToSurface);
        float lambert =
            sat(-dot(N, lightToSurface) *
                inversesqrt(distanceSquared));

        if (lambert > 0) {
            float outsideInnerDistance =
                max(0, sqrt(distanceSquared)-colorInnerDistance.w);

            float attenuation =
                sat(B(0x3f808102) /
                    (1 + positionFalloff.w *
                         outsideInnerDistance * outsideInnerDistance)
                    + B(0xbb810205));

            pointLighting +=
                colorInnerDistance.rgb * (lambert*attenuation);
        }
    }
}


// m1.s: material and directional lighting

vec3 cameraPosition = F3(PSView, 0x220);
vec3 cameraToSurface = position-cameraPosition;
float cameraDistanceSquared = dot(cameraToSurface, cameraToSurface);
vec3 awayFromCamera =
    cameraToSurface * inversesqrt(cameraDistanceSquared);

vec3 direction0 = F3(PSV[0x30], 0x080);
vec3 direction1 = F3(PSV[0x30], 0x090);

float ndotl0 = sat(-dot(direction0, N));
float ndotl1 = sat(-dot(direction1, N));

vec3 halfDirection0 = awayFromCamera + direction0;

// The second specular lobe does not use direction1 to form its half vector.
vec3 halfDirection1 =
    awayFromCamera + vec3(0, B(0xbe99999a), 0);

float specularExponent = F(PSV[0x70], 0x0ac);

// Original pow is exp2(exponent * log2(saturated cosine)).
float lobe0 = exp2(
    specularExponent *
    log2(sat(-dot(N, halfDirection0) *
             inversesqrt(dot(halfDirection0, halfDirection0))))
);

float lobe1 = exp2(
    specularExponent *
    log2(sat(-dot(N, halfDirection1) *
             inversesqrt(dot(halfDirection1, halfDirection1))))
);

vec3 lightColor0 = F3(PSV[0x30], 0x000);
vec3 lightColor1 = F3(PSV[0x30], 0x010);

vec3 diffuseColor =
    F3(PSV[0x70],0x090) * F3(PSV[0x60],0x000);

vec3 specularColor =
    F3(PSV[0x70],0x0a0) * F3(PSV[0x60],0x010);

vec3 ambientTextureColor =
    F3(PSV[0x70],0x0b0) * F(PSV[0x60],0x01c);

vec4 base =
    sample(T[0x000], S[0x000], uv0);

vec3 diffuseDetail =
    sample(T[0x020], S[0x010], uv1).rgb;

vec3 specularMaterial =
    sample(T[0x060], S[0x030],
           uv0 * F2(PSV[0x70],0x040)).rgb;

vec3 specularDetail =
    sample(T[0x080], S[0x040],
           uv1 * F2(PSV[0x70],0x050)).rgb;

vec4 screenLighting =
    sample(T[0x1c0], S[0x0e0], rasterUV) /
    F(PSV[0x10],0x008);

vec3 adjustedBase =
    sq(base.rgb) *
    (vec3(1)-flatten*(vec3(1)-sq(diffuseDetail)));

vec3 adjustedSpecular =
    specularMaterial *
    (vec3(1)-flatten*(vec3(1)-specularDetail));

vec3 vertexTint = attr1.rgb * shadowCoverage;
vec3 indirectTint = indirectShadowTint * shadowCoverage;

vec3 bakedAndPoint = attr4.xyz + pointLighting;
vec3 fogTransmission = attr5.xyz;

vec3 directionalDiffuse =
    lightColor0 * directionalShadowTint * ndotl0 +
    lightColor1 * shadowG * ndotl1;

vec3 diffuseLighting =
    vertexTint *
        (ambientTextureColor +
         F3(PSV[0x10],0x100) +
         diffuseColor * directionalDiffuse);

vec3 indirectDiffuse =
    fogTransmission * diffuseColor * bakedAndPoint +
    screenLighting.rgb * shadowG;

float screenBlendBase = F(PSV[0x00],0x04c);
float indirectDiffuseBlend =
    screenBlendBase + (1-screenBlendBase)*shadowG;

diffuseLighting +=
    indirectDiffuseBlend * indirectTint * indirectDiffuse;

vec3 directionalSpecular =
    lightColor0 * directionalShadowTint * lobe0 +
    lightColor1 *
        (2*F(PSV[0x30],0x0f0)) * lobe1 * ndotl1;

vec3 specularLighting =
    vertexTint * specularColor * directionalSpecular;

// Screen-lighting A is broadcast across all three specular channels.
vec3 indirectSpecular =
    fogTransmission * specularColor * bakedAndPoint +
    vec3(screenLighting.a * shadowG);

specularLighting +=
    indirectTint * lobe1 * indirectSpecular;

vec3 linearMaterialColor =
    adjustedBase * diffuseLighting +
    adjustedSpecular * specularLighting;


// m1.s: tone curve, vertex scattering, distance fog

float toneChannel(float linearValue) {
    float x = 2*F(PSV[0x10],0x000)*linearValue;

    float numerator =
        x*(B(0x3e19999a)*x + B(0x3d4ccccd)) +
        B(0x3b83126f);

    float denominator =
        x*(B(0x3e19999a)*x + 0.5) +
        B(0x3d75c290);

    return sqrt(sat(
        B(0x3fb0852e)*(numerator/denominator) +
        B(0xbdbc49ca)
    ));
}

vec3 toneColor = vec3(
    toneChannel(linearMaterialColor.r),
    toneChannel(linearMaterialColor.g),
    toneChannel(linearMaterialColor.b)
);

vec3 outputColor = toneColor;
float distanceFogAmount = 0;

if (has(pixelFlags, 0x00000004))
    outputColor += attr2.rgb;

if (has(pixelFlags, 0x00400000) &&
    !has(pixelFlags, 0x00000004)) {
    float distanceToCamera = sqrt(cameraDistanceSquared);

    distanceFogAmount =
        sat(F(PSV[0x10],0x028)*distanceToCamera +
            F(PSV[0x10],0x02c));

    outputColor =
        toneColor*(1-distanceFogAmount) +
        F3(PSV[0x10],0x010)*distanceFogAmount;
}


// m1.s: material alpha and optional output scales
//
// Alpha uses base texture A, not vertex alpha or shadowCoverage.

float materialAlpha =
    F(PSV[0x50],0x01c) *
    F(PSV[0x70],0x09c) *
    F(PSV[0x60],0x00c) *
    base.a;

if (!has(pixelFlags, 0x00000008)) {
    outputColor *= F3(PSV[0x10],0x030);
    materialAlpha *= F(PSV[0x10],0x03c);
}


// m1.s: auxiliary screen mask exported in mrt1 R

float auxiliaryMask = 0;

if (!has(pixelFlags, 0x02000000)) {
    uint screenMaskFlags = U(PSV[0x00], 0x00c);

    float depthAdd = F(PSV[0x00],0x02c);
    float depthNumerator = F(PSV[0x00],0x03c);

    float depthAt(vec2 coordinates) {
        return depthNumerator /
            (sample(T[0x100],S[0x080],coordinates).r + depthAdd);
    }

    float reconstructedDepth = depthAt(rasterUV);

    vec3 screenNormal =
        2*sample(T[0x1e0],S[0x0f0],rasterUV).rgb - vec3(1);

    float distanceStart = F(PSV[0x00],0x2d4);
    float distanceNear  = F(PSV[0x00],0x2d8);
    float distanceEnd   = F(PSV[0x00],0x2dc);
    float minimumRadius = F(PSV[0x00],0x2e0);

    float radiusFade =
        sat(1-(reconstructedDepth-distanceStart) /
              (distanceEnd-distanceStart));

    float nearRamp =
        sat((reconstructedDepth-distanceNear) /
            (distanceStart-distanceNear));

    vec2 screenDirection =
        rasterToUV * vec2(screenNormal.x, -screenNormal.y);

    float normalRadius =
        nearRamp *
        max(minimumRadius, F(PSV[0x00],0x2bc)*radiusFade);

    // Both samples execute even when screenMaskFlags bit2 is clear.
    vec3 normalPlus =
        sample(T[0x1e0], S[0x0f0],
               rasterUV+screenDirection*normalRadius).rgb;

    vec3 normalMinus =
        sample(T[0x1e0], S[0x0f0],
               rasterUV-screenDirection*normalRadius).rgb;

    float responsePlus =
        dot(screenNormal, 2*normalPlus-vec3(1));

    float responseMinus =
        dot(screenNormal, 2*normalMinus-vec3(1));

    float normalEdgeMask = 0;
    float depthEdgeMask = 0;
    float facingMask = 0;

    if (has(screenMaskFlags, 0x4)) {
        float agreement =
            0.25*(responsePlus+1)*(responseMinus+1);

        float agreementSquared = agreement*agreement;

        float threshold =
            1-F(PSV[0x00],0x2c4)*nearRamp;

        float thresholdPassed =
            (agreementSquared+B(0x3b808081) >= threshold) ? 1 : 0;

        float suppression =
            F(PSV[0x00],0x2c0) +
            (1-F(PSV[0x00],0x2c0)) *
            sat(thresholdPassed+agreementSquared);

        normalEdgeMask = nearRamp*(1-suppression);
    }

    if (has(screenMaskFlags, 0x8)) {
        float depthRadius =
            nearRamp *
            max(minimumRadius,
                F(PSV[0x00],0x2b0)*radiusFade);

        float depthPlus =
            depthAt(rasterUV+screenDirection*depthRadius);

        float depthMinus =
            depthAt(rasterUV-screenDirection*depthRadius);

        float depthResponse =
            1-0.5*(-2*reconstructedDepth+depthPlus+depthMinus);

        float threshold =
            1-F(PSV[0x00],0x2b8)*nearRamp;

        float thresholdPassed =
            (depthResponse+B(0x3b808081) >= threshold) ? 1 : 0;

        float suppression =
            F(PSV[0x00],0x2b4) +
            (1-F(PSV[0x00],0x2b4)) *
            sat(thresholdPassed+depthResponse);

        depthEdgeMask = nearRamp*(1-suppression);
    }

    if (has(screenMaskFlags, 0x2)) {
        float facing = sat(-dot(N, awayFromCamera));

        float facingFar = F(PSV[0x00],0x2f0);
        float facingNear =
            min(F(PSV[0x00],0x2f4),
                facingFar+B(0xba83126f));

        float facingDistanceRamp =
            sat((reconstructedDepth-facingNear) /
                (facingFar-facingNear));

        float width =
            F(PSV[0x00],0x2f8) +
            facingDistanceRamp *
            (F(PSV[0x00],0x2cc)-F(PSV[0x00],0x2f8));

        float normalizedFacing =
            sat((facing+B(0x3b808081)) /
                (nearRamp*width+B(0x3b808081)));

        float facingSuppression =
            sat(F(PSV[0x00],0x2c8) +
                (1-F(PSV[0x00],0x2c8)) *
                normalizedFacing*normalizedFacing);

        float candidate =
            nearRamp*(1-facingSuppression);

        float normalResponseThreshold =
            F(PSV[0x00],0x2fc) +
            facingDistanceRamp *
            (F(PSV[0x00],0x2d0)-F(PSV[0x00],0x2fc));

        float normalResponse =
            1-responsePlus*responseMinus;

        facingMask =
            normalResponseThreshold > 0
            ? candidate *
                ((normalResponse >= normalResponseThreshold) ? 1 : 0)
            : candidate;
    }

    auxiliaryMask =
        max(depthEdgeMask, max(facingMask, normalEdgeMask));

    float distanceFade =
        max(1-F(PSV[0x00],0x2e8),
            sat(1-(reconstructedDepth-distanceStart) /
                  (F(PSV[0x00],0x2e4)-distanceStart)));

    auxiliaryMask *= distanceFade;
    auxiliaryMask *= 1-distanceFogAmount;

    auxiliaryMask *=
        (F(PSV[0x00],0x2ec) <= auxiliaryMask) ? 1 : 0;
}


// m1.s: final blend and exports

float mrt1Alpha = materialAlpha;

if (has(pixelFlags, 0x00800000)) {
    vec3 blendColor = F3(PSView,0x230);
    float blendWeight = F(PSView,0x23c);

    outputColor =
        blendColor + blendWeight*outputColor -
        blendColor*blendWeight;

    auxiliaryMask *= blendWeight;
    mrt1Alpha = materialAlpha*blendWeight;
}

// This selection uses the pre-final-blend materialAlpha.
// Consequently mrt0 A and mrt1 A can differ even without rasterThird mode.
float mrt0Alpha =
    has(U(PSView,0x22c),0x1)
    ? sat(B(0x3a4d9a68)*rasterThird)
    : materialAlpha;

// Logical shader outputs before target-format conversion/component swapping:
//
//   mrt0 = (outputColor.r, outputColor.g, outputColor.b, mrt0Alpha)
//   mrt1 = (auxiliaryMask, 0, 0, mrt1Alpha)
//
// Exact compressed-export packing:
//
// H2_RTZ(a,b) packs float32 a into the low 16 bits and b into the high
// 16 bits using v_cvt_pkrtz_f16_f32: float16 conversion toward zero.
// Ordinary GLSL packHalf2x16 is not necessarily equivalent.

uint packedMaskAndZero =
    H2_RTZ(auxiliaryMask, 0);

uint packedZeroAndAlpha =
    H2_RTZ(0, mrt1Alpha);

uint packedRedGreen =
    H2_RTZ(outputColor.r, outputColor.g);

uint packedBlueAlpha =
    H2_RTZ(outputColor.b, mrt0Alpha);

// exp mrt1 compr:
//   first packed word  = packedMaskAndZero  => R,G
//   second packed word = packedZeroAndAlpha => B,A
//
// exp mrt0 compr:
//   first packed word  = packedRedGreen     => R,G
//   second packed word = packedBlueAlpha    => B,A
//
// The shader restores its original execution mask before exporting.
// Hardware target formats and component-swap state determine memory layout;
// those states are outside the supplied disassemblies.
```
