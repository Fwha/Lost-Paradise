codex
# s_b1dzt[skin] / sm_o1dzt[skin] and s_b1dst / sm_o1dzt: a reading

Both supplied vertex programs perform **four-bone skinning**. In particular, `vs_rigid.s` is not a rigid object-matrix transform despite its filename.

The readings below share common code where the programs share the calculation. `SKIN` means `vs_skin.s` / `ps_skin.s`; `RIGID` means `vs_rigid.s` / `ps_sm_o1dzt.s`. PC pairs are written **`SKIN / RIGID`**, in hexadecimal.

```glsl
// Notation and scope
//
// All offsets are bytes.
//
// VS:
//   VSDirect0 = entry s[4:7].
//   VSDirect1 = entry s[8:11].
//   VSV[k]    = V# loaded from entry s[12:13] + k.
//
// PS:
//   PSV[k]    = V# loaded from entry s[12:13] + k.
//   PSView    = PSV[0xc0].
//   PSFlags   = PSV[0xd0].
//   T[k]      = T# loaded from entry s[8:9] + k.
//   S[k]      = S# loaded from entry s[10:11] + k.
//
// Two PS samplers are direct entry bindings, not table loads:
//   ScreenNormalSampler = entry s[0:3].
//   ScreenLightSampler  = entry s[4:7].
//
// Names always refer to entry bindings, before SGPR reuse.
//
// F(buffer,offset) reads float32.
// U(buffer,offset) reads uint32.
// I(buffer,offset) reads int32.
// F2/F3/F4 read consecutive components.
// B(hex) = uintBitsToFloat(hex).
//
// sat(x) = clamp(x,0,1).
// has(f,m) = (f&m) != 0.
//
// add32/sub32/mul32 preserve individual FP32 instruction boundaries.
// mad32(a,b,c) denotes the original GCN multiply-add calculation.
// mac32(a,b,c) denotes the original accumulating multiply-add calculation.
// Do not reassociate their chains, or silently substitute fused GLSL fma.
//
// rcp32/rsq32/log2_32/exp2_32/sqrt32 denote the original instructions.
// Native GLSL operations express their meaning but need not reproduce
// the hardware approximation bit for bit.
//
// Vector operations below apply independently to each component.
//
// D3(a,b) means this ordered dot product:
//
//   t = mul32(a.x,b.x);
//   t = mac32(a.y,b.y,t);
//   t = mac32(a.z,b.z,t);
//
// NORM(a):
//   r = rsq32(D3(a,a));
//   return mul32(a,r);
//
// POW(a,e):
//   t = log2_32(a);
//   t = mul32(t,e);
//   return exp2_32(t);
//
// P5(a):
//   a2 = mul32(a,a);
//   a4 = mul32(a2,a2);
//   return mul32(a4,a);
//
// sample() means image_sample with implicit LOD.
// Descriptor format, swizzle, filtering and addressing remain significant.
//
// The load manifests give every s_buffer_load PC and its complete loaded
// range. A reference to F(buffer,offset) below uses the corresponding
// manifest entry; vector overfetch does not imply every component is used.
//
// Post-fetch VS VGPRs are specified because fetch-shader bodies were not
// supplied. PS rasterXY = entry vec2(v2,v3), rasterThird = entry v4.
//
// Names such as "ambient", "ramp" and "grading" describe calculations,
// not recovered engine field names.
```

```glsl
// VS constant-load manifest
//
// Buffer        Offset / number of dwords       PC: SKIN / RIGID
//
// VSV[0x00]     0x12c / 1                       114 / 114
//               signed deformation count
//
// VSV[0x00]     0x0d0 + 16*i / 4                2d4 / 2dc
//               deformation plane coefficients
// VSV[0x00]     0x050 + 16*i / 4                2d8 / 2e0
//               deformation plane origin
// VSV[0x00]     0x090 + 16*i / 4                2dc / 2e4
//               displacement.xyz, displacement scale.w
//
// VSV[0x40]     0x000 / 1                       358 / 360
//               vertex flags
// VSV[0x10]     0x170 / 4                       35c / 364
//               position offset.xyz
// VSV[0x30]     0x220 / 4                       3ac / 3b4
//               camera.xyz; conditional position-path load
// VSV[0x10]     0x160 / 4                       3d0 / 3d8
//               lighting-position offset.xyz
//
// VSV[0x30]     0x030 / 16                      40c / 414
//               projection W row, then view X/Y/Z rows
// VSV[0x30]     0x220 / 4                       410 / 418
//               camera.xyz; unconditional second load
//
// VSDirect1    0x080 / 4                       488 / 490
//               fog direction.xyz
// VSDirect0    0x040 / 16                      48c / 494
//               four fog vec4s
//
// VSV[0x30]     0x000 / 8                       5b8 / 5c0
//               projection X/Y rows
// VSV[0x30]     0x020 / 4                       5bc / 5c4
//               projection Z row
//
// Palette V#:
//   VSV[0x20], descriptor load PC 14 in both programs.
//   This is used by tbuffer_load_format_xyzw, not s_buffer_load.
//
// No VS UV-offset, vertex-colour multiplier, normal-flatten,
// or baked-light constants are loaded.
```

```glsl
// VS post-fetch inputs, both programs
//
// vec3 localPosition  = vec3(v4,v5,v6);
//
// vec4 fetchedWeights = vec4(v8,v9,v10,v11);
// uvec4 boneIndex     = uvec4(v12,v13,v14,v15); // integer bits
//
// vec4 encodedNormal  = vec4(v16,v17,v18,v19);
// vec4 encodedTangent = vec4(v20,v21,v22,v23);
// vec2 materialUV     = vec2(v24,v25);
//
// fetchedWeights.x is ignored.
// encodedNormal.w is exported only by SKIN.
// No vertex colour or baked-light input participates.
```

```glsl
// VS palette fetch and four-bone blend
//
// Both shaders fetch exactly four bones, three float4 rows per bone.
// There is no bone-count branch.
//
// For each index:
//
//   byteAddress = mul_lo_i32(48,boneIndex[k]);
//   element     = arithmeticShiftRight(byteAddress,4);
//   byteOffset  = byteAddress & 15;
//
// Each row is a typed float4 load with idxen+offen:
//   row r uses element+r and the same byteOffset.
//
// Thus, for ordinary nonnegative indices and a 16-byte palette stride,
// bone k occupies 48 bytes:
//
//   row0 = F4(palette,48*index+0);
//   row1 = F4(palette,48*index+16);
//   row2 = F4(palette,48*index+32);
//
// Preserve the descriptor addressing if its actual stride differs;
// the program constructs an element index, not simply a raw byte address.
//
// Typed-load PCs, identical in both VS programs:
//
//              row0    row1    row2
//   bone 0      084     074     07c
//   bone 1      09c     08c     094
//   bone 2      0c8     0a4     0c0
//   bone 3      0f8     0e4     0ec

float w1 = fetchedWeights.y;
float w2 = fetchedWeights.z;
float w3 = fetchedWeights.w;

float weightSum = add32(w1,w2);       // 104
weightSum = add32(weightSum,w3);      // 108
float w0 = sub32(1,weightSum);        // 10c

// No clamping or weight renormalization.

vec4 bone[4][3] = fetchPaletteRows(boneIndex);

// Blend each linear coefficient in bone order 0,1,2,3.
// Keep each multiply/add boundary; do not form four transformed positions
// and blend those instead.
float M[3][3];

for (int r=0; r<3; ++r) {
    for (int c=0; c<3; ++c) {
        float a = mul32(bone[0][r][c],w0);
        a = mad32(bone[1][r][c],w1,a);
        a = mad32(bone[2][r][c],w2,a);
        a = mad32(bone[3][r][c],w3,a);
        M[r][c] = a;
    }
}

// Here mad32 denotes each original MAD/MAC step.
// The two VS schedules use different temporary registers and, at several
// sites, MAD versus MAC, but preserve this coefficient accumulation order.

// Floor-relative translation:
// The anchor is floor(bone 0 translation), separately for every row.
// It is not floor(the weighted translation), and not one anchor per bone.
vec3 skinnedPosition;

for (int r=0; r<3; ++r) {
    float anchor = floor(bone[0][r].w);

    float t = sub32(bone[0][r].w,anchor);
    t = mad32(t,w0,anchor);

    float d1 = sub32(bone[1][r].w,anchor);
    t = mac32(w1,d1,t);

    float d2 = sub32(bone[2][r].w,anchor);
    t = mac32(w2,d2,t);

    float d3 = sub32(bone[3][r].w,anchor);
    t = mac32(w3,d3,t);

    t = mac32(M[r][0],localPosition.x,t);
    t = mac32(M[r][1],localPosition.y,t);
    t = mac32(M[r][2],localPosition.z,t);

    skinnedPosition[r] = t;
}

// Final position MAC PCs:
//
//                 x          y          z
//   SKIN       258,278,28c 23c,26c,284 24c,274,288
//   RIGID      264,280,294 24c,270,28c 25c,278,290
```

```glsl
// VS deformation loop
//
// This modifies position only. It does not modify the blended matrix,
// normal or tangent.
//
// All tests use the original skinnedPosition. Displacements accumulate,
// then are added once after the loop.

int deformationCount = I(VSV[0x00],0x12c); // 114 / 114

if (deformationCount > 0) {
    vec3 displacementSum = vec3(0);

    for (int i=0; i<deformationCount; ++i) {
        vec4 plane = F4(VSV[0x00],0x0d0+16*i); // 2d4 / 2dc
        vec4 origin = F4(VSV[0x00],0x050+16*i); // 2d8 / 2e0
        vec4 shift = F4(VSV[0x00],0x090+16*i);  // 2dc / 2e4

        // Deliberately not dot(plane.xyz,position-origin.xyz).
        //
        // The unusual coordinate order is present in the program:
        // plane.x multiplies position.x,
        // plane.y multiplies position.y,
        // plane.z multiplies position.z.
        float side = mul32(plane.x,skinnedPosition.x);
        side = mac32(-plane.x,origin.x,side);
        side = mac32(plane.y,skinnedPosition.y,side);
        side = mac32(-plane.y,origin.y,side);
        side = mac32(plane.z,skinnedPosition.z,side);
        side = mac32(-plane.z,origin.z,side);

        float selected = side >= 0 ? 1 : 0;
        vec3 shiftScaled = mul32(shift.xyz,shift.w);

        displacementSum =
            mac32(selected,shiftScaled,displacementSum);
    }

    skinnedPosition = add32(displacementSum,skinnedPosition);
}
```

```glsl
// VS position, view vector, projection and fog

uint vertexFlags = U(VSV[0x40],0);

vec3 worldPosition =
    add32(F3(VSV[0x10],0x170),skinnedPosition);

vec3 clipInputPosition = worldPosition;

if (!(has(vertexFlags,0x00080000) &&
     !has(vertexFlags,0x00100000))) {
    // Conditional load 3ac / 3b4.
    clipInputPosition =
        sub32(worldPosition,F3(VSV[0x30],0x220));
}

vec3 lightingPosition = worldPosition;

if (has(vertexFlags,0x00080000)) {
    lightingPosition =
        add32(F3(VSV[0x10],0x160),worldPosition);
}

// The second camera load, 410 / 418, executes unconditionally.
// The view vector is computed even when vertex fog is disabled.
vec3 cameraDelta =
    sub32(lightingPosition,F3(VSV[0x30],0x220));

float cameraDistance2 = mul32(cameraDelta.x,cameraDelta.x);
cameraDistance2 =
    mad32(cameraDelta.y,cameraDelta.y,cameraDistance2);
cameraDistance2 =
    mac32(cameraDelta.z,cameraDelta.z,cameraDistance2);

vec3 awayFromCamera =
    mul32(rsq32(cameraDistance2),cameraDelta);

vec4 viewX = F4(VSV[0x30],0x40);
vec4 viewY = F4(VSV[0x30],0x50);
vec4 viewZ = F4(VSV[0x30],0x60);

float ex = mul32(viewX.x,clipInputPosition.x);
ex = mac32(viewX.y,clipInputPosition.y,ex);
ex = mac32(viewX.z,clipInputPosition.z,ex);

float ey = mul32(viewY.x,clipInputPosition.x);
ey = mac32(viewY.y,clipInputPosition.y,ey);
ey = mac32(viewY.z,clipInputPosition.z,ey);

float ez = mul32(viewZ.x,clipInputPosition.x);
ez = mac32(viewZ.y,clipInputPosition.y,ez);
ez = mad32(viewZ.z,clipInputPosition.z,ez);

vec4 projectionW = F4(VSV[0x30],0x30);

float cw = mad32(projectionW.x,ex,projectionW.w);
cw = mac32(projectionW.y,ey,cw);
cw = mad32(projectionW.z,ez,cw);

vec4 projectionX = F4(VSV[0x30],0x00);
vec4 projectionY = F4(VSV[0x30],0x10);
vec4 projectionZ = F4(VSV[0x30],0x20);

float cx = mad32(projectionX.x,ex,projectionX.w);
float cy = mad32(projectionY.x,ex,projectionY.w);
float cz = mac32(projectionZ.x,ex,projectionZ.w);

cx = mac32(projectionX.y,ey,cx);
cy = mac32(projectionY.y,ey,cy);
cz = mac32(projectionZ.y,ey,cz);

cx = mac32(projectionX.z,ez,cx);
cy = mac32(projectionY.z,ez,cy);
cz = mac32(projectionZ.z,ez,cz);

pos0 = vec4(cx,cy,cz,cw); // export 644 / 64c

vec3 transmission = vec3(1);
vec3 scattering = vec3(0);

if (has(vertexFlags,0x00020000)) {
    vec4 fog0 = F4(VSDirect0,0x40);
    vec4 fog1 = F4(VSDirect0,0x50);
    vec4 fog2 = F4(VSDirect0,0x60);
    vec4 fog3 = F4(VSDirect0,0x70);

    float directionDot =
        D3(F3(VSDirect1,0x80),awayFromCamera);

    float denominator =
        mac32(fog2.w,-directionDot,fog1.w);

    float r = rsq32(denominator);
    float r2 = mul32(r,r);
    float r3 = mul32(r2,r);

    float angular =
        mad32(directionDot,directionDot,fog3.w);

    vec3 phase = mul32(fog1.xyz,angular);
    phase = mac32(fog2.xyz,r3,phase);

    vec3 tintedPhase = mul32(fog3.xyz,phase);

    vec3 exponent = mul32(fog0.xyz,cw);
    exponent = mul32(B(0x3fb8aa3b),exponent);
    transmission = exp2_32(exponent);

    // Preserve multiply, then subtracting MAC.
    scattering = mul32(fog3.w,tintedPhase);
    scattering =
        mac32(tintedPhase,-transmission,scattering);
}

if (has(vertexFlags,0x00040000))
    scattering = vec3(0);
```

```glsl
// VS normal/tangent skinning and exports
//
// Decode before applying the blended linear matrix.
// No normalization or inverse transpose is performed in the VS.
// No deformation-loop correction is applied to either direction.

vec3 objectNormal =
    mad32(B(0x40008102),encodedNormal.xyz,vec3(-1));

vec3 objectTangent =
    mad32(B(0x40008102),encodedTangent.xyz,vec3(-1));

float handedness =
    mad32(B(0x40008102),encodedTangent.w,-1);

vec3 skinnedNormal;
vec3 skinnedTangent;

for (int r=0; r<3; ++r) {
    float n = mul32(M[r][0],objectNormal.x);
    n = mad32(M[r][1],objectNormal.y,n);
    n = mac32(M[r][2],objectNormal.z,n);

    float t = mul32(M[r][0],objectTangent.x);
    t = mad32(M[r][1],objectTangent.y,t);
    t = mac32(M[r][2],objectTangent.z,t);

    skinnedNormal[r] = n;
    skinnedTangent[r] = t;
}

// VS export param N <-> PS attrN, both pairs:
//
// param0 / attr0:
//   xyz = vertex fog transmission
//   w   = lightingPosition.z
//
// param1 / attr1:
//   xyz = vertex fog scattering
//   w   = 0
//
// param2 / attr2:
//   xy  = materialUV, with no shader UV offset
//   z   = SKIN: encodedNormal.w, unchanged
//         RIGID: 0
//   w   = 0
//
// param3 / attr3:
//   xyz = awayFromCamera, normalized at the vertex
//   w   = lightingPosition.x
//
// param4 / attr4:
//   xyz = skinnedNormal
//   w   = lightingPosition.y
//
// param5 / attr5:
//   xyz = skinnedTangent
//   w   = handedness
//
// No param6 or param7 exports.
//
// Export PCs:
//
//              SKIN   RIGID
//   param5      64c    654
//   param4      654    65c
//   param3      65c    664
//   param2      664    66c
//   param1      66c    674
//   param0      674    67c
//
// PS reads attr0.xyzw, attr1.xyz, attr2.xy,
// attr3.xyzw, attr4.xyzw, attr5.xyzw.
// SKIN additionally reads attr2.z; RIGID does not.
```

```glsl
// PS constant-load manifest
//
// Each row is buffer, byte offset, dword count, SKIN PC / RIGID PC.
// "-" means no corresponding load in that program.
//
// PSV[0x60]  0x0dc             1       024 / 020
// PSV[0x60]  0x0d0+16*i        4       074 / 074
// PSV[0x60]  0x050+16*i        4       078 / 078
// PSV[0x60]  0x090+16*i        4       0d4 / 0d4
//
// PSFlags    0x000             1       120 / 180
// PSView     0x218             2       124 / 1cc
// PSV[0x10]  0x0f0             4       16c / 2c8
// PSV[0x70]  0x0ac             1       258 / 1ac
// PSV[0x90]  0x070             4       260 / -
// PSV[0x10]  0x000             4       270 / a24
// PSV[0x70]  0x088             1       274 / b6c
//
// PSV[0x30]  0x0a0             4       43c / 378
// PSV[0x30]  0x020             4       440 / 37c
// PSV[0x30]  0x0b0             4       56c / 49c
// PSV[0x30]  0x030             4       570 / 4a0
// PSV[0x30]  0x0c0             4       6a0 / 5c4
// PSV[0x30]  0x040             4       6a4 / 5c8
// PSV[0x30]  0x0d0             4       7e0 / 6f8
// PSV[0x30]  0x050             4       7e4 / 6fc
//
// PSView     0x22c             1       b2c / 908
// PSV[0x20]  0x4f0            16       b30 / 8f8
// PSV[0x20]  0x530            16       b3c / 900
// PSV[0x30]  0x080             8       bbc / 9ec
// PSV[0x70]  0x060             8       bd0 / -
// PSV[0x70]  0x068             2       -   / 9c4
// PSV[0x00]  0x040             2       be4 / 914
// PSV[0x00]  0x048             1       be8 / 9bc
// PSV[0x90]  0x134             1       bec / 9c8
// PSV[0x30]  0x060             4       bf4 / 9fc
// PSV[0x00]  0x00c             1       c0c / a20
// PSV[0x30]  0x120             4       c10 / a28
// PSV[0x90]  0x020             4       c44 / -
// PSV[0x20]  0x5b0             4       c94 / d84
// PSV[0x60]  0x110             4       d8c / b84
// PSV[0x30]  0x100             4       da8 / 9f8
// PSV[0x30]  0x000             8       e58 / a10
// PSV[0x30]  0x0e0             4       e64 / 9e8
// PSV[0x70]  0x090             8       e68 / aa4
// PSV[0x60]  0x000             8       e6c / b70
// PSV[0x20]  0x570            16       e70 / cc0
//
// PSV[0x00]  0x000            16      12fc / 100c
//
// PSView     0x220             4      143c / 114c
// PSV[0x10]  0x010             8      1440 / 1150
//
// PSV[0x50]  0x01c             1      14c4 / 11d4
// PSV[0x70]  0x09c             1      14c8 / 11d8
// PSV[0x60]  0x00c             1      14cc / 11dc
// PSFlags    0x000             1      14d0 / 11e0
// PSV[0x10]  0x030             4      14d4 / 11e4
//
// PSView     0x218             2      153c / 124c
// PSV[0x00]  0x02c             1      1540 / 1250
// PSV[0x00]  0x284             4      1544 / 1254
// PSV[0x00]  0x03c             1      1548 / 1258
// PSV[0x00]  0x26c             1      154c / 125c
// PSV[0x00]  0x00c             1      1550 / 1260
//
// PSV[0x00]  0x270             2      1694 / 13a4
// PSV[0x00]  0x054             1      1698 / 13a8
// PSV[0x00]  0x260             2      1700 / 1410
// PSV[0x00]  0x268             1      1704 / 1414
// PSV[0x00]  0x2a0             4      17e0 / 14e8
// PSV[0x00]  0x278             2      17e4 / 14ec
// PSV[0x00]  0x058             1      17e8 / 14f0
// PSV[0x00]  0x280             1      17ec / 14f4
// PSV[0x00]  0x294             2      18a8 / 15b8
// PSV[0x00]  0x29c             1      18ac / 15bc
//
// PSFlags    0x000             1      18fc / 160c
// PSV[0x60]  0x11c             1      1900 / 1610
// PSView     0x230             4      1928 / 1638
// PSView     0x22c             1      19a0 / 16b0
//
// Important component meanings:
//
// PSV[0x10]:
//   0x000 exposure
//   0x008 screen-lighting divisor
//   0x00c multiplier for the material's Fresnel floor
//   0x010..0x018 distance-fog RGB
//   0x028 distance-fog scale
//   0x02c distance-fog bias
//   0x030..0x03c final RGB/alpha scales
//   0x0f0..0x0fc screen-shadow RG scale/bias
//
// PSV[0x20]:
//   0x4f0 ambient-polynomial gain
//   0x500..0x53c red polynomial coefficients
//   0x540..0x57c green polynomial coefficients
//   0x580..0x5bc blue polynomial coefficients
//   0x4f4..0x4fc are overfetch, not lighting terms here.
//
// PSV[0x30]:
//   0x000,0x010 directional-light RGBs
//   0x020..0x050 point RGB/inner-distance vec4s
//   0x060..0x068 upper ambient tint
//   0x080,0x090 directional vectors
//   0x0a0..0x0d0 point position/falloff vec4s
//   0x0e0..0x0e8 lower ambient tint
//   0x100..0x108 grazing/base-colour effect RGB
//   0x10c grazing/base-colour effect gain
//   0x120..0x12c material occlusion-control vec4
//
// PSV[0x60]:
//   0x000..0x008 diffuse multiplier
//   0x00c alpha multiplier
//   0x010..0x018 specular multiplier
//   0x110..0x118 linear colour-blend RGB
//   0x11c linear colour-blend weight;
//         also suppresses the auxiliary output mask
//
// PSV[0x70]:
//   0x068,0x06c grazing-effect controls
//   0x078,0x07c SKIN ramp Y endpoints
//   0x088 material Fresnel-floor control
//   0x090..0x098 diffuse RGB
//   0x09c alpha factor
//   0x0ac exponent scale
//
// PSV[0x90]:
//   0x024 SKIN ramp-correction gain
//   0x02c SKIN ramp terminator control
//   0x070..0x07c SKIN extra-lobe controls
//   0x134 secondary directional diffuse gain
```

```glsl
// PS texture bindings and every sample PC
//
// Meaning             Texture   Sampler                 SKIN / RIGID
//
// Tangent normal XY   T[0x0a0]   S[0x050]                 1c8 / 150
// Specular material   T[0x040]   S[0x020]                 1d0 / 158
// Screen shadow RG    T[0x1a0]   S[0x0d0]                 154 / 2b0
// Material control   T[0x020]   S[0x010]                 c3c / a04
// Base RGBA           T[0x000]   S[0x000]                 d6c / a30
// Screen lighting     T[0x1c0]   ScreenLightSampler       da0 / 90c
//
// Directional ramp    T[0x0c0]   S[0x060]:
//   directional 0                                     cbc / -
//   directional 1                                     cc4 / -
//   shadow-modified directional 0                     e50 / -
//
// Screen depth        T[0x100]   S[0x080]:
//   centre                                           1560 / 1270
//   plus offset                                      1728 / 1438
//   minus offset                                     1740 / 1450
//
// Screen normal       T[0x1e0]   ScreenNormalSampler:
//   centre                                           1568 / 1278
//   plus offset                                      1600 / 1310
//   minus offset                                     1618 / 1328
//
// Normal, specular material, material control and base use attr2.xy
// directly: no shader UV-scale constants.
//
// Specular-material channels:
//   R: scalar specular contribution
//   G: diffuse-colour multiplier and grazing-effect input
//   B: exponent multiplier
//
// Material-control channels:
//   R: blends screen-shadow effects
//   G: weights the additive grazing/base-colour effect
//   B: modulates normal-edge suppression and facing-mask width
//   A: material occlusion input
//
// Screen-shadow requests RG only. It does not read coverage/alpha.
//
// Directional ramp requests RGB; its Y coordinate is derived from
// attr2.z and PSV[0x70] 0x078/0x07c.
```

```glsl
// PS flags
//
// pixelFlags = U(PSFlags,0):
//
//   bit 0,  0x00000001:
//       enable the entire point-light calculation;
//       individual bits 16..19 still select its four lights.
//
//   bit 2,  0x00000004:
//       add attr1.rgb after tone/grading;
//       suppress PS distance fog.
//
//   bit 3,  0x00000008:
//       sample/remap screen-shadow RG;
//       bypass final output RGB/alpha scales.
//
//   bits 16..19, 0x00010000 << i:
//       enable point light i, provided bit0 is also set.
//
//   bit 22, 0x00400000:
//       enable PS distance fog when bit2 is clear.
//
//   bit 23, 0x00800000:
//       final output-colour blend;
//       scale auxiliary mask and mrt1 alpha by its weight.
//
//   bit 25, 0x02000000:
//       skip the screen-space mask.
//
// U(PSView,0x22c):
//   bit0: mrt0 alpha uses scaled rasterThird.
//   bit1: combine remapped shadow RG by multiplication;
//         otherwise combine them with min.
//
// U(PSV[0x00],0x00c):
//   bit0: optional post-tone colour grading.
//   bit1: facing mask.
//   bit2: screen-normal edge mask.
//   bit3: screen-depth edge mask.
//
// Additional branches:
//   deformation-loop continuation and plane-band rejection;
//   SKIN extra-lobe amplitude > 0;
//   facing normal-response threshold > 0.
//
// Point attenuation is computed for each enabled light without the
// stage shader's "lambert > 0" branch.
```

```glsl
// PS deformation inversion and plane-band rejection
//
// Executed before material shading.
// This is a pixel-side inverse walk of the vertex displacement scheme.
//
// The position used by lighting later remains the original interpolated
// position. Only the temporary "unfolded" position is changed here.

vec3 position = vec3(attr3.w,attr4.w,attr0.w);
vec3 unfoldedPosition = position;

float deformationCount = F(PSV[0x60],0x0dc); // 024 / 020

for (int i=0; deformationCount > float(i); ++i) {
    vec4 plane =
        F4(PSV[0x60],0x0d0+16*i); // 074 / 074

    vec4 originAndBand =
        F4(PSV[0x60],0x050+16*i); // 078 / 078

    float side = mul32(plane.x,unfoldedPosition.x);
    side = mac32(-plane.x,originAndBand.x,side);
    side = mac32(plane.y,unfoldedPosition.y,side);
    side = mac32(-plane.y,originAndBand.y,side);
    side = mac32(plane.z,unfoldedPosition.z,side);
    side = mac32(-plane.z,originAndBand.z,side);

    if (abs(side) < originAndBand.w)
        discard;

    vec4 shift =
        F4(PSV[0x60],0x090+16*i); // 0d4 / 0d4

    float selected = side >= 0 ? 1 : 0;
    vec3 scaledShift = mul32(shift.xyz,shift.w);

    unfoldedPosition =
        mac32(selected,-scaledShift,unfoldedPosition);
}

// Implementation detail of the original:
// rejected lanes are removed from saved s[14:15], while WQM execution
// can continue for derivatives. The saved surviving mask is restored
// immediately before exports.
//
// Thus there is real pixel rejection here even though there is no
// ordinary kill instruction in the listing.
```

```glsl
// PS normal frame, view and material samples

uint pixelFlags = U(PSFlags,0);
uint viewFlags = U(PSView,0x22c);

vec2 uv = attr2.xy;
vec2 rasterToUV = F2(PSView,0x218);
vec2 rasterUV = mul32(rasterToUV,rasterXY);

vec3 geometricNormal = attr4.xyz;
vec3 tangent = attr5.xyz;

// Cross order: geometric normal × tangent.
// Each subtraction is the original negative-product MAC.
vec3 bitangent;
bitangent.x = mul32(geometricNormal.y,tangent.z);
bitangent.x =
    mac32(tangent.y,-geometricNormal.z,bitangent.x);
bitangent.y = mul32(geometricNormal.z,tangent.x);
bitangent.y =
    mac32(tangent.z,-geometricNormal.x,bitangent.y);
bitangent.z = mul32(geometricNormal.x,tangent.y);
bitangent.z =
    mac32(tangent.x,-geometricNormal.y,bitangent.z);

bitangent = mul32(attr5.w,bitangent);

vec2 encodedXY =
    sample(T[0x0a0],S[0x050],uv).rg; // 1c8 / 150

vec3 specMaterial =
    sample(T[0x040],S[0x020],uv).rgb; // 1d0 / 158

vec2 normalXY = mad32(2,encodedXY,vec2(-1));

float xy2 = mul32(normalXY.x,normalXY.x);
xy2 = mac32(normalXY.y,normalXY.y,xy2);
float normalZ = sqrt32(sat(sub32(1,xy2)));

vec3 unnormalizedN = mul32(tangent,normalXY.x);
unnormalizedN =
    mad32(bitangent,normalXY.y,unnormalizedN);
unnormalizedN =
    mac32(geometricNormal,normalZ,unnormalizedN);

vec3 N = NORM(unnormalizedN);

// Interpolating a vertex-normalized vector does not preserve its length.
vec3 V = NORM(attr3.xyz); // direction away from camera

float exponent =
    max(1,mul32(F(PSV[0x70],0x0ac),specMaterial.b));

float NV = D3(N,V);
float twiceNV = mul32(2,NV);

vec3 reflectedV = mad32(-N,twiceNV,V);

float specularNormalization =
    mad32(B(0x3d22ca2f),exponent,B(0x3daf7aac));

float fresnelFloor =
    mul32(F(PSV[0x10],0x00c),F(PSV[0x70],0x088));

float oneMinusFresnelFloor = sub32(1,fresnelFloor);
```

```glsl
// PS screen-shadow factors
//
// One remap, not the stage shader's two remaps.

float shadowR = 1;
float shadowG = 1;

if (has(pixelFlags,8)) {
    vec2 shadow =
        sample(T[0x1a0],S[0x0d0],rasterUV).rg; // 154 / 2b0

    vec4 remap = F4(PSV[0x10],0x0f0);

    shadowR = mad32(remap.x,shadow.r,remap.y);
    shadowG = mad32(shadow.g,remap.z,remap.w);
}

// No saturation of the remapped values.
float combinedShadow =
    has(viewFlags,2)
    ? mul32(shadowR,shadowG)
    : min(shadowR,shadowG);
```

```glsl
// SKIN-only extra specular lobe
//
// RIGID has no corresponding constants, branch or term.

float extraSpecular = 0;

#ifdef SKIN
vec4 extra = F4(PSV[0x90],0x070); // 260

if (extra.x > 0) {
    vec3 extraDirection;

    extraDirection.x = mad32(-extra.z,V.z,V.x);
    extraDirection.y = add32(extra.w,V.y);
    extraDirection.z = mad32(extra.z,V.x,V.z);

    float len2 = D3(extraDirection,extraDirection);
    float nd = D3(extraDirection,N);

    float lobe = sat(mul32(-rsq32(len2),nd));
    lobe = log2_32(lobe);
    lobe = mul32(extra.y,lobe);
    lobe = mul32(lobe,exponent);
    lobe = exp2_32(lobe);

    extraSpecular = mul32(extra.x,lobe);
}
#endif
```

```glsl
// PS point lighting
//
// L is light-to-surface, so Lambert uses -N·L.
//
// Per-light constant-load PCs:
//
//                   SKIN              RIGID
//   light0 pos/RGB  43c / 440          378 / 37c
//   light1 pos/RGB  56c / 570          49c / 4a0
//   light2 pos/RGB  6a0 / 6a4          5c4 / 5c8
//   light3 pos/RGB  7e0 / 7e4          6f8 / 6fc

vec3 pointDiffuse = vec3(0);
vec3 pointSpecular = vec3(0);
float pointOcclusionLight = 0;

if (has(pixelFlags,1)) {
    vec3 attenuatedColor[4] = vec3[4](
        vec3(0),vec3(0),vec3(0),vec3(0));

    vec3 L[4] = vec3[4](
        vec3(0),vec3(0),vec3(0),vec3(0));

    vec3 H[4] = vec3[4](
        vec3(0),vec3(0),vec3(0),vec3(0));

    float lambert[4] = float[4](0,0,0,0);
    float occlusionLight[4] = float[4](0,0,0,0);

    for (int i=0; i<4; ++i) {
        if (has(pixelFlags,0x00010000u<<i)) {
            vec4 pp = F4(PSV[0x30],0x0a0+16*i);
            vec4 pc = F4(PSV[0x30],0x020+16*i);

            vec3 delta = sub32(position,pp.xyz);
            float distance2 = D3(delta,delta);

            float outside =
                max(0,sub32(sqrt32(distance2),pc.w));

            float falloff = mul32(pp.w,outside);
            float denominator = mad32(falloff,outside,1);

            float attenuation =
                sat(mad32(B(0x3f808102),
                          rcp32(denominator),
                          B(0xbb810205)));

            float inverseDistance = rsq32(distance2);
            L[i] = mul32(delta,inverseDistance);

            lambert[i] = min(1,max(0,-D3(N,L[i])));

            attenuatedColor[i] = mul32(pc.rgb,attenuation);

            vec3 diffuse =
                max(vec3(0),mul32(lambert[i],attenuatedColor[i]));

            pointDiffuse = add32(pointDiffuse,diffuse);

            // Original luminance order and literal bits.
            float luminance =
                mul32(B(0x3e991687),diffuse.r);

            luminance =
                mad32(diffuse.g,B(0x3f1645a2),luminance);

            luminance =
                mad32(diffuse.b,B(0x3de978d5),luminance);

            // An additional Lambert factor is present here.
            occlusionLight[i] = mul32(lambert[i],luminance);

            // Build from delta*inverseDistance + V, preserving the
            // multiply-add construction before normalization.
            H[i] = NORM(mad32(delta,inverseDistance,V));
        }
    }

    for (int i=0; i<4; ++i) {
        float s;

#ifdef SKIN
        float reflectedLobe =
            POW(min(1,max(0,-D3(reflectedV,L[i]))),exponent);

        reflectedLobe =
            mul32(reflectedLobe,specularNormalization);

        float fresnelArgument = sub32(1,D3(H[i],V));
        float fresnel =
            mad32(P5(fresnelArgument),
                  oneMinusFresnelFloor,
                  fresnelFloor);

        s = mul32(reflectedLobe,fresnel);
        s = mul32(s,lambert[i]);
#else
        s = POW(min(1,max(0,-D3(H[i],N))),exponent);
#endif

        if (i==0)
            pointSpecular = mul32(attenuatedColor[i],s);
        else
            pointSpecular =
                mac32(attenuatedColor[i],s,pointSpecular);
    }

    pointSpecular = max(vec3(0),pointSpecular);

    // Original accumulation order, also used when bit0 is clear:
    // (((light0 + light1) + light2) + light3).
    pointOcclusionLight =
        add32(occlusionLight[0],occlusionLight[1]);
    pointOcclusionLight =
        add32(pointOcclusionLight,occlusionLight[2]);
    pointOcclusionLight =
        add32(pointOcclusionLight,occlusionLight[3]);
}
```

```glsl
// PS ambient polynomial
//
// A second normalization of N is used for this calculation.
// Directional Lambert/specular calculations continue to use N above.
//
// The coefficients are a general quadratic polynomial in the direction;
// describing this as an SH-like ambient representation does not require
// assuming a particular coefficient convention.
//
// Each colour owns four consecutive vec4s:
//
//              A       B       C       D
//   red       0x500   0x510   0x520   0x530
//   green     0x540   0x550   0x560   0x570
//   blue      0x580   0x590   0x5a0   0x5b0
//
// Loads are split across 4f0/530/570/5b0, as in the manifest.

vec3 ambientDirection = NORM(N);

float ambientPolynomial(int channel,vec3 n) {
    int baseOffset = 0x500+0x40*channel;

    vec4 A = F4(PSV[0x20],baseOffset+0x00);
    vec4 Bc = F4(PSV[0x20],baseOffset+0x10);
    vec4 C = F4(PSV[0x20],baseOffset+0x20);
    vec4 D = F4(PSV[0x20],baseOffset+0x30);

    float b = D3(Bc.xyz,n);
    float c = D3(C.xyz,n);
    float a = D3(A.xyz,n);

    float linear = add32(D.x,A.w);
    linear = add32(linear,a);

    float value = D.w;
    value = mac32(D.y,n.y,value);
    value = mac32(D.z,n.z,value);
    value = mac32(b,n.y,value);
    value = mac32(Bc.w,n.y,value);
    value = mac32(c,n.z,value);
    value = mac32(C.w,n.z,value);
    value = mac32(linear,n.x,value);

    return max(B(0x3d800000),value); // minimum 1/16
}

vec3 polynomial = vec3(
    ambientPolynomial(0,ambientDirection),
    ambientPolynomial(1,ambientDirection),
    ambientPolynomial(2,ambientDirection));

vec3 lowerTint = F3(PSV[0x30],0x0e0);
vec3 upperTint = F3(PSV[0x30],0x060);

vec3 ambient = mad32(F(PSV[0x20],0x4f0),
                     polynomial,lowerTint);

vec3 tintDifference = sub32(upperTint,lowerTint);

#ifdef SKIN
// SKIN computes the red hemisphere contribution this way:
float redHemisphere = tintDifference.r;
redHemisphere =
    mac32(tintDifference.r,N.y,redHemisphere);
ambient.r = mad32(0.5,redHemisphere,ambient.r);

// Green/blue use the half-range Y expression.
float hemisphere = mad32(0.5,N.y,0.5);
ambient.gb =
    mac32(hemisphere,tintDifference.gb,ambient.gb);
#else
float hemisphere = mad32(0.5,N.y,0.5);
ambient = mac32(hemisphere,tintDifference,ambient);
#endif
```

```glsl
// PS directional lighting, screen light and material control

vec3 direction0 = F3(PSV[0x30],0x080);
vec3 direction1 = F3(PSV[0x30],0x090);

vec3 lightColor0 = F3(PSV[0x30],0x000);
vec3 lightColor1 = F3(PSV[0x30],0x010);

float secondaryDiffuseGain = F(PSV[0x90],0x134);

float rawNL0 = D3(direction0,N);
float rawNL1 = D3(direction1,N);

float nl0 = min(1,max(0,-rawNL0));
float nl1 = min(1,max(0,-rawNL1));

vec4 control =
    sample(T[0x020],S[0x010],uv); // c3c / a04

vec4 base =
    sample(T[0x000],S[0x000],uv); // d6c / a30

vec4 screen =
    sample(T[0x1c0],ScreenLightSampler,rasterUV); // da0 / 90c

float inverseScreenDivisor =
    rcp32(F(PSV[0x10],0x008));

vec3 indirectDiffuse =
    mad32(screen.rgb,inverseScreenDivisor,pointDiffuse);

vec3 indirectSpecular =
    mac32(inverseScreenDivisor,vec3(screen.a),pointSpecular);

float materialShadow =
    mad32(control.r,sub32(combinedShadow,1),1);

vec3 primaryDiffuse = mul32(lightColor0,nl0);
primaryDiffuse = mul32(materialShadow,primaryDiffuse);

vec3 secondaryDiffuse = mul32(lightColor1,nl1);

vec3 directionalDiffuse =
    mac32(secondaryDiffuseGain,secondaryDiffuse,primaryDiffuse);

vec3 diffuseLighting = add32(ambient,directionalDiffuse);
```

```glsl
// SKIN-only directional ramp correction
//
// All three lookups execute in the SKIN program.
// The correction is not gated by a pixelFlags bit.

#ifdef SKIN
float rampY =
    mad32(sub32(F(PSV[0x70],0x078),
                F(PSV[0x70],0x07c)),
          attr2.z,
          F(PSV[0x70],0x07c));

vec3 ramp0 =
    sample(T[0x0c0],S[0x060],
           vec2(mad32(-0.5,rawNL0,0.5),rampY)).rgb; // cbc

vec3 ramp1 =
    sample(T[0x0c0],S[0x060],
           vec2(mad32(-0.5,rawNL1,0.5),rampY)).rgb; // cc4

float shadowRBlend =
    mul32(control.r,sub32(shadowR,1));

float modifiedIncidence =
    mad32(-rawNL0,shadowRBlend,-rawNL0);

float terminatorIncidence =
    mul32(F(PSV[0x90],0x02c),-rawNL0);

float shadowRampX =
    mad32(0.5,min(terminatorIncidence,modifiedIncidence),0.5);

vec3 shadowRamp =
    sample(T[0x0c0],S[0x060],
           vec2(shadowRampX,rampY)).rgb; // e50

vec3 adjustedRamp =
    mad32(ramp0,materialShadow,-shadowRamp);

adjustedRamp =
    mac32(nl0,adjustedRamp,shadowRamp);

vec3 rampDiffuse = mul32(lightColor0,adjustedRamp);
rampDiffuse = mac32(lightColor1,ramp1,rampDiffuse);

// Subtract the ordinary directional diffuse already included above.
vec3 rampCorrection =
    sub32(rampDiffuse,directionalDiffuse);

float shadowGBlend =
    mul32(control.r,sub32(shadowG,1));

float rampGain =
    mad32(F(PSV[0x90],0x024),
          shadowGBlend,
          F(PSV[0x90],0x024));

diffuseLighting =
    mac32(rampCorrection,rampGain,diffuseLighting);
#endif

// Directional-response ramp, shared by both shaders.
//
// Constants are PSV[0x00] 40/44/48, not the stage's
// diffuseScreenBlendBase at 4c.

float responseInverseWidth =
    rcp32(sub32(F(PSV[0x00],0x048),
                F(PSV[0x00],0x044)));

float response =
    mul32(-rawNL0,responseInverseWidth);

response =
    sat(mac32(-F(PSV[0x00],0x044),
              responseInverseWidth,response));

float directionalResponse =
    mad32(F(PSV[0x00],0x040),sub32(response,1),1);

diffuseLighting =
    mad32(diffuseLighting,directionalResponse,indirectDiffuse);
```

```glsl
// PS directional specular
//
// Both programs use reflected-direction power lobes for directional
// lights, with normalization, fifth-power Fresnel, and Lambert.
//
// RIGID point specular above uses a different, half-vector power lobe.

float directionalSpecular(vec3 direction,float nl) {
    float lobe =
        POW(min(1,max(0,-D3(direction,reflectedV))),exponent);

    lobe = mul32(lobe,specularNormalization);

    vec3 halfDirection = NORM(add32(direction,V));
    float f = sub32(1,D3(halfDirection,V));

    float fresnel =
        mad32(P5(f),oneMinusFresnelFloor,fresnelFloor);

    lobe = mul32(lobe,fresnel);
    return mul32(lobe,nl);
}

float ds0 = directionalSpecular(direction0,nl0);
float ds1 = directionalSpecular(direction1,nl1);

vec3 directionalSpecularLighting =
    mul32(lightColor0,materialShadow);

directionalSpecularLighting =
    mad32(directionalSpecularLighting,ds0,vec3(extraSpecular));

directionalSpecularLighting =
    mac32(lightColor1,ds1,directionalSpecularLighting);

vec3 specularMultiplier =
    mul32(F3(PSV[0x60],0x010),F3(PSV[0x70],0x0a0));

vec3 specularLighting =
    mul32(specularMultiplier,directionalSpecularLighting);

#ifdef SKIN
float specularFloor = sat(sub32(specMaterial.g,attr2.z));

float primarySpecularWeight =
    mad32(materialShadow,nl0,specularFloor);

float primaryShadowedLambert = mul32(materialShadow,nl0);

primarySpecularWeight =
    mac32(primaryShadowedLambert,
          -specularFloor,
          primarySpecularWeight);

specularLighting =
    mad32(specularLighting,primarySpecularWeight,indirectSpecular);
#else
specularLighting =
    add32(specularLighting,indirectSpecular);
#endif
```

```glsl
// PS grazing/base-colour effect
//
// This is an additive change to base.rgb squared.
// It is not an extra light added after material multiplication.

float geometricFacing =
    sat(add32(1,D3(geometricNormal,V)));

float normalFacing = sat(add32(1,NV));
float facingProduct = mul32(geometricFacing,normalFacing);

float effectExponent = add32(2,F(PSV[0x70],0x06c));

float effect = log2_32(facingProduct);
effect = mul32(effect,effectExponent);
effect = exp2_32(effect);

effect = mul32(specMaterial.g,effect);
effect = mul32(F(PSV[0x30],0x10c),effect);

#ifdef SKIN
effect = mul32(effect,add32(2,F(PSV[0x70],0x068)));
#else
effect = mul32(effect,add32(B(0x40200000),
                          F(PSV[0x70],0x068))); // 2.5 + constant
effect = mul32(B(0x3dcccccd),effect);            // 0.1
#endif

vec3 adjustedBaseSquared = mul32(base.rgb,base.rgb);

vec3 effectColor =
    mul32(F3(PSV[0x30],0x100),control.g);

adjustedBaseSquared =
    mac32(effect,effectColor,adjustedBaseSquared);
```

```glsl
// PS material occlusion adjustment
//
// This uses control.a and a light-dependent suppression.
// No stage-style directional/indirect shadow-tint RGBs are present.

float primaryLightLuminance =
    mul32(B(0x3e991687),lightColor0.r);

primaryLightLuminance =
    mad32(lightColor0.g,B(0x3f1645a2),primaryLightLuminance);

primaryLightLuminance =
    mad32(lightColor0.b,B(0x3de978d5),primaryLightLuminance);

float occlusionLight =
    max(0,sub32(F(PSV[0x30],0x124),rawNL0));

occlusionLight =
    mul32(F(PSV[0x30],0x128),occlusionLight);

occlusionLight =
    mac32(pointOcclusionLight,
          rcp32(primaryLightLuminance),
          occlusionLight);

float remainingShadow =
    mad32(-F(PSV[0x30],0x12c),
          materialShadow,
          F(PSV[0x30],0x12c));

float upperLimit = sub32(1,remainingShadow);

float occlusionSuppression =
    max(0,min(upperLimit,occlusionLight));

float materialOcclusion = sub32(1,control.a);
materialOcclusion =
    mac32(occlusionSuppression,
          -materialOcclusion,
          materialOcclusion);

float occlusionFactor =
    sat(mad32(-F(PSV[0x30],0x120),materialOcclusion,1));
```

```glsl
// PS material composition, linear colour blend and tone curve

vec3 diffuseMultiplier =
    mul32(F3(PSV[0x70],0x090),specMaterial.g);

diffuseMultiplier =
    mul32(F3(PSV[0x60],0x000),diffuseMultiplier);

vec3 diffuseMaterial =
    mul32(diffuseMultiplier,adjustedBaseSquared);

vec3 linearColor = mul32(diffuseMaterial,diffuseLighting);

linearColor =
    mac32(specularLighting,specMaterial.r,linearColor);

linearColor = mul32(occlusionFactor,linearColor);

// Separate material-level blend before exposure/tone mapping.
vec3 linearBlendColor = F3(PSV[0x60],0x110);
float linearBlendWeight = F(PSV[0x60],0x11c);

linearColor =
    mad32(linearBlendColor,linearBlendWeight,linearColor);

linearColor =
    mac32(-linearBlendWeight,
          // This operand is the pre-blend, occlusion-scaled colour.
          occlusionScaledMaterialColor,
          linearColor);

// Equivalent meaning:
//   blendColor*weight + oldColor - weight*oldColor.
// Preserve the two multiply-add instructions in an implementation.

float toneComponent(float material,float transmission) {
    float exposureTransmission =
        mul32(F(PSV[0x10],0),transmission);

    // Original v_mul with mul:2 output modifier.
    float x = mul32(exposureTransmission,material);
    x = mul32(2,x);

    float denominatorInner = mad32(B(0x3e19999a),x,0.5);
    float denominator =
        mad32(x,denominatorInner,B(0x3d75c290));

    float numeratorInner =
        mad32(B(0x3e19999a),x,B(0x3d4ccccd));

    float numerator =
        mad32(numeratorInner,x,B(0x3b83126f));

    float ratio = mul32(rcp32(denominator),numerator);

    return sqrt32(
        sat(mad32(B(0x3fb0852e),ratio,B(0xbdbc49ca))));
}

// Vertex fog transmission multiplies the entire linear result here.
// It is not confined to baked/point lighting.
vec3 toned = vec3(
    toneComponent(linearColor.r,attr0.r),
    toneComponent(linearColor.g,attr0.g),
    toneComponent(linearColor.b,attr0.b));
```

```glsl
// PS optional post-tone colour grading
//
// Controlled by PSV[0x00] flags bit0, not pixelFlags bit0.
//
// The brightness driver uses unshadowed directional Lambert RGB,
// point diffuse/specular, and divided screen RGB/alpha.
// It does not include ambient polynomial or material RGB multipliers.

uint screenMaskFlags = U(PSV[0x00],0x00c);

if (has(screenMaskFlags,1)) {
    vec3 brightnessRGB =
        add32(indirectDiffuse,indirectSpecular);

    brightnessRGB =
        add32(brightnessRGB,mul32(lightColor0,nl0));

    brightnessRGB =
        add32(brightnessRGB,mul32(lightColor1,nl1));

    float gb = add32(brightnessRGB.g,brightnessRGB.b);
    float rThird = mul32(B(0x3eaaaaab),brightnessRGB.r);

    float brightness =
        mad32(gb,B(0x3eaaaaab),rThird);

    float oneMinus = sub32(1,brightness);
    float lowWeight = sat(mul32(oneMinus,oneMinus));

    float middleWeight = mad32(-oneMinus,oneMinus,1);
    middleWeight =
        sat(mac32(brightness,-brightness,middleWeight));

    float highWeight = sat(mul32(brightness,brightness));

    // Loaded as one 16-dword block:
    //   SKIN 12fc, RIGID 100c.
    vec3 baseExponent = F3(PSV[0x00],0x00);
    vec4 redControls = F4(PSV[0x00],0x10);
    vec4 greenControls = F4(PSV[0x00],0x20);
    vec4 blueControls = F4(PSV[0x00],0x30);

    // The common black-level parameter is redControls.w (+0x1c).
    float black = redControls.w;

    float inverseRange = sub32(1,black);
    inverseRange = sub32(inverseRange,black);
    inverseRange = rcp32(inverseRange);

    vec3 gradeExponent;
    for (int c=0; c<3; ++c) {
        vec3 controls =
            c==0 ? redControls.xyz :
            c==1 ? greenControls.xyz :
                   blueControls.xyz;

        float e = mul32(controls.x,lowWeight);
        e = mac32(controls.y,middleWeight,e);
        e = mac32(controls.z,highWeight,e);
        gradeExponent[c] = add32(baseExponent[c],e);
    }

    vec3 adjusted = sat(mad32(inverseRange,toned,vec3(-black)));

    toned = sat(exp2_32(mul32(log2_32(adjusted),gradeExponent)));
}
```

```glsl
// PS scattering, distance fog, alpha and output scales

vec3 outputColor = toned;
float distanceFogAmount = 0;

if (has(pixelFlags,4))
    outputColor = add32(toned,attr1.rgb);

if (has(pixelFlags,0x00400000) && !has(pixelFlags,4)) {
    vec3 camera = F3(PSView,0x220); // 143c / 114c

    vec3 delta = sub32(camera,position);
    float distance = sqrt32(D3(delta,delta));

    distanceFogAmount =
        sat(mac32(F(PSV[0x10],0x028),
                  distance,
                  F(PSV[0x10],0x02c)));

    outputColor =
        mad32(F3(PSV[0x10],0x010),distanceFogAmount,toned);

    outputColor =
        mac32(distanceFogAmount,-toned,outputColor);
}

float materialAlpha =
    mul32(F(PSV[0x70],0x09c),F(PSV[0x50],0x01c));

materialAlpha =
    mul32(F(PSV[0x60],0x00c),materialAlpha);

materialAlpha = mul32(materialAlpha,base.a);

if (!has(pixelFlags,8)) {
    outputColor =
        mul32(F3(PSV[0x10],0x030),outputColor);

    materialAlpha =
        mul32(F(PSV[0x10],0x03c),materialAlpha);
}
```

```glsl
// PS auxiliary screen-space mask
//
// The overall structure resembles the stage mask, but the constant
// layout and two control-texture terms differ.
//
// All PCs below are SKIN / RIGID.

float auxiliaryMask = 0;

if (!has(pixelFlags,0x02000000)) {
    vec2 maskRasterToUV = F2(PSView,0x218); // 153c / 124c
    vec2 maskUV = mul32(maskRasterToUV,rasterXY);

    float depthAdd = F(PSV[0x00],0x02c);       // 1540 / 1250
    float depthNumerator = F(PSV[0x00],0x03c); // 1548 / 1258

    vec4 distanceControls =
        F4(PSV[0x00],0x284); // 1544 / 1254

    float distanceStart = distanceControls.x;
    float distanceNear = distanceControls.y;
    float distanceEnd = distanceControls.z;
    float minimumRadius = distanceControls.w;

    float depthSample =
        sample(T[0x100],S[0x080],maskUV).r; // 1560 / 1270

    float inverseDepthDenominator =
        rcp32(add32(depthAdd,depthSample));

    float depthMinusStart =
        mad32(depthNumerator,
              inverseDepthDenominator,
              -distanceStart);

    float negativeDepth =
        mul32(-depthNumerator,inverseDepthDenominator);

    float radiusFade =
        sat(mad32(-depthMinusStart,
                  rcp32(sub32(distanceEnd,distanceStart)),1));

    float nearInverse =
        rcp32(sub32(distanceStart,distanceNear));

    float nearRamp = mul32(-negativeDepth,nearInverse);
    nearRamp =
        sat(mac32(-distanceNear,nearInverse,nearRamp));

    vec3 encodedScreenNormal =
        sample(T[0x1e0],ScreenNormalSampler,maskUV).rgb;
        // 1568 / 1278

    vec3 screenNormal =
        mad32(2,encodedScreenNormal,vec3(-1));

    vec2 screenDirection =
        mul32(maskRasterToUV,
              vec2(screenNormal.x,-screenNormal.y));

    float normalRadius =
        mul32(F(PSV[0x00],0x26c),radiusFade);
        // 154c / 125c

    normalRadius = max(minimumRadius,normalRadius);
    normalRadius = mul32(nearRamp,normalRadius);

    vec3 normalPlus =
        sample(T[0x1e0],ScreenNormalSampler,
               mad32(screenDirection,normalRadius,maskUV)).rgb;
        // 1600 / 1310

    vec3 normalMinus =
        sample(T[0x1e0],ScreenNormalSampler,
               mad32(-screenDirection,normalRadius,maskUV)).rgb;
        // 1618 / 1328

    // Both offset samples occur even when all feature bits are clear.
    //
    // Preserve the original response construction:
    // multiply encoded samples by screenNormal, double the products,
    // sum them, then subtract sum(screenNormal).
    float normalResponse(vec3 encoded) {
        float yz = mul32(encoded.y,screenNormal.y);
        float zz = mul32(encoded.z,screenNormal.z);

        float response = mul32(encoded.x,screenNormal.x);
        response = mul32(2,response);
        response = mac32(2,yz,response);
        response = mac32(2,zz,response);

        float normalSum = add32(screenNormal.x,screenNormal.y);
        normalSum = add32(normalSum,screenNormal.z);

        return sub32(response,normalSum);
    }

    float responsePlus = normalResponse(normalPlus);
    float responseMinus = normalResponse(normalMinus);

    float normalEdge = 0;
    float depthEdge = 0;
    float facingMask = 0;

    uint maskFlags = U(PSV[0x00],0x00c); // 1550 / 1260

    if (has(maskFlags,4)) {
        vec2 k = F2(PSV[0x00],0x270); // 1694 / 13a4
        float controlScale = F(PSV[0x00],0x054);
        // 1698 / 13a8

        float agreement = mad32(0.5,responseMinus,0.5);

        // The original MAC has div:2 on its result.
        agreement = mac32(responsePlus,agreement,agreement);
        agreement = mul32(0.5,agreement);

        float testedAgreement =
            mad32(agreement,agreement,B(0x3b808081));

        float threshold = mad32(-k.y,nearRamp,1);
        float passed = testedAgreement >= threshold ? 1 : 0;

        float gate = sat(mac32(agreement,agreement,passed));
        float suppression = mad32(sub32(1,k.x),gate,k.x);

        normalEdge =
            mad32(-nearRamp,suppression,nearRamp);

        // New relative to the stage mask:
        normalEdge =
            mul32(mad32(-control.b,controlScale,1),normalEdge);
    }

    if (has(maskFlags,8)) {
        vec2 k = F2(PSV[0x00],0x260); // 1700 / 1410
        float thresholdScale = F(PSV[0x00],0x268);
        // 1704 / 1414

        float radius = mul32(k.x,radiusFade);
        radius = max(minimumRadius,radius);
        radius = mul32(nearRamp,radius);

        float zp =
            sample(T[0x100],S[0x080],
                   mad32(screenDirection,radius,maskUV)).r;
            // 1728 / 1438

        float zm =
            sample(T[0x100],S[0x080],
                   mad32(-screenDirection,radius,maskUV)).r;
            // 1740 / 1450

        float response = mul32(2,negativeDepth);

        response =
            mac32(depthNumerator,
                  rcp32(add32(depthAdd,zp)),response);

        response =
            mac32(depthNumerator,
                  rcp32(add32(depthAdd,zm)),response);

        response = mad32(-0.5,response,1);

        float threshold =
            mad32(-thresholdScale,nearRamp,1);

        float passed =
            add32(response,B(0x3b808081)) >= threshold ? 1 : 0;

        float gate = sat(add32(passed,response));
        float suppression = mad32(sub32(1,k.y),gate,k.y);

        depthEdge =
            mad32(-nearRamp,suppression,nearRamp);
    }

    if (has(maskFlags,2)) {
        vec4 distanceFacing = F4(PSV[0x00],0x2a0);
        // 17e0 / 14e8

        vec2 facingControls = F2(PSV[0x00],0x278);
        // 17e4 / 14ec

        float controlScale = F(PSV[0x00],0x058);
        // 17e8 / 14f0

        float farThreshold = F(PSV[0x00],0x280);
        // 17ec / 14f4

        float facing = min(1,max(0,-D3(N,V)));

        float farDistance = distanceFacing.x;

        float nearDistance =
            min(distanceFacing.y,
                add32(farDistance,B(0xba83126f)));

        float invRange =
            rcp32(sub32(farDistance,nearDistance));

        float facingDistance =
            mul32(-negativeDepth,invRange);

        facingDistance =
            sat(mac32(nearDistance,-invRange,facingDistance));

        float width =
            mad32(facingDistance,
                  sub32(facingControls.y,distanceFacing.z),
                  distanceFacing.z);

        // New relative to the stage mask:
        float widthRamp =
            mad32(-control.b,controlScale,1);

        widthRamp = mul32(widthRamp,nearRamp);

        float denominator =
            mad32(widthRamp,width,B(0x3b808081));

        float normalizedFacing =
            sat(mul32(rcp32(denominator),
                      add32(facing,B(0x3b808081))));

        float gain =
            mul32(sub32(1,facingControls.x),normalizedFacing);

        float suppression =
            sat(mad32(gain,normalizedFacing,facingControls.x));

        float threshold =
            mad32(sub32(farThreshold,distanceFacing.w),
                  facingDistance,distanceFacing.w);

        float response =
            mad32(-responsePlus,responseMinus,1);

        float candidate =
            mac32(suppression,-nearRamp,nearRamp);

        float gated =
            mul32(response >= threshold ? 1 : 0,candidate);

        facingMask = threshold > 0 ? gated : candidate;
    }

    auxiliaryMask = max(depthEdge,max(facingMask,normalEdge));

    vec2 fade = F2(PSV[0x00],0x294); // 18a8 / 15b8
    float outputThreshold = F(PSV[0x00],0x29c);
    // 18ac / 15bc

    float fadeInverse = rcp32(sub32(fade.x,distanceStart));

    float distanceFade =
        sat(mad32(-depthMinusStart,fadeInverse,1));

    distanceFade = max(sub32(1,fade.y),distanceFade);

    auxiliaryMask = mul32(distanceFade,auxiliaryMask);
    auxiliaryMask =
        mac32(auxiliaryMask,-distanceFogAmount,auxiliaryMask);

    auxiliaryMask =
        mul32(outputThreshold <= auxiliaryMask ? 1 : 0,
              auxiliaryMask);
}

// This suppression is outside the bit25 mask branch.
auxiliaryMask =
    mul32(sub32(1,F(PSV[0x60],0x11c)),auxiliaryMask);
```

```glsl
// PS final blend and exports

float mrt1Alpha = materialAlpha;

if (has(pixelFlags,0x00800000)) {
    vec4 blend = F4(PSView,0x230); // 1928 / 1638

    outputColor = mad32(blend.w,outputColor,blend.rgb);
    outputColor = mac32(-blend.rgb,blend.w,outputColor);

    auxiliaryMask = mul32(blend.w,auxiliaryMask);
    mrt1Alpha = mul32(blend.w,materialAlpha);
}

float rasterAlpha =
    sat(mul32(B(0x3a4d9a68),rasterThird));

float mrt0Alpha =
    has(U(PSView,0x22c),1) ? rasterAlpha : materialAlpha;
    // final flag load: 19a0 / 16b0

// Logical outputs:
//
//   mrt0 = vec4(outputColor,mrt0Alpha)
//   mrt1 = vec4(auxiliaryMask,0,0,mrt1Alpha)
//
// Compressed export packing:
//
// H2_RTZ(a,b) converts each component to half toward zero,
// placing a in low 16 bits and b in high 16 bits.

uint maskZero = H2_RTZ(auxiliaryMask,0);
uint zeroAlpha = H2_RTZ(0,mrt1Alpha);
uint redGreen = H2_RTZ(outputColor.r,outputColor.g);
uint blueAlpha = H2_RTZ(outputColor.b,mrt0Alpha);

// Restore the saved surviving execution mask.
//
// SKIN:
//   19c0 exp mrt1 compr: maskZero,zeroAlpha
//   19c8 exp mrt0 compr: redGreen,blueAlpha
//
// RIGID:
//   16d0 exp mrt1 compr: maskZero,zeroAlpha
//   16d8 exp mrt0 compr: redGreen,blueAlpha
//
// Target format/component-swap state remains outside these listings.
```

In the linear-colour block, `occlusionScaledMaterialColor` means the saved value immediately before the `PSV[0x60]+0x110` blend:

```glsl
vec3 occlusionScaledMaterialColor = linearColor;

linearColor =
    mad32(linearBlendColor,linearBlendWeight,
          occlusionScaledMaterialColor);

linearColor =
    mac32(-linearBlendWeight,occlusionScaledMaterialColor,linearColor);
```

## Differences from material.cpp's kFs/kVs

For **both pairs**, the vertex port needs these changes:

- Replace the object matrix in `VSV[0x20]` with a four-bone palette descriptor. Fetch position, weights, integer bone indices, encoded normal, encoded tangent and UV.
- Compute weights as `1-((w.y+w.z)+w.w), w.y, w.z, w.w`; do not use fetched `w.x`.
- Blend the nine linear coefficients and the floor-relative translation before transforming position.
- Add the signed-count deformation loop using `VSV[0x00]+0x12c`, `+0x50`, `+0x90`, and `+0xd0`.
- Retain position-offset, camera-subtraction, lighting-offset and vertex-fog flag meanings, but use the new PCs in the manifest.
- Load the second camera value unconditionally and compute a normalized vertex view direction.
- Skin normal and tangent with the blended palette matrix. Do not apply the stage object matrix.
- Remove vertex colour, baked-light decoding, secondary UV, UV offsets, vertex-alpha scaling and normal flattening.
- Replace the eight-param stage interface with the six-param layout above.
- Export fog transmission directly in `param0.xyz`. It multiplies the entire final linear material result in the PS.
- Keep the original projection/view chains. Neither supplied VS contains the shadow port’s `max(0,clip.z)`.

For **both pairs**, the fragment port needs these changes:

- Read flags/view from `PSV[0xd0]` / `PSV[0xc0]`, with the new load PCs.
- Read UV from `attr2.xy`, normal from `attr4.xyz`, tangent/handedness from `attr5`, view direction from `attr3.xyz`, and position from three `.w` components.
- Use the normal map directly; there is no flattening, detail-normal blend or normal UV-scale constant.
- Replace the stage specular texture at `T[0x60]` with `T[0x40]`: R is scalar specular, G multiplies diffuse material and the grazing effect, B scales the exponent.
- Use `T[0x20]` as an RGBA material-control texture on the primary UV. It is not a secondary-UV detail texture.
- Remove both stage detail textures and their modulation/blending calculations.
- Sample screen lighting with the direct entry sampler `s[4:7]`; sample screen normals with direct `s[0:3]`.
- Sample screen-shadow RG only, remap once with `PSV[0x10]+0xf0`, then select `min(R,G)` or `R*G` using `PSView+0x22c` bit1.
- Remove screen-shadow coverage and the stage directional/indirect shadow-tint calculations.
- Gate all point lighting with pixel flag bit0 as well as individual bits16–19.
- Remove the stage per-light `lambert>0` branch. Add point specular and the separate Lambert-weighted point-luminance accumulation.
- Replace baked lighting and the stage ambient-texture term with the quadratic ambient polynomial from `PSV[0x20]+0x500..0x5bc`, its `+0x4f0` gain, and the upper/lower tints at `PSV[0x30]+0x60/+0xe0`.
- Use the actual second directional vector for both its diffuse and specular calculations. The stage’s fixed secondary specular direction is absent.
- Use normalized reflected-direction lobes, fifth-power Fresnel and Lambert for directional specular.
- Read the secondary directional **diffuse** gain from `PSV[0x90]+0x134`. The stage `PSV[0x30]+0xf0` secondary-specular gain is absent.
- Add the grazing-angle modification to base RGB squared.
- Add the directional-response ramp using `PSV[0x00]+0x40/+0x44/+0x48`.
- Add material occlusion using control A and `PSV[0x30]+0x120..0x12c`.
- Add the pre-tone colour blend using `PSV[0x60]+0x110..0x11c`.
- Apply `attr0.rgb` transmission to the whole linear result before the existing tone curve.
- Add optional post-tone grading controlled by `PSV[0x00]+0xc` bit0.
- Add the pixel deformation inverse walk and plane-band rejection.
- Keep distance fog, output scaling, final blend and alpha-selection structures, but use the manifest’s bindings/PCs.
- Remap the screen-mask constants:

| Stage offset in `PSV[0x00]` | Character offset | Meaning |
|---|---:|---|
| `0x2b0/2b4/2b8` | `0x260/264/268` | Depth-edge radius/minimum/threshold scale |
| `0x2bc/2c0/2c4` | `0x26c/270/274` | Normal-edge radius/minimum/threshold scale |
| `0x2c8/2cc/2d0` | `0x278/27c/280` | Facing minimum/far width/far threshold |
| `0x2d4/2d8/2dc/2e0` | `0x284/288/28c/290` | Distance start/near/end/minimum radius |
| `0x2e4/2e8/2ec` | `0x294/298/29c` | Distance-fade end/amount/output threshold |
| `0x2f0/2f4/2f8/2fc` | `0x2a0/2a4/2a8/2ac` | Facing far/near/near width/near threshold |

- Additionally multiply normal-edge strength by `1-control.b*F(PSV[0x00],0x54)`.
- Additionally multiply the facing-width near ramp by `1-control.b*F(PSV[0x00],0x58)`.
- Suppress the final auxiliary mask by `1-F(PSV[0x60],0x11c)`.
- Preserve half conversion toward zero if exact compressed-export matching is required; `kFs` currently uses ordinary `packHalf2x16`.

For the **RIGID-labelled pair** specifically:

- `param2.z` is zero and the PS does not read it.
- Point specular is `pow(sat(-N·normalize(L+V)),exponent)` times attenuated point RGB. It has no directional-style normalization, Fresnel or extra Lambert factor.
- The grazing-effect scale is `0.1*(2.5+F(PSV[0x70],0x68))`.
- There are no directional-ramp texture samples or extra-lobe constants.

For the **SKIN pair** specifically:

- Export raw `encodedNormal.w` in `param2.z`.
- Add the three `T[0xc0]` directional-ramp samples and their correction calculation.
- Load ramp Y controls from `PSV[0x70]+0x78/+0x7c`, and ramp correction/terminator controls from `PSV[0x90]+0x24/+0x2c`.
- Add the amplitude-tested extra specular lobe using `PSV[0x90]+0x70..0x7c`.
- Use reflected-direction point specular with normalization, fifth-power Fresnel and Lambert.
- Weight directional specular by `a+f-a*f`, where `a=materialShadow*nl0` and `f=sat(specMaterial.g-attr2.z)`.
- Use grazing-effect scale `2+F(PSV[0x70],0x68)`.
- Preserve the red ambient hemisphere’s different multiply/add sequence.

These are separate choices from skinning itself: a useful define split is a shared character-material path, a palette vertex path, and individual defines for the skin ramp, extra lobe, point-specular model and vertex-controlled specular weight.

## Comparison with shadow.cpp's kSkinVs

The **skinning scheme matches structurally**:

- Four influences.
- Three float4 rows per bone.
- Index construction from `48*boneIndex`, shifted by four with the low four bits supplied as the byte offset.
- Implicit first weight `1-((w.y+w.z)+w.w)`.
- A separate floor anchor from bone0’s translation for each row.
- Linear matrix coefficients blended in bone order.

It is **not an operation-for-operation position match to the handwritten shadow GLSL**. `kSkinVs` adds `p.off` to the floor anchor before accumulating translation residuals, and its host code folds camera subtraction into that offset. These material shaders finish the floor-relative translation and local-position transform first, then add `VSV[0x10]+0x170`, then conditionally subtract the camera. The shadow port also combines view/projection on the host and clamps clip Z; these material shaders execute their explicit view/projection chains.

Beyond the shadow port, both material VS programs compute the deformation displacement, skin normal and tangent, decode tangent handedness, compute a normalized view vector, compute vertex fog, and export the packed lighting position. Only `vs_skin.s` exports encoded normal W for the skin-material controls.

## Repeated and overlapping loads

Preserve these when constructing the new `Load` tables:

- **VS camera:** `3ac / 3b4` is conditional; `410 / 418` is unconditional. Both read `VSV[0x30]+0x220`. Map both to the camera slot, or use the unconditional load as the canonical source.
- **PS flags:** `120,14d0,18fc` / `180,11e0,160c` all read `PSFlags+0`.
- **PS raster scale:** `124 / 1cc` is unconditional; `153c / 124c` is inside the screen-mask branch.
- **PS view flags:** `b2c / 908` and `19a0 / 16b0` both read `PSView+0x22c`.
- **Material alpha:** `PSV[0x70]+0x9c` is inside the eight-dword material load `e68 / aa4`, then loaded separately at `14c8 / 11d8`.
- **Diffuse alpha multiplier:** `PSV[0x60]+0xc` is inside `e6c / b70`, then separately loaded at `14cc / 11dc`.
- **Linear blend weight:** `PSV[0x60]+0x11c` is inside `d8c / b84`, then separately loaded at `1900 / 1610`.
- **Mask/grading flags:** `c0c / a20` and `1550 / 1260` read `PSV[0x00]+0xc`; the optional grading block `12fc / 100c` also includes that offset.
- **Depth reconstruction:** the optional grading block `12fc / 100c` includes `PSV[0x00]+0x2c` and `+0x3c`; the mask branch separately loads them at `1540,1548` / `1250,1258`.

The last two cases are especially relevant to zero-filled untaken branches: the grading block must not overwrite live mask constants with zeros, or vice versa. The existing nonzero-preserving merge in `material.cpp` addresses that capture representation, provided all overlapping loads map to the same destination components.
tokens used
