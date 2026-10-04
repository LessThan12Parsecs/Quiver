/**
 * Water surface shaders (CDLOD ocean mesh, see OceanMesh.ts).
 *
 * Vertex: each instanced patch vertex is morphed toward the next coarser grid near the end of
 * its LOD range (CDLOD, LOD distances scaled by the field of view), then displaced with
 * OCEAN_GLSL: wind chop (Lagrangian, components shorter than ~3 grid cells faded out —
 * identical to oceanSurface() where the grid resolves them) and the shoaling swell, plus the
 * render-only breaking geometry (BREAKER_GLSL: pitching lip, foam roller mound). The swell
 * gradient is taken by central differences over one grid cell (where the breaker geometry is
 * active, the normal of the displaced surface comes from the same differences: vLip); the chop
 * gradient is analytic in the fragment shader. Far levels (grid >= 90 m) skip the swell.
 *
 * Fragment:
 *  - Pixel footprint from the camera model and the interpolated swell normal (smooth; dFdx of
 *    the displaced position jumps at triangle edges and is constant per 2x2 quad).
 *  - Normal = swell slope (interpolated) + analytic chop slope (components finer than the
 *    footprint fade, their slope variance goes into roughness) + short-wave detail from the
 *    spectral ripple tiles (DetailNormals.ts) at up to three scales, applied in the tangent frame
 *    of the macro surface. The tiles store slope moments (LEAN mapping), so filtered fetches
 *    with analytic gradients return the slope variance lost to filtering -> GGX roughness.
 *    Detail amplitude varies with drifting wind gusts and wind-aligned slicks; on steep faces a
 *    stretched copy of the detail adds streaks running down the face.
 *  - Back faces: the surface seen from below when the camera is under water (uCameraUnderwater,
 *    set from the CPU surface); above water they are the underside of a breaking lip (shaded
 *    with the flipped lip normal) or crest-silhouette slivers (shaded like the front face).
 *  - Schlick Fresnel, PMREM sky reflection, GGX sun glitter, water body colour from single
 *    scattering + Beer-Lambert over the sand seabed along the refracted ray, subsurface glow of
 *    thin backlit crests and lips, foam (breaking crests/faces, trailing bores, lingering
 *    surf-zone lace, whitecaps) with a lumpy relief: bump-mapped lumps and bubble domes,
 *    crease occlusion, thin translucent vs thick opaque foam, and aerial perspective.
 *
 * Shading-quality defines: DETAIL_LAYERS (1-3), FOAM_DETAIL (0/1), FACE_DETAIL (0/1),
 * WIND_PATTERNS (0/1: drifting gusts, slicks and the detail domain warp — 5 value-noise calls
 * per pixel).
 */
import { OCEAN_GLSL } from '../../ocean/waveGLSL';
import { BREAKER_GLSL } from './breaker';
import { ENV_GLSL, NOISE_GLSL } from './common';

export const MAX_LODS = 16;

const VARYINGS = /* glsl */ `
varying vec3 vWorldPos;
varying vec2 vRest;
varying vec2 vSwellSlope;
varying vec4 vSwellA;   // swell eta, local wave height, breaking, fullness
varying vec4 vSwellB;   // cos psi, sin psi, ratio, still-water depth
varying vec4 vSwellC;   // dominant direction xz, phase speed, wavenumber
varying vec2 vLod;      // lod level + morph, grid spacing
varying vec4 vLip;      // breaker geometry: normal of the displaced surface (xyz), weight (w)
varying float vLipClear; // share of the breaker offset that is the thrown (clear-water) lip
`;

export const OCEAN_VERTEX = /* glsl */ `
${OCEAN_GLSL}
${NOISE_GLSL}
${BREAKER_GLSL}
#define MAX_LODS ${MAX_LODS}
uniform float uTime;
uniform vec4 uLodOrigin;   // camera x, |y|, z; LOD distance scale (tan(fov/2) / tan(30 deg))
uniform vec2 uLodMorph[MAX_LODS];
attribute vec4 aNode;   // patch origin x, z, grid spacing, lod level
${VARYINGS}

// oceanWind() with every component faded out once its wavelength drops below ~3 grid cells.
// Op-for-op identical to oceanWind() where all components are resolved (lod factor exactly 1).
vec3 oceanWindLod(vec2 x0, float spacing) {
  float fade = smoothstep(0.1, 1.5, oceanField(x0).x);
  vec3 d = vec3(0.0);
  for (int j = 0; j < OCEAN_MAX_WIND; j++) {
    if (j >= uWindCount) break;
    vec4 A = uWindA[j];
    vec4 B = uWindB[j];
    float lod = smoothstep(2.0, 4.0, OCEAN_TWO_PI / (A.z * spacing));
    float ph = A.z * (A.x * x0.x + A.y * x0.y) + A.w;
    float s = sin(ph);
    float c = cos(ph);
    float f = fade * lod;
    float hd = -B.y * f * s;
    d.x += A.x * hd;
    d.z += A.y * hd;
    d.y += B.x * f * c;
  }
  return d;
}

void main() {
  vec2 g = position.xz;
  float cell = aNode.z;
  vec2 rest0 = aNode.xy + g * cell;
  vec3 dl = vec3(rest0.x - uLodOrigin.x, uLodOrigin.y, rest0.y - uLodOrigin.z);
  vec2 mr = uLodMorph[int(aNode.w + 0.5)];
  float morph = clamp((length(dl) * uLodOrigin.w - mr.x) * mr.y, 0.0, 1.0);
  vec2 rest = rest0 - fract(g * 0.5) * 2.0 * morph * cell;
  float spacing = cell * (1.0 + morph);

  vec3 w = oceanWindLod(rest, spacing);
  vec2 p = rest + w.xz;
  // Far away (grid of 40-90 m: ~2-3 cells per swell wavelength) the swell would alias: fade it
  // out; beyond that skip it (uniform per patch: no divergence).
  float swellFade = 1.0 - smoothstep(40.0, 90.0, spacing);
  vec3 pos = vec3(p.x, w.y, p.y);
  vSwellSlope = vec2(0.0);
  vSwellA = vec4(0.0);
  vSwellB = vec4(1.0, 0.0, 0.0, 0.0);
  vSwellC = vec4(1.0, 0.0, 0.0, 0.0);
  vLip = vec4(0.0, 1.0, 0.0, 0.0);
  vLipClear = 0.0;
  if (swellFade > 0.0) {
    OceanSwell sw = oceanSwell(p);
    float h = 0.5 * spacing;
    OceanSwell sxp = oceanSwell(p + vec2(h, 0.0));
    OceanSwell sxm = oceanSwell(p - vec2(h, 0.0));
    OceanSwell szp = oceanSwell(p + vec2(0.0, h));
    OceanSwell szm = oceanSwell(p - vec2(0.0, h));
    vSwellSlope = vec2(sxp.eta - sxm.eta, szp.eta - szm.eta) * (swellFade / (2.0 * h));
    pos.y += sw.eta * swellFade;
    vSwellA = vec4(sw.eta * swellFade, sw.height * swellFade, sw.breaking, sw.fullness);
    vSwellB = vec4(cos(sw.psi), sin(sw.psi), min(sw.ratio, 8.0), sw.depth);
    vSwellC = vec4(sw.dir, sw.speed, sw.k);
    if (breakerActive(sw)) {
      // breaking lip / roller: displace, and take the normal of the displaced surface
      float lipClear;
      vec3 b0 = breakerOffset(sw, p, uTime, spacing, lipClear);
      vLipClear = lipClear;
      vec3 bxp = breakerOffset(sxp, p + vec2(h, 0.0), uTime, spacing);
      vec3 bxm = breakerOffset(sxm, p - vec2(h, 0.0), uTime, spacing);
      vec3 bzp = breakerOffset(szp, p + vec2(0.0, h), uTime, spacing);
      vec3 bzm = breakerOffset(szm, p - vec2(0.0, h), uTime, spacing);
      pos += b0;
      vec3 dx = vec3(2.0 * h, sxp.eta - sxm.eta, 0.0) + bxp - bxm;
      vec3 dz = vec3(0.0, szp.eta - szm.eta, 2.0 * h) + bzp - bzm;
      float act = length(b0) + length(bxp - bxm) + length(bzp - bzm);
      vLip = vec4(normalize(cross(dz, dx)), smoothstep(0.01, 0.06, act));
    }
  }
  vWorldPos = pos;
  vRest = rest;
  vLod = vec2(aNode.w + morph, spacing);
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
}
`;

export const OCEAN_FRAGMENT = /* glsl */ `
${NOISE_GLSL}
${ENV_GLSL}
${OCEAN_GLSL}
#define PI 3.14159265
#ifndef DETAIL_LAYERS
#define DETAIL_LAYERS 3
#endif
#ifndef FOAM_DETAIL
#define FOAM_DETAIL 1
#endif
#ifndef FACE_DETAIL
#define FACE_DETAIL 1
#endif
#ifndef WIND_PATTERNS
#define WIND_PATTERNS 1
#endif

uniform float uTime;
uniform vec2 uWindDrift;                   // wind direction * gust advection speed (m/s)
uniform float uPixelAngle;                 // angular size of a pixel (rad)
uniform sampler2D uDetail0;                // ripple tiles: (slope u, slope v, slope^2, height)
uniform sampler2D uDetail1;
uniform vec4 uDetailDecS;                  // texel decode: value = texel * S + B
uniform vec4 uDetailDecB;
uniform vec2 uDetailOrigin;                // world origin of the detail coordinates
uniform vec4 uDetailL0;                    // per layer: tile u axis (world xz), 1 / tile size (1/m), slope rms
uniform vec4 uDetailL1;
uniform vec4 uDetailL2;
uniform vec4 uDetailOff01;                 // uv offsets of layers 0 (xy) and 1 (zw)
uniform vec2 uDetailOff2;
uniform float uFaceStreaks;                // slope rms of the streaks on steep faces
uniform vec3 uAbsorption;                  // clear offshore water absorption a, 1/m (R, G, B)
uniform vec3 uBackscatter;                 // clear offshore water backscattering bb, 1/m
uniform vec3 uSurfAbsorption;              // extra a in the surf zone (CDOM, suspended sediment)
uniform vec3 uSurfBackscatter;             // extra bb in the surf zone (suspended sand)
uniform float uTurbidity;                  // scales the surf-zone extras
uniform vec3 uBubbleBackscatter;           // bb of fully aerated water (whitewater bubbles)
uniform float uBottomAlbedo;               // wet sand albedo multiplier
uniform float uFoamIntensity;
uniform float uSssIntensity;
uniform float uCausticsIntensity;
uniform float uRoughness;                  // base GGX alpha
uniform float uCameraUnderwater;           // 1 when the camera is below the water surface (CPU)
uniform int uDebug;
${VARYINGS}

// World-space offsets of the surface point for a one-pixel step in screen x / y, on the plane
// through P with normal n (ray differentials). Smooth everywhere, unlike dFdx of the interpolated
// position (which jumps at triangle edges and is constant per 2x2 pixel quad).
void pixelFootprint(vec3 P, vec3 n, out vec3 dpx, out vec3 dpy) {
  vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 up = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  vec3 fwd = -vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);
  vec3 d = P - cameraPosition;
  float z = max(dot(d, fwd), 1e-3);
  vec3 du = d / z;                      // ray at unit view depth
  float nd = dot(n, du);
  float lim = 0.03 * length(du);        // grazing limit (~33:1 stretch)
  nd = nd < 0.0 ? min(nd, -lim) : max(nd, lim);
  vec3 ex = right * uPixelAngle;
  vec3 ey = up * uPixelAngle;
  dpx = z * (ex - du * (dot(n, ex) / nd));
  dpy = z * (ey - du * (dot(n, ey) / nd));
}

// Fraction of a wave component (wavenumber k, direction d) kept at the pixel footprint: the
// footprint is projected onto the component's direction (exact anisotropic filtering of a
// sinusoid); components with fewer than ~5 px per wavelength fade out.
float keepFrac(float k, vec2 d, vec2 dpx, vec2 dpy) {
  float fpd = max(abs(dot(d, dpx)), abs(dot(d, dpy))) + 1e-4;
  return smoothstep(2.0, 5.0, OCEAN_TWO_PI / (k * fpd));
}

// World-space slope of the wind chop at rest point x0 (analytic Lagrangian Jacobian, same maths as
// OceanModel.sample). Components finer than the footprint fade; their slope variance is returned.
vec2 chopSlope(vec2 x0, vec2 dpx, vec2 dpy, out float jdet, out float lostVar) {
  float fade = smoothstep(0.1, 1.5, oceanDepth(x0));
  float jxx = 0.0;
  float jxz = 0.0;
  float jzz = 0.0;
  float gx = 0.0;
  float gz = 0.0;
  lostVar = 0.0;
  for (int j = 0; j < OCEAN_MAX_WIND; j++) {
    if (j >= uWindCount) break;
    vec4 A = uWindA[j];
    vec4 B = uWindB[j];
    float aa = keepFrac(A.z, A.xy, dpx, dpy);
    float ph = A.z * (A.x * x0.x + A.y * x0.y) + A.w;
    float s = sin(ph);
    float c = cos(ph);
    float a = B.x * fade * aa;
    float ah = B.y * fade * aa;
    float jd = -ah * A.z * c;
    jxx += jd * A.x * A.x;
    jxz += jd * A.x * A.y;
    jzz += jd * A.y * A.y;
    float gy = -a * A.z * s;
    gx += gy * A.x;
    gz += gy * A.y;
    float sl = B.x * fade * A.z;
    lostVar += 0.5 * sl * sl * (1.0 - aa * aa);
  }
  float a11 = 1.0 + jxx;
  float a22 = 1.0 + jzz;
  jdet = a11 * a22 - jxz * jxz;
  float inv = 1.0 / max(jdet, 0.08);
  return vec2(a22 * gx - jxz * gz, -jxz * gx + a11 * gz) * inv;
}

// One detail layer (tile 'tex' with axis/scale/rms A at uv offset 'off'): world-space slope (xy)
// and height (z) of the layer, scaled by amp. The slope variance lost to filtering (LEAN: mean
// squared slope minus squared mean slope of the filtered texel) is added to 'lostVar'. Layers
// whose features are all far below the footprint are not fetched: their whole variance is lost.
vec3 detailLayer(sampler2D tex, vec4 A, vec2 off, vec2 x, vec2 gx, vec2 gy, float fp, float amp, inout float lostVar) {
  float sw = A.w * amp;
  float keep = 1.0 - smoothstep(0.06, 0.12, fp * A.z);
  if (keep <= 0.0) {
    lostVar += sw * sw;
    return vec3(0.0);
  }
  vec2 a = A.xy;
  vec2 b = vec2(-A.y, A.x);
  vec2 uv = vec2(dot(x, a), dot(x, b)) * A.z + off;
  vec2 ux = vec2(dot(gx, a), dot(gx, b)) * A.z;
  vec2 uy = vec2(dot(gy, a), dot(gy, b)) * A.z;
  vec4 t = textureGrad(tex, uv, ux, uy) * uDetailDecS + uDetailDecB;
  float m2 = dot(t.xy, t.xy);
  lostVar += (max(t.z - m2, 0.0) + m2 * (1.0 - keep * keep)) * sw * sw;
  return vec3((a * t.x + b * t.y) * (sw * keep), t.w * keep);
}

// Whitewater advected at velocity vel (flow-map style: two phases cross-faded).
vec3 flowWhitewater(vec2 x, vec2 vel, float t, float fp) {
  const float T = 3.0;
  float p0 = fract(t / T);
  float p1 = fract(t / T + 0.5);
  float w0 = 1.0 - abs(2.0 * p0 - 1.0);
  vec3 f0 = whitewaterField(x - vel * (p0 * T), t, fp);
  vec3 f1 = whitewaterField(x - vel * (p1 * T) + vec2(41.3, 17.9), t, fp);
  return mix(f1, f0, w0);
}

#if FOAM_DETAIL
// Whitewater relief: (relative height ~[-1, 1], slope d/dx, slope d/dz) and crease mask (out).
// Billow noise (sum of |signed value noise| octaves, ~60 cm down to ~6 cm): rounded lumps of
// every size separated by sharp creases, like cauliflower; domain-warped and churning. Octaves
// fade out with the pixel footprint fp (m) before they can alias; the thin crease lines with the
// footprint's long axis fpLong (they alias first at grazing angles).
vec3 foamRelief(vec2 x, float t, float fp, float fpLong, out float crease) {
  vec2 p = x * 1.6 + vec2(0.0, 0.25 * t);
  mat2 j = mat2(1.6);
  mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
  float f = 1.6;
  float amp = 1.0;
  vec3 r = vec3(0.0);
  crease = 0.0;
  for (int i = 0; i < 4; i++) {
    float aa = 1.0 - smoothstep(0.18, 0.45, fp * f);
    if (aa <= 0.0) break;
    vec3 n = vnoiseD(p);
    float sn = 2.0 * n.x - 1.0;
    float b = abs(sn);
    // world slope of a billow of height 0.07 m * amp
    vec2 g = (sign(sn) * 2.0 * 0.07 * amp) * (n.yz * j);
    r += aa * vec3(amp * (b - 0.5), g);
    crease = max(crease, (1.0 - smoothstep(0.03, 0.09, fpLong * f)) * amp * (1.0 - smoothstep(0.0, 0.18, b)));
    p = rot * p * 2.1 + vec2(13.7, 5.1) + vec2(0.37, -0.21) * t;
    j = rot * j * 2.1;
    f *= 2.1;
    amp *= 0.5;
  }
  return r;
}
#endif

// Animated caustic network on the seabed (mean ~1); sharp only in very shallow, clear water:
// suspended sand and bubbles in the surf zone (uTurbidity) blur and dim it with depth.
float caustics(vec2 x, float t, float fp, float depth) {
  float aa = (1.0 - smoothstep(0.01, 0.035, fp)) * (1.0 - smoothstep(0.3, 1.8, depth)) * exp(-0.9 * uTurbidity * depth);
  if (aa <= 0.01) return 1.0;
  vec2 warp = vec2(vnoise(x * 0.9 + 0.3 * t), vnoise(x * 0.9 + 9.1 - 0.3 * t)) - 0.5;
  vec2 w1 = worley(x * 2.3 + warp, t * 1.2);
  vec2 w2 = worley(x * 3.1 + warp * 1.3 + 4.7, -t * 1.0);
  float blur = 0.16 + 0.4 * uTurbidity * depth;
  float e1 = 1.0 - smoothstep(0.0, blur, w1.y - w1.x);
  float e2 = 1.0 - smoothstep(0.0, blur, w2.y - w2.x);
  float c = 0.55 * e1 + 0.45 * e2 + 1.2 * e1 * e2;
  return mix(1.0, 0.76 + 0.95 * c, aa);
}

float ggxD(float NdotH, float a2) {
  float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
float smithG(float NdotV, float NdotL, float a2) {
  float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
  float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}
// Henyey-Greenstein phase function.
float phaseHG(float cosT, float g) {
  float d = 1.0 + g * g - 2.0 * g * cosT;
  return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}

// Perturb normal n by a world-space height gradient g (xz), projected onto the tangent plane.
vec3 bumpNormal(vec3 n, vec2 g) {
  vec3 gw = vec3(g.x, 0.0, g.y);
  return normalize(n - (gw - n * dot(n, gw)));
}

void main() {
  vec3 P = vWorldPos;
  vec3 toCam = cameraPosition - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 L = uSunDirection;

  // Breaking lip / roller geometry (render only): its own normal; the folded underside of a lip
  // is seen from its back side.
  float lipW = gl_FrontFacing || uCameraUnderwater < 0.5 ? vLip.w : 0.0;
  vec3 Nlip = normalize(vLip.xyz) * (gl_FrontFacing ? 1.0 : -1.0);

  // Pixel footprint (ray differentials on the plane of the interpolated swell normal), in metres.
  vec3 Nsw = normalize(vec3(-vSwellSlope.x, 1.0, -vSwellSlope.y));
  if (lipW > 0.0) Nsw = normalize(mix(Nsw, Nlip, lipW));
  vec3 dpx;
  vec3 dpy;
  pixelFootprint(P, Nsw, dpx, dpy);
  float fx = length(dpx);
  float fy = length(dpy);
  float fp = max(sqrt(fx * fy), max(fx, fy) * 0.125) + 1e-4;

  // ---- swell state ---------------------------------------------------------------------
  float etaS = vSwellA.x;
  float H = vSwellA.y;
  float brk = clamp(vSwellA.z, 0.0, 1.0);
  float full = vSwellA.w;
  float psi = atan(vSwellB.y, vSwellB.x);
  float ratio = vSwellB.z;
  float depth = oceanDepth(P.xz);
  float column = max(P.y + depth, 0.0);   // water column under this surface point
  float c = vSwellC.z;
  float omega = max(c * vSwellC.w, 0.2);
  float shallow = 1.0 - smoothstep(2.0, 12.0, depth);
  float steepS = length(vSwellSlope);

  // ---- foam masks -------------------------------------------------------------------------
  // Active whitewater (the roller): a ragged line along the lip at breaking onset, spreading down
  // the face as the breaker develops. Its width scales with the wave height (in metres, converted
  // to waveform phase; the forward lean compresses the face ~2x in phase).
  float bs = smoothstep(0.02, 0.8, brk);
  float kDom = max(vSwellC.w, 0.02);
  float faceExt = clamp(2.0 * kDom * (0.5 + 2.4 * H), 0.2, 1.8) * mix(0.35, 1.0, bs);
  float backExt = clamp(kDom * (0.4 + 1.2 * H), 0.08, 0.5);
  // ragged edges: jitter the phase window with ~1-4 m noise along the crest
  float psiJ = psi;
  if (brk > 0.0) psiJ += (vnoise(P.xz * vec2(0.35, 0.25)) - 0.5) * 0.6 * faceExt + (vnoise(P.xz * 1.1 + 7.0) - 0.5) * 0.22 * faceExt;
  // (the front transition is wide: the foam pattern, thresholded by the coverage, cuts a chunky
  // broken leading edge into it)
  float onFace = smoothstep(-backExt - 0.15, -backExt * 0.3, psiJ) * (1.0 - smoothstep(faceExt * 0.45, faceExt * 1.15, psiJ));
  float white = onFace * smoothstep(0.0, 0.45, brk) * smoothstep(0.05, 0.7, H);
  // the thrown lip is a clear sheet of water (it turns white where it lands: the roller)
  white *= 1.0 - 0.85 * smoothstep(0.05, 0.4, vLipClear);
  // Trailing foam carpet left behind the roller, thinning out with time since the crest passed;
  // bigger breakers leave more and longer-lasting foam.
  float age = mod(-psi, 2.0 * PI) / omega;
  float broken = smoothstep(0.92, 1.35, ratio) * smoothstep(0.35, 0.7, full + brk);
  float trail = broken * exp(-age / (1.5 + 2.5 * H)) * smoothstep(0.08, 0.9, H);
  // Lingering lace over the inner surf zone.
  float surf = 0.3 * smoothstep(1.0, 2.2, ratio) * smoothstep(0.1, 0.6, H);

  // ---- normal: swell + chop (geometry) + spectral detail ------------------------------------
  // Wind gusts / cat's paws (slowly drifting rougher patches) and wind-aligned slicks.
#if WIND_PATTERNS
  vec2 gp = (P.xz - uWindDrift * uTime) / 60.0;
  float gust = 0.5 + 1.1 * vnoise(gp) * (0.55 + 0.45 * vnoise(gp * 2.7 + 5.0));
  vec2 wd = normalize(uWindDrift + vec2(1e-5, 0.0));
  vec2 sp = vec2(dot(P.xz, wd), dot(P.xz, vec2(-wd.y, wd.x))) * vec2(1.0 / 90.0, 1.0 / 8.0);
  float slick = smoothstep(0.72, 0.84, vnoise(sp + vec2(3.1 + 0.004 * uTime, 0.03 * uTime)));
  slick *= 1.0 - smoothstep(0.8, 2.5, fp);
#else
  float gust = 0.92;   // mean of the gust field
  float slick = 0.0;
#endif
  // no ripples under dense whitewater
  float detAmp = gust * (1.0 - 0.55 * slick) * (1.0 - 0.9 * smoothstep(0.5, 0.9, max(white, trail)));

  float jdet;
  float chopLost;
  vec2 macro = vSwellSlope + chopSlope(vRest, dpx.xz, dpy.xz, jdet, chopLost);
  vec3 Ng = normalize(vec3(-macro.x, 1.0, -macro.y));
  vec3 Tx = normalize(vec3(1.0, macro.x, 0.0));
  vec3 Tz = normalize(vec3(0.0, macro.y, 1.0));
  if (lipW > 0.0) {
    // lip / roller: the chop rides on the displaced surface's normal
    Ng = normalize(mix(Ng, bumpNormal(Nlip, macro - vSwellSlope), lipW));
    Tx = normalize(Tx - Ng * dot(Tx, Ng));
    Tz = normalize(Tz - Ng * dot(Tz, Ng));
  }

  // Detail coordinates follow the wave profile (x - 0.6 y ~ arc length on the front face, so
  // steep faces aren't smeared), with a slow domain warp (no straight infinite crests).
  vec2 xd = vec2(P.x - 0.6 * P.y, P.z) - uDetailOrigin;
  vec2 gx = vec2(dpx.x - 0.6 * dpx.y, dpx.z);
  vec2 gy = vec2(dpy.x - 0.6 * dpy.y, dpy.z);
#if WIND_PATTERNS
  vec2 wq = P.xz * (1.0 / 13.0) + uTime * 0.015;
  xd += (vec2(vnoise(wq), vnoise(wq + 7.31)) - 0.5) * 0.8;
#endif
  float detVar = 0.0;
  // the 4.2 m layer (the most visible scale near the camera) is always sampled
  vec3 det = detailLayer(uDetail1, uDetailL1, uDetailOff01.zw, xd, gx, gy, fp, detAmp, detVar);
#if DETAIL_LAYERS > 1
  det += detailLayer(uDetail0, uDetailL0, uDetailOff01.xy, xd, gx, gy, fp, detAmp, detVar);
#else
  detVar += uDetailL0.w * uDetailL0.w * detAmp * detAmp;
#endif
#if DETAIL_LAYERS > 2
  det += detailLayer(uDetail1, uDetailL2, uDetailOff2, xd, gx, gy, fp, detAmp, detVar);
#else
  detVar += uDetailL2.w * uDetailL2.w * detAmp * detAmp;
#endif
#if FACE_DETAIL
  // Streaks running down steep faces: the detail stretched ~8x along the wave direction.
  float faceMask = smoothstep(0.3, 0.9, steepS) * smoothstep(0.35, 0.8, full) * smoothstep(-0.2, 0.4, sin(psi));
  faceMask *= 1.0 - smoothstep(0.2, 0.6, max(white, trail));
  if (faceMask > 0.01 && uFaceStreaks > 0.0) {
    vec2 wdir = vSwellC.xy;
    vec2 wper = vec2(-wdir.y, wdir.x);
    vec2 sc = vec2(1.0 / 2.2, 1.0 / 16.0);
    vec2 suv = vec2(dot(xd, wper), dot(xd, wdir)) * sc + vec2(0.37, 0.61);
    vec2 sux = vec2(dot(gx, wper), dot(gx, wdir)) * sc;
    vec2 suy = vec2(dot(gy, wper), dot(gy, wdir)) * sc;
    vec4 st = textureGrad(uDetail1, suv, sux, suy) * uDetailDecS + uDetailDecB;
    // patchy: streaks come and go along the crest
    float patchy = smoothstep(0.25, 0.75, vnoise(vec2(dot(P.xz, wper) * 0.18, dot(P.xz, wdir) * 0.05 + 3.7)));
    float k = uFaceStreaks * faceMask * (0.35 + 0.65 * patchy);
    det.xy += (wper * st.x + wdir * (st.y * 0.14)) * k;
    detVar += max(st.z - dot(st.xy, st.xy), 0.0) * k * k;
  }
#endif
  vec3 N = normalize(Ng - Tx * det.x - Tz * det.y);

  // Back faces: the surface seen from below only when the camera is under water. Above water
  // they are a lip's underside (lip normal, flipped above) or slivers of the far side of a crest
  // winning the depth test at its silhouette — shaded like the front face (no dark specks).
  if (!gl_FrontFacing && uCameraUnderwater > 0.5) {
    // Seen from below (camera underwater): Snell's window to the sky, total internal reflection
    // of the dark water outside it, attenuated along the in-water path to the eye.
    vec3 Nd = -N;
    vec3 Tu = refract(-V, Nd, 1.333);
    vec3 kapU = uAbsorption + uSurfAbsorption * uTurbidity + 20.0 * (uBackscatter + uSurfBackscatter * uTurbidity);
    vec3 inWater = (uSunColor * max(uSunDirection.y, 0.0) + uSkyIrradiance) * (uBackscatter + uSurfBackscatter) * 2.0 / kapU;
    vec3 colU = inWater;
    if (dot(Tu, Tu) > 0.0) {
      float Fu = 0.02 + 0.98 * pow(1.0 - clamp(dot(Nd, V), 0.0, 1.0), 5.0);
      colU = mix(envSample(normalize(Tu), 0.15) * (1.0 - Fu), inWater, Fu);
    }
    colU *= exp(-kapU * min(dist, 60.0));
    colU += inWater * (1.0 - exp(-kapU * min(dist, 60.0)));
    gl_FragColor = vec4(colU, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    return;
  }
  float NdotV0 = dot(N, V);
  // Keep the shading normal visible from the camera (steep faces seen from behind).
  if (NdotV0 < 0.02) N = normalize(N + V * (0.02 - NdotV0));

  // Roughness: base + slope variance lost to footprint filtering (chop: analytic, detail: LEAN).
  float a2 = uRoughness * uRoughness + chopLost + detVar + 1.5e-6 * dist;
  a2 = clamp(a2, 2e-4, 0.5);
  float perceptual = sqrt(sqrt(a2));

  float NdotV = clamp(dot(N, V), 1e-3, 1.0);
  float F = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
  F = mix(F, 0.02 + 0.3 * (1.0 - NdotV), clamp(a2 * 2.0, 0.0, 0.4)); // rough surfaces dull the grazing rim

  // Whitecaps where the wind chop's Jacobian folds.
  float caps = smoothstep(0.62, 0.3, jdet);
  float cov = max(max(white, trail * 0.95), max(surf, caps * 0.7));
  float aer = clamp(max(white, trail * 0.75), 0.0, 1.0);

  // ---- water body: Lee et al. (1999) shallow-water reflectance along the refracted ray ------
  vec3 T = refract(-V, N, 1.0 / 1.333);
  float mu = clamp(-T.y, 0.2, 1.0);
  vec2 q = P.xz + T.xz * min(column / mu, 60.0);
  float Hb = max(P.y + oceanDepth(q), 0.0);
  q = P.xz + T.xz * min(Hb / mu, 60.0);
  Hb = max(P.y + oceanDepth(q), 0.0);

  float turb = uTurbidity * shallow;
  vec3 aW = uAbsorption + uSurfAbsorption * turb;
  vec3 bb = uBackscatter + uSurfBackscatter * turb + uBubbleBackscatter * (aer * aer);
  vec3 kap = aW + bb;
  vec3 u = bb / kap;
  vec3 rdp = (0.084 + 0.17 * u) * u;
  vec3 duC = 1.03 * sqrt(1.0 + 2.4 * u);
  vec3 duB = 1.04 * sqrt(1.0 + 5.4 * u);
  float sunUp = max(L.y, 0.0);
  float muS = sqrt(1.0 - (1.0 - sunUp * sunUp) / (1.333 * 1.333)); // refracted sun cosine
  vec3 attC = exp(-(1.0 / muS + duC / mu) * kap * Hb);
  vec3 attB = exp(-(1.0 / muS + duB / mu) * kap * Hb);
  vec3 EdSun = uSunColor * sunUp;
  vec3 Ed = EdSun + uSkyIrradiance;
  float sunShare = dot(EdSun, vec3(0.33)) / max(dot(Ed, vec3(0.33)), 1e-4);
  float cw = uCausticsIntensity * smoothstep(0.02, 0.2, Hb);
  float caust = cw > 0.0 ? mix(1.0, caustics(q, uTime, fp, Hb), cw) : 1.0;
  // (the seabed is invisible under deep water: skip its procedural albedo there)
  vec3 rho = max(attB.g, attB.b) > 1e-4 ? sandAlbedo(q, fp) * uBottomAlbedo : vec3(0.0);
  vec3 rrs = rdp * (1.0 - attC) + rho / PI * attB * (1.0 - sunShare + sunShare * caust);
  vec3 Rrs = 0.52 * rrs / (1.0 - 1.7 * rrs);
  vec3 body = Rrs * Ed;

  // ---- subsurface glow of thin crests and lips (backlit by the sun and the bright horizon) ----
  float crestH = clamp(etaS / max(H, 0.05) + 0.35, 0.0, 1.0);
  float steep = clamp(steepS * 1.2, 0.0, 1.0);
  float thin = crestH * crestH * (0.35 + 0.65 * steep) * smoothstep(0.15, 0.9, H);
  // the upper face of a steep, nearly breaking wave is a thin sheet of water (the lip)
  float lip = crestH * crestH * crestH * steep * smoothstep(0.55, 0.95, full) * smoothstep(0.3, 1.0, H);
  float path = mix(0.3 + 2.5 * (1.0 - crestH) * max(H, 0.3), 0.15 + 0.8 * (1.0 - crestH) * H, lip);
  vec3 trans = exp(-(aW + 4.0 * bb) * path);
  vec3 Ls = normalize(vec3(L.x, L.y * 0.35, L.z));    // refraction flattens the sun inside the wave
  float cosB = dot(V, -Ls);
  float back = pow(clamp(cosB, 0.0, 1.0), 3.0) * 0.12 + phaseHG(cosB, 0.6) * 0.25 * lip;
  float facing = clamp(0.5 - dot(N, L), 0.0, 1.0) + 0.25;
  vec3 scatterTint = bb / kap * 6.0 + vec3(0.1, 0.22, 0.18);
  vec3 sss = (uSunColor * (back * facing) + horizonColor(-V) * 0.25) * trans * scatterTint;
  body += sss * (thin + 0.8 * lip) * uSssIntensity;

  // ---- reflection + sun glitter ------------------------------------------------------------
  vec3 R = reflect(-V, N);
  float down = smoothstep(0.06, -0.12, R.y);
  R.y = max(R.y, 0.0);
  R = normalize(R + vec3(0.0, 0.002, 0.0));
  vec3 refl = envSample(R, perceptual);
  refl = mix(refl, body * 1.5 + horizonColor(R) * 0.1, down);

  float NdotL = dot(N, L);
  vec3 spec = vec3(0.0);
  if (NdotL > 0.0 && L.y > -0.02) {
    vec3 Hh = normalize(L + V);
    float NdotH = max(dot(N, Hh), 0.0);
    float VdotH = max(dot(V, Hh), 0.0);
    float Fs = 0.02 + 0.98 * pow(1.0 - VdotH, 5.0);
    float sa2 = max(a2, 0.0012);
    spec = uSunColor * (ggxD(NdotH, sa2) * smithG(NdotV, NdotL, sa2) * Fs * NdotL);
    spec = min(spec, vec3(3000.0));
  }

  // ---- foam ---------------------------------------------------------------------------------
  float foam = 0.0;
  vec3 foamCol = vec3(0.0);
  if (cov > 0.003) {
    float t = uTime;
    // Foam coordinates follow the wave profile; filtered with the geometric-mean footprint
    // (sharper at grazing angles; it is low contrast).
    vec2 fuv = vec2(P.x - 0.6 * P.y, P.z);
    float ffp = sqrt(fx * fy) + 1e-4;
    // dense, freshly churned whitewater (roller + the bore just behind it) vs decaying foam: a
    // mat with holes that opens up into a lace network as the coverage drops
    float fresh = max(white, trail * smoothstep(0.4, 0.9, trail));
    float act = smoothstep(0.05, 0.36, fresh);
    vec3 ww = act > 0.0 ? flowWhitewater(fuv * vec2(0.8, 1.0), vec2(c * 0.7 * fresh, 0.0), t, ffp) : vec3(0.5, 0.0, 0.0);
    float lace = act < 1.0 ? laceField(fuv * vec2(0.7, 1.0) + vec2(-0.1 * t, 0.025 * t), t, ffp) : 0.0;
    if (act < 1.0) {
      // foam left by the bore lies in streaks parallel to the crest; they carry the structure
      // where the lace strands are already filtered out (distance)
      vec2 cd = vSwellC.xy;
      vec2 sx = vec2(dot(P.xz, cd) * 0.55, dot(P.xz, vec2(-cd.y, cd.x)) * 0.06);
      float streak = fbm2(sx + 4.1, ffp * 0.55);
      lace = clamp(lace + (streak - 0.5) * 0.7 * smoothstep(0.03, 0.15, ffp), 0.0, 1.0);
    }
    vec3 rel = vec3(0.0);
    float crease = 0.0;
#if FOAM_DETAIL
    // isotropic relief is filtered with the long footprint axis (no streaks at grazing angles)
    rel = foamRelief(fuv, t, min(max(fx, fy), 2.5 * ffp), max(fx, fy), crease);
    // lace strands vary in width and break up into bubbly clumps
    lace *= 0.9 + 0.35 * rel.x;
#endif
    float fpat = mix(lace, ww.x, act);
    // The coverage threshold cuts through the billows and lumps: chunky, broken edges.
    float pat = fpat + 0.25 * rel.x * act - 0.12 * crease;
    float thr = 1.0 - cov;
    float soft = mix(0.11, 0.035, act) + 0.25 * min(ffp, 0.25);
    // at grazing angles the pattern changes much faster across screen rows than the footprint
    // suggests: widen the threshold transition to ~1 px of the pattern's own change (no stair steps)
    soft = max(soft, 0.7 * fwidth(pat));
    float edge = smoothstep(thr - soft, thr + soft, pat) * smoothstep(0.0, 0.2, cov);
    float thick = smoothstep(thr, thr + 0.35, pat) * mix(0.5, 1.0, smoothstep(0.25, 0.75, fpat)) * mix(0.6, 1.0, act);
    // the active roller itself is continuous along the breaking section (its texture comes from
    // the relief shading); the pattern only breaks up the leading edge and the trailing foam
    float core = smoothstep(0.8, 1.0, white) * smoothstep(0.25, 0.9, H);
    thick = max(thick, core * smoothstep(0.2, 0.6, ww.x + 0.3 * rel.x));
    // thin foam is a translucent film of bubbles with holes opening along the creases; thick
    // foam is opaque
    edge *= 1.0 - 0.75 * (1.0 - thick) * crease;
    foam = edge * mix(0.25, 1.0, thick) * mix(0.8, 1.0, act) * smoothstep(0.0, 0.35, cov);
    foam = max(foam, core * (0.88 + 0.12 * ww.x));
    foam = clamp(foam * uFoamIntensity, 0.0, 1.0);

    // relief: billows (advected with the bore) + micro lumps/bubbles; height drives occlusion
    float hgt = (ww.x - 0.5) * act * 1.4 + rel.x + 0.25 * (thick - 0.5);
    vec2 bump = (ww.yz * vec2(0.8, 1.0) * (0.45 * act) + rel.yz) * mix(0.35, 1.0, thick);
    vec3 Nf = bumpNormal(Ng, bump);
    float ao = mix(0.42, 1.0, smoothstep(-0.55, 0.35, hgt)) * (1.0 - 0.35 * crease);
    ao = mix(1.0, ao, mix(0.4, 1.0, thick));
    float ndl = dot(Nf, L);
    // thick foam: matte with self-shadowing lumps; thin foam: wrapped (light diffuses through)
    float sunTerm = mix(max(ndl * 0.6 + 0.4, 0.0), max(ndl, 0.0) * 0.9 + 0.1 * ao, thick) * mix(0.6, 1.0, ao);
    vec3 alb = vec3(0.86, 0.9, 0.93) * (0.85 + 0.15 * smoothstep(-0.3, 0.4, hgt));
    foamCol = alb / PI * (uSunColor * sunTerm + envIrradiance(Nf) * ao);
    // creases are bluish-grey: their light has travelled through aerated water
    foamCol *= mix(vec3(0.72, 0.86, 0.95), vec3(1.0), ao);
    // backlit thin foam glows (forward scattering through a few bubble layers)
    foamCol += uSunColor * alb * (phaseHG(dot(V, -L), 0.55) * 0.35 * (1.0 - 0.75 * thick));
  }

  vec3 col = refl * F + body * (1.0 - F) / 0.98 + spec * (1.0 - foam);
  col = mix(col, foamCol, foam);

  col = applyHaze(col, P, cameraPosition);

  if (uDebug == 1) {
    col = 0.5 + 0.5 * cos(6.2831 * (vLod.x * 0.17 + vec3(0.0, 0.33, 0.67)));
  } else if (uDebug == 2) {
    col = N * 0.5 + 0.5;
  } else if (uDebug == 3) {
    col = vec3(foam, white, trail);
  } else if (uDebug == 4) {
    col = vec3(brk, full * 0.5, clamp(ratio * 0.3, 0.0, 1.0));
  } else if (uDebug == 5) {
    col = body * 4.0;
  } else if (uDebug == 6) {
    col = vec3(0.0);
  } else if (uDebug == 7) {
    col = vec3(sqrt(a2) * 4.0, sqrt(detVar) * 4.0, sqrt(chopLost) * 4.0);
  } else if (uDebug == 8) {
    col = 0.5 + 0.5 * cos(6.2831 * (log2(fp) * 0.25 + vec3(0.0, 0.33, 0.67)));
  }

  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;
