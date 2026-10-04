/**
 * Shared GLSL helpers for the render module (WebGL2 / GLSL ES 3.00 through THREE.ShaderMaterial).
 *
 *  NOISE_GLSL     hashes, value noise, fbm, Worley (cellular) noise.
 *  ENV_GLSL       environment uniforms (sun, sky, PMREM env map) + sampling, haze (aerial
 *                 perspective) and sand albedo. Requires the PMREM defines from
 *                 Environment.envDefines and `#include <cube_uv_reflection_fragment>` before it
 *                 (done by ENV_GLSL itself).
 */

export const NOISE_GLSL = /* glsl */ `
#ifndef QUIVER_NOISE_GLSL
#define QUIVER_NOISE_GLSL
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
// Value noise in [0, 1] with quintic fade.
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = hash12(i);
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// Value noise with its analytic gradient: returns (value, d/dx, d/dy).
vec3 vnoiseD(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  float a = hash12(i);
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));
  float k1 = b - a;
  float k2 = c - a;
  float k4 = a - b - c + d;
  return vec3(a + k1 * u.x + k2 * u.y + k4 * u.x * u.y, du * vec2(k1 + k4 * u.y, k2 + k4 * u.x));
}
// fbm of value noise; octaves whose cell size is below ~2 pixels (fp = world size of a pixel at
// unit frequency) fade to their mean so distant patterns don't alias.
float fbm4(vec2 p, float fp) {
  float s = 0.0;
  float a = 0.5;
  float w = 0.0;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 4; i++) {
    float aa = 1.0 - smoothstep(0.25, 0.6, fp);
    s += a * mix(0.5, vnoise(p), aa);
    w += a;
    p = r * p * 2.03 + 17.1;
    fp *= 2.03;
    a *= 0.5;
  }
  return s / w;
}
// fbm4 with its analytic gradient: (value, d/dx, d/dy), footprint-filtered like fbm4.
vec3 fbm4D(vec2 p, float fp) {
  vec3 s = vec3(0.0);
  float a = 0.5;
  float w = 0.0;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  mat2 j = mat2(1.0);
  for (int i = 0; i < 4; i++) {
    float aa = 1.0 - smoothstep(0.25, 0.6, fp);
    vec3 n = vnoiseD(p);
    s += a * vec3(mix(0.5, n.x, aa), (n.yz * j) * aa);
    w += a;
    p = r * p * 2.03 + 17.1;
    j = r * j * 2.03;
    fp *= 2.03;
    a *= 0.5;
  }
  return s / w;
}
// Worley noise: (F1, F2) distances to the two nearest jittered feature points; jitter animates
// with phase t (radians).
vec2 worley(vec2 p, float t) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float f1 = 8.0;
  float f2 = 8.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 g = vec2(float(x), float(y));
      vec2 h = hash22(i + g);
      vec2 o = 0.5 + 0.38 * sin(t + 6.2831 * h);
      vec2 r = g + o - f;
      float d = dot(r, r);
      if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
    }
  }
  return sqrt(vec2(f1, f2));
}

// Dense whitewater: billowy, churning (domain-warped fbm), no cellular structure.
// Returns (density, d height/dx, d height/dy) where height is a smooth lumpy field for bump shading.
vec3 whitewaterField(vec2 x, float t, float fp) {
  vec2 w = vec2(fbm4(x * 0.3 + vec2(0.0, 0.35 * t), fp * 0.3), fbm4(x * 0.3 + vec2(5.2, 1.3 - 0.3 * t), fp * 0.3));
  vec3 lumps = fbm4D(x * 0.75 + w * 2.2 + vec2(0.2 * t, 0.0), fp * 0.75);
  float fine = fbm4(x * 3.1 + w * 1.5 - vec2(0.0, 0.5 * t), fp * 3.1);
  return vec3(clamp(0.75 * lumps.x + 0.35 * fine - 0.05, 0.0, 1.0), lumps.yz * 0.75);
}

// Decaying foam: lacy cellular network carried by blotchy patches; thresholding at
// (1 - coverage) leaves lacy patches and streaks as coverage drops.
float laceField(vec2 x, float t, float fp) {
  float n = fbm4(x * 0.11 + vec2(0.0, 0.01 * t), fp * 0.11);
  float patches = smoothstep(0.3, 0.75, n);
  vec2 warp = vec2(fbm4(x * 0.35, fp * 0.35), fbm4(x * 0.35 + 13.7, fp * 0.35)) - 0.5;
  float aa1 = 1.0 - smoothstep(0.2, 0.55, fp * 0.9);
  float aa2 = 1.0 - smoothstep(0.2, 0.55, fp * 2.7);
  vec2 w1 = worley(x * 0.9 + warp * 3.0, t * 0.2);
  float fine = fbm4(x * 1.3, fp * 1.3);
  float e1 = (1.0 - smoothstep(0.0, 0.16 + 0.35 * patches, w1.y - w1.x)) * smoothstep(0.3, 0.6, fine + 0.2 * patches);
  vec2 w2 = worley(x * 2.7 + warp * 4.0 + 7.3, t * 0.35);
  float e2 = 1.0 - smoothstep(0.0, 0.2 + 0.2 * patches, w2.y - w2.x);
  float lace = max(mix(0.4, e1, aa1), 0.8 * mix(0.4, e2, aa2));
  return clamp(patches * (0.45 + 0.55 * lace) * (0.75 + 0.25 * fine) + 0.25 * n * lace, 0.0, 1.0);
}
#endif
`;

export const ENV_GLSL = /* glsl */ `
#ifndef QUIVER_ENV_GLSL
#define QUIVER_ENV_GLSL
#include <cube_uv_reflection_fragment>
uniform vec3 uSunDirection;     // unit vector toward the sun
uniform vec3 uSunColor;         // sun irradiance at normal incidence (linear, sky units)
uniform vec3 uSkyIrradiance;    // sky irradiance on a horizontal surface
uniform sampler2D uEnvMap;      // PMREM (cube UV) of the sky without the sun disc
uniform float uHazeDensity;     // aerial-perspective extinction, 1/m
uniform float uEnvIntensity;

#ifdef ENVMAP_TYPE_CUBE_UV
vec3 envSample(vec3 dir, float roughness) {
  return textureCubeUV(uEnvMap, dir, roughness).rgb * uEnvIntensity;
}
#else
vec3 envSample(vec3 dir, float roughness) { return uSkyIrradiance / 3.14159265; }
#endif

// Cosine-weighted sky irradiance arriving at a surface with normal n.
vec3 envIrradiance(vec3 n) {
  return 3.14159265 * envSample(n, 1.0);
}

// Sky colour just above the horizon in the azimuth of dir (what distant things fade into).
vec3 horizonColor(vec3 dir) {
  vec2 h = dir.xz;
  float l = length(h);
  h = l > 1e-4 ? h / l : vec2(1.0, 0.0);
  return envSample(normalize(vec3(h.x, 0.035, h.y)), 0.18);
}

// Aerial perspective: blend toward the horizon sky with distance.
vec3 applyHaze(vec3 col, vec3 worldPos, vec3 camPos) {
  vec3 d = worldPos - camPos;
  float dist = length(d);
  float t = exp(-uHazeDensity * dist);
  return mix(horizonColor(d / max(dist, 1e-3)), col, t);
}

// Dry beach sand albedo (linear) with subtle large-scale variation; fp = metres per pixel.
vec3 sandAlbedo(vec2 xz, float fp) {
  float n = fbm4(xz * 0.045, fp * 0.045);
  float m = fbm4(xz * 0.9 + 31.0, fp * 0.9);
  vec3 base = vec3(0.6, 0.47, 0.3);
  vec3 tint = vec3(0.52, 0.44, 0.32);
  vec3 c = mix(base, tint, smoothstep(0.35, 0.7, n));
  return c * (0.86 + 0.22 * n) * (0.94 + 0.12 * m);
}
#endif
`;
