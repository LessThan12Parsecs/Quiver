/**
 * GLSL mirror of the CPU wave model (waveModel.ts / oceanTables.ts / dispersion.ts).
 *
 * Every function here is a line-by-line translation of its TypeScript counterpart; if you
 * change one, change the other and run `npm run check:gpu` (tools/gpuCheck) which compares the
 * two on the GPU.
 *
 * Usage (WebGL2 / GLSL ES 3.00, e.g. THREE.ShaderMaterial with glslVersion = THREE.GLSL3 or the
 * default WebGL2 shader chunk path):
 *
 *   const ocean = new OceanShaderData(model);
 *   material.uniforms = { ...material.uniforms, ...ocean.uniforms };
 *   vertexShader = OCEAN_GLSL + `...  OceanPoint p = oceanSurface(restXZ); ...`;
 *   every frame:  model.setTime(t); ocean.update(model);
 *   after model.rebuild(): ocean.rebuild(model);
 */
import * as THREE from 'three';
import { MAX_SWELLS, MAX_WIND_WAVES, SET_LENGTH } from './oceanConfig';
import { BREAK_FULL, BREAK_START, TimeSlot, type OceanModel } from './waveModel';

const f = (v: number): string => (Number.isInteger(v) ? v.toFixed(1) : String(v));

export const OCEAN_GLSL = /* glsl */ `
#ifndef OCEAN_WAVES_GLSL
#define OCEAN_WAVES_GLSL
#define OCEAN_MAX_SWELLS ${MAX_SWELLS}
#define OCEAN_MAX_WIND ${MAX_WIND_WAVES}
#define OCEAN_SET_LENGTH ${SET_LENGTH}
#define OCEAN_PI 3.141592653589793
#define OCEAN_TWO_PI 6.283185307179586

uniform int uSwellCount;
uniform vec4 uSwellA[OCEAN_MAX_SWELLS];      // omega, ky, amp0, envelope offset
uniform vec4 uSwellB[OCEAN_MAX_SWELLS];      // sinT0, cosT0, cg0, c0
uniform float uSwellHDeep[OCEAN_MAX_SWELLS];
uniform float uSetPattern[OCEAN_MAX_SWELLS * OCEAN_SET_LENGTH];
uniform int uWindCount;
uniform vec4 uWindA[OCEAN_MAX_WIND];         // dirX, dirZ, k, phase offset
uniform vec4 uWindB[OCEAN_MAX_WIND];         // amp, ampH, omega, 0
uniform vec4 uOceanParams0;                  // g, breakerIndex, peakingMax, minDepth
uniform vec4 uOceanParams1;                  // ursellStart, ursellFull, skewMax, skewPhase
uniform highp sampler2D uPhaseTable;         // RGBA32F: Phi, kx, baseDepth, baseCap
uniform vec4 uPhaseInfo;                     // xMin, dx, size, 0
uniform highp sampler2D uOceanField;         // RG32F: depth, cap
uniform vec4 uFieldInfo;                     // xMin, zMin, dx, dz
uniform vec2 uFieldSize;                     // nx, nz

struct OceanSwell {
  float eta;
  vec2 vel;        // horizontal water velocity
  float depth;
  float height;
  float ratio;
  float fullness;
  float breaking;
  float peaking;
  float skew;
  float psi;       // dominant waveform phase, 0 = crest, (0, pi) = front face
  float speed;     // dominant phase speed
  vec2 dir;        // dominant direction
  float k;         // dominant local wavenumber
};

struct OceanPoint {
  vec3 position;   // world position of the displaced rest point
  vec3 wind;       // wind-chop displacement (xz) and elevation (y)
  OceanSwell swell;
};

vec2 oceanPhaseLookup(int row, float x) {
  float n = uPhaseInfo.z;
  float dx = uPhaseInfo.y;
  float u = (x - uPhaseInfo.x) / dx;
  if (u <= 0.0) {
    vec4 t0 = texelFetch(uPhaseTable, ivec2(0, row), 0);
    return vec2(t0.x + t0.y * (x - uPhaseInfo.x), t0.y);
  }
  if (u >= n - 1.0) {
    vec4 t1 = texelFetch(uPhaseTable, ivec2(int(n) - 1, row), 0);
    return vec2(t1.x + t1.y * (x - (uPhaseInfo.x + (n - 1.0) * dx)), t1.y);
  }
  float jf = floor(u);
  float t = u - jf;
  int j = int(jf);
  vec4 a = texelFetch(uPhaseTable, ivec2(j, row), 0);
  vec4 b = texelFetch(uPhaseTable, ivec2(j + 1, row), 0);
  float t2 = t * t;
  float t3 = t2 * t;
  float phi = (2.0 * t3 - 3.0 * t2 + 1.0) * a.x + (t3 - 2.0 * t2 + t) * dx * a.y
            + (-2.0 * t3 + 3.0 * t2) * b.x + (t3 - t2) * dx * b.y;
  return vec2(phi, a.y + (b.y - a.y) * t);
}

// x: depth (positive below sea level), y: breaking-history cap.
vec2 oceanField(vec2 p) {
  float fu = (p.x - uFieldInfo.x) / uFieldInfo.z;
  float fv = (p.y - uFieldInfo.y) / uFieldInfo.w;
  if (fu >= 0.0 && fv >= 0.0 && fu <= uFieldSize.x - 1.0 && fv <= uFieldSize.y - 1.0) {
    int i = min(int(floor(fu)), int(uFieldSize.x) - 2);
    int j = min(int(floor(fv)), int(uFieldSize.y) - 2);
    float tx = fu - float(i);
    float tz = fv - float(j);
    vec2 a0 = texelFetch(uOceanField, ivec2(i, j), 0).xy;
    vec2 a1 = texelFetch(uOceanField, ivec2(i + 1, j), 0).xy;
    vec2 b0 = texelFetch(uOceanField, ivec2(i, j + 1), 0).xy;
    vec2 b1 = texelFetch(uOceanField, ivec2(i + 1, j + 1), 0).xy;
    vec2 a = a0 + (a1 - a0) * tx;
    vec2 b = b0 + (b1 - b0) * tx;
    return a + (b - a) * tz;
  }
  float n = uPhaseInfo.z;
  float u = clamp((p.x - uPhaseInfo.x) / uPhaseInfo.y, 0.0, n - 1.0);
  int j = min(int(floor(u)), int(n) - 2);
  float t = u - float(j);
  vec4 a = texelFetch(uPhaseTable, ivec2(j, 0), 0);
  vec4 b = texelFetch(uPhaseTable, ivec2(j + 1, 0), 0);
  return a.zw + (b.zw - a.zw) * t;
}

float oceanDepth(vec2 p) { return oceanField(p).x; }

float oceanWaveNumber(float omega, float h, float g) {
  float kh0 = omega * omega * h / g;
  float kh = kh0 / pow(1.0 - exp(-pow(kh0, 1.25)), 0.4);
  for (int i = 0; i < 2; i++) {
    float t = tanh(kh);
    float fv = kh * t - kh0;
    float df = t + kh * (1.0 - t * t);
    kh -= fv / df;
  }
  return kh / h;
}

float oceanWrapPi(float a) { return a - OCEAN_TWO_PI * floor((a + OCEAN_PI) / OCEAN_TWO_PI); }

float oceanSoftCap(float r) { return r <= 0.8 ? r : 1.0 - 0.2 * exp(-(r - 0.8) / 0.2); }

float oceanPeakMean(float p) {
  float q = p + 0.25;
  return 1.0 / sqrt(OCEAN_PI * (q + 1.0 / (32.0 * q)));
}

float oceanSolveSkew(float phi, float kappa, float delta) {
  float M = oceanWrapPi(phi - delta - 0.5 * OCEAN_PI);
  float E = M + 0.85 * kappa * (sin(M) >= 0.0 ? 1.0 : -1.0);
  for (int i = 0; i < 4; i++) {
    float s = sin(E);
    float c = cos(E);
    float fv = E - kappa * s - M;
    float fp = 1.0 - kappa * c;
    float fpp = kappa * s;
    E -= fv / (fp - 0.5 * fv * fpp / fp);
  }
  return oceanWrapPi(E + delta + 0.5 * OCEAN_PI);
}

float oceanSetEnvelope(int row, float s) {
  float L = float(OCEAN_SET_LENGTH);
  float fi = floor(s);
  float fr = s - fi;
  int base = row * OCEAN_SET_LENGTH;
  float p0 = uSetPattern[base + int(mod(fi - 1.0, L))];
  float p1 = uSetPattern[base + int(mod(fi, L))];
  float p2 = uSetPattern[base + int(mod(fi + 1.0, L))];
  float p3 = uSetPattern[base + int(mod(fi + 2.0, L))];
  float f2 = fr * fr;
  float f3 = f2 * fr;
  float v = 0.5 * (2.0 * p1 + (-p0 + p2) * fr + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
          + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3);
  return max(v, 0.0);
}

OceanSwell oceanSwell(vec2 p) {
  float g = uOceanParams0.x;
  float gamma = uOceanParams0.y;
  vec2 fld = oceanField(p);
  float h = fld.x;
  float hEff = max(h, uOceanParams0.w);
  float sH[OCEAN_MAX_SWELLS];
  float sPhi[OCEAN_MAX_SWELLS];
  float sC[OCEAN_MAX_SWELLS];
  float sK[OCEAN_MAX_SWELLS];
  vec2 sD[OCEAN_MAX_SWELLS];
  float hLin = 0.0;
  float k0 = 1.0;
  int dom = 0;
  for (int i = 0; i < OCEAN_MAX_SWELLS; i++) {
    sH[i] = 0.0; sPhi[i] = 0.0; sC[i] = 0.0; sK[i] = 1.0; sD[i] = vec2(1.0, 0.0);
    if (i >= uSwellCount) continue;
    vec4 A = uSwellA[i];
    vec4 B = uSwellB[i];
    vec2 lk = oceanPhaseLookup(i, p.x);
    float kxT = lk.y;
    float phase = lk.x + A.y * p.y + A.w;
    float env = oceanSetEnvelope(i, phase / OCEAN_TWO_PI);
    float hc = min(hEff, uSwellHDeep[i]);
    float k = oceanWaveNumber(A.x, hc, g);
    float kh = k * hc;
    float c = A.x / k;
    float cg = 0.5 * (1.0 + 2.0 * kh / sinh(2.0 * kh)) * c;
    float sinT = B.x * c / B.w;
    float cosT = sqrt(max(1.0 - sinT * sinT, 1e-4));
    float K = sqrt(B.z / cg) * sqrt(B.y / cosT);
    if (i == 0) k0 = K;
    float H = 2.0 * A.z * env * K;
    float kg = sqrt(kxT * kxT + A.y * A.y);
    sH[i] = H;
    sPhi[i] = phase;
    sC[i] = c;
    sK[i] = k;
    sD[i] = vec2(kxT, A.y) / kg;
    hLin += H;
    if (H > sH[dom]) dom = i;
  }
  float hLim = fld.y * k0;
  float ratio = hLin / max(hLim, 1e-4);
  float capScale = ratio > 0.8 ? oceanSoftCap(ratio) / ratio : 1.0;
  float height = hLin * capScale;
  float fullness = height / max(gamma * h, 1e-4);
  float breaking = smoothstep(${f(BREAK_START)}, ${f(BREAK_FULL)}, ratio) * smoothstep(0.65, 0.9, fullness);
  float kDom = uSwellCount > 0 ? sK[dom] : 1.0;
  float lambda = OCEAN_TWO_PI / kDom;
  float ursell = height * lambda * lambda / (hEff * hEff * hEff);
  float pk = 1.0 + (uOceanParams0.z - 1.0) * smoothstep(uOceanParams1.x, uOceanParams1.y, ursell);
  pk += (1.6 - pk) * 0.6 * breaking;
  float kappa = uOceanParams1.z * smoothstep(0.5, 0.97, fullness);
  float mean = oceanPeakMean(pk);
  float eta = 0.0;
  vec2 vel = vec2(0.0);
  float domPsi = 0.0;
  for (int i = 0; i < OCEAN_MAX_SWELLS; i++) {
    if (i >= uSwellCount) continue;
    float psi = oceanSolveSkew(sPhi[i], kappa, uOceanParams1.w);
    float y = sH[i] * capScale * (pow(0.5 + 0.5 * cos(psi), pk) - mean);
    eta += y;
    float c = sC[i];
    float u = c - sqrt(max(c * c - 2.0 * g * y, 0.0025 * c * c));
    vel += sD[i] * u;
    if (i == dom) domPsi = psi;
  }
  if (uSwellCount > 0 && breaking > 0.0) {
    float c = sC[dom];
    float along = dot(vel, sD[dom]);
    float target = 0.9 * c;
    float mask = smoothstep(0.3, 0.9, cos(domPsi - 0.6));
    if (along < target) vel += sD[dom] * ((target - along) * breaking * mask);
  }
  OceanSwell o;
  o.eta = eta;
  o.vel = vel;
  o.depth = h;
  o.height = height;
  o.ratio = ratio;
  o.fullness = fullness;
  o.breaking = breaking;
  o.peaking = pk;
  o.skew = kappa;
  o.psi = domPsi;
  o.speed = uSwellCount > 0 ? sC[dom] : 0.0;
  o.dir = uSwellCount > 0 ? sD[dom] : vec2(1.0, 0.0);
  o.k = uSwellCount > 0 ? sK[dom] : 0.0;
  return o;
}

// Wind chop at rest point x0: xz = horizontal displacement, y = elevation.
vec3 oceanWind(vec2 x0) {
  float fade = smoothstep(0.1, 1.5, oceanField(x0).x);
  vec3 d = vec3(0.0);
  for (int j = 0; j < OCEAN_MAX_WIND; j++) {
    if (j >= uWindCount) break;
    vec4 A = uWindA[j];
    vec4 B = uWindB[j];
    float ph = A.z * (A.x * x0.x + A.y * x0.y) + A.w;
    float s = sin(ph);
    float c = cos(ph);
    float hd = -B.y * fade * s;
    d.x += A.x * hd;
    d.z += A.y * hd;
    d.y += B.x * fade * c;
  }
  return d;
}

// Full surface point for a rest grid point x0 (what the ocean mesh renders).
OceanPoint oceanSurface(vec2 x0) {
  OceanPoint o;
  o.wind = oceanWind(x0);
  vec2 p = x0 + o.wind.xz;
  o.swell = oceanSwell(p);
  o.position = vec3(p.x, o.wind.y + o.swell.eta, p.y);
  return o;
}
#endif
`;

/** Uniform values + data textures for OCEAN_GLSL, kept in sync with an OceanModel. */
export class OceanShaderData {
  readonly uniforms: Record<string, THREE.IUniform>;
  private phaseTexture!: THREE.DataTexture;
  private fieldTexture!: THREE.DataTexture;

  constructor(model: OceanModel) {
    this.uniforms = {
      uSwellCount: { value: 0 },
      uSwellA: { value: Array.from({ length: MAX_SWELLS }, () => new THREE.Vector4()) },
      uSwellB: { value: Array.from({ length: MAX_SWELLS }, () => new THREE.Vector4()) },
      uSwellHDeep: { value: new Float32Array(MAX_SWELLS) },
      uSetPattern: { value: new Float32Array(MAX_SWELLS * SET_LENGTH) },
      uWindCount: { value: 0 },
      uWindA: { value: Array.from({ length: MAX_WIND_WAVES }, () => new THREE.Vector4()) },
      uWindB: { value: Array.from({ length: MAX_WIND_WAVES }, () => new THREE.Vector4()) },
      uOceanParams0: { value: new THREE.Vector4() },
      uOceanParams1: { value: new THREE.Vector4() },
      uPhaseTable: { value: null },
      uPhaseInfo: { value: new THREE.Vector4() },
      uOceanField: { value: null },
      uFieldInfo: { value: new THREE.Vector4() },
      uFieldSize: { value: new THREE.Vector2() },
    };
    this.rebuild(model);
  }

  /** Re-upload tables and constants after OceanModel.rebuild(). */
  rebuild(model: OceanModel): void {
    const u = this.uniforms;
    const t = model.tables;
    this.phaseTexture?.dispose();
    this.fieldTexture?.dispose();
    this.phaseTexture = makeFloatTexture(t.phase, t.phaseSize, MAX_SWELLS, THREE.RGBAFormat);
    this.fieldTexture = makeFloatTexture(t.field, t.fieldNx, t.fieldNz, THREE.RGFormat);
    u.uPhaseTable.value = this.phaseTexture;
    u.uOceanField.value = this.fieldTexture;
    (u.uPhaseInfo.value as THREE.Vector4).set(t.phaseXMin, t.phaseDx, t.phaseSize, 0);
    (u.uFieldInfo.value as THREE.Vector4).set(t.fieldXMin, t.fieldZMin, t.fieldDx, t.fieldDz);
    (u.uFieldSize.value as THREE.Vector2).set(t.fieldNx, t.fieldNz);

    const c = model.config;
    (u.uOceanParams0.value as THREE.Vector4).set(c.gravity, c.breakerIndex, c.peakingMax, c.minDepth);
    (u.uOceanParams1.value as THREE.Vector4).set(c.ursellStart, c.ursellFull, c.skewMax, c.skewPhase);

    u.uSwellCount.value = model.swells.length;
    const hDeep = u.uSwellHDeep.value as Float32Array;
    const pattern = u.uSetPattern.value as Float32Array;
    const B = u.uSwellB.value as THREE.Vector4[];
    model.swells.forEach((s, i) => {
      B[i].set(s.sinT0, s.cosT0, s.cg0, s.c0);
      hDeep[i] = s.hDeep;
      pattern.set(s.pattern, i * SET_LENGTH);
    });

    u.uWindCount.value = model.wind.length;
    const WB = u.uWindB.value as THREE.Vector4[];
    model.wind.forEach((w, j) => WB[j].set(w.amp, w.ampH, w.omega, 0));
    this.update(model);
  }

  /** Upload the time-dependent phase offsets (call after model.setTime each frame). */
  update(model: OceanModel): void {
    const u = this.uniforms;
    const A = u.uSwellA.value as THREE.Vector4[];
    const so = model.swellOffsets[TimeSlot.Now];
    model.swells.forEach((s, i) => A[i].set(s.omega, s.ky, s.amp0, so[i]));
    const WA = u.uWindA.value as THREE.Vector4[];
    const wo = model.windOffsets[TimeSlot.Now];
    model.wind.forEach((w, j) => WA[j].set(w.dirX, w.dirZ, w.k, wo[j]));
  }

  dispose(): void {
    this.phaseTexture?.dispose();
    this.fieldTexture?.dispose();
  }
}

function makeFloatTexture(
  data: Float32Array,
  width: number,
  height: number,
  format: THREE.PixelFormat,
): THREE.DataTexture {
  const tex = new THREE.DataTexture(data, width, height, format, THREE.FloatType);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}
