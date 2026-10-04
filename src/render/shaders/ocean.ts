/**
 * Water surface shaders (CDLOD ocean mesh, see OceanMesh.ts).
 *
 * Vertex: each instanced patch vertex is morphed toward the next coarser grid near the end of
 * its LOD range (CDLOD), then displaced with OCEAN_GLSL: wind chop (Lagrangian, components
 * shorter than ~3 grid cells faded out — identical to oceanSurface() where the grid resolves
 * them) and the shoaling swell. The swell gradient is taken by central differences over one
 * grid cell; the chop gradient is analytic in the fragment shader.
 *
 * Fragment: per-pixel normal = swell slope (interpolated) + analytic chop slope + capillary
 * ripples (both filtered by the pixel footprint, lost slope variance goes into roughness);
 * Schlick Fresnel, PMREM sky reflection, GGX sun glitter, water body colour from single
 * scattering + Beer-Lambert over the sand seabed along the refracted ray, subsurface glow of
 * thin backlit crests, foam (breaking crests/faces, trailing bores, lingering surf-zone lace,
 * whitecaps, swash) and aerial perspective toward the horizon sky.
 */
import { OCEAN_GLSL } from '../../ocean/waveGLSL';
import { ENV_GLSL, NOISE_GLSL } from './common';

export const RIPPLE_COUNT = 24;
export const MAX_LODS = 16;

const VARYINGS = /* glsl */ `
varying vec3 vWorldPos;
varying vec2 vRest;
varying vec2 vSwellSlope;
varying vec4 vSwellA;   // swell eta, local wave height, breaking, fullness
varying vec4 vSwellB;   // cos psi, sin psi, ratio, still-water depth
varying vec4 vSwellC;   // dominant direction xz, phase speed, wavenumber
varying vec2 vLod;      // lod level + morph, grid spacing
`;

export const OCEAN_VERTEX = /* glsl */ `
${OCEAN_GLSL}
#define MAX_LODS ${MAX_LODS}
uniform vec3 uLodOrigin;
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
  float morph = clamp((length(dl) - mr.x) * mr.y, 0.0, 1.0);
  vec2 rest = rest0 - fract(g * 0.5) * 2.0 * morph * cell;
  float spacing = cell * (1.0 + morph);

  vec3 w = oceanWindLod(rest, spacing);
  vec2 p = rest + w.xz;
  OceanSwell sw = oceanSwell(p);
  // Far away (grid >~ 20 m) the swell is sub-pixel: fade it rather than alias it.
  float swellFade = 1.0 - smoothstep(16.0, 40.0, spacing);
  float h = 0.5 * spacing;
  float ex = oceanSwell(p + vec2(h, 0.0)).eta - oceanSwell(p - vec2(h, 0.0)).eta;
  float ez = oceanSwell(p + vec2(0.0, h)).eta - oceanSwell(p - vec2(0.0, h)).eta;
  vSwellSlope = vec2(ex, ez) * (swellFade / (2.0 * h));

  vec3 pos = vec3(p.x, w.y + sw.eta * swellFade, p.y);
  vWorldPos = pos;
  vRest = rest;
  vSwellA = vec4(sw.eta * swellFade, sw.height * swellFade, sw.breaking, sw.fullness);
  vSwellB = vec4(cos(sw.psi), sin(sw.psi), min(sw.ratio, 8.0), sw.depth);
  vSwellC = vec4(sw.dir, sw.speed, sw.k);
  vLod = vec2(aNode.w + morph, spacing);
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
}
`;

export const OCEAN_FRAGMENT = /* glsl */ `
${NOISE_GLSL}
${ENV_GLSL}
${OCEAN_GLSL}
#define RIPPLE_COUNT ${RIPPLE_COUNT}
#define PI 3.14159265

uniform float uTime;
uniform vec4 uRipA[RIPPLE_COUNT];          // dirX, dirZ, k, phase (relative to uRipOrigin)
uniform vec4 uRipAmp[RIPPLE_COUNT / 4];    // slope amplitude a*k, packed by 4
uniform vec2 uRipOrigin;
uniform vec2 uWindDrift;                   // wind direction * gust advection speed (m/s)
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
uniform int uDebug;
${VARYINGS}

float rippleAmp(int i) { return uRipAmp[i / 4][i - (i / 4) * 4]; }

// Fraction of a wave component (wavenumber k, direction d) kept at the pixel footprint: the
// footprint is projected onto the component's direction (exact anisotropic filtering of a
// sinusoid), components with fewer than ~3 px per wavelength fade out.
float keepFrac(float k, vec2 d, vec2 dpx, vec2 dpy) {
  float fpd = max(abs(dot(d, dpx)), abs(dot(d, dpy))) + 1e-4;
  return smoothstep(1.6, 4.5, OCEAN_TWO_PI / (k * fpd));
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

// Capillary-gravity ripples (lambda < ~1.2 m), normals only.
vec2 rippleSlope(vec2 x, vec2 dpx, vec2 dpy, float gust, out float lostVar) {
  vec2 s = vec2(0.0);
  lostVar = 0.0;
  for (int i = 0; i < RIPPLE_COUNT; i++) {
    vec4 R = uRipA[i];
    float amp = rippleAmp(i) * gust;
    float aa = keepFrac(R.z, R.xy, dpx, dpy);
    float ph = R.z * dot(R.xy, x) + R.w;
    s -= R.xy * (amp * aa * sin(ph));
    lostVar += 0.5 * amp * amp * (1.0 - aa * aa);
  }
  return s;
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

// Animated caustic network on the seabed (mean ~1); sharp in very shallow water, blurring with depth.
float caustics(vec2 x, float t, float fp, float depth) {
  float aa = (1.0 - smoothstep(0.015, 0.06, fp)) * (1.0 - smoothstep(0.4, 2.5, depth));
  if (aa <= 0.0) return 1.0;
  vec2 warp = vec2(vnoise(x * 0.9 + 0.3 * t), vnoise(x * 0.9 + 9.1 - 0.3 * t)) - 0.5;
  vec2 w1 = worley(x * 2.3 + warp, t * 1.2);
  vec2 w2 = worley(x * 3.1 + warp * 1.3 + 4.7, -t * 1.0);
  float e1 = 1.0 - smoothstep(0.0, 0.16, w1.y - w1.x);
  float e2 = 1.0 - smoothstep(0.0, 0.16, w2.y - w2.x);
  float c = 0.55 * e1 + 0.45 * e2 + 1.2 * e1 * e2;
  return mix(1.0, 0.72 + 1.1 * c, aa);
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

  // Pixel footprint in metres (anisotropy limited to 8:1 so grazing angles don't over-blur).
  vec3 dpx = dFdx(P);
  vec3 dpy = dFdy(P);
  float fx = length(dpx);
  float fy = length(dpy);
  float fp = max(sqrt(fx * fy), max(fx, fy) * 0.125) + 1e-4;

  // Wind gusts / cat's paws: slowly drifting patches of rougher water.
  vec2 gp = (P.xz - uWindDrift * uTime) / 60.0;
  float gust = 0.4 + 1.2 * vnoise(gp) * (0.55 + 0.45 * vnoise(gp * 2.7 + 5.0));

  float jdet;
  float chopLost;
  float ripLost;
  vec2 slope = vSwellSlope + chopSlope(vRest, dpx.xz, dpy.xz, jdet, chopLost);
  vec2 xr = vec2(P.x - P.y, P.z) - uRipOrigin;
  slope += rippleSlope(xr, dpx.xz, dpy.xz, gust, ripLost);
  // Irregular capillary texture on top of the discrete ripples (breaks up their interference).
  vec3 cap = fbm4D(xr * 4.0 + uWindDrift * (uTime * 1.6), fp * 4.0);
  slope += cap.yz * (0.012 * gust);
  vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
  if (!gl_FrontFacing) {
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

  // Roughness: base + footprint-filtered slope variance (mss ~ alpha^2) + geometric specular AA.
  vec3 dnx = dFdx(N);
  vec3 dny = dFdy(N);
  float geoVar = 0.5 * max(dot(dnx, dnx), dot(dny, dny));
  float a2 = uRoughness * uRoughness + (chopLost + ripLost) + min(geoVar, 0.1) + 1.5e-6 * dist;
  a2 = clamp(a2, 2e-4, 0.5);
  float perceptual = sqrt(sqrt(a2));

  float NdotV = clamp(dot(N, V), 1e-3, 1.0);
  float F = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
  F = mix(F, 0.02 + 0.3 * (1.0 - NdotV), clamp(a2 * 2.0, 0.0, 0.4)); // rough surfaces dull the grazing rim

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

  // ---- foam -----------------------------------------------------------------------------
  // Active whitewater (the roller): a ragged line along the lip at breaking onset, spreading down
  // the face as the breaker develops. Its width scales with the wave height (in metres, converted
  // to waveform phase; the forward lean compresses the face ~2x in phase).
  float bs = smoothstep(0.02, 0.8, brk);
  float kDom = max(vSwellC.w, 0.02);
  float faceExt = clamp(2.0 * kDom * (0.5 + 2.4 * H), 0.2, 1.8) * mix(0.35, 1.0, bs);
  float backExt = clamp(kDom * (0.4 + 1.2 * H), 0.08, 0.5);
  // ragged edges: jitter the phase window with ~2-4 m noise along the crest
  float psiJ = psi + (vnoise(P.xz * vec2(0.35, 0.25)) - 0.5) * 0.6 * faceExt + (vnoise(P.xz * 1.1 + 7.0) - 0.5) * 0.15;
  float onFace = smoothstep(-backExt - 0.15, -backExt * 0.3, psiJ) * (1.0 - smoothstep(faceExt * 0.7, faceExt, psiJ));
  float white = onFace * smoothstep(0.0, 0.45, brk) * smoothstep(0.05, 0.7, H);
  // Trailing foam carpet left behind the roller, thinning out with time since the crest passed;
  // bigger breakers leave more and longer-lasting foam.
  float age = mod(-psi, 2.0 * PI) / omega;
  float broken = smoothstep(0.92, 1.35, ratio) * smoothstep(0.35, 0.7, full + brk);
  float trail = broken * exp(-age / (1.5 + 2.5 * H)) * smoothstep(0.08, 0.9, H);
  // Lingering lace over the inner surf zone.
  float surf = 0.3 * smoothstep(1.0, 2.2, ratio) * smoothstep(0.1, 0.6, H);
  // Whitecaps where the wind chop's Jacobian folds.
  float caps = smoothstep(0.62, 0.3, jdet);
  float cov = max(max(white, trail * 0.95), max(surf, caps * 0.7));
  float foam = 0.0;
  float aer = 0.0;
  float foamShade = 1.0;
  vec3 Nf = N;
  if (cov > 0.003) {
    float t = uTime;
    // Foam coordinates follow the wave profile (x - y) so steep faces aren't smeared. Foam is
    // filtered with the geometric-mean footprint (sharper at grazing angles; it is low contrast).
    vec2 fuv = vec2(P.x - P.y, P.z);
    float ffp = sqrt(fx * fy) + 1e-4;
    float act = smoothstep(0.04, 0.3, max(white, trail));
    vec3 ww = act > 0.0 ? flowWhitewater(fuv * vec2(0.8, 1.0), vec2(c * 0.7 * max(white, trail), 0.0), t, ffp) : vec3(0.0);
    float lace = act < 1.0 ? laceField(fuv * vec2(0.7, 1.0) + vec2(-0.1 * t, 0.025 * t), t, ffp) : 0.0;
    float fpat = mix(lace, ww.x, act);
    float thr = 1.0 - cov;
    float edge = smoothstep(thr - 0.05, thr + 0.08, fpat) * smoothstep(0.0, 0.2, cov);
    float thick = smoothstep(thr, thr + 0.35, fpat) * mix(0.55, 1.0, smoothstep(0.25, 0.75, fpat));
    foam = edge * (0.3 + 0.7 * thick) * smoothstep(0.0, 0.35, cov);
    foam = max(foam, smoothstep(0.85, 1.0, white) * smoothstep(0.25, 0.9, H) * (0.8 + 0.2 * ww.x));
    foam = clamp(foam * uFoamIntensity, 0.0, 1.0);
    foamShade = 0.78 + 0.22 * fpat;
    aer = clamp(max(white, trail * 0.75), 0.0, 1.0);
    // Lumpy whitewater: bump from the analytic gradient of the billow height.
    Nf = bumpNormal(N, ww.yz * vec2(0.8, 1.0) * (0.35 * act));
  }

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
  vec3 rho = sandAlbedo(q, fp) * uBottomAlbedo;
  vec3 rrs = rdp * (1.0 - attC) + rho / PI * attB * (1.0 - sunShare + sunShare * caust);
  vec3 Rrs = 0.52 * rrs / (1.0 - 1.7 * rrs);
  vec3 body = Rrs * Ed;

  // ---- subsurface glow of thin crests (backlit by the sun and the bright horizon) ----------
  float crestH = clamp(etaS / max(H, 0.05) + 0.35, 0.0, 1.0);
  float steep = clamp(length(vSwellSlope) * 1.2, 0.0, 1.0);
  float thin = crestH * crestH * (0.35 + 0.65 * steep) * smoothstep(0.15, 0.9, H);
  float path = 0.3 + 2.5 * (1.0 - crestH) * max(H, 0.3);
  vec3 trans = exp(-(aW + 4.0 * bb) * path);
  float back = pow(clamp(dot(V, -normalize(vec3(L.x, L.y * 0.35, L.z))), 0.0, 1.0), 3.0);
  float facing = clamp(0.5 - dot(N, L), 0.0, 1.0) + 0.25;
  vec3 scatterTint = bb / kap * 6.0 + vec3(0.1, 0.22, 0.18);
  vec3 sss = (uSunColor * (back * facing * 0.12) + horizonColor(-V) * 0.25) * trans * scatterTint;
  body += sss * thin * uSssIntensity;

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

  vec3 col = refl * F + body * (1.0 - F) / 0.98 + spec * (1.0 - foam);

  // ---- foam shading (matte, slightly translucent: wrap lighting) ---------------------------
  if (foam > 0.0) {
    float ndl = dot(Nf, L);
    vec3 foamLight = uSunColor * max(ndl * 0.7 + 0.3, 0.0) + envIrradiance(Nf);
    vec3 foamCol = vec3(0.8, 0.84, 0.86) * foamShade * foamLight / PI;
    col = mix(col, foamCol, foam);
  }

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
  }

  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;
