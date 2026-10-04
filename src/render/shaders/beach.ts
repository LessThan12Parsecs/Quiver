/**
 * Sand / terrain shaders (BeachMesh.ts).
 *
 * Vertex: static terrain (seabed + beach from the bathymetry, procedural dunes and hinterland
 * further back). Per vertex the swash state for its alongshore position is evaluated from the
 * swell (OCEAN_GLSL): run-up level (m above still water) and backwash phase.
 *
 * Fragment: dry sand (procedural albedo + small wind ripples), wet sand band with sky sheen,
 * thin swash sheet with foam line at the run-up front and lace left by the backwash, dune grass
 * / scrub further up, sun + sky lighting and aerial perspective.
 */
import { OCEAN_GLSL } from '../../ocean/waveGLSL';
import { ENV_GLSL, NOISE_GLSL } from './common';

const VARYINGS = /* glsl */ `
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec2 vSwash;   // run-up level above still water (m), swash phase 0..1 (0 = bore arrives)
varying float vVeg;    // hinterland: 0 beach, 1 dunes, 2 inland scrub/forest
`;

export const BEACH_VERTEX = /* glsl */ `
${OCEAN_GLSL}
uniform vec2 uSwashX;     // x of the swell probe offshore of the beach, x of the shoreline probe
attribute float aVeg;
${VARYINGS}
void main() {
  vec3 pos = position;
  vWorldPos = pos;
  vNormal = normal;
  vVeg = aVeg;
  // Swash: incoming bore height a little offshore drives the run-up; the phase is taken where
  // the bore reaches the shoreline (crest passage = start of the uprush).
  vec2 sw = vec2(0.0);
  if (pos.y > -1.5 && pos.y < 2.5 && pos.x > 60.0 && pos.x < 260.0) {
    OceanSwell off = oceanSwell(vec2(uSwashX.x, pos.z));
    OceanSwell sh = oceanSwell(vec2(uSwashX.y, pos.z));
    float phase = mod(-sh.psi, 6.2831853) / 6.2831853;
    sw = vec2(0.08 + 0.55 * off.height, phase);
  }
  vSwash = sw;
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
}
`;

export const BEACH_FRAGMENT = /* glsl */ `
${NOISE_GLSL}
${ENV_GLSL}
#define PI 3.14159265
uniform float uTime;
uniform float uWetLevel;     // top of the wet band (m above still water)
uniform vec2 uWindDir;
uniform int uDebug;
uniform float uUnderwater;   // 1 when the camera is below the water surface
${VARYINGS}

// Swash front height above still water for run-up amplitude R at swash phase ph (0..1).
float swashLevel(float R, float ph) {
  float up = sin(clamp(ph / 0.28, 0.0, 1.0) * 1.5707963);
  float down = 1.0 - smoothstep(0.28, 0.92, ph);
  return R * (ph < 0.28 ? up : down) - 0.06;
}

void main() {
  vec3 P = vWorldPos;
  vec3 toCam = cameraPosition - P;
  float dist = length(toCam);
  vec3 V = toCam / dist;
  vec3 L = uSunDirection;
  vec3 dpx = dFdx(P);
  vec3 dpy = dFdy(P);
  float fx = length(dpx);
  float fy = length(dpy);
  float fp = max(sqrt(fx * fy), max(fx, fy) * 0.125) + 1e-4;
  float h = P.y;

  // ---- swash: thin film between the sand and the run-up front ----------------------------
  float sl = swashLevel(vSwash.x, vSwash.y);
  float film = sl - h;                               // > 0 under the swash sheet (vertical m)
  float sheet = smoothstep(0.0, 0.01, film) * step(0.01, vSwash.x);
  // Permanently wet band below the high-water mark (ragged), drying out above it.
  float wetTop = uWetLevel + 0.25 * (vnoise(P.xz * 0.035) - 0.5);
  float dry = smoothstep(wetTop - 0.2, wetTop + 0.3, h);
  float wet = max(1.0 - dry, sheet);

  // ---- normals: wind ripples + lumps on dry sand, smoothed when wet -----------------------
  vec3 N = normalize(vNormal);
  vec2 wd = uWindDir;
  vec2 wp = vec2(dot(P.xz, wd), dot(P.xz, vec2(-wd.y, wd.x)));
  float rippleAA = 1.0 - smoothstep(0.012, 0.045, fp);
  float warp = vnoise(wp * vec2(0.6, 0.9)) * 1.4;
  float rph = (wp.x + warp) * (6.2831853 / 0.11);
  vec2 bump = wd * (cos(rph) * 0.14 * rippleAA * dry);
  vec3 lumps = fbm4D(P.xz * 0.9, fp * 0.9);
  bump += lumps.yz * (0.22 * (0.35 + 0.65 * dry));
  N = normalize(N - vec3(bump.x, 0.0, bump.y) * 0.9);

  // ---- albedo ------------------------------------------------------------------------------
  vec3 albedo = sandAlbedo(P.xz, fp);
  float grainAA = 1.0 - smoothstep(0.004, 0.02, fp);
  albedo *= mix(1.0, 0.86 + 0.28 * hash12(floor(P.xz * 60.0)), grainAA);
  albedo *= 0.92 + 0.16 * lumps.x;
  // Vegetation: marram grass on the dunes (sparse on steep sand faces), scrub and forest inland.
  float flatness = smoothstep(0.75, 0.93, normalize(vNormal).y);
  float dunes = clamp(vVeg, 0.0, 1.0);
  float inland = clamp(vVeg - 1.0, 0.0, 1.0);
  float vn = fbm4(P.xz * 0.025, fp * 0.025);
  float vn2 = fbm4(P.xz * 0.13 + 3.0, fp * 0.13);
  float grassCover = dunes * smoothstep(0.22, 0.5, vn + 0.25 * flatness + 0.4 * inland) * (0.6 + 0.4 * smoothstep(0.3, 0.7, vn2));
  // effective albedos include the blades' self-shadowing (much darker than the material)
  float tuft = fbm4(P.xz * 0.9 + 7.0, fp * 0.9);
  vec3 marram = mix(vec3(0.075, 0.085, 0.035), vec3(0.17, 0.165, 0.08), vn2) * (0.7 + 0.6 * tuft);
  vec3 scrub = mix(vec3(0.035, 0.055, 0.02), vec3(0.08, 0.095, 0.035), vn2) * (0.75 + 0.5 * tuft);
  vec3 forest = vec3(0.022, 0.036, 0.016) * (0.8 + 0.4 * vn2);
  grassCover *= smoothstep(0.2, 0.45, tuft + 0.25 * vn2);
  vec3 vegCol = mix(marram, scrub, smoothstep(0.05, 0.45, inland + 0.3 * (vn - 0.5)));
  vegCol = mix(vegCol, forest, smoothstep(0.4, 0.8, inland) * smoothstep(0.35, 0.55, vn + 0.15));
  float veg = clamp(grassCover + inland, 0.0, 1.0);
  albedo = mix(albedo, vegCol, veg);
  // Wet sand: darker and slightly cooler; the film adds a faint water tint.
  vec3 wetAlbedo = albedo * vec3(0.48, 0.5, 0.55);
  vec3 base = mix(albedo, wetAlbedo, wet * (1.0 - veg));

  // ---- lighting ------------------------------------------------------------------------------
  float NdotL = max(dot(N, L), 0.0);
  // Mild opposition effect: dry sand is brighter looking away from the sun.
  float opp = mix(1.0, 0.85 + 0.3 * pow(max(dot(V, L), 0.0), 2.0), dry);
  vec3 diffuse = base / PI * (uSunColor * NdotL * opp + envIrradiance(N));

  // Specular: the swash film is a near-mirror, wet sand a rough sheen.
  vec3 Ns = normalize(mix(N, normalize(vNormal), sheet * 0.8));
  float NdotV = clamp(dot(Ns, V), 1e-3, 1.0);
  float rough = mix(0.5, 0.1, sheet);
  float F = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
  F *= mix(0.3, 1.0, sheet) * wet;
  vec3 R = reflect(-V, Ns);
  R.y = max(R.y, 0.01);
  vec3 spec = envSample(normalize(R), rough) * F;
  vec3 Hh = normalize(L + V);
  float NdotH = max(dot(Ns, Hh), 0.0);
  float a2 = rough * rough * rough * rough;
  float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  spec += uSunColor * (a2 / (PI * d * d)) * 0.25 * F * max(dot(Ns, L), 0.0);
  vec3 col = diffuse * (1.0 - F) + spec;

  // ---- swash foam: ragged bubbly line at the run-up front, sparse lace on the backwash -------
  float front = smoothstep(-0.006, 0.002, film) * (1.0 - smoothstep(0.01, 0.04, film));
  float foam = 0.0;
  if (vSwash.x > 0.01 && (front > 0.0 || sheet > 0.0)) {
    float lace = laceField(P.xz * 1.7 + vec2(0.0, 0.2 * uTime), uTime * 0.3, fp * 1.7);
    foam = front * smoothstep(0.1, 0.45, lace + 0.15);
    foam = max(foam, sheet * smoothstep(0.35, 0.95, vSwash.y) * smoothstep(0.62, 0.9, lace) * 0.85);
    foam *= 1.0 - smoothstep(0.05, 0.2, fp);
  }
  vec3 foamCol = vec3(0.8, 0.84, 0.86) / PI * (uSunColor * max(dot(N, L) * 0.7 + 0.3, 0.0) + envIrradiance(N));
  col = mix(col, foamCol, foam);

  if (uUnderwater > 0.5) {
    // camera underwater: in-water attenuation and scattering toward the eye (matches the water's
    // default optics roughly); terrain above the water stays dry-lit but is seen through the surface.
    vec3 kapU = vec3(0.34, 0.09, 0.1) + 20.0 * vec3(0.011, 0.014, 0.016);
    vec3 inWater = (uSunColor * max(uSunDirection.y, 0.0) + uSkyIrradiance) * vec3(0.011, 0.014, 0.016) * 2.0 / kapU;
    vec3 tr = exp(-kapU * min(dist, 80.0));
    col = col * tr * vec3(0.7, 0.85, 0.9) + inWater * (1.0 - tr);
  } else {
    col = applyHaze(col, P, cameraPosition);
  }
  if (uDebug == 1) col = vec3(clamp(vVeg, 0.0, 1.0), clamp(vVeg - 1.0, 0.0, 1.0), veg);
  else if (uDebug == 2) col = vec3(sheet, foam, wet);
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;
