/**
 * Render-only breaking-wave geometry (VIS: the swell is a single-valued height field — exact CPU
 * queries for the physics, see src/ocean — so on its own a breaking wave is a smooth hump with
 * a foam decal). The water mesh and the spray add this offset to the height-field position;
 * nothing else (physics, CPU queries, `check:gpu`) sees it.
 *
 *  - Lip: as a wave is about to break and while its whitewater is just starting (fullness → 1:
 *    the section ahead of the curl of a peeling wave), the upper part of the crest (above ≈ 40 %
 *    of the waveform height, a little of its back) is thrown forward along the wave direction
 *    (up to 1.8 H at the crest, ∝ w²) and droops (0.6 H, ∝ w³). The top moves further than the
 *    steep lower face below it, so the mesh folds into an overhang: a pitching lip with a cavity
 *    under it. The folded part is seen from its back side (the water shader shades it with the
 *    flipped lip normal). The lip collapses into the roller as the whitewater grows.
 *  - Roller: once broken, the whitewater on the crest and upper face is a churning mound of foam
 *    (≈ 0.25 H thick, lumpy outline, pushed a little forward) instead of paint on the hump.
 *
 * Both fade with the grid spacing (they need ~0.25–1 m cells; far away they are sub-pixel and
 * would only alias). Needs OCEAN_GLSL (OceanSwell) and NOISE_GLSL (vnoise) before it.
 */
export const BREAKER_GLSL = /* glsl */ `
#ifndef QUIVER_BREAKER_GLSL
#define QUIVER_BREAKER_GLSL
// Is any breaker geometry possible for this swell state (cheap pre-test)?
bool breakerActive(OceanSwell s) {
  return s.height > 0.35 && (s.fullness > 0.86 || s.breaking > 0.15);
}

// World offset of the surface point at x with swell state s; t = time (s), spacing = grid cell (m).
// lipOut = how much of the offset is the thrown lip (0..1: clear water, not foam).
vec3 breakerOffset(OceanSwell s, vec2 x, float t, float spacing, out float lipOut) {
  lipOut = 0.0;
  float H = s.height;
  if (H < 0.35) return vec3(0.0);
  float b = clamp(s.breaking, 0.0, 1.0);
  float psi = s.psi;
  vec3 dir = vec3(s.dir.x, 0.0, s.dir.y);
  vec3 off = vec3(0.0);
  // lip: thrown as the wave is about to break, collapsing into the roller as the whitewater grows
  float lip = smoothstep(0.86, 0.97, s.fullness) * (1.0 - smoothstep(0.3, 0.75, b));
  lip *= smoothstep(0.5, 1.1, H) * (1.0 - smoothstep(1.0, 3.0, spacing));
  if (lip > 0.0) {
    // weight: height within the waveform ((1 + cos psi) / 2)^p (1 = crest), from 40 % up; behind
    // the crest only its first metre or so
    float hq = pow(0.5 + 0.5 * cos(psi), max(s.peaking, 1.0));
    float w = smoothstep(0.4, 1.0, hq) * (psi < 0.0 ? smoothstep(-0.45, -0.02, psi) : 1.0);
    float w2 = w * w;
    off = dir * (1.8 * H * lip * w2);
    off.y -= 0.6 * H * lip * w2 * w;
    lipOut = lip * w;
  }
  // roller: a lumpy mound of foam over the crest and the upper face
  float roll = smoothstep(0.15, 0.7, b) * smoothstep(0.4, 1.0, H) * (1.0 - smoothstep(3.0, 8.0, spacing));
  if (roll > 0.0) {
    float kd = max(s.k, 0.02);
    float faceExt = clamp(2.0 * kd * (0.5 + 2.4 * H), 0.2, 1.8);
    // ragged outline: the phase window wanders along the crest
    float jit = (vnoise(x * vec2(0.3, 0.22) + vec2(0.0, 0.05 * t)) - 0.5) * 0.5 * faceExt;
    float m = smoothstep(-0.5, -0.08, psi + jit) * (1.0 - smoothstep(0.3 * faceExt, 0.95 * faceExt, psi + jit));
    // churning lumps (~1-3 m) tumbling forward with the bore
    vec2 q = x * 0.55 - s.dir * (0.6 * t);
    float lump = 0.45 + 0.75 * vnoise(q) + 0.35 * vnoise(q * 2.3 + 5.1);
    off.y += 0.25 * H * roll * m * lump;
    off += dir * (0.12 * H * roll * m);
    lipOut *= 1.0 - roll * m;
  }
  return off;
}

vec3 breakerOffset(OceanSwell s, vec2 x, float t, float spacing) {
  float lip;
  return breakerOffset(s, x, t, spacing, lip);
}
#endif
`;
