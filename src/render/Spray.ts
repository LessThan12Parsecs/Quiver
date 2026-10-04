/**
 * GPU spray and mist off breaking crests (stateless particles, no CPU simulation).
 *
 * Each particle has a fixed random seed point in the surf zone and a looping lifetime. In the
 * vertex shader the seed is mapped onto its nearest swell crest (inverting the waveform's forward
 * lean, refined once with the crest's local state), the breaking state of the crest and of the
 * roller just ahead of it gates visibility, and the particle follows a ballistic
 * path with drag relative to where the crest was when it was "spawned" (the crest has since moved
 * on at the phase speed, so droplets thrown off the lip fall behind it); the crest point includes
 * the render-only breaker geometry (thrown lip, roller mound: shaders/breaker.ts). Two
 * populations: fine droplets thrown off the lip, and soft mist puffs hanging over the roller.
 * Lighting (per vertex: a sprite is small): sun with a strong forward-scattering lobe (backlit
 * spray glows) plus sky irradiance, aerial perspective.
 *
 * Cost: particles whose seed sits in small, unbroken water (lulls, outside the sets) are dropped
 * after one swell evaluation; mist puffs are capped at 64 px (large sprites pop at the screen
 * edge — points are clipped by their centre — and cost overdraw).
 */
import * as THREE from 'three';
import { mulberry32 } from '../ocean/waveModel';
import { OCEAN_GLSL, type OceanShaderData } from '../ocean/waveGLSL';
import type { Environment } from './Environment';
import { BREAKER_GLSL } from './shaders/breaker';
import { ENV_GLSL, NOISE_GLSL } from './shaders/common';

/** Spray draws after every other transparent object (board underwater pass 20, rider 30). */
export const SPRAY_RENDER_ORDER = 40;

export interface SprayRegion {
  xMin: number;
  xMax: number;
  zMin: number;
  zMax: number;
  /** Relative share of the particles. */
  weight: number;
}

/** The default surf spot's breaking zones: the A-frame bar and the shore break. */
export const DEFAULT_SPRAY_REGIONS: SprayRegion[] = [
  { xMin: -75, xMax: 45, zMin: -230, zMax: 230, weight: 0.7 },
  { xMin: 60, xMax: 118, zMin: -300, zMax: 300, weight: 0.3 },
];

export interface SprayOptions {
  /** Number of particles (droplets + lip spray + mist). */
  count?: number;
  /** Seed regions (world xz) covering the breaking zones; particles map onto the nearest crest. */
  regions?: SprayRegion[];
  /** Overall opacity multiplier. */
  intensity?: number;
  /** Wind direction (deg from +X toward +Z) and speed (m/s): drifts the mist. */
  windDirectionDeg?: number;
  windSpeed?: number;
}

const SPRAY_VERTEX = /* glsl */ `
${OCEAN_GLSL}
${NOISE_GLSL}
${ENV_GLSL}
${BREAKER_GLSL}
attribute vec4 aSeed;        // seed x, seed z, random, random
attribute vec2 aRand;        // random, kind (0 droplet, 1 lip spray, 2 mist)
uniform float uTime;
uniform float uPixelScale;   // pixels per metre at unit distance
uniform vec2 uWind;          // wind velocity (m/s)
uniform float uIntensity;
varying float vAlpha;
varying float vKind;
varying vec3 vWorld;
varying vec3 vColor;

void main() {
  float kind = aRand.y;
  float r0 = aSeed.w;
  float life = kind > 1.5 ? mix(2.0, 4.0, r0) : kind > 0.5 ? mix(0.7, 1.4, r0) : mix(0.6, 1.2, r0);
  float tau = mod(uTime + aSeed.z * 97.0, life);
  float u = tau / life;

  // Nearest crest of the dominant swell: undo the forward-lean warp phi = psi + kappa cos(psi - d).
  // Wavenumber and lean change a lot toward a breaking crest (shoaling), so the first estimate
  // from the seed's own state lands a few metres short; one step with the local state fixes it.
  OceanSwell s0 = oceanSwell(aSeed.xy);
  if (s0.height < 0.3 && s0.breaking < 0.005) {
    // small unbroken water (a lull): nothing breaks near this seed
    vAlpha = 0.0;
    vKind = kind;
    vWorld = vec3(0.0);
    vColor = vec3(0.0);
    gl_PointSize = 1.0;
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  float d = uOceanParams1.w;
  float phi0 = s0.psi + s0.skew * (cos(s0.psi - d) - cos(-d));
  vec2 anchor = aSeed.xy - s0.dir * (phi0 / max(s0.k, 0.01));
  OceanSwell s1 = oceanSwell(anchor);
  float phi1 = s1.psi + s1.skew * (cos(s1.psi - d) - cos(-d));
  anchor -= s1.dir * (phi1 / max(s1.k, 0.01));
  OceanSwell s = oceanSwell(anchor);
  // Breaking crests emit (the crest itself or the roller just ahead of it, where the seed lies);
  // fade out before the seed's nearest crest switches (seed in a trough).
  float brkC = max(s.breaking, s0.breaking * (1.0 - smoothstep(0.6, 1.6, abs(s0.psi))));
  // (thresholds match the water shader's roller mask: whitewater shows from breaking ~0.05)
  float emit = smoothstep(0.03, 0.35, brkC) * smoothstep(0.3, 0.9, s.height);
  emit *= 1.0 - smoothstep(2.4, 3.0, abs(s0.psi));

  float r1 = fract(aSeed.z * 13.37 + r0 * 7.1);
  float r2 = fract(aSeed.z * 5.71 + aRand.x * 3.3);
  float r3 = fract(aRand.x * 17.3 + r0 * 2.9);
  float H = s.height;
  vec3 dir = vec3(s.dir.x, 0.0, s.dir.y);
  vec3 side = vec3(-s.dir.y, 0.0, s.dir.x);
  vec3 base = vec3(anchor.x, s.eta, anchor.y) + side * (r3 - 0.5) * 1.5 + breakerOffset(s, anchor, uTime, 0.25);
  vec3 pos;
  float size;
  float alpha;
  if (kind < 1.5) {
    // droplets / lip spray: thrown forward and up off the lip with drag, then fall back.
    // The crest has moved on at the phase speed since the particle left it.
    float kd = kind > 0.5 ? 2.8 : 1.6;
    float drag = (1.0 - exp(-kd * tau)) / kd;
    vec3 v0 = dir * (s.speed * mix(0.85, 1.3, r1)) + vec3(0.0, mix(1.0, 3.2, r2) * sqrt(H), 0.0);
    pos = base + vec3(0.0, 0.1 * H, 0.0) - dir * (s.speed * tau) + v0 * drag;
    pos.y -= 4.9 * tau * tau * (kind > 0.5 ? 0.45 : 0.8);
    pos.xz += uWind * (tau * 0.4);
    size = kind > 0.5 ? mix(0.1, 0.4, r1) * (0.7 + 0.6 * u) * sqrt(H) : mix(0.02, 0.06, r1);
    alpha = kind > 0.5 ? 0.7 : 0.9;
  } else {
    // mist: a veil hanging over the roller and just behind it, rising and drifting with the wind
    float behind = mix(-0.4, 3.0, r1 * r1) * H;
    pos = base + vec3(0.0, mix(0.2, 0.8, r2) * H, 0.0) - dir * behind;
    pos += vec3(uWind.x * 0.5, 0.35, uWind.y * 0.5) * tau - dir * (0.2 * s.speed * tau);
    size = mix(0.8, 2.4, r3) * (0.6 + 0.7 * u) * sqrt(H);
    alpha = 0.22 * (1.0 - 0.5 * r1);
  }
  float fade = smoothstep(0.0, 0.15, u) * (1.0 - smoothstep(0.5, 1.0, u));
  vAlpha = emit * fade * uIntensity * alpha;
  vKind = kind;
  vWorld = pos;
  vec4 mv = viewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  float px = size * uPixelScale / max(-mv.z, 0.1);
  // sub-pixel particles keep 1.5 px and fade instead of shrinking (no shimmering)
  vAlpha *= clamp(px / 1.5, 0.0, 1.0);
  gl_PointSize = clamp(px, 1.5, kind > 1.5 ? 64.0 : 128.0);
  if (vAlpha < 0.002) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vColor = vec3(0.0);
    return;
  }
  // lighting: forward-scattering lobe (Henyey-Greenstein g = 0.75) + isotropic part, haze
  vec3 V = normalize(cameraPosition - pos);
  float cosT = dot(-V, uSunDirection);
  float g = 0.75;
  float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosT, 1.5) / (4.0 * 3.14159265);
  vColor = applyHaze(uSunColor * (0.1 + hg * 1.1) * 0.9 + uSkyIrradiance * 0.32, pos, cameraPosition);
}
`;

const SPRAY_FRAGMENT = /* glsl */ `
${NOISE_GLSL}
varying float vAlpha;
varying float vKind;
varying vec3 vWorld;
varying vec3 vColor;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  // lip spray: clumpy droplet clusters; mist: gaussian puffs that blend into a veil
  float soft = vKind > 1.5 ? (exp(-3.5 * r2) - 0.03) * (0.7 + 0.3 * vnoise(c * 1.7 + vWorld.xz * 0.7))
             : vKind > 0.5 ? pow(1.0 - r2, 1.5) * (0.4 + 0.6 * smoothstep(0.15, 0.85, vnoise(c * 1.8 + vWorld.xz * 2.3)))
             : smoothstep(1.0, 0.4, r2);
  gl_FragColor = vec4(vColor, vAlpha * soft);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

export class Spray {
  readonly object3d: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly material: THREE.ShaderMaterial;
  readonly count: number;

  constructor(shaderData: OceanShaderData, env: Environment, opts: SprayOptions = {}) {
    const n = opts.count ?? 48000;
    this.count = n;
    const regions = opts.regions ?? DEFAULT_SPRAY_REGIONS;
    const total = regions.reduce((a, r) => a + r.weight, 0);
    const rng = mulberry32(4242);
    const seed = new Float32Array(n * 4);
    const rnd = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      let pick = rng() * total;
      let reg = regions[0];
      for (const r of regions) {
        reg = r;
        pick -= r.weight;
        if (pick <= 0) break;
      }
      seed[i * 4] = reg.xMin + rng() * (reg.xMax - reg.xMin);
      seed[i * 4 + 1] = reg.zMin + rng() * (reg.zMax - reg.zMin);
      seed[i * 4 + 2] = rng();
      seed[i * 4 + 3] = rng();
      rnd[i * 2] = rng();
      const k = rng();
      rnd[i * 2 + 1] = k < 0.42 ? 1 : k < 0.77 ? 2 : 0;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 4));
    geo.setAttribute('aRand', new THREE.BufferAttribute(rnd, 2));
    const wd = THREE.MathUtils.degToRad(opts.windDirectionDeg ?? 25);
    const ws = opts.windSpeed ?? 4;
    this.material = new THREE.ShaderMaterial({
      name: 'Spray',
      uniforms: {
        ...shaderData.uniforms,
        ...env.uniforms,
        uTime: { value: 0 },
        uPixelScale: { value: 600 },
        uWind: { value: new THREE.Vector2(Math.cos(wd) * ws, Math.sin(wd) * ws) },
        uIntensity: { value: opts.intensity ?? 1 },
      },
      defines: { ...env.envDefines },
      vertexShader: SPRAY_VERTEX,
      fragmentShader: SPRAY_FRAGMENT,
      transparent: true,
      depthWrite: false,
    });
    this.object3d = new THREE.Points(geo, this.material);
    this.object3d.name = 'spray';
    this.object3d.frustumCulled = false;
    // last in the transparent queue (after the board's underwater pass, 20, and the rider,
    // RIDER_RENDER_ORDER 30, which write colour over whatever was blended before them): spray
    // writes no depth, and still depth-tests against the rider, so only spray in front shows
    this.object3d.renderOrder = SPRAY_RENDER_ORDER;
  }

  get intensity(): number {
    return this.material.uniforms.uIntensity.value as number;
  }

  set intensity(v: number) {
    this.material.uniforms.uIntensity.value = v;
  }

  /** Per frame: animation time (s) and the camera (for point-size scaling). */
  update(camera: THREE.PerspectiveCamera, time: number, viewportHeightPx: number): void {
    const u = this.material.uniforms;
    u.uTime.value = time;
    u.uPixelScale.value = (camera.projectionMatrix.elements[5] * viewportHeightPx) / 2;
  }

  dispose(): void {
    this.object3d.geometry.dispose();
    this.material.dispose();
    this.object3d.removeFromParent();
  }
}
