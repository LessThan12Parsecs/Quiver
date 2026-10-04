/**
 * Detail normals for the water surface: animated, periodic slope tiles of short wind waves
 * (capillary-gravity ripples, lambda ~2 cm - 5 m), generated on the GPU every frame and sampled by
 * the ocean fragment shader at several scales.
 *
 * Why a tile instead of analytic sinusoids: a handful of long-crested sinusoids reads as a regular
 * woven texture and, in the sun glitter, as a rectangular lattice (two near-perpendicular short
 * components beat against each other). Real short waves are short-crested and spectrally dense.
 * Each tile is the sum of `components` (default 128) waves with integer wave vectors on the tile
 * torus (seamless), log-uniform wavenumbers over ~4.4 octaves, a broad wind-centred directional
 * spread with some counter-wind energy, Rayleigh amplitudes (equal slope variance per log-k band:
 * saturation range) and the capillary-gravity dispersion relation for its world size.
 *
 * Texel layout (LEAN mapping, Olano & Baker 2010):  RGBA = (s_u, s_v, s_u^2 + s_v^2, eta)
 *   s   slope of the normalised field (total slope variance 1), tile axes (u = wind direction);
 *   eta normalised height (variance 1).
 * Mipmaps average the moments, so a footprint-filtered fetch returns the mean slope and the mean
 * squared slope; their difference is exactly the slope variance lost to filtering, which the
 * water shader adds to the GGX roughness (no shimmer, energy-conserving sun glitter at distance).
 *
 * Float render targets need EXT_color_buffer_float / _half_float; without them the tile is
 * stored as 8-bit with a scale/bias (`decode`).
 */
import * as THREE from 'three';
import { mulberry32 } from '../ocean/waveModel';

export interface RippleTileOptions {
  /** Texels per side (power of two). */
  size?: number;
  /** Number of wave components (<= MAX_TILE_COMPONENTS). */
  components?: number;
  /** World size of the tile (m) used for the dispersion relation (animation speed). */
  worldSize?: number;
  /** Wavenumber range in cycles per tile (kMax <= size / 4 keeps >= 4 texels per wavelength). */
  kMin?: number;
  kMax?: number;
  /** Spectral tilt: slope variance per log-k band ~ k^tilt (0 = saturation range). */
  tilt?: number;
  /** Share of components spread uniformly in direction (rest: ~N(0, spread) around the wind). */
  isotropicShare?: number;
  /** Std-dev of the wind-centred directional spread (rad). */
  spread?: number;
  /** Anisotropic filtering level (clamped to the device maximum; fixed for the tile's life). */
  anisotropy?: number;
  seed?: number;
}

export const MAX_TILE_COMPONENTS = 128;

const TILE_VERTEX = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const TILE_FRAGMENT = /* glsl */ `
#define N_COMP ${MAX_TILE_COMPONENTS}
uniform vec4 uComp[N_COMP];   // m, n (cycles per tile), slope amplitude / |k|, phase
uniform int uCount;
uniform float uSize;
uniform float uHeightNorm;
uniform vec4 uEncode;         // value * x + y (rgb), alpha: * z + w (8-bit fallback)
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  vec2 s = vec2(0.0);
  float h = 0.0;
  for (int i = 0; i < N_COMP; i++) {
    if (i >= uCount) break;
    vec4 c = uComp[i];
    // integer wave vector: fract() keeps the argument small and the tile exactly periodic
    float ph = 6.2831853 * fract(dot(c.xy, uv)) + c.w;
    float sn = sin(ph);
    float cs = cos(ph);
    s -= c.xy * (c.z * sn);
    h += c.z * cs;   // height amplitude = slope amplitude / (2 pi |k|); 2 pi is in uHeightNorm
  }
  h *= uHeightNorm;
  vec4 o = vec4(s, dot(s, s), h);
  gl_FragColor = vec4(o.xy * uEncode.x + uEncode.y, o.z * uEncode.z, o.w * uEncode.x + uEncode.y);
}
`;

/** One animated detail tile (render target + generator). */
export class RippleTile {
  readonly target: THREE.WebGLRenderTarget;
  /** Decode of a fetched texel: value = texel * decodeScale + decodeBias (identity for float). */
  readonly decodeScale = new THREE.Vector4(1, 1, 1, 1);
  readonly decodeBias = new THREE.Vector4(0, 0, 0, 0);
  readonly worldSize: number;
  readonly count: number;

  private readonly material: THREE.ShaderMaterial;
  private readonly mesh: THREE.Mesh;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.Camera();
  private readonly m: Float64Array;
  private readonly n: Float64Array;
  private readonly amp: Float64Array;
  private readonly omega: Float64Array;
  private readonly phase0: Float64Array;
  private lastTime = Number.NaN;

  constructor(renderer: THREE.WebGLRenderer, opts: RippleTileOptions = {}) {
    const size = opts.size ?? 256;
    const count = Math.min(opts.components ?? 128, MAX_TILE_COMPONENTS);
    this.count = count;
    this.worldSize = opts.worldSize ?? 4;
    const kMin = opts.kMin ?? 3;
    const kMax = Math.min(opts.kMax ?? size / 4, size / 2.5);
    const tilt = opts.tilt ?? 0;
    const iso = opts.isotropicShare ?? 0.14;
    const spread = opts.spread ?? 0.75;
    const rng = mulberry32(opts.seed ?? 7);

    this.m = new Float64Array(count);
    this.n = new Float64Array(count);
    this.amp = new Float64Array(count);
    this.omega = new Float64Array(count);
    this.phase0 = new Float64Array(count);
    const used = new Set<string>();
    const g = 9.81;
    const sigma = 0.074 / 1025;
    let slopeVar = 0;
    let heightVar = 0;
    for (let i = 0; i < count; i++) {
      // stratified log-uniform wavenumber, wind-centred direction, unique integer wave vector
      let m = 0;
      let n = 0;
      for (let tries = 0; tries < 64; tries++) {
        const f = (i + rng()) / count;
        const kap = kMin * Math.pow(kMax / kMin, f);
        let th: number;
        if (rng() < iso) th = (rng() * 2 - 1) * Math.PI;
        else {
          // Box-Muller normal, wrapped
          const u1 = Math.max(rng(), 1e-9);
          th = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rng()) * spread;
        }
        m = Math.round(kap * Math.cos(th));
        n = Math.round(kap * Math.sin(th));
        const key = `${m},${n}`;
        if ((m !== 0 || n !== 0) && !used.has(key) && !used.has(`${-m},${-n}`)) {
          used.add(key);
          break;
        }
      }
      const kap = Math.hypot(m, n);
      const cosTh = m / kap;
      // Rayleigh amplitude; counter-wind waves carry less energy
      const ray = Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-6))) / Math.SQRT2;
      const dirW = 0.55 + 0.45 * Math.max(cosTh, 0) + 0.15 * Math.min(cosTh, 0);
      const a = ray * dirW * Math.pow(kap / kMin, tilt * 0.5);
      this.m[i] = m;
      this.n[i] = n;
      this.amp[i] = a; // slope amplitude
      const k = (2 * Math.PI * kap) / this.worldSize;
      this.omega[i] = Math.sqrt(g * k + sigma * k * k * k);
      this.phase0[i] = rng() * Math.PI * 2;
      slopeVar += 0.5 * a * a;
      heightVar += 0.5 * (a / (2 * Math.PI * kap)) ** 2;
    }
    const norm = 1 / Math.sqrt(slopeVar);
    for (let i = 0; i < count; i++) this.amp[i] *= norm;
    const heightNorm = 1 / (Math.sqrt(heightVar) * norm * 2 * Math.PI);

    const ext = renderer.extensions;
    const floatOk = ext.has('EXT_color_buffer_float') || ext.has('EXT_color_buffer_half_float');
    const encode = new THREE.Vector4(1, 0, 1, 0);
    if (!floatOk) {
      // 8-bit fallback: slopes and height in [-4, 4] sigma, squared slope in [0, 16]
      encode.set(1 / 8, 0.5, 1 / 16, 0);
      this.decodeScale.set(8, 8, 16, 8);
      this.decodeBias.set(-4, -4, 0, -4);
    }
    this.target = new THREE.WebGLRenderTarget(size, size, {
      type: floatOk ? THREE.HalfFloatType : THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      anisotropy: Math.max(1, Math.min(opts.anisotropy ?? 4, renderer.capabilities.getMaxAnisotropy())),
    });
    this.target.texture.name = 'RippleTile';

    const comp = Array.from({ length: MAX_TILE_COMPONENTS }, () => new THREE.Vector4());
    for (let i = 0; i < count; i++) comp[i].set(this.m[i], this.n[i], this.amp[i] / Math.hypot(this.m[i], this.n[i]), 0);
    this.material = new THREE.ShaderMaterial({
      name: 'RippleTileGen',
      uniforms: {
        uComp: { value: comp },
        uCount: { value: count },
        uSize: { value: size },
        uHeightNorm: { value: heightNorm },
        uEncode: { value: encode },
      },
      vertexShader: TILE_VERTEX,
      fragmentShader: TILE_FRAGMENT,
      depthTest: false,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
  }

  get texture(): THREE.Texture {
    return this.target.texture;
  }

  /** Regenerate the tile for animation time `time` (s); no-op if unchanged. */
  update(renderer: THREE.WebGLRenderer, time: number): void {
    if (time === this.lastTime) return;
    this.lastTime = time;
    const comp = this.material.uniforms.uComp.value as THREE.Vector4[];
    const TWO_PI = Math.PI * 2;
    for (let i = 0; i < this.count; i++) {
      let ph = this.phase0[i] - this.omega[i] * time;
      ph -= TWO_PI * Math.floor(ph / TWO_PI);
      comp[i].w = ph;
    }
    const prev = renderer.getRenderTarget();
    const prevXr = renderer.xr.enabled;
    renderer.xr.enabled = false;
    renderer.setRenderTarget(this.target);
    renderer.render(this.scene, this.camera);
    renderer.setRenderTarget(prev);
    renderer.xr.enabled = prevXr;
  }

  dispose(): void {
    this.target.dispose();
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}
