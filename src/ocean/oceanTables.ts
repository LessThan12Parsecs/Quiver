import { Bathymetry } from './bathymetry';
import { deepWaterDepth, deepWaveNumber, localWave, waveNumber, type LocalWave } from './dispersion';
import { MAX_SWELLS, type OceanConfig } from './oceanConfig';

/** Minimum depth used when integrating the phase over land (amplitude is zero there anyway). */
const PHASE_MIN_DEPTH = 0.3;
/** "No limit" value for the breaking-history cap offshore. */
export const NO_CAP = 1e4;

/**
 * Pre-sampled lookup tables shared by the CPU wave model and the GPU shader.
 *
 * phase : RGBA float texture, width = phaseSize, height = MAX_SWELLS.
 *         texel (j, row i) = [Phi_i(x_j), kx_i(x_j), baseDepth(x_j), baseCap(x_j)]
 *         Phi_i is the cross-shore phase integral  int kx dx  (Phi_i(0) = 0) for swell i,
 *         so the swell phase is  Phi_i(x) + ky_i * z + offset_i(t).
 *         baseDepth / baseCap (stored in every row, read from row 0) describe the bar-free
 *         profile used outside the 2D field.
 * field : RG float texture nx * nz over the surf zone: [depth(x, z), cap(x, z)].
 *
 * cap(x, z) is the breaking history: the largest *deep-water-equivalent* wave height that can
 * reach (x, z) without having broken on the way in,
 *     cap(x) = min over x' <= x of  gamma * depth(x') / K(x')
 * where K = shoaling * refraction coefficient of the primary swell. A wave's local height is
 * limited to cap * K(x), so a wave that broke on the bar arrives in the trough smaller.
 *
 * Lookups (CPU here, GPU in waveGLSL.ts) use identical interpolation:
 *   - phase: cubic Hermite in x with kx*dx as tangents; linear extrapolation outside.
 *   - kx: linear.
 *   - field: bilinear inside, else linear base profile from the phase table.
 */
export class OceanTables {
  readonly phase: Float32Array;
  readonly phaseXMin: number;
  readonly phaseDx: number;
  readonly phaseSize: number;

  /** Interleaved [depth, cap] per node, row-major in z. */
  readonly field: Float32Array;
  readonly fieldXMin: number;
  readonly fieldZMin: number;
  readonly fieldDx: number;
  readonly fieldDz: number;
  readonly fieldNx: number;
  readonly fieldNz: number;

  constructor(
    readonly config: OceanConfig,
    readonly bathymetry: Bathymetry,
  ) {
    const t = config.tables;
    const g = config.gravity;
    const n = t.phaseSize;
    this.phaseSize = n;
    this.phaseXMin = t.phaseXMin;
    this.phaseDx = (t.phaseXMax - t.phaseXMin) / (n - 1);
    this.phase = new Float32Array(n * MAX_SWELLS * 4);

    const swells = config.swells.slice(0, MAX_SWELLS);
    const primary = swells[0];
    const lw: LocalWave = { k: 0, c: 0, cg: 0, kh: 0, shoaling: 1, refraction: 1 };
    // Per-depth breaking limit expressed as deep-water-equivalent height for the primary swell.
    const capOf = (depth: number): number => {
      if (!primary) return NO_CAP;
      if (depth <= 0) return 0;
      const omega = (2 * Math.PI) / primary.period;
      const sinT0 = Math.sin((primary.directionDeg * Math.PI) / 180);
      localWave(omega, Math.max(depth, config.minDepth), g, sinT0, lw);
      return (config.breakerIndex * depth) / (lw.shoaling * lw.refraction);
    };

    const baseDepth = new Float64Array(n);
    const baseCap = new Float64Array(n);
    let running = NO_CAP;
    for (let j = 0; j < n; j++) {
      baseDepth[j] = bathymetry.baseDepth(this.phaseXMin + j * this.phaseDx);
      running = Math.min(running, capOf(baseDepth[j]));
      baseCap[j] = running;
    }

    for (let row = 0; row < MAX_SWELLS; row++) {
      const base = row * n * 4;
      for (let j = 0; j < n; j++) {
        this.phase[base + j * 4 + 2] = baseDepth[j];
        this.phase[base + j * 4 + 3] = baseCap[j];
      }
      const sw = swells[row];
      if (!sw) continue;
      const omega = (2 * Math.PI) / sw.period;
      const ky = deepWaveNumber(omega, g) * Math.sin((sw.directionDeg * Math.PI) / 180);
      const hDeep = deepWaterDepth(omega, g);
      const kxAt = (x: number): number => {
        const h = Math.min(Math.max(bathymetry.depth(x, t.phaseLineZ), PHASE_MIN_DEPTH), hDeep);
        const k = waveNumber(omega, h, g);
        return Math.sqrt(Math.max(k * k - ky * ky, 1e-8));
      };
      const kx = new Float64Array(n);
      for (let j = 0; j < n; j++) kx[j] = kxAt(this.phaseXMin + j * this.phaseDx);
      // Integrate with Simpson's rule, anchored so Phi(x = 0) = 0.
      const phi = new Float64Array(n);
      const j0 = Math.min(Math.max(Math.round(-this.phaseXMin / this.phaseDx), 0), n - 1);
      const xj0 = this.phaseXMin + j0 * this.phaseDx;
      phi[j0] = kxAt(0) * xj0; // so Phi(x = 0) is ~0 even when x = 0 falls between samples
      const seg = (j: number): number => {
        const xm = this.phaseXMin + (j + 0.5) * this.phaseDx;
        return (this.phaseDx / 6) * (kx[j] + 4 * kxAt(xm) + kx[j + 1]);
      };
      for (let j = j0; j < n - 1; j++) phi[j + 1] = phi[j] + seg(j);
      for (let j = j0; j > 0; j--) phi[j - 1] = phi[j] - seg(j - 1);
      for (let j = 0; j < n; j++) {
        this.phase[base + j * 4] = phi[j];
        this.phase[base + j * 4 + 1] = kx[j];
      }
    }

    this.fieldNx = t.fieldNx;
    this.fieldNz = t.fieldNz;
    this.fieldXMin = t.fieldXMin;
    this.fieldZMin = t.fieldZMin;
    this.fieldDx = (t.fieldXMax - t.fieldXMin) / (t.fieldNx - 1);
    this.fieldDz = (t.fieldZMax - t.fieldZMin) / (t.fieldNz - 1);
    this.field = new Float32Array(t.fieldNx * t.fieldNz * 2);
    // Waves entering the field have already been limited by the offshore base profile.
    const capEntering = baseCapAt(this, baseCap, t.fieldXMin);
    for (let jz = 0; jz < t.fieldNz; jz++) {
      const z = this.fieldZMin + jz * this.fieldDz;
      let cap = capEntering;
      for (let ix = 0; ix < t.fieldNx; ix++) {
        const x = this.fieldXMin + ix * this.fieldDx;
        const d = bathymetry.depth(x, z);
        cap = Math.min(cap, capOf(d));
        const k = (jz * t.fieldNx + ix) * 2;
        this.field[k] = d;
        this.field[k + 1] = cap;
      }
    }
  }

  /**
   * Cross-shore phase integral for swell `row` at x, written to out[0], and kx to out[1].
   * Mirrors `oceanPhaseLookup` in GLSL.
   */
  phaseLookup(row: number, x: number, out: Float64Array | number[]): void {
    const n = this.phaseSize;
    const p = this.phase;
    const base = row * n * 4;
    const u = (x - this.phaseXMin) / this.phaseDx;
    if (u <= 0) {
      const kx = p[base + 1];
      out[0] = p[base] + kx * (x - this.phaseXMin);
      out[1] = kx;
      return;
    }
    if (u >= n - 1) {
      const i = base + (n - 1) * 4;
      const kx = p[i + 1];
      out[0] = p[i] + kx * (x - (this.phaseXMin + (n - 1) * this.phaseDx));
      out[1] = kx;
      return;
    }
    const j = Math.floor(u);
    const t = u - j;
    const i0 = base + j * 4;
    const i1 = i0 + 4;
    const t2 = t * t;
    const t3 = t2 * t;
    const dx = this.phaseDx;
    out[0] =
      (2 * t3 - 3 * t2 + 1) * p[i0] +
      (t3 - 2 * t2 + t) * dx * p[i0 + 1] +
      (-2 * t3 + 3 * t2) * p[i1] +
      (t3 - t2) * dx * p[i1 + 1];
    out[1] = p[i0 + 1] + (p[i1 + 1] - p[i0 + 1]) * t;
  }

  /**
   * Depth (out[0]) and breaking-history cap (out[1]) at (x, z). Mirrors `oceanField` in GLSL.
   */
  fieldAt(x: number, z: number, out: Float64Array | number[]): void {
    const fu = (x - this.fieldXMin) / this.fieldDx;
    const fv = (z - this.fieldZMin) / this.fieldDz;
    if (fu >= 0 && fv >= 0 && fu <= this.fieldNx - 1 && fv <= this.fieldNz - 1) {
      const i = Math.min(Math.floor(fu), this.fieldNx - 2);
      const j = Math.min(Math.floor(fv), this.fieldNz - 2);
      const tx = fu - i;
      const tz = fv - j;
      const d = this.field;
      const r0 = (j * this.fieldNx + i) * 2;
      const r1 = r0 + this.fieldNx * 2;
      for (let c = 0; c < 2; c++) {
        const a = d[r0 + c] + (d[r0 + 2 + c] - d[r0 + c]) * tx;
        const b = d[r1 + c] + (d[r1 + 2 + c] - d[r1 + c]) * tx;
        out[c] = a + (b - a) * tz;
      }
      return;
    }
    const n = this.phaseSize;
    const u = Math.min(Math.max((x - this.phaseXMin) / this.phaseDx, 0), n - 1);
    const j = Math.min(Math.floor(u), n - 2);
    const t = u - j;
    const p = this.phase;
    out[0] = p[j * 4 + 2] + (p[(j + 1) * 4 + 2] - p[j * 4 + 2]) * t;
    out[1] = p[j * 4 + 3] + (p[(j + 1) * 4 + 3] - p[j * 4 + 3]) * t;
  }

  private readonly tmp = new Float64Array(2);

  /** Water depth at (x, z) (positive below sea level). */
  depthAt(x: number, z: number): number {
    this.fieldAt(x, z, this.tmp);
    return this.tmp[0];
  }
}

function baseCapAt(t: OceanTables, baseCap: Float64Array, x: number): number {
  const u = Math.min(Math.max((x - t.phaseXMin) / t.phaseDx, 0), t.phaseSize - 1);
  return baseCap[Math.floor(u)];
}
