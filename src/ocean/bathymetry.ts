import type { BathymetryConfig } from './oceanConfig';

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

/**
 * Analytic surf-spot seabed: a monotone-cubic shore profile along x plus an A-frame sandbar
 * (a ">" shaped bar pointing out to sea, shallowest at z = 0) that makes waves peel both ways.
 *
 * This is the *source* of the bathymetry. Runtime queries that must match the GPU go through
 * OceanTables (sampled grids), not through this class.
 */
export class Bathymetry {
  private readonly xs: Float64Array;
  private readonly ds: Float64Array;
  private readonly ms: Float64Array;

  constructor(readonly config: BathymetryConfig) {
    const p = config.profile;
    if (p.length < 2) throw new Error('Bathymetry profile needs at least 2 points');
    const n = p.length;
    this.xs = new Float64Array(n);
    this.ds = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      this.xs[i] = p[i][0];
      this.ds[i] = p[i][1];
      if (i > 0 && this.xs[i] <= this.xs[i - 1]) throw new Error('Bathymetry profile must be sorted by x');
    }
    this.ms = pchipSlopes(this.xs, this.ds);
  }

  /** Base (bar-free) water depth at x, positive below sea level. Flat beyond the end points. */
  baseDepth(x: number): number {
    const xs = this.xs;
    const n = xs.length;
    if (x <= xs[0]) return this.ds[0];
    if (x >= xs[n - 1]) return this.ds[n - 1];
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid;
      else hi = mid;
    }
    const h = xs[hi] - xs[lo];
    const t = (x - xs[lo]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (
      (2 * t3 - 3 * t2 + 1) * this.ds[lo] +
      (t3 - 2 * t2 + t) * h * this.ms[lo] +
      (-2 * t3 + 3 * t2) * this.ds[hi] +
      (t3 - t2) * h * this.ms[hi]
    );
  }

  /** Height the sandbar raises the seabed by at (x, z). Zero outside the bar. */
  barBump(x: number, z: number): number {
    const b = this.config.bar;
    if (!b.enabled) return 0;
    const az = Math.sqrt(z * z + b.roundness * b.roundness) - b.roundness;
    const bx = b.x0 + b.sweep * az;
    const crestDepth = b.peakDepth + b.depthSlope * az;
    const fade = 1 - smoothstep(b.fadeStartZ, b.fadeEndZ, Math.abs(z));
    const height = Math.max(0, this.baseDepth(bx) - crestDepth) * fade;
    if (height <= 0) return 0;
    const u = (x - bx) / b.width;
    return height * Math.exp(-u * u);
  }

  /** Water depth at (x, z) relative to mean sea level (negative on dry land). */
  depth(x: number, z: number): number {
    return this.baseDepth(x) - this.barBump(x, z);
  }

  /** Seabed elevation (y coordinate) at (x, z). */
  seabedY(x: number, z: number): number {
    return -this.depth(x, z);
  }
}

/** Fritsch-Carlson monotone cubic slopes. */
function pchipSlopes(xs: Float64Array, ys: Float64Array): Float64Array {
  const n = xs.length;
  const m = new Float64Array(n);
  const d = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) d[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) {
      m[i] = 0;
    } else {
      const h0 = xs[i] - xs[i - 1];
      const h1 = xs[i + 1] - xs[i];
      const w1 = 2 * h1 + h0;
      const w2 = h1 + 2 * h0;
      m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
    }
  }
  return m;
}
