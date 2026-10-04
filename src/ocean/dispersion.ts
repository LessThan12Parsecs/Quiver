/**
 * Linear wave theory helpers. Every function here that the GPU also needs is mirrored
 * operation-for-operation in waveGLSL.ts (`oceanWaveNumber` etc.) so CPU physics and GPU
 * rendering agree.
 */

/**
 * Solve the dispersion relation  omega^2 = g k tanh(k h)  for k.
 * Guo (2002) explicit approximation followed by two Newton iterations (relative error < 1e-7).
 * `h` must be > 0.
 */
export function waveNumber(omega: number, h: number, g: number): number {
  const kh0 = (omega * omega * h) / g; // deep-water k times depth
  let kh = kh0 / Math.pow(1 - Math.exp(-Math.pow(kh0, 1.25)), 0.4);
  for (let i = 0; i < 2; i++) {
    const t = Math.tanh(kh);
    const f = kh * t - kh0;
    const df = t + kh * (1 - t * t);
    kh -= f / df;
  }
  return kh / h;
}

/** Ratio of group to phase velocity, n = 0.5 (1 + 2kh / sinh 2kh). kh is clamped by callers. */
export function groupRatio(kh: number): number {
  return 0.5 * (1 + (2 * kh) / Math.sinh(2 * kh));
}

/** Deep-water wavenumber for angular frequency omega. */
export function deepWaveNumber(omega: number, g: number): number {
  return (omega * omega) / g;
}

/** Depth beyond which a component is treated as deep water (k h = 2 pi, tanh = 0.99999). */
export function deepWaterDepth(omega: number, g: number): number {
  return (2 * Math.PI) / deepWaveNumber(omega, g);
}

export interface LocalWave {
  k: number;
  c: number;
  cg: number;
  kh: number;
  /** Shoaling coefficient sqrt(cg0 / cg). */
  shoaling: number;
  /** Refraction coefficient sqrt(cos(theta0) / cos(theta)) from Snell's law. */
  refraction: number;
}

/** Local linear-wave properties for a swell of angular frequency omega at depth h (h > 0). */
export function localWave(
  omega: number,
  h: number,
  g: number,
  sinTheta0: number,
  out: LocalWave,
): LocalWave {
  const hc = Math.min(h, deepWaterDepth(omega, g));
  const k = waveNumber(omega, hc, g);
  const kh = k * hc;
  const c = omega / k;
  const cg = groupRatio(kh) * c;
  const cg0 = (0.5 * g) / omega;
  const c0 = g / omega;
  const sinT = (sinTheta0 * c) / c0;
  const cosT = Math.sqrt(Math.max(1 - sinT * sinT, 1e-4));
  const cosT0 = Math.sqrt(Math.max(1 - sinTheta0 * sinTheta0, 1e-4));
  out.k = k;
  out.c = c;
  out.cg = cg;
  out.kh = kh;
  out.shoaling = Math.sqrt(cg0 / cg);
  out.refraction = Math.sqrt(cosT0 / cosT);
  return out;
}
