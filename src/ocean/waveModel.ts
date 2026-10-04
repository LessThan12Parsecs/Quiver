/**
 * CPU wave model — the single source of truth for the water surface.
 *
 * The surface is the sum of two parts:
 *
 *  1. Shoaling swell (Eulerian height field).
 *     Each swell component has a cross-shore phase integral Phi(x) from the dispersion relation
 *     over the bathymetry (refraction via Snell: ky constant). Its local height comes from linear
 *     shoaling + refraction, a per-wave "set" envelope, and a depth-limited breaking cap with
 *     history (see OceanTables: a wave that broke on the bar stays smaller inside it).
 *     The waveform is shaped with
 *        y = H * (((1 + cos psi) / 2)^p - mean_p)       (crest peaking, p from Ursell number)
 *        phi = psi + kappa * cos(psi - skewPhase)        (Kepler warp: steep front face)
 *     Because this part is an explicit height field it can never fold over, and the CPU can
 *     evaluate it at any world (x, z) exactly.
 *
 *  2. Wind chop (small Lagrangian Gerstner waves). A rest point x0 is displaced horizontally by
 *     D(x0) and vertically by Y(x0). The CPU inverts x = x0 + D(x0) by fixed-point iteration
 *     (a contraction because the total steepness is limited by `wind.steepnessBudget`).
 *
 * GPU vertex position for rest grid point x0 (see waveGLSL.ts, mirrored op-for-op):
 *     p.xz = x0 + D_wind(x0)
 *     p.y  = Y_wind(x0) + eta_swell(p.xz)
 *
 * Water velocity (for physics) is physical rather than Lagrangian:
 *   - swell: Bernoulli along the surface in the wave frame, u = c - sqrt(c^2 - 2 g eta),
 *     plus a whitewater push toward 0.9 c on the crest/face of broken waves, and on the crest of
 *     a wave close to its breaking limit (crest kinematics: the crest water speeds up toward the
 *     phase speed as the wave nears breaking — u_crest → c is the kinematic breaking criterion —
 *     which the Bernoulli speed of the rounded model crest, ≈ 0.5 c, misses);
 *   - chop: linear orbital velocity u = omega * eta;
 *   - vertical: from the kinematic free-surface condition w = d(eta)/dt + u . grad(eta),
 *     with derivatives taken by finite differences of the exact surface, so a body moving with
 *     the water stays on the surface.
 */
import { Bathymetry } from './bathymetry';
import { deepWaterDepth, deepWaveNumber, waveNumber } from './dispersion';
import {
  DEFAULT_OCEAN_CONFIG,
  MAX_SWELLS,
  MAX_WIND_WAVES,
  SET_LENGTH,
  cloneOceanConfig,
  type OceanConfig,
  type WindWaveConfig,
} from './oceanConfig';
import { OceanTables } from './oceanTables';

const TWO_PI = Math.PI * 2;
/** Envelope offsets are wrapped to this period so they stay exact in float32 on the GPU. */
export const ENVELOPE_PERIOD = TWO_PI * SET_LENGTH;
/** Time step used for the central difference d(eta)/dt. */
export const FD_DT = 1 / 120;
/** Spatial step used for the central difference grad(eta). */
export const FD_DX = 0.05;
/** Fixed-point iterations used to invert the wind-chop displacement. */
export const WIND_INVERSE_ITERATIONS = 4;
/** Ratio H/(gamma h) where whitewater starts / is fully developed. */
export const BREAK_START = 0.95;
export const BREAK_FULL = 1.3;
/** Ratio H/(gamma h) where the crest water starts speeding up toward the phase speed (it reaches
 * CREST_PUSH of the way from its Bernoulli speed to the whitewater speed 0.9 c at BREAK_START). */
export const CREST_START = 0.85;
export const CREST_PUSH = 0.85;

/** Derived per-swell constants (also uploaded to the GPU). */
export interface SwellParams {
  omega: number;
  ky: number;
  /** Deep-water amplitude H0 / 2. */
  amp0: number;
  sinT0: number;
  cosT0: number;
  /** Deep-water group velocity g / (2 omega). */
  cg0: number;
  /** Deep-water phase velocity g / omega. */
  c0: number;
  /** Depth clamp beyond which the component is deep water. */
  hDeep: number;
  phase0: number;
  pattern: Float32Array;
}

/** A deep-water wind-chop component (also uploaded to the GPU). */
export interface WindComponent {
  dirX: number;
  dirZ: number;
  k: number;
  omega: number;
  /** Vertical amplitude. */
  amp: number;
  /** Horizontal (Gerstner) amplitude. */
  ampH: number;
  phase0: number;
}

/** Result of evaluating the swell height field at a world point. */
export interface SwellEval {
  /** Swell elevation (sum of components). */
  eta: number;
  /** Horizontal water velocity from the swell. */
  ux: number;
  uz: number;
  /** Seabed depth at the point (positive below mean sea level). */
  depth: number;
  /** Total local swell height after the breaking cap. */
  height: number;
  /**
   * Linear height / breaking-history limit: < ~0.8 unbroken, ~0.9-1 about to break,
   * > 1 broken (here or further out).
   */
  ratio: number;
  /** Actual height / (gamma * h): how close the wave is to the local depth limit (0..~1). */
  fullness: number;
  /** Whitewater intensity 0..1. */
  breaking: number;
  /** Crest peakedness exponent used. */
  peaking: number;
  /** Forward lean kappa used. */
  skew: number;
  /** Dominant component: waveform phase psi in (-pi, pi] (0 = crest, (0, pi) = front face). */
  domPsi: number;
  domSpeed: number;
  domDirX: number;
  domDirZ: number;
  domK: number;
}

export interface WindEval {
  /** Rest point that was evaluated. */
  x0: number;
  z0: number;
  /** Horizontal displacement of the rest point. */
  dx: number;
  dz: number;
  /** Vertical displacement. */
  dy: number;
  /** Orbital velocity (horizontal). */
  ux: number;
  uz: number;
}

/** Wind chop with analytic Lagrangian derivatives (depth fade treated as locally constant). */
export interface WindDerivs extends WindEval {
  /** d(displacement)/d(rest) — symmetric 2x2 (identity not included). */
  jxx: number;
  jxz: number;
  jzz: number;
  /** d(dy)/d(rest). */
  gx: number;
  gz: number;
  /** Time derivatives of the displacement at fixed rest point. */
  dxdt: number;
  dzdt: number;
  dydt: number;
}

/** Full surface query result for physics and gameplay. */
export interface SurfaceSample {
  height: number;
  /** Unit surface normal. */
  normalX: number;
  normalY: number;
  normalZ: number;
  /** Surface gradient d(eta)/dx, d(eta)/dz. */
  slopeX: number;
  slopeZ: number;
  /** Eulerian d(eta)/dt at the point. */
  dEtaDt: number;
  /** Water particle velocity at the surface. */
  velX: number;
  velY: number;
  velZ: number;
  depth: number;
  waveHeight: number;
  /** See SwellEval.ratio. */
  ratio: number;
  /** See SwellEval.fullness: steepness / closeness to breaking. */
  fullness: number;
  /** Whitewater intensity 0..1. */
  breaking: number;
  /** Dominant swell: local phase speed, direction, waveform phase (0 = crest, (0, pi) = face). */
  phaseSpeed: number;
  dirX: number;
  dirZ: number;
  wavePhase: number;
}

export function createSwellEval(): SwellEval {
  return {
    eta: 0, ux: 0, uz: 0, depth: 0, height: 0, ratio: 0, fullness: 0, breaking: 0, peaking: 1, skew: 0,
    domPsi: 0, domSpeed: 0, domDirX: 1, domDirZ: 0, domK: 0,
  };
}

export function createWindEval(): WindEval {
  return { x0: 0, z0: 0, dx: 0, dz: 0, dy: 0, ux: 0, uz: 0 };
}

export function createWindDerivs(): WindDerivs {
  return {
    x0: 0, z0: 0, dx: 0, dz: 0, dy: 0, ux: 0, uz: 0,
    jxx: 0, jxz: 0, jzz: 0, gx: 0, gz: 0, dxdt: 0, dzdt: 0, dydt: 0,
  };
}

export function createSurfaceSample(): SurfaceSample {
  return {
    height: 0, normalX: 0, normalY: 1, normalZ: 0, slopeX: 0, slopeZ: 0, dEtaDt: 0,
    velX: 0, velY: 0, velZ: 0, depth: 0, waveHeight: 0, ratio: 0, fullness: 0, breaking: 0,
    phaseSpeed: 0, dirX: 1, dirZ: 0, wavePhase: 0,
  };
}

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

/** Wrap to [-pi, pi). */
export function wrapPi(a: number): number {
  return a - TWO_PI * Math.floor((a + Math.PI) / TWO_PI);
}

/** Positive modulo. */
function pmod(a: number, m: number): number {
  return a - m * Math.floor(a / m);
}

/** Soft saturation used by the breaking cap: identity below 0.8, smoothly approaches 1. */
export function softCap(r: number): number {
  return r <= 0.8 ? r : 1 - 0.2 * Math.exp(-(r - 0.8) / 0.2);
}

/** Mean of ((1 + cos psi) / 2)^p over a period: Gamma(p+1/2) / (sqrt(pi) Gamma(p+1)), approx. */
export function peakMean(p: number): number {
  const q = p + 0.25;
  return 1 / Math.sqrt(Math.PI * (q + 1 / (32 * q)));
}

/**
 * Solve phi = psi + kappa * cos(psi - delta) for psi (kappa in [0, 0.99]).
 * Rewritten as Kepler's equation E - kappa sin E = M with E = psi - delta - pi/2,
 * M = phi - delta - pi/2 (wrapped); Danby starter + 4 Halley iterations.
 * Returns psi wrapped to [-pi, pi).
 */
export function solveSkew(phi: number, kappa: number, delta: number): number {
  const M = wrapPi(phi - delta - Math.PI / 2);
  const sM = Math.sin(M);
  let E = M + 0.85 * kappa * (sM >= 0 ? 1 : -1);
  for (let i = 0; i < 4; i++) {
    const s = Math.sin(E);
    const c = Math.cos(E);
    const f = E - kappa * s - M;
    const fp = 1 - kappa * c;
    const fpp = kappa * s;
    E -= f / (fp - (0.5 * f * fpp) / fp);
  }
  return wrapPi(E + delta + Math.PI / 2);
}

/** Periodic Catmull-Rom interpolation of a set pattern at continuous wave index s. */
export function setEnvelope(pattern: ArrayLike<number>, s: number): number {
  const L = SET_LENGTH;
  const i = Math.floor(s);
  const f = s - i;
  const p0 = pattern[pmod(i - 1, L)];
  const p1 = pattern[pmod(i, L)];
  const p2 = pattern[pmod(i + 1, L)];
  const p3 = pattern[pmod(i + 2, L)];
  const f2 = f * f;
  const f3 = f2 * f;
  const v =
    0.5 *
    (2 * p1 +
      (-p0 + p2) * f +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * f3);
  return Math.max(v, 0);
}

/** Small deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Build wind-chop components from a Pierson-Moskowitz spectrum (log-spaced frequencies,
 * cos^2 directional spreading). Horizontal amplitudes are scaled to respect the steepness budget.
 */
export function buildWindComponents(cfg: WindWaveConfig, g: number): { comps: WindComponent[]; steepness: number } {
  const n = Math.max(0, Math.min(cfg.count, MAX_WIND_WAVES));
  const comps: WindComponent[] = [];
  if (n === 0 || cfg.amplitudeScale <= 0) return { comps, steepness: 0 };
  const rng = mulberry32(cfg.seed);
  const wMin = Math.sqrt((TWO_PI * g) / cfg.maxWavelength);
  const wMax = Math.sqrt((TWO_PI * g) / cfg.minWavelength);
  const U = Math.max(cfg.speed, 0.5);
  const wp = (0.877 * g) / U;
  const logSpan = Math.log(wMax / wMin);
  let sumKA = 0;
  for (let i = 0; i < n; i++) {
    const frac = (i + 0.5 + (rng() - 0.5) * 0.8) / n;
    const w = wMin * Math.exp(logSpan * frac);
    const dw = (w * logSpan) / n;
    const S = 0.0081 * g * g * Math.pow(w, -5) * Math.exp(-1.25 * Math.pow(wp / w, 4));
    const amp = Math.sqrt(2 * S * dw) * cfg.amplitudeScale;
    // cos^2 spreading by rejection sampling
    let u = 0;
    for (let tries = 0; tries < 32; tries++) {
      u = rng() * 2 - 1;
      const c = Math.cos((u * Math.PI) / 2);
      if (rng() < c * c) break;
    }
    const theta = ((cfg.directionDeg + u * cfg.spreadDeg) * Math.PI) / 180;
    const k = (w * w) / g;
    comps.push({
      dirX: Math.cos(theta),
      dirZ: Math.sin(theta),
      k,
      omega: w,
      amp,
      ampH: amp * cfg.choppiness,
      phase0: rng() * TWO_PI,
    });
    sumKA += k * amp * cfg.choppiness;
  }
  const scale = sumKA > cfg.steepnessBudget ? cfg.steepnessBudget / sumKA : 1;
  let steepness = 0;
  for (const c of comps) {
    c.ampH *= scale;
    steepness += c.k * c.ampH;
  }
  return { comps, steepness };
}

/** Time slots for which phase offsets are kept: t - FD_DT, t, t + FD_DT. */
export const TimeSlot = { Prev: 0, Now: 1, Next: 2 } as const;
export type TimeSlot = (typeof TimeSlot)[keyof typeof TimeSlot];

export class OceanModel {
  config!: OceanConfig;
  bathymetry!: Bathymetry;
  tables!: OceanTables;
  swells: SwellParams[] = [];
  wind: WindComponent[] = [];
  /** Total horizontal steepness sum(k * ampH) of the wind chop. */
  windSteepness = 0;
  /** Simulation time in seconds (double precision). */
  time = 0;

  /** Envelope offsets (phase0 - omega t, wrapped to ENVELOPE_PERIOD) per time slot and swell. */
  readonly swellOffsets = [new Float64Array(MAX_SWELLS), new Float64Array(MAX_SWELLS), new Float64Array(MAX_SWELLS)];
  /** Wind phase offsets (phase0 - omega t, wrapped to 2 pi) per time slot and component. */
  readonly windOffsets = [
    new Float64Array(MAX_WIND_WAVES),
    new Float64Array(MAX_WIND_WAVES),
    new Float64Array(MAX_WIND_WAVES),
  ];

  // scratch
  private readonly lookup = new Float64Array(2);
  private readonly field = new Float64Array(2);
  private readonly windField = new Float64Array(2);
  private readonly sH = new Float64Array(MAX_SWELLS);
  private readonly sPhi = new Float64Array(MAX_SWELLS);
  private readonly sC = new Float64Array(MAX_SWELLS);
  private readonly sK = new Float64Array(MAX_SWELLS);
  private readonly sDx = new Float64Array(MAX_SWELLS);
  private readonly sDz = new Float64Array(MAX_SWELLS);
  private readonly swellTmp = createSwellEval();
  private readonly windTmp = createWindEval();
  private readonly sampleSwell = createSwellEval();
  private readonly sampleWind = createWindDerivs();

  constructor(config: OceanConfig = DEFAULT_OCEAN_CONFIG) {
    this.rebuild(config);
  }

  /** (Re)build tables and derived constants. Cheap enough to call from a debug GUI. */
  rebuild(config: OceanConfig = this.config): void {
    this.config = cloneOceanConfig(config);
    const g = this.config.gravity;
    this.bathymetry = new Bathymetry(this.config.bathymetry);
    this.tables = new OceanTables(this.config, this.bathymetry);
    this.swells = this.config.swells.slice(0, MAX_SWELLS).map((sw) => {
      const omega = TWO_PI / sw.period;
      const theta = (sw.directionDeg * Math.PI) / 180;
      const pattern = new Float32Array(SET_LENGTH);
      const src = sw.setPattern.length > 0 ? sw.setPattern : [1];
      for (let i = 0; i < SET_LENGTH; i++) pattern[i] = src[i % src.length];
      return {
        omega,
        ky: deepWaveNumber(omega, g) * Math.sin(theta),
        amp0: sw.height / 2,
        sinT0: Math.sin(theta),
        cosT0: Math.cos(theta),
        cg0: (0.5 * g) / omega,
        c0: g / omega,
        hDeep: deepWaterDepth(omega, g),
        phase0: sw.phase,
        pattern,
      };
    });
    const w = buildWindComponents(this.config.wind, g);
    this.wind = w.comps;
    this.windSteepness = w.steepness;
    this.setTime(this.time);
  }

  /** Set the simulation time and refresh the phase offsets for t - FD_DT, t and t + FD_DT. */
  setTime(t: number): void {
    this.time = t;
    for (let slot = 0; slot < 3; slot++) {
      const ts = t + (slot - 1) * FD_DT;
      const so = this.swellOffsets[slot];
      for (let i = 0; i < this.swells.length; i++) {
        const s = this.swells[i];
        so[i] = pmod(s.phase0 - s.omega * ts, ENVELOPE_PERIOD);
      }
      const wo = this.windOffsets[slot];
      for (let j = 0; j < this.wind.length; j++) {
        const c = this.wind[j];
        wo[j] = pmod(c.phase0 - c.omega * ts, TWO_PI);
      }
    }
  }

  /** Still-water depth at (x, z) from the shared tables (matches the GPU). */
  depthAt(x: number, z: number): number {
    return this.tables.depthAt(x, z);
  }

  /** Evaluate the shoaling swell height field at world (x, z). Mirrors `oceanSwell` in GLSL. */
  evalSwell(x: number, z: number, out: SwellEval, slot: TimeSlot = TimeSlot.Now): SwellEval {
    const cfg = this.config;
    const g = cfg.gravity;
    this.tables.fieldAt(x, z, this.field);
    const h = this.field[0];
    const cap = this.field[1];
    const hEff = Math.max(h, cfg.minDepth);
    const offs = this.swellOffsets[slot];
    const n = this.swells.length;
    let hLin = 0;
    let k0 = 1;
    let dom = 0;
    for (let i = 0; i < n; i++) {
      const s = this.swells[i];
      this.tables.phaseLookup(i, x, this.lookup);
      const kxT = this.lookup[1];
      const phase = this.lookup[0] + s.ky * z + offs[i];
      const env = setEnvelope(s.pattern, phase / TWO_PI);
      const hc = Math.min(hEff, s.hDeep);
      const k = waveNumber(s.omega, hc, g);
      const kh = k * hc;
      const c = s.omega / k;
      const cg = 0.5 * (1 + (2 * kh) / Math.sinh(2 * kh)) * c;
      const sinT = (s.sinT0 * c) / s.c0;
      const cosT = Math.sqrt(Math.max(1 - sinT * sinT, 1e-4));
      const K = Math.sqrt(s.cg0 / cg) * Math.sqrt(s.cosT0 / cosT);
      if (i === 0) k0 = K;
      const H = 2 * s.amp0 * env * K;
      const kg = Math.sqrt(kxT * kxT + s.ky * s.ky);
      this.sH[i] = H;
      this.sPhi[i] = phase;
      this.sC[i] = c;
      this.sK[i] = k;
      this.sDx[i] = kxT / kg;
      this.sDz[i] = s.ky / kg;
      hLin += H;
      if (H > this.sH[dom]) dom = i;
    }

    // Depth-limited breaking with history: the height can't exceed what survived the shallowest
    // point crossed so far (cap * K), which is never more than gamma * h.
    const hLim = cap * k0;
    const ratio = hLin / Math.max(hLim, 1e-4);
    const capScale = ratio > 0.8 ? softCap(ratio) / ratio : 1;
    const height = hLin * capScale;
    const fullness = height / Math.max(cfg.breakerIndex * h, 1e-4);
    const breaking = smoothstep(BREAK_START, BREAK_FULL, ratio) * smoothstep(0.65, 0.9, fullness);

    // Shape: crest peaking from the Ursell number of the dominant component, forward lean as
    // the wave approaches the local depth limit.
    const kDom = n > 0 ? this.sK[dom] : 1;
    const lambda = TWO_PI / kDom;
    const ursell = (height * lambda * lambda) / (hEff * hEff * hEff);
    let p = 1 + (cfg.peakingMax - 1) * smoothstep(cfg.ursellStart, cfg.ursellFull, ursell);
    p += (1.6 - p) * 0.6 * breaking;
    const kappa = cfg.skewMax * smoothstep(0.5, 0.97, fullness);
    const mean = peakMean(p);

    let eta = 0;
    let ux = 0;
    let uz = 0;
    let domPsi = 0;
    for (let i = 0; i < n; i++) {
      const psi = solveSkew(this.sPhi[i], kappa, cfg.skewPhase);
      const y = this.sH[i] * capScale * (Math.pow(0.5 + 0.5 * Math.cos(psi), p) - mean);
      eta += y;
      const c = this.sC[i];
      const u = c - Math.sqrt(Math.max(c * c - 2 * g * y, 0.0025 * c * c));
      ux += this.sDx[i] * u;
      uz += this.sDz[i] * u;
      if (i === dom) domPsi = psi;
    }
    // Whitewater: on the crest and face of a broken wave the water is carried along at ~0.9 c.
    // Near the breaking limit the crest (and the top of the face, where the lip will form)
    // already moves at up to CREST_PUSH of the way to that speed.
    const steep = smoothstep(CREST_START, BREAK_START, ratio) * smoothstep(0.65, 0.9, fullness);
    if (n > 0 && (breaking > 0 || steep > 0)) {
      const c = this.sC[dom];
      const dx = this.sDx[dom];
      const dz = this.sDz[dom];
      const along = ux * dx + uz * dz;
      const target = 0.9 * c;
      const mask = smoothstep(0.3, 0.9, Math.cos(domPsi - 0.6));
      const crest = CREST_PUSH * steep * smoothstep(0.54, 0.955, Math.cos(domPsi - 0.2));
      if (along < target) {
        const push = (target - along) * Math.max(breaking * mask, crest);
        ux += dx * push;
        uz += dz * push;
      }
    }

    out.eta = eta;
    out.ux = ux;
    out.uz = uz;
    out.depth = h;
    out.height = height;
    out.ratio = ratio;
    out.fullness = fullness;
    out.breaking = breaking;
    out.peaking = p;
    out.skew = kappa;
    out.domPsi = domPsi;
    out.domSpeed = n > 0 ? this.sC[dom] : 0;
    out.domDirX = n > 0 ? this.sDx[dom] : 1;
    out.domDirZ = n > 0 ? this.sDz[dom] : 0;
    out.domK = n > 0 ? this.sK[dom] : 0;
    return out;
  }

  /** Evaluate wind chop at rest point (x0, z0). Mirrors `oceanWind` in GLSL. */
  evalWind(x0: number, z0: number, out: WindEval, slot: TimeSlot = TimeSlot.Now): WindEval {
    this.tables.fieldAt(x0, z0, this.windField);
    const fade = smoothstep(0.1, 1.5, this.windField[0]);
    const offs = this.windOffsets[slot];
    let dx = 0;
    let dz = 0;
    let dy = 0;
    let ux = 0;
    let uz = 0;
    for (let j = 0; j < this.wind.length; j++) {
      const w = this.wind[j];
      const ph = w.k * (w.dirX * x0 + w.dirZ * z0) + offs[j];
      const s = Math.sin(ph);
      const c = Math.cos(ph);
      const a = w.amp * fade;
      const hd = -w.ampH * fade * s;
      dx += w.dirX * hd;
      dz += w.dirZ * hd;
      dy += a * c;
      const u = w.omega * a * c;
      ux += w.dirX * u;
      uz += w.dirZ * u;
    }
    out.x0 = x0;
    out.z0 = z0;
    out.dx = dx;
    out.dz = dz;
    out.dy = dy;
    out.ux = ux;
    out.uz = uz;
    return out;
  }

  /** evalWind plus analytic derivatives with respect to the rest point and time. */
  evalWindDerivs(x0: number, z0: number, out: WindDerivs, slot: TimeSlot = TimeSlot.Now): WindDerivs {
    this.tables.fieldAt(x0, z0, this.windField);
    const fade = smoothstep(0.1, 1.5, this.windField[0]);
    const offs = this.windOffsets[slot];
    let dx = 0, dz = 0, dy = 0, ux = 0, uz = 0;
    let jxx = 0, jxz = 0, jzz = 0, gx = 0, gz = 0, dxdt = 0, dzdt = 0, dydt = 0;
    for (let j = 0; j < this.wind.length; j++) {
      const w = this.wind[j];
      const ph = w.k * (w.dirX * x0 + w.dirZ * z0) + offs[j];
      const s = Math.sin(ph);
      const c = Math.cos(ph);
      const a = w.amp * fade;
      const ah = w.ampH * fade;
      const hd = -ah * s;
      dx += w.dirX * hd;
      dz += w.dirZ * hd;
      dy += a * c;
      const u = w.omega * a * c;
      ux += w.dirX * u;
      uz += w.dirZ * u;
      const jd = -ah * w.k * c;
      jxx += jd * w.dirX * w.dirX;
      jxz += jd * w.dirX * w.dirZ;
      jzz += jd * w.dirZ * w.dirZ;
      const gy = -a * w.k * s;
      gx += gy * w.dirX;
      gz += gy * w.dirZ;
      const ht = ah * w.omega * c;
      dxdt += w.dirX * ht;
      dzdt += w.dirZ * ht;
      dydt += a * w.omega * s;
    }
    out.x0 = x0; out.z0 = z0;
    out.dx = dx; out.dz = dz; out.dy = dy; out.ux = ux; out.uz = uz;
    out.jxx = jxx; out.jxz = jxz; out.jzz = jzz; out.gx = gx; out.gz = gz;
    out.dxdt = dxdt; out.dzdt = dzdt; out.dydt = dydt;
    return out;
  }

  /**
   * Wind chop at world point (x, z): finds the rest point whose displaced position is (x, z)
   * and evaluates it there.
   */
  windAtWorld(x: number, z: number, out: WindEval, slot: TimeSlot = TimeSlot.Now): WindEval {
    let x0 = x;
    let z0 = z;
    for (let it = 0; it < WIND_INVERSE_ITERATIONS; it++) {
      this.evalWind(x0, z0, out, slot);
      x0 = x - out.dx;
      z0 = z - out.dz;
    }
    return this.evalWind(x0, z0, out, slot);
  }

  /** Rest point (x0, z0) of the wind-chop displacement for world point (x, z). */
  windRestPoint(x: number, z: number, out: WindEval, slot: TimeSlot = TimeSlot.Now): WindEval {
    let x0 = x;
    let z0 = z;
    for (let it = 0; it < WIND_INVERSE_ITERATIONS; it++) {
      this.evalWind(x0, z0, out, slot);
      x0 = x - out.dx;
      z0 = z - out.dz;
    }
    out.x0 = x0;
    out.z0 = z0;
    return out;
  }

  /** Surface elevation at world (x, z). */
  heightAt(x: number, z: number, slot: TimeSlot = TimeSlot.Now): number {
    const w = this.windAtWorld(x, z, this.windTmp, slot);
    return w.dy + this.evalSwell(x, z, this.swellTmp, slot).eta;
  }

  /**
   * Full surface query: height, normal, gradient, d(eta)/dt, water velocity and wave state.
   * The swell part uses central finite differences of the exact height field; the wind part
   * uses analytic Lagrangian derivatives. Cost ~ 7 swell evaluations + 1 wind inversion
   * (~10 us); physics should sample a coarse probe grid and interpolate.
   */
  sample(x: number, z: number, out: SurfaceSample = createSurfaceSample()): SurfaceSample {
    const d = FD_DX;
    const sw = this.evalSwell(x, z, this.sampleSwell, TimeSlot.Now);
    const rest = this.windRestPoint(x, z, this.windTmp, TimeSlot.Now);
    const wd = this.evalWindDerivs(rest.x0, rest.z0, this.sampleWind, TimeSlot.Now);
    // Eulerian wind gradient: grad(eta) = J^-T grad0(Y), J = I + dD/dx0 (symmetric here).
    const a11 = 1 + wd.jxx;
    const a12 = wd.jxz;
    const a22 = 1 + wd.jzz;
    const det = a11 * a22 - a12 * a12;
    const gwx = (a22 * wd.gx - a12 * wd.gz) / det;
    const gwz = (-a12 * wd.gx + a11 * wd.gz) / det;
    const etwT = wd.dydt - (gwx * wd.dxdt + gwz * wd.dzdt);

    const st = this.swellTmp;
    const sxp = this.evalSwell(x + d, z, st).eta;
    const sxm = this.evalSwell(x - d, z, st).eta;
    const szp = this.evalSwell(x, z + d, st).eta;
    const szm = this.evalSwell(x, z - d, st).eta;
    const stp = this.evalSwell(x, z, st, TimeSlot.Next).eta;
    const stm = this.evalSwell(x, z, st, TimeSlot.Prev).eta;

    const ex = gwx + (sxp - sxm) / (2 * d);
    const ez = gwz + (szp - szm) / (2 * d);
    const et = etwT + (stp - stm) / (2 * FD_DT);
    const ux = sw.ux + wd.ux;
    const uz = sw.uz + wd.uz;
    const inv = 1 / Math.sqrt(ex * ex + 1 + ez * ez);
    out.height = sw.eta + wd.dy;
    out.normalX = -ex * inv;
    out.normalY = inv;
    out.normalZ = -ez * inv;
    out.slopeX = ex;
    out.slopeZ = ez;
    out.dEtaDt = et;
    out.velX = ux;
    out.velY = et + ux * ex + uz * ez;
    out.velZ = uz;
    out.depth = sw.depth;
    out.waveHeight = sw.height;
    out.ratio = sw.ratio;
    out.fullness = sw.fullness;
    out.breaking = sw.breaking;
    out.phaseSpeed = sw.domSpeed;
    out.dirX = sw.domDirX;
    out.dirZ = sw.domDirZ;
    out.wavePhase = sw.domPsi;
    return out;
  }
}
