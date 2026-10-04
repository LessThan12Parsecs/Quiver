import { describe, expect, it } from 'vitest';
import { waveNumber } from '../src/ocean/dispersion';
import { Bathymetry } from '../src/ocean/bathymetry';
import { DEFAULT_OCEAN_CONFIG, SET_LENGTH } from '../src/ocean/oceanConfig';
import {
  OceanModel,
  createSurfaceSample,
  createSwellEval,
  createWindEval,
  setEnvelope,
  solveSkew,
} from '../src/ocean/waveModel';

const g = 9.81;

function exactK(omega: number, h: number): number {
  let k = (omega * omega) / g;
  for (let i = 0; i < 100; i++) {
    const t = Math.tanh(k * h);
    const f = g * k * t - omega * omega;
    const df = g * t + g * k * h * (1 - t * t);
    k -= f / df;
  }
  return k;
}

describe('dispersion', () => {
  it('matches the exact dispersion relation across shallow to deep water', () => {
    for (const T of [4, 7, 11, 16]) {
      const omega = (2 * Math.PI) / T;
      for (const h of [0.15, 0.5, 1, 2, 5, 20, 100]) {
        const k = waveNumber(omega, h, g);
        expect(Math.abs(k - exactK(omega, h)) / exactK(omega, h)).toBeLessThan(1e-6);
      }
    }
  });
});

describe('bathymetry', () => {
  const b = new Bathymetry(DEFAULT_OCEAN_CONFIG.bathymetry);
  it('passes through the profile control points', () => {
    for (const [x, d] of DEFAULT_OCEAN_CONFIG.bathymetry.profile) expect(b.baseDepth(x)).toBeCloseTo(d, 9);
  });
  it('has the bar crest at the configured depth at the peak', () => {
    const bar = DEFAULT_OCEAN_CONFIG.bathymetry.bar;
    expect(b.depth(bar.x0, 0)).toBeCloseTo(bar.peakDepth, 6);
  });
  it('makes the bar deeper away from the peak and fades it out', () => {
    expect(b.depth(-20, 0)).toBeLessThan(b.depth(-20 + 0.45 * 52, 60));
    expect(b.barBump(0, 300)).toBe(0);
  });
});

describe('tables', () => {
  const m = new OceanModel();
  it('phase lookup is continuous with derivative ~ kx', () => {
    const out = [0, 0];
    const out2 = [0, 0];
    for (const x of [-1700, -1600, -900.3, -40.1, 0, 12.7, 150, 399.9, 400, 450]) {
      m.tables.phaseLookup(0, x, out);
      m.tables.phaseLookup(0, x + 1e-3, out2);
      expect(Math.abs((out2[0] - out[0]) / 1e-3 - out[1])).toBeLessThan(2e-3);
    }
  });
  it('field depth matches the analytic bathymetry', () => {
    for (let i = 0; i < 200; i++) {
      const x = -250 + i * 2.2;
      const z = -300 + i * 3.1;
      expect(Math.abs(m.depthAt(x, z) - m.bathymetry.depth(x, z))).toBeLessThan(0.02);
    }
  });
});

describe('waveform helpers', () => {
  it('set envelope interpolates the pattern and is periodic', () => {
    const p = DEFAULT_OCEAN_CONFIG.swells[0].setPattern;
    for (let i = 0; i < SET_LENGTH; i++) {
      expect(setEnvelope(p, i)).toBeCloseTo(p[i], 6);
      expect(setEnvelope(p, i + 0.37)).toBeCloseTo(setEnvelope(p, i + 0.37 + 3 * SET_LENGTH), 6);
      expect(setEnvelope(p, -i - 0.21)).toBeCloseTo(setEnvelope(p, SET_LENGTH - i - 0.21), 6);
    }
  });
  it('solves the skew (Kepler) equation for every phase and lean', () => {
    for (const kappa of [0, 0.3, 0.7, 0.9, 0.95]) {
      for (let i = 0; i <= 400; i++) {
        const phi = -Math.PI + (2 * Math.PI * i) / 400;
        const psi = solveSkew(phi, kappa, -0.35);
        const back = psi + kappa * Math.cos(psi + 0.35);
        const d = Math.atan2(Math.sin(back - phi), Math.cos(back - phi));
        expect(Math.abs(d)).toBeLessThan(1e-9);
      }
    }
  });
});

describe('ocean model', () => {
  const m = new OceanModel();
  const se = createSwellEval();
  const we = createWindEval();
  const s = createSurfaceSample();

  it('inverts the wind-chop displacement', () => {
    m.setTime(12.3);
    for (let i = 0; i < 300; i++) {
      const x = -300 + i * 1.37;
      const z = -50 + i * 0.71;
      const r = m.windRestPoint(x, z, we);
      m.evalWind(r.x0, r.z0, we);
      expect(Math.hypot(r.x0 + we.dx - x, r.z0 + we.dz - z)).toBeLessThan(1e-4);
    }
  });

  it('is finite everywhere with unit normals', () => {
    for (const t of [0, 55.5, 999.9]) {
      m.setTime(t);
      for (let i = 0; i < 2000; i++) {
        const x = -2000 + (i % 100) * 23.3;
        const z = -700 + Math.floor(i / 100) * 71;
        m.sample(x, z, s);
        for (const v of [s.height, s.velX, s.velY, s.velZ, s.slopeX, s.slopeZ, s.ratio, s.breaking]) {
          expect(Number.isFinite(v)).toBe(true);
        }
        expect(Math.hypot(s.normalX, s.normalY, s.normalZ)).toBeCloseTo(1, 9);
      }
    }
  });

  it('water velocity keeps a floating particle on the surface (kinematic condition)', () => {
    const dt = 1 / 240;
    let worst = 0;
    for (const t of [20, 67.4, 140]) {
      for (let i = 0; i < 300; i++) {
        const x = -80 + (i % 60) * 2.1;
        const z = -40 + Math.floor(i / 60) * 17;
        m.setTime(t);
        m.sample(x, z, s);
        const px = x + s.velX * dt;
        const py = s.height + s.velY * dt;
        const pz = z + s.velZ * dt;
        m.setTime(t + dt);
        worst = Math.max(worst, Math.abs(m.heightAt(px, pz) - py));
      }
    }
    // second-order error only: well under a millimetre per step
    expect(worst).toBeLessThan(1e-3);
  });

  it('does not break in deep water and breaks set waves on the bar', () => {
    let maxBreakDeep = 0;
    let barBreak = 0;
    for (let t = 0; t < 16 * 11; t += 0.5) {
      m.setTime(t);
      for (let x = -400; x < -120; x += 4) maxBreakDeep = Math.max(maxBreakDeep, m.evalSwell(x, 0, se).breaking);
      for (let x = -40; x < -10; x += 1) barBreak = Math.max(barBreak, m.evalSwell(x, 0, se).breaking);
    }
    expect(maxBreakDeep).toBe(0);
    expect(barBreak).toBeGreaterThan(0.9);
  });

  it('never lets a wave exceed the depth limit', () => {
    for (let t = 0; t < 176; t += 3.7) {
      m.setTime(t);
      for (let x = -100; x < 130; x += 0.7) {
        m.evalSwell(x, (x * 7) % 90, se);
        // bilinear interpolation of depth and cap can overshoot by a hair between grid nodes
        expect(se.height).toBeLessThanOrEqual(1.01 * DEFAULT_OCEAN_CONFIG.breakerIndex * Math.max(se.depth, 0) + 5e-3);
      }
    }
  });

  it('loses height crossing the bar (energy dissipated by breaking)', () => {
    let before = 0;
    let after = 0;
    for (let t = 0; t < 176; t += 0.5) {
      m.setTime(t);
      before = Math.max(before, m.evalSwell(-45, 0, se).height);
      after = Math.max(after, m.evalSwell(30, 0, se).height);
    }
    expect(after).toBeLessThan(0.75 * before);
  });

  it('peels: the break point moves shoreward away from the peak', () => {
    const firstBreak = (z: number): number => {
      let xb = Infinity;
      for (let t = 0; t < 176; t += 0.5) {
        m.setTime(t);
        for (let x = -80; x < 60; x += 0.5) {
          if (m.evalSwell(x, z, se).breaking > 0.3) {
            xb = Math.min(xb, x);
            break;
          }
        }
      }
      return xb;
    };
    const b0 = firstBreak(0);
    const b30 = firstBreak(30);
    const b60 = firstBreak(-60);
    expect(b0).toBeLessThan(b30);
    expect(b30).toBeLessThan(b60);
  });
});
