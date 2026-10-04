/**
 * Spawn helpers.
 *
 * lineupSpawn: prone, still, outside the peak (x ≈ −75, z ≈ 5), facing the beach.
 * waveSpawn:   standing on the upper part of the steepening, still unbroken face of the next set
 *              wave down the line from the peak (≈20–30 m to one side, ahead of the peeling
 *              section), board flat on the face, angled 55° from the wave direction toward the
 *              unbroken shoulder, a little slower than the wave (velocity component along the wave
 *              direction 0.8 × the local phase speed: the face's slope brings it up to speed — a
 *              settled trim rather than a board shot across the face faster than it can sustain).
 */
import { createSwellEval, type OceanModel } from '../ocean/waveModel';
import type { Spawn } from './SurfSim';

/** Prone, still, outside the peak, facing the beach (+X). */
export function lineupSpawn(ocean: OceanModel, time: number = ocean.time): Spawn {
  return { x: -75, z: 5, headingRad: 0, stance: 'prone', speed: 0, time };
}

export interface WaveSpawnOptions {
  /** Start searching at this time (default: ocean.time). */
  fromTime?: number;
  /** Which side of the peak: +1 (+Z, rides toward +Z) or −1. Default +1. */
  side?: 1 | -1;
  /** Lateral distances from the peak (|z|) to try, m. Default [24, 30, 20]: down the line, so
   * the ride starts ahead of the peeling section (right next to the peak the wave is about to
   * break on the rider). */
  offsets?: number[];
  /** Minimum local wave height, m. Default 1.2 (set waves; regular waves are ≈1 m at the bar). */
  minHeight?: number;
  /** Fullness window (closeness to breaking). Default [0.5, 0.75]: a face that is steepening
   * but still makeable (≥ 0.8 it is a critical, about-to-break face). */
  minFullness?: number;
  maxFullness?: number;
  /** Maximum whitewater at the spot. Default 0.05. */
  maxBreaking?: number;
  /** Heading angle from the wave direction toward the shoulder, rad. Default 55°, i.e. ~35° off
   * the crest line ("angled 35° down the line"). Note: 30° from the fall line runs nearly straight
   * down these steep near-breaking faces and usually ends in a pearl at the bottom. */
  angleRad?: number;
  /** How far ahead in time to search, s. Default 420 (more than two set cycles). */
  searchSeconds?: number;
  /** Search time step, s. Default 0.25. */
  timeStep?: number;
  /** Waveform phase window on the front face (0 = crest, π/2 ≈ mid face). Default [0.25, 0.7]:
   * the upper face (lower down, a no-input rider soon drops to the steep bottom and pearls). */
  minPhase?: number;
  maxPhase?: number;
  /** Velocity component along the wave direction, × the local phase speed. Default 0.8. */
  speedScale?: number;
  /** Cross-shore search range, m. Default [−70, −10]. */
  xMin?: number;
  xMax?: number;
}

export interface WaveSpawnResult {
  spawn: Spawn;
  /** Simulation time of the spawn (= spawn.time). */
  time: number;
  /** Wave state at the spawn point. */
  waveHeight: number;
  fullness: number;
  phaseSpeed: number;
  /** Face slope along the wave direction at the spawn point (negative = descending shoreward). */
  slope: number;
}

/**
 * Find the next set wave's steepening unbroken face down the line from the peak and return a
 * standing spawn on it.
 * Conditions at the spot: height > minHeight, fullness in [minFullness, maxFullness],
 * breaking < maxBreaking, not broken further out (ratio < 0.95), on the upper/middle front face
 * (waveform phase 0.25–0.7 rad). Among candidates at the first qualifying time, the one nearest
 * the middle of that phase window is used. The board moves with the wave: the velocity component
 * along the wave direction is speedScale × the local phase speed, angled `angleRad` toward the
 * shoulder (away from the peak). Returns null if nothing qualifies within `searchSeconds`.
 */
export function waveSpawn(ocean: OceanModel, opts: WaveSpawnOptions = {}): WaveSpawnResult | null {
  const side = opts.side ?? 1;
  const offsets = opts.offsets ?? [24, 30, 20];
  const minH = opts.minHeight ?? 1.2;
  const fMin = opts.minFullness ?? 0.5;
  const fMax = opts.maxFullness ?? 0.75;
  const bMax = opts.maxBreaking ?? 0.05;
  const angle = opts.angleRad ?? (55 * Math.PI) / 180;
  const t0 = opts.fromTime ?? ocean.time;
  const span = opts.searchSeconds ?? 420;
  const dt = opts.timeStep ?? 0.25;
  const xMin = opts.xMin ?? -70;
  const xMax = opts.xMax ?? -10;
  const psiMin = opts.minPhase ?? 0.25;
  const psiMax = opts.maxPhase ?? 0.7;
  const speedScale = opts.speedScale ?? 0.8;
  const psiTarget = 0.5 * (psiMin + psiMax);
  const se = createSwellEval();
  const savedTime = ocean.time;
  let result: WaveSpawnResult | null = null;
  try {
    for (let t = t0; t <= t0 + span && !result; t += dt) {
      ocean.setTime(t);
      let best: { x: number; z: number; slope: number; h: number; f: number; c: number; dx: number; dz: number; score: number } | null = null;
      for (const off of offsets) {
        const z = side * off;
        // coarse scan for a tall, steepening wave
        for (let xc = xMin; xc <= xMax; xc += 1) {
          ocean.evalSwell(xc, z, se);
          if (se.height < minH || se.fullness < fMin - 0.08) continue;
          // refine around xc
          for (let x = xc - 0.5; x <= xc + 0.5; x += 0.25) {
            ocean.evalSwell(x, z, se);
            if (se.height < minH || se.fullness < fMin || se.fullness > fMax) continue;
            if (se.breaking > bMax || se.ratio > 0.95) continue;
            if (se.domPsi < psiMin || se.domPsi > psiMax) continue;
            const h = se.height;
            const f = se.fullness;
            const c = se.domSpeed;
            const dx = se.domDirX;
            const dz = se.domDirZ;
            const psi = se.domPsi;
            const e1 = ocean.evalSwell(x + 0.25 * dx, z + 0.25 * dz, se).eta;
            const e0 = ocean.evalSwell(x - 0.25 * dx, z - 0.25 * dz, se).eta;
            const slope = (e1 - e0) / 0.5;
            // prefer the middle of the face
            const score = Math.abs(psi - psiTarget);
            if (!best || score < best.score) best = { x, z, slope, h, f, c, dx, dz, score };
          }
        }
        if (best) break; // prefer the first offset that works
      }
      if (best) {
        const waveDir = Math.atan2(best.dz, best.dx);
        const heading = waveDir + side * angle;
        // the board moves with the water across its heading (SurfSim.reset); pick the speed along
        // the heading that makes the ground velocity along the wave speedScale × the phase speed
        const ws = ocean.sample(best.x, best.z);
        const wLat = -Math.sin(heading) * ws.velX + Math.cos(heading) * ws.velZ;
        const speed = (speedScale * best.c + wLat * side * Math.sin(angle)) / Math.cos(angle);
        result = {
          spawn: {
            x: best.x,
            z: best.z,
            headingRad: heading,
            stance: 'standing',
            speed,
            onWave: true,
            time: t,
          },
          time: t,
          waveHeight: best.h,
          fullness: best.f,
          phaseSpeed: best.c,
          slope: best.slope,
        };
      }
    }
  } finally {
    ocean.setTime(savedTime);
  }
  return result;
}
