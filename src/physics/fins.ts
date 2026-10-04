/**
 * Fin hydrodynamics: a fin is a low-aspect-ratio lifting surface.
 *
 * Coefficients (α = angle of attack in (-π, π], positive when the flow crosses the fin along +normal):
 *   attached:   C_L = a sinα|cosα|,   a = 2π·AR/(AR + 2)  (lifting-line slope for low AR)
 *               C_D = C_D0 + C_L²/(π e AR)
 *   flat plate: C_L = C_Nmax sinα|cosα|,  C_D = C_D0 + C_Nmax sin²α
 *   blended with a logistic stall weight σ(|α'|) (α' = incidence folded to [0, π/2]) centred at
 *   16° (width 2°), which puts the lift peak at ≈ 14°; the force is smooth through the stall and
 *   sensible for reversed flow.
 *
 * Force on the fin from water moving at relative velocity w (= −v_rel of the fin, spanwise part
 * removed): F = q A (C_L l̂ + C_D ŵ), q = ½ρ|w|², ŵ = w/|w|, l̂ = unit part of the fin normal
 * perpendicular to ŵ. With this construction the lift always pushes the fin *with* the cross-flow,
 * i.e. it resists sideslip, and the sign does not depend on which way the normal points.
 */
import type { Vector3 } from 'three';
import { RHO_WATER } from './constants';

export const FIN_MODEL = {
  /** Centre of the logistic stall blend, rad (the lift peak lands at ≈ 14°). */
  stallAngle: (16 * Math.PI) / 180,
  /** Width of the stall transition, rad. */
  stallWidth: (2 * Math.PI) / 180,
  /** Profile drag coefficient. */
  cd0: 0.012,
  /** Oswald efficiency. */
  oswald: 0.85,
  /** Flat-plate normal force coefficient after stall (low aspect ratio). */
  cnMax: 1.1,
};

export interface FinCoefficients {
  cl: number;
  cd: number;
  /** Attached-flow weight 1 → 0 through the stall. */
  attached: number;
  /** Local lift slope dC_L/dα used for the implicit damping. */
  slope: number;
}

export function finLiftSlope(aspectRatio: number): number {
  return (2 * Math.PI * aspectRatio) / (aspectRatio + 2);
}

/** Lift/drag coefficients of a fin at angle of attack alpha (rad) for an effective aspect ratio. */
export function finCoefficients(alpha: number, aspectRatio: number, out: FinCoefficients): FinCoefficients {
  const m = FIN_MODEL;
  const s = Math.sin(alpha);
  const c = Math.abs(Math.cos(alpha));
  const inc = Math.atan2(Math.abs(s), c); // incidence 0..π/2
  const sig = 1 / (1 + Math.exp((inc - m.stallAngle) / m.stallWidth));
  const ar = Math.max(aspectRatio, 0.05);
  const a = finLiftSlope(ar);
  const clAtt = a * s * c;
  const clFp = m.cnMax * s * c;
  out.cl = sig * clAtt + (1 - sig) * clFp;
  out.cd = m.cd0 + sig * ((clAtt * clAtt) / (Math.PI * m.oswald * ar)) + (1 - sig) * m.cnMax * s * s;
  out.attached = sig;
  out.slope = sig * a + (1 - sig) * m.cnMax * 0.5;
  return out;
}

export interface FinForceResult {
  /** Force on the fin, world. */
  fx: number;
  fy: number;
  fz: number;
  /** Angle of attack, rad. */
  alpha: number;
  /** In-plane relative flow speed, m/s. */
  speed: number;
  /** Damping coefficient along the fin normal for the implicit solver, N·s/m. */
  damping: number;
  attached: number;
}

const coeff: FinCoefficients = { cl: 0, cd: 0, attached: 1, slope: 0 };

/**
 * Hydrodynamic force on a fin.
 * @param vrx..vrz  velocity of the fin centre relative to the water (world)
 * @param span      unit span direction (world)
 * @param forward   unit chord direction trailing → leading edge (world)
 * @param normal    unit fin normal (world)
 * @param area      wetted area (m²), already scaled by the submerged span fraction
 * @param aspectRatio effective aspect ratio of the wetted part
 */
export function finForce(
  vrx: number,
  vry: number,
  vrz: number,
  span: Vector3,
  forward: Vector3,
  normal: Vector3,
  area: number,
  aspectRatio: number,
  out: FinForceResult,
): FinForceResult {
  // remove spanwise flow; w = water velocity relative to the fin
  const vs = vrx * span.x + vry * span.y + vrz * span.z;
  const wx = -(vrx - vs * span.x);
  const wy = -(vry - vs * span.y);
  const wz = -(vrz - vs * span.z);
  const speed = Math.sqrt(wx * wx + wy * wy + wz * wz);
  out.fx = 0;
  out.fy = 0;
  out.fz = 0;
  out.alpha = 0;
  out.speed = speed;
  out.damping = 0;
  out.attached = 1;
  if (speed < 1e-6 || area <= 0) return out;
  const ux = wx / speed, uy = wy / speed, uz = wz / speed;
  // chord direction leading → trailing edge is −forward
  const wc = -(ux * forward.x + uy * forward.y + uz * forward.z);
  const wn = ux * normal.x + uy * normal.y + uz * normal.z;
  const alpha = Math.atan2(wn, wc);
  finCoefficients(alpha, aspectRatio, coeff);
  const q = 0.5 * RHO_WATER * speed * speed * area;
  // l̂ = normal − (normal·ŵ) ŵ
  let lx = normal.x - wn * ux;
  let ly = normal.y - wn * uy;
  let lz = normal.z - wn * uz;
  const ll = Math.sqrt(lx * lx + ly * ly + lz * lz);
  let cl = coeff.cl;
  if (ll > 1e-9) {
    lx /= ll;
    ly /= ll;
    lz /= ll;
  } else {
    cl = 0;
  }
  out.fx = q * (cl * lx + coeff.cd * ux);
  out.fy = q * (cl * ly + coeff.cd * uy);
  out.fz = q * (cl * lz + coeff.cd * uz);
  out.alpha = alpha;
  out.damping = 0.5 * RHO_WATER * area * speed * Math.max(coeff.slope, 0.3);
  out.attached = coeff.attached;
  return out;
}
