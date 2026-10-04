/**
 * Hydrodynamic hull of a board: planform cells, a coarse water probe grid and the per-cell
 * water forces.
 *
 * Cells (nu × nv over u × v) store, in the board local frame: the bottom point at the cell
 * centroid, the deck height above it, the exact cell area and volume (sub-integrated so the cells
 * add up to the board's volume), the outward bottom/deck normals (rocker slope, rail tuck/bevel)
 * and, for the outer column, the rail side area and its outward normal.
 *
 * Water: `sampleWater()` calls `ocean.sample()` at a probe grid (pu × pv points on the bottom,
 * moving with the board) once per step; each cell and fin interpolates height, slope, water
 * velocity and still-water depth bilinearly in (u, v) and corrects the height for its exact
 * horizontal position with the local slope.
 *
 * Forces per cell (see docs/DESIGN.md "Physics"):
 *  1. Hydrostatics: the bottom→deck segment's submerged fraction f (any orientation) gives the
 *     displaced volume V·f at its submerged centroid. The force is the free-surface pressure
 *     gradient: ρ g V f cosθ n̂ (n̂ = water surface normal, cosθ = n̂_y), which is exactly
 *     Archimedes in still water and, on a sloping wave face, reproduces the "sliding down the
 *     face" push felt by a floating object (Bernoulli along the surface: a co-moving floater
 *     accelerates with the water at g sinθ along the face).
 *  2. Planing/impact pressure on wetted bottom (and deck) cells: v_rel = v_cell − v_water,
 *     v_n = v_rel·n_out: F = −n_out ½ρA_w (C_p1|v_rel|v_n + C_p2 v_n²) for v_n > 0, a small
 *     suction for v_n < 0, plus a low-speed radiation damping term −n_out B A_w v_n.
 *  3. Skin friction (ITTC 1957 × form factor) on the wetted area, tangential relative flow.
 *  4. Rail side pressure on submerged rail cells moving sideways into the water.
 *  5. Fins: low-aspect lifting surfaces (fins.ts), area scaled by the submerged span.
 *  6. Angular damping ∝ wetted area.
 *  7. Added mass of the wetted bottom (flat plate at the free surface, ρπb²/8 per metre of
 *     wetted length, distributed over the wetted cells along their normals) enters the implicit
 *     mass matrix, with the matching diffraction force m_a·(dw_n/dt) from the water's normal
 *     acceleration under each cell. It makes the light board respond to chop and slams like a
 *     real hull (≈ 100 kg of water moves with a planing funboard in heave).
 * All stiff velocity-dependent terms also report their linearisation to the ImplicitSystem.
 */
import { Vector3 } from 'three';
import type { OceanModel, SurfaceSample } from '../ocean/waveModel';
import { createSurfaceSample } from '../ocean/waveModel';
import type { BoardShape, FinGeometry } from './boardShape';
import { GRAVITY, NU_WATER, RHO_WATER } from './constants';
import { finForce, type FinForceResult } from './fins';
import type { ImplicitSystem } from './implicit';
import type { RigidBody } from './RigidBody';

/** Tunable hydrodynamic coefficients (shared by all hulls; a debug GUI may edit them). */
export const HYDRO = {
  /** Planing pressure coefficient on |v_rel|·v_n. */
  cp1: 1.0,
  /** Impact pressure coefficient on v_n². */
  cp2: 1.0,
  /** Suction coefficient for surfaces moving away from the water (v_n < 0). */
  suction: 0.15,
  /** Cross-flow drag coefficient of a deeply submerged surface moving along its normal (the
   * flow separates round the board; replaces the planing law once the deck is well under water). */
  crossflowCd: 1.0,
  /** Lift (planing-law) coefficient of a deeply submerged surface relative to the planing value:
   * a low-aspect-ratio plate in unbounded fluid has a smaller lift slope (≈ πAR/2 per rad). */
  deepLift: 0.4,
  /** Low-speed radiation (wave-making) damping of normal motion, kg/(m² s). */
  radiation: 500,
  /** Tangential speed at which the radiation damping has halved, m/s. */
  radiationFadeSpeed: 1.5,
  /** Form factor on the ITTC flat-plate friction. */
  formFactor: 1.3,
  /** Rail side pressure coefficients (as cp1, cp2). */
  railCp1: 0.8,
  railCp2: 1.0,
  /** Angular damping per m² of wetted area, N·m·s. */
  angularDamping: 12,
  /** Immersion over which a cell goes from dry to fully wetted, m. */
  wetDepth: 0.012,
  /** Added mass coefficient: a wetted strip of beam b carries addedMass·ρ·π·b²/8 per metre (flat
   * plate at the free surface), rising to twice that (ρ·π·b²/4, plate in unbounded fluid) once the
   * deck is deeper than half the beam; distributed over its wetted cells along the bottom normal. */
  addedMass: 1,
  /** Seabed contact stiffness per m² of cell area, N/m³, and damping, N·s/m³. */
  seabedStiffness: 4e5,
  seabedDamping: 4e3,
  seabedFriction: 0.5,
};

/** Totals of the water forces for one step (world frame). */
export interface HullDiagnostics {
  buoyancy: Vector3;
  /** Planing / impact pressure + suction + radiation (bottom and deck). */
  pressure: Vector3;
  friction: Vector3;
  rail: Vector3;
  fin: Vector3;
  seabed: Vector3;
  /** Displaced volume of the board, m³. */
  submergedVolume: number;
  /** Wetted bottom area, m². */
  wettedArea: number;
  /** Wetted deck area, m². */
  deckWettedArea: number;
  /** Wetted length along the board, m. */
  wettedLength: number;
  /** Fin angle of attack (rad), in-plane flow speed and attached-flow weight, per fin. */
  finAlpha: number[];
  finSpeed: number[];
  finAttached: number[];
  finSubmerged: number[];
}

export function createHullDiagnostics(nFins: number): HullDiagnostics {
  return {
    buoyancy: new Vector3(),
    pressure: new Vector3(),
    friction: new Vector3(),
    rail: new Vector3(),
    fin: new Vector3(),
    seabed: new Vector3(),
    submergedVolume: 0,
    wettedArea: 0,
    deckWettedArea: 0,
    wettedLength: 0,
    finAlpha: new Array<number>(nFins).fill(0),
    finSpeed: new Array<number>(nFins).fill(0),
    finAttached: new Array<number>(nFins).fill(1),
    finSubmerged: new Array<number>(nFins).fill(0),
  };
}

export interface HullOptions {
  /** Cells along the board. */
  nu?: number;
  /** Cells across the board. */
  nv?: number;
  /** Probes along the board. */
  pu?: number;
  /** Probes across the board. */
  pv?: number;
}

/** Interpolated water state at a point (scratch). */
interface WaterAt {
  h: number;
  sx: number;
  sz: number;
  vx: number;
  vy: number;
  vz: number;
  seabed: number;
}

export class BoardHull {
  readonly shape: BoardShape;
  readonly nu: number;
  readonly nv: number;
  readonly n: number;
  // --- cell geometry, local frame
  /** Cell centroid x, z and bottom / deck y. */
  readonly cx: Float64Array;
  readonly cz: Float64Array;
  readonly cyb: Float64Array;
  readonly cyd: Float64Array;
  readonly cu: Float64Array;
  readonly cv: Float64Array;
  readonly area: Float64Array;
  readonly volume: Float64Array;
  /** Outward bottom normal (3 per cell). */
  readonly nb: Float64Array;
  /** Outward deck normal (3 per cell). */
  readonly nd: Float64Array;
  /** −1 left rail cell, +1 right rail cell, 0 inner. */
  readonly rail: Int8Array;
  readonly sideArea: Float64Array;
  /** Rail side outward normal (3 per cell). */
  readonly sn: Float64Array;
  /** Rail point x, z and rail bottom / deck y (per cell, used when rail != 0). */
  readonly rx: Float64Array;
  readonly rz: Float64Array;
  readonly ryb: Float64Array;
  readonly ryd: Float64Array;
  // --- probes
  readonly pu: number;
  readonly pv: number;
  readonly np: number;
  /** Probe parametric positions. */
  readonly probeU: Float64Array;
  readonly probeV: Float64Array;
  /** Probe local points (3 per probe). */
  readonly probeLocal: Float64Array;
  /** Probe world positions (3 per probe), refreshed by sampleWater. */
  readonly probeWorld: Float64Array;
  readonly probeSamples: SurfaceSample[];
  // per-probe water arrays
  private readonly ph: Float64Array;
  private readonly psx: Float64Array;
  private readonly psz: Float64Array;
  private readonly pvx: Float64Array;
  private readonly pvy: Float64Array;
  private readonly pvz: Float64Array;
  private readonly pseabed: Float64Array;
  /** Bilinear weights (4 per cell) and probe indices. */
  private readonly cwIdx: Int32Array;
  private readonly cwW: Float64Array;
  private readonly rwIdx: Int32Array;
  private readonly rwW: Float64Array;
  private readonly fwIdx: Int32Array;
  private readonly fwW: Float64Array;
  /** Per-cell force (3 per cell) written when `recordCellForces` is true (debug arrows). */
  readonly cellForce: Float64Array;
  /** Per-cell submerged fraction of the last step. */
  readonly cellSubmerged: Float64Array;
  recordCellForces = false;
  readonly fins: FinGeometry[];
  /** Per-fin force (3 per fin) and application point (3 per fin, world) of the last step. */
  readonly finForceWorld: Float64Array;
  readonly finPointWorld: Float64Array;

  private readonly water: WaterAt = { h: 0, sx: 0, sz: 0, vx: 0, vy: 0, vz: 0, seabed: -100 };
  // per-step scratch (pass 1 → pass 2)
  private readonly sRb: Float64Array;
  private readonly sDb: Float64Array;
  private readonly sDd: Float64Array;
  private readonly sW: Float64Array;
  /** Planing pressure distribution factor per cell (mean 1 over a strip's wetted length). */
  readonly sPf: Float64Array;
  /** Local board beam at each cell, m. */
  private readonly beam: Float64Array;
  /** Water velocity at each cell in the previous step (for the added-mass / diffraction force). */
  private readonly prevWater: Float64Array;
  private readonly prevWet: Uint8Array;
  /** Step used for the water acceleration, s. */
  waterDt = 1 / 240;
  /** Total added mass of the last step (heave, kg). */
  addedMassTotal = 0;
  private readonly finRes: FinForceResult = { fx: 0, fy: 0, fz: 0, alpha: 0, speed: 0, damping: 0, attached: 1 };
  private readonly tmpSpan = new Vector3();
  private readonly tmpFwd = new Vector3();
  private readonly tmpNrm = new Vector3();
  /** Smoothed wetted length used for the friction Reynolds number. */
  private wetLength = 1;

  constructor(shape: BoardShape, opts: HullOptions = {}) {
    this.shape = shape;
    const nu = (this.nu = opts.nu ?? 24);
    const nv = (this.nv = opts.nv ?? 6);
    const n = (this.n = nu * nv);
    this.cx = new Float64Array(n);
    this.cz = new Float64Array(n);
    this.cyb = new Float64Array(n);
    this.cyd = new Float64Array(n);
    this.cu = new Float64Array(n);
    this.cv = new Float64Array(n);
    this.area = new Float64Array(n);
    this.volume = new Float64Array(n);
    this.nb = new Float64Array(3 * n);
    this.nd = new Float64Array(3 * n);
    this.rail = new Int8Array(n);
    this.sideArea = new Float64Array(n);
    this.sn = new Float64Array(3 * n);
    this.rx = new Float64Array(n);
    this.rz = new Float64Array(n);
    this.ryb = new Float64Array(n);
    this.ryd = new Float64Array(n);
    this.cellForce = new Float64Array(3 * n);
    this.sRb = new Float64Array(3 * n);
    this.sDb = new Float64Array(n);
    this.sDd = new Float64Array(n);
    this.sW = new Float64Array(6 * n);
    this.sPf = new Float64Array(n).fill(1);
    this.beam = new Float64Array(n);
    this.prevWater = new Float64Array(3 * n);
    this.prevWet = new Uint8Array(n);
    this.cellSubmerged = new Float64Array(n);
    this.fins = shape.fins;
    this.finForceWorld = new Float64Array(3 * this.fins.length);
    this.finPointWorld = new Float64Array(3 * this.fins.length);

    const L = shape.length;
    const tmp = new Vector3();
    const sub = 6;
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const k = i * nv + j;
        // sub-integrate area, volume and area centroid
        let A = 0;
        let V = 0;
        let mx = 0;
        let mv = 0;
        for (let a = 0; a < sub; a++) {
          const u = (i + (a + 0.5) / sub) / nu;
          const hw = shape.halfWidth(u);
          for (let b = 0; b < sub; b++) {
            const v = -1 + (2 * (j + (b + 0.5) / sub)) / nv;
            const dA = (L / (nu * sub)) * ((2 * hw) / (nv * sub));
            A += dA;
            V += dA * shape.thickness(u, v);
            mx += dA * u;
            mv += dA * v;
          }
        }
        const uc = A > 0 ? mx / A : (i + 0.5) / nu;
        const vc = A > 0 ? mv / A : -1 + (2 * (j + 0.5)) / nv;
        this.cu[k] = uc;
        this.cv[k] = vc;
        this.beam[k] = 2 * shape.halfWidth(uc);
        this.area[k] = A;
        this.volume[k] = V;
        this.cx[k] = shape.x(uc);
        this.cz[k] = shape.z(uc, vc);
        this.cyb[k] = shape.bottomY(uc, vc);
        this.cyd[k] = shape.bottomY(uc, vc) + (A > 0 ? V / A : shape.thickness(uc, vc));
        shape.bottomNormal(uc, vc, tmp);
        this.nb[3 * k] = tmp.x;
        this.nb[3 * k + 1] = tmp.y;
        this.nb[3 * k + 2] = tmp.z;
        shape.deckNormal(uc, vc, tmp);
        this.nd[3 * k] = tmp.x;
        this.nd[3 * k + 1] = tmp.y;
        this.nd[3 * k + 2] = tmp.z;
        const side = j === 0 ? -1 : j === nv - 1 ? 1 : 0;
        this.rail[k] = side;
        if (side !== 0) {
          let As = 0;
          for (let a = 0; a < sub; a++) {
            const u0 = (i + a / sub) / nu;
            const u1 = (i + (a + 1) / sub) / nu;
            const um = 0.5 * (u0 + u1);
            const ds = Math.hypot((u1 - u0) * L, shape.halfWidth(u1) - shape.halfWidth(u0));
            As += ds * shape.thickness(um, side);
          }
          this.sideArea[k] = As;
          const um = (i + 0.5) / nu;
          shape.railNormal(um, side, tmp);
          this.sn[3 * k] = tmp.x;
          this.sn[3 * k + 1] = tmp.y;
          this.sn[3 * k + 2] = tmp.z;
          this.rx[k] = shape.x(um);
          this.rz[k] = shape.z(um, side);
          this.ryb[k] = shape.bottomY(um, side);
          this.ryd[k] = shape.deckY(um, side);
        }
      }
    }

    // probe grid
    const pu = (this.pu = opts.pu ?? 8);
    const pv = (this.pv = opts.pv ?? 3);
    const np = (this.np = pu * pv);
    this.probeU = new Float64Array(pu);
    this.probeV = new Float64Array(pv);
    for (let i = 0; i < pu; i++) this.probeU[i] = 0.03 + (0.94 * i) / (pu - 1);
    for (let j = 0; j < pv; j++) this.probeV[j] = pv === 1 ? 0 : -0.85 + (1.7 * j) / (pv - 1);
    this.probeLocal = new Float64Array(3 * np);
    this.probeWorld = new Float64Array(3 * np);
    this.probeSamples = [];
    for (let i = 0; i < pu; i++) {
      for (let j = 0; j < pv; j++) {
        const p = i * pv + j;
        shape.bottomPoint(this.probeU[i], this.probeV[j], tmp);
        this.probeLocal[3 * p] = tmp.x;
        this.probeLocal[3 * p + 1] = tmp.y;
        this.probeLocal[3 * p + 2] = tmp.z;
        this.probeSamples.push(createSurfaceSample());
      }
    }
    this.ph = new Float64Array(np);
    this.psx = new Float64Array(np);
    this.psz = new Float64Array(np);
    this.pvx = new Float64Array(np);
    this.pvy = new Float64Array(np);
    this.pvz = new Float64Array(np);
    this.pseabed = new Float64Array(np);

    this.cwIdx = new Int32Array(4 * n);
    this.cwW = new Float64Array(4 * n);
    this.rwIdx = new Int32Array(4 * n);
    this.rwW = new Float64Array(4 * n);
    for (let k = 0; k < n; k++) {
      this.weights(this.cu[k], this.cv[k], this.cwIdx, this.cwW, k);
      if (this.rail[k] !== 0) this.weights((Math.floor(k / nv) + 0.5) / nu, this.rail[k], this.rwIdx, this.rwW, k);
    }
    const nf = this.fins.length;
    this.fwIdx = new Int32Array(4 * nf);
    this.fwW = new Float64Array(4 * nf);
    for (let f = 0; f < nf; f++) {
      const fin = this.fins[f];
      const u = shape.uAtX(fin.base.x);
      const hw = Math.max(shape.halfWidth(u), 1e-3);
      this.weights(u, Math.max(-1, Math.min(1, fin.base.z / hw)), this.fwIdx, this.fwW, f);
    }
  }

  /** Forget the previous-step water velocities (after a teleport/reset). */
  resetWaterHistory(): void {
    this.prevWet.fill(0);
  }

  /** Forget all history (water under the cells, smoothed wetted length): a fresh start. */
  resetHistory(): void {
    this.prevWet.fill(0);
    this.wetLength = 1;
  }

  /** Bilinear probe weights (with linear extrapolation outside the grid) for (u, v). */
  private weights(u: number, v: number, idx: Int32Array, w: Float64Array, k: number): void {
    const pu = this.pu;
    const pv = this.pv;
    const fu = ((u - this.probeU[0]) / (this.probeU[pu - 1] - this.probeU[0])) * (pu - 1);
    const i0 = Math.min(Math.max(Math.floor(fu), 0), pu - 2);
    const t = fu - i0;
    let j0 = 0;
    let s = 0;
    if (pv > 1) {
      const fv = ((v - this.probeV[0]) / (this.probeV[pv - 1] - this.probeV[0])) * (pv - 1);
      j0 = Math.min(Math.max(Math.floor(fv), 0), pv - 2);
      s = fv - j0;
    }
    const j1 = pv > 1 ? j0 + 1 : j0;
    idx[4 * k] = i0 * pv + j0;
    idx[4 * k + 1] = (i0 + 1) * pv + j0;
    idx[4 * k + 2] = i0 * pv + j1;
    idx[4 * k + 3] = (i0 + 1) * pv + j1;
    w[4 * k] = (1 - t) * (1 - s);
    w[4 * k + 1] = t * (1 - s);
    w[4 * k + 2] = (1 - t) * s;
    w[4 * k + 3] = t * s;
  }

  /** Sample the ocean at every probe for the body's current pose. */
  sampleWater(ocean: OceanModel, body: RigidBody): void {
    const R = body.R;
    const p = body.position;
    for (let i = 0; i < this.np; i++) {
      const lx = this.probeLocal[3 * i], ly = this.probeLocal[3 * i + 1], lz = this.probeLocal[3 * i + 2];
      const X = p.x + R[0] * lx + R[1] * ly + R[2] * lz;
      const Y = p.y + R[3] * lx + R[4] * ly + R[5] * lz;
      const Z = p.z + R[6] * lx + R[7] * ly + R[8] * lz;
      this.probeWorld[3 * i] = X;
      this.probeWorld[3 * i + 1] = Y;
      this.probeWorld[3 * i + 2] = Z;
      const s = ocean.sample(X, Z, this.probeSamples[i]);
      this.ph[i] = s.height;
      this.psx[i] = s.slopeX;
      this.psz[i] = s.slopeZ;
      this.pvx[i] = s.velX;
      this.pvy[i] = s.velY;
      this.pvz[i] = s.velZ;
      this.pseabed[i] = -s.depth;
    }
  }

  /** Interpolated water at world point (X, Z) using weights k of the given table. */
  private waterAt(idx: Int32Array, wt: Float64Array, k: number, X: number, Z: number, out: WaterAt): WaterAt {
    let h = 0, sx = 0, sz = 0, vx = 0, vy = 0, vz = 0, px = 0, pz = 0, sb = 0;
    for (let c = 0; c < 4; c++) {
      const i = idx[4 * k + c];
      const w = wt[4 * k + c];
      h += w * this.ph[i];
      sx += w * this.psx[i];
      sz += w * this.psz[i];
      vx += w * this.pvx[i];
      vy += w * this.pvy[i];
      vz += w * this.pvz[i];
      px += w * this.probeWorld[3 * i];
      pz += w * this.probeWorld[3 * i + 2];
      sb += w * this.pseabed[i];
    }
    out.h = h + sx * (X - px) + sz * (Z - pz);
    out.sx = sx;
    out.sz = sz;
    out.vx = vx;
    out.vy = vy;
    out.vz = vz;
    out.seabed = sb;
    return out;
  }

  /** Water height (and slope/velocity) interpolated at local station (u, v) for world (X, Z). */
  waterHeightAtCell(k: number, X: number, Z: number): number {
    return this.waterAt(this.cwIdx, this.cwW, k, X, Z, this.water).h;
  }

  /**
   * Accumulate all water forces on the board into `sys` (generalised forces + linearised damping)
   * and write totals to `diag`. Requires `sampleWater` for the same pose.
   */
  computeForces(body: RigidBody, sys: ImplicitSystem, diag: HullDiagnostics): void {
    const H = HYDRO;
    const rho = RHO_WATER;
    const g = GRAVITY;
    const R = body.R;
    const px = body.position.x, py = body.position.y, pz = body.position.z;
    const vx0 = body.velocity.x, vy0 = body.velocity.y, vz0 = body.velocity.z;
    const wx = body.angularVelocity.x, wy = body.angularVelocity.y, wz = body.angularVelocity.z;
    const W = this.water;
    const rec = this.recordCellForces;

    diag.buoyancy.set(0, 0, 0);
    diag.pressure.set(0, 0, 0);
    diag.friction.set(0, 0, 0);
    diag.rail.set(0, 0, 0);
    diag.fin.set(0, 0, 0);
    diag.seabed.set(0, 0, 0);
    let subVol = 0;
    let addedMass = 0;
    let wetArea = 0;
    let deckWet = 0;
    let uMin = 2;
    let uMax = -1;

    // ITTC friction coefficient from the board's speed relative to the water under it
    let relSpeed = 0;
    {
      const c = this.np >> 1;
      const dvx = vx0 - this.pvx[c], dvy = vy0 - this.pvy[c], dvz = vz0 - this.pvz[c];
      relSpeed = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
    }
    const Re = Math.max((relSpeed * Math.max(this.wetLength, 0.1)) / NU_WATER, 2e4);
    const lg = Math.log10(Re) - 2;
    const cfHalfRho = 0.5 * rho * H.formFactor * (0.075 / (lg * lg));

    // --- pass 1: cell geometry (world, relative to the COM), water and immersion
    const n = this.n;
    for (let k = 0; k < n; k++) {
      const lx = this.cx[k], lyb = this.cyb[k], lz = this.cz[k];
      const lt = this.cyd[k] - lyb;
      const rbx = R[0] * lx + R[1] * lyb + R[2] * lz;
      const rby = R[3] * lx + R[4] * lyb + R[5] * lz;
      const rbz = R[6] * lx + R[7] * lyb + R[8] * lz;
      this.waterAt(this.cwIdx, this.cwW, k, px + rbx, pz + rbz, W);
      const db = W.h - (py + rby);
      this.sRb[3 * k] = rbx;
      this.sRb[3 * k + 1] = rby;
      this.sRb[3 * k + 2] = rbz;
      this.sDb[k] = db;
      this.sDd[k] = db + W.sx * R[1] * lt + W.sz * R[7] * lt - R[4] * lt;
      this.sW[6 * k] = W.sx;
      this.sW[6 * k + 1] = W.sz;
      this.sW[6 * k + 2] = W.vx;
      this.sW[6 * k + 3] = W.vy;
      this.sW[6 * k + 4] = W.vz;
      this.sW[6 * k + 5] = W.seabed;
    }

    // --- planing pressure distribution along each longitudinal strip: peaks at the leading edge
    // of the wetted length (spray root) and falls to zero at the trailing edge, 2(1 − ξ) with mean
    // 1, so the centre of pressure sits ≈ 1/3 of the wetted length behind the leading edge.
    // Blended out when the flow is not along the board (sideslip, slow drift).
    {
      const c = this.np >> 1;
      const fx = (vx0 - this.pvx[c]) * R[0] + (vy0 - this.pvy[c]) * R[3] + (vz0 - this.pvz[c]) * R[6];
      const along2 = relSpeed > 1e-6 ? Math.min((fx * fx) / (relSpeed * relSpeed), 1) : 0;
      const forward = fx >= 0;
      const nu = this.nu;
      const nv = this.nv;
      for (let j = 0; j < nv; j++) {
        let iMin = nu;
        let iMax = -1;
        for (let i = 0; i < nu; i++) {
          if (this.sDb[i * nv + j] > 0) {
            if (i < iMin) iMin = i;
            iMax = i;
          }
        }
        const len = iMax - iMin + 1;
        for (let i = 0; i < nu; i++) {
          const k = i * nv + j;
          if (i < iMin || i > iMax) {
            this.sPf[k] = 1;
            continue;
          }
          const xi = forward ? (iMax - i + 0.5) / len : (i - iMin + 0.5) / len;
          this.sPf[k] = 1 + along2 * (1 - 2 * xi);
        }
      }
    }

    // --- pass 2: forces
    for (let k = 0; k < n; k++) {
      const lt = this.cyd[k] - this.cyb[k];
      const rbx = this.sRb[3 * k], rby = this.sRb[3 * k + 1], rbz = this.sRb[3 * k + 2];
      const tx = R[1] * lt, ty = R[4] * lt, tz = R[7] * lt;
      const Yb = py + rby;
      const db = this.sDb[k];
      const dd = this.sDd[k];
      W.sx = this.sW[6 * k];
      W.sz = this.sW[6 * k + 1];
      W.vx = this.sW[6 * k + 2];
      W.vy = this.sW[6 * k + 3];
      W.vz = this.sW[6 * k + 4];
      W.seabed = this.sW[6 * k + 5];
      let fx = 0, fy = 0, fz = 0;
      let f = 0;
      // --- 1. hydrostatics
      if (db > 0 || dd > 0) {
        let c0: number; // centroid position along bottom→deck, 0..1
        if (db >= 0 && dd >= 0) {
          f = 1;
          c0 = 0.5;
        } else if (db > 0) {
          f = db / (db - dd);
          c0 = 0.5 * f;
        } else {
          f = dd / (dd - db);
          c0 = 1 - 0.5 * f;
        }
        const Vs = this.volume[k] * f;
        subVol += Vs;
        const kb = (rho * g * Vs) / (1 + W.sx * W.sx + W.sz * W.sz);
        const bx = -W.sx * kb, by = kb, bz = -W.sz * kb;
        sys.addBoardForce(bx, by, bz, rbx + c0 * tx, rby + c0 * ty, rbz + c0 * tz);
        diag.buoyancy.x += bx;
        diag.buoyancy.y += by;
        diag.buoyancy.z += bz;
        fx += bx;
        fy += by;
        fz += bz;
      }
      this.cellSubmerged[k] = f;

      // depth regimes (0 = at the free surface, 1 = deep, water on both faces):
      //  - motion along the normal (heave, rising/sinking): deep once the deck is under by half the
      //    beam — impact pressure → cross-flow drag, free-surface → unbounded-fluid added mass;
      //  - lift from the forward motion (planing law ∝ |v|·v_n): the free surface matters over a
      //    chord, i.e. the board length — deep once the deck is under by half the length, where a
      //    low-aspect-ratio plate lifts with a smaller slope (deepLift × the planing value).
      const dq = dd > 0 ? Math.min(dd / (0.5 * this.beam[k]), 1) : 0;
      const deep = dq * dq * (3 - 2 * dq);
      const dl = dd > 0 ? Math.min(dd / (0.5 * this.shape.length), 1) : 0;
      const liftScale = 1 - (1 - H.deepLift) * dl * dl * (3 - 2 * dl);

      // --- 2+3. bottom pressure and friction
      const wetB = Math.min(Math.max(db / H.wetDepth, 0), 1);
      if (wetB > 0) {
        const a = this.area[k] * wetB;
        wetArea += a;
        const u = this.cu[k];
        if (u < uMin) uMin = u;
        if (u > uMax) uMax = u;
        const nlx = this.nb[3 * k], nly = this.nb[3 * k + 1], nlz = this.nb[3 * k + 2];
        const nx = R[0] * nlx + R[1] * nly + R[2] * nlz;
        const ny = R[3] * nlx + R[4] * nly + R[5] * nlz;
        const nz = R[6] * nlx + R[7] * nly + R[8] * nlz;
        const vrx = vx0 + wy * rbz - wz * rby - W.vx;
        const vry = vy0 + wz * rbx - wx * rbz - W.vy;
        const vrz = vz0 + wx * rby - wy * rbx - W.vz;
        this.surfaceForce(sys, diag, a, nx, ny, nz, vrx, vry, vrz, rbx, rby, rbz, cfHalfRho, this.sPf[k], deep, liftScale);
        fx += this.sf[0];
        fy += this.sf[1];
        fz += this.sf[2];
        // added mass of the wetted bottom (normal direction) and the diffraction force from the
        // water's normal acceleration under the cell
        // (free-surface value ρπb²/8 per metre → unbounded-fluid value ρπb²/4 when deep)
        const ma = H.addedMass * rho * (Math.PI / 8) * this.beam[k] * a * (1 + deep);
        addedMass += ma;
        sys.addBoardMass(nx, ny, nz, rbx, rby, rbz, ma);
        if (this.prevWet[k]) {
          const dwn = ((W.vx - this.prevWater[3 * k]) * nx + (W.vy - this.prevWater[3 * k + 1]) * ny + (W.vz - this.prevWater[3 * k + 2]) * nz) / this.waterDt;
          const fd = ma * clampAbs(dwn, 4 * g);
          sys.addBoardForce(nx * fd, ny * fd, nz * fd, rbx, rby, rbz);
          fx += nx * fd;
          fy += ny * fd;
          fz += nz * fd;
        }
      }
      this.prevWet[k] = wetB > 0 ? 1 : 0;
      this.prevWater[3 * k] = W.vx;
      this.prevWater[3 * k + 1] = W.vy;
      this.prevWater[3 * k + 2] = W.vz;
      // --- deck pressure (nose diving, upside down, washed over)
      const wetD = Math.min(Math.max(dd / H.wetDepth, 0), 1);
      if (wetD > 0) {
        const a = this.area[k] * wetD;
        deckWet += a;
        const nlx = this.nd[3 * k], nly = this.nd[3 * k + 1], nlz = this.nd[3 * k + 2];
        const nx = R[0] * nlx + R[1] * nly + R[2] * nlz;
        const ny = R[3] * nlx + R[4] * nly + R[5] * nlz;
        const nz = R[6] * nlx + R[7] * nly + R[8] * nlz;
        const rdx = rbx + tx, rdy = rby + ty, rdz = rbz + tz;
        const vrx = vx0 + wy * rdz - wz * rdy - W.vx;
        const vry = vy0 + wz * rdx - wx * rdz - W.vy;
        const vrz = vz0 + wx * rdy - wy * rdx - W.vz;
        this.surfaceForce(sys, diag, a, nx, ny, nz, vrx, vry, vrz, rdx, rdy, rdz, cfHalfRho, 1, deep, liftScale);
        fx += this.sf[0];
        fy += this.sf[1];
        fz += this.sf[2];
      }

      // --- 4. rail side pressure
      const side = this.rail[k];
      if (side !== 0) {
        const lrx = this.rx[k], lrz = this.rz[k];
        const lrb = this.ryb[k], lrd = this.ryd[k];
        const rrx = R[0] * lrx + R[1] * lrb + R[2] * lrz;
        const rry = R[3] * lrx + R[4] * lrb + R[5] * lrz;
        const rrz = R[6] * lrx + R[7] * lrb + R[8] * lrz;
        const ht = lrd - lrb;
        const rtx = R[1] * ht, rty = R[4] * ht, rtz = R[7] * ht;
        this.waterAt(this.rwIdx, this.rwW, k, px + rrx, pz + rrz, W);
        const d0 = W.h - (py + rry);
        const d1 = W.h + W.sx * rtx + W.sz * rtz - (py + rry + rty);
        let fr = 0;
        let c0 = 0.5;
        if (d0 >= 0 && d1 >= 0) fr = 1;
        else if (d0 > 0) {
          fr = d0 / (d0 - d1);
          c0 = 0.5 * fr;
        } else if (d1 > 0) {
          fr = d1 / (d1 - d0);
          c0 = 1 - 0.5 * fr;
        }
        if (fr > 0) {
          const ax = rrx + c0 * rtx, ay = rry + c0 * rty, az = rrz + c0 * rtz;
          const snx = this.sn[3 * k], sny = this.sn[3 * k + 1], snz = this.sn[3 * k + 2];
          const nx = R[0] * snx + R[1] * sny + R[2] * snz;
          const ny = R[3] * snx + R[4] * sny + R[5] * snz;
          const nz = R[6] * snx + R[7] * sny + R[8] * snz;
          const vrx = vx0 + wy * az - wz * ay - W.vx;
          const vry = vy0 + wz * ax - wx * az - W.vy;
          const vrz = vz0 + wx * ay - wy * ax - W.vz;
          const vn = vrx * nx + vry * ny + vrz * nz;
          if (vn > 0) {
            const vm = Math.sqrt(vrx * vrx + vry * vry + vrz * vrz);
            const As = this.sideArea[k] * fr;
            const p = 0.5 * rho * As * (H.railCp1 * vm * vn + H.railCp2 * vn * vn);
            sys.addBoardForce(-nx * p, -ny * p, -nz * p, ax, ay, az);
            sys.addBoardDamping(nx, ny, nz, ax, ay, az, 0.5 * rho * As * (H.railCp1 * (vm + (vn * vn) / Math.max(vm, 1e-6)) + 2 * H.railCp2 * vn));
            diag.rail.x -= nx * p;
            diag.rail.y -= ny * p;
            diag.rail.z -= nz * p;
            fx -= nx * p;
            fy -= ny * p;
            fz -= nz * p;
          }
        }
      }

      // --- seabed contact (bottom or deck point below the sand)
      const pen = Math.max(W.seabed - Yb, W.seabed - (Yb + ty));
      if (pen > 0) {
        const a = this.area[k];
        const vry = vy0 + wz * rbx - wx * rbz;
        const fn = Math.max(a * (H.seabedStiffness * pen - H.seabedDamping * vry), 0);
        const vtx = vx0 + wy * rbz - wz * rby;
        const vtz = vz0 + wx * rby - wy * rbx;
        const vt = Math.sqrt(vtx * vtx + vtz * vtz) + 0.05;
        const ff = (H.seabedFriction * fn) / vt;
        sys.addBoardForce(-vtx * ff, fn, -vtz * ff, rbx, rby, rbz);
        sys.addBoardDamping(0, 1, 0, rbx, rby, rbz, a * H.seabedDamping);
        diag.seabed.x -= vtx * ff;
        diag.seabed.y += fn;
        diag.seabed.z -= vtz * ff;
        fx -= vtx * ff;
        fy += fn;
        fz -= vtz * ff;
      }

      if (rec) {
        this.cellForce[3 * k] = fx;
        this.cellForce[3 * k + 1] = fy;
        this.cellForce[3 * k + 2] = fz;
      }
    }

    // --- 5. fins
    for (let f = 0; f < this.fins.length; f++) {
      const fin = this.fins[f];
      const b = fin.base;
      const t = fin.tip;
      const rbx = R[0] * b.x + R[1] * b.y + R[2] * b.z;
      const rby = R[3] * b.x + R[4] * b.y + R[5] * b.z;
      const rbz = R[6] * b.x + R[7] * b.y + R[8] * b.z;
      const rtx = R[0] * t.x + R[1] * t.y + R[2] * t.z;
      const rty = R[3] * t.x + R[4] * t.y + R[5] * t.z;
      const rtz = R[6] * t.x + R[7] * t.y + R[8] * t.z;
      this.waterAt(this.fwIdx, this.fwW, f, px + rbx, pz + rbz, W);
      const d0 = W.h - (py + rby);
      const d1 = W.h + W.sx * (rtx - rbx) + W.sz * (rtz - rbz) - (py + rty);
      let fs = 0;
      let sx0 = rbx, sy0 = rby, sz0 = rbz; // submerged end
      let ex = rtx, ey = rty, ez = rtz; // other end
      if (d0 >= 0 && d1 >= 0) fs = 1;
      else if (d0 > 0) fs = d0 / (d0 - d1);
      else if (d1 > 0) {
        fs = d1 / (d1 - d0);
        sx0 = rtx; sy0 = rty; sz0 = rtz;
        ex = rbx; ey = rby; ez = rbz;
      }
      diag.finSubmerged[f] = fs;
      if (fs <= 0) {
        diag.finAlpha[f] = 0;
        diag.finSpeed[f] = 0;
        diag.finAttached[f] = 1;
        this.finForceWorld[3 * f] = 0;
        this.finForceWorld[3 * f + 1] = 0;
        this.finForceWorld[3 * f + 2] = 0;
        continue;
      }
      body.localDirToWorld(fin.span, this.tmpSpan);
      body.localDirToWorld(fin.forward, this.tmpFwd);
      body.localDirToWorld(fin.normal, this.tmpNrm);
      // hydrodynamic centre of the wetted part (≈ 42 % of the wetted span, quarter chord)
      const cf = 0.42 * fs;
      const chordShift = -0.12 * fin.spec.baseChord;
      const cx = sx0 + cf * (ex - sx0) + chordShift * this.tmpFwd.x;
      const cy = sy0 + cf * (ey - sy0) + chordShift * this.tmpFwd.y;
      const cz = sz0 + cf * (ez - sz0) + chordShift * this.tmpFwd.z;
      const vrx = vx0 + wy * cz - wz * cy - W.vx;
      const vry = vy0 + wz * cx - wx * cz - W.vy;
      const vrz = vz0 + wx * cy - wy * cx - W.vz;
      const res = finForce(vrx, vry, vrz, this.tmpSpan, this.tmpFwd, this.tmpNrm, fin.spec.area * fs, fin.aspectRatio * fs, this.finRes);
      sys.addBoardForce(res.fx, res.fy, res.fz, cx, cy, cz);
      sys.addBoardDamping(this.tmpNrm.x, this.tmpNrm.y, this.tmpNrm.z, cx, cy, cz, res.damping);
      diag.fin.x += res.fx;
      diag.fin.y += res.fy;
      diag.fin.z += res.fz;
      diag.finAlpha[f] = res.alpha;
      diag.finSpeed[f] = res.speed;
      diag.finAttached[f] = res.attached;
      this.finForceWorld[3 * f] = res.fx;
      this.finForceWorld[3 * f + 1] = res.fy;
      this.finForceWorld[3 * f + 2] = res.fz;
      this.finPointWorld[3 * f] = px + cx;
      this.finPointWorld[3 * f + 1] = py + cy;
      this.finPointWorld[3 * f + 2] = pz + cz;
    }

    // --- 6. angular damping ∝ wetted area
    const totalWet = wetArea + deckWet;
    if (totalWet > 0) {
      const c = H.angularDamping * totalWet;
      sys.addBoardTorque(-c * wx, -c * wy, -c * wz);
      sys.addAngularDamping(c);
    }

    diag.submergedVolume = subVol;
    this.addedMassTotal = addedMass;
    diag.wettedArea = wetArea;
    diag.deckWettedArea = deckWet;
    diag.wettedLength = uMax >= uMin ? (uMax - uMin + 1 / this.nu) * this.shape.length : 0;
    // smoothed wetted length for the next step's Reynolds number
    this.wetLength += 0.05 * (Math.max(diag.wettedLength, 0.1) - this.wetLength);
  }

  /** Force of the last `surfaceForce` call. */
  private readonly sf = new Float64Array(3);

  /** Pressure (planing/impact/suction/radiation) + friction on a wetted surface element. */
  private surfaceForce(
    sys: ImplicitSystem,
    diag: HullDiagnostics,
    a: number,
    nx: number,
    ny: number,
    nz: number,
    vrx: number,
    vry: number,
    vrz: number,
    rx: number,
    ry: number,
    rz: number,
    cfHalfRho: number,
    pf: number,
    deep: number,
    liftScale: number,
  ): void {
    const H = HYDRO;
    const rho = RHO_WATER;
    const vn = vrx * nx + vry * ny + vrz * nz;
    const vm = Math.sqrt(vrx * vrx + vry * vry + vrz * vrz);
    const tx = vrx - vn * nx, ty = vry - vn * ny, tz = vrz - vn * nz;
    const vt = Math.sqrt(tx * tx + ty * ty + tz * tz);
    const brad = (1 - deep) * H.radiation / (1 + (vt / H.radiationFadeSpeed) * (vt / H.radiationFadeSpeed));
    // pressure along −n (positive = pushing the surface out of the water)
    let p: number;
    let c: number;
    if (vn > 0) {
      // planing / lift (∝ |v|·v_n, distributed along the wetted length by pf, reduced deep) +
      // impact (∝ v_n², local: not redistributed) at the surface → cross-flow drag deep
      const cp1 = liftScale * H.cp1;
      const cp2 = (1 - deep) * H.cp2 + deep * H.crossflowCd;
      p = 0.5 * rho * a * (pf * cp1 * vm * vn + cp2 * vn * vn) + brad * a * vn;
      c = 0.5 * rho * a * (pf * cp1 * (vm + (vn * vn) / Math.max(vm, 1e-6)) + 2 * cp2 * vn) + brad * a;
    } else {
      p = -0.5 * rho * a * H.suction * vn * vn + brad * a * vn;
      c = rho * a * H.suction * -vn + brad * a;
    }
    const fpx = -nx * p, fpy = -ny * p, fpz = -nz * p;
    // friction
    const ff = cfHalfRho * a * vt;
    const ffx = -ff * tx, ffy = -ff * ty, ffz = -ff * tz;
    sys.addBoardForce(fpx + ffx, fpy + ffy, fpz + ffz, rx, ry, rz);
    sys.addBoardDamping(nx, ny, nz, rx, ry, rz, c);
    diag.pressure.x += fpx;
    diag.pressure.y += fpy;
    diag.pressure.z += fpz;
    diag.friction.x += ffx;
    diag.friction.y += ffy;
    diag.friction.z += ffz;
    this.sf[0] = fpx + ffx;
    this.sf[1] = fpy + ffy;
    this.sf[2] = fpz + ffz;
  }
}

function clampAbs(x: number, m: number): number {
  return x > m ? m : x < -m ? -m : x;
}
