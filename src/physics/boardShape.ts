/**
 * Surfboard geometry, shared by the physics hull and the render mesh.
 *
 * Parameterisation
 *   u ∈ [0, 1]  along the board, tail end (u = 0) → nose tip (u = 1)
 *   v ∈ [-1, 1] across the board, left rail (v = -1) → stringer (0) → right rail (v = +1)
 *
 * Shape frame (used internally): x = u·L from the tail end, y up with y = 0 at the lowest point
 * of the bottom rocker, z = v·halfWidth(u).
 *
 * Board local frame (what the physics and the mesh use): origin at the board's centre of mass,
 * +X toward the nose, +Y out of the deck, +Z toward the right rail. Use the local-frame functions
 * `x(u)`, `z(u, v)`, `bottomY(u, v)`, `deckY(u, v)` (or `bottomPoint` / `deckPoint`) to build
 * meshes; `comShape` is the offset between the two frames (local = shape − comShape).
 *
 * Cross-section: the deck is domed toward the rails, the bottom tucks up near the rail edge;
 * at v = ±1 deck and bottom are `railApex · stringerThickness(u)` apart (the rail apex; a mesh
 * closes the rail by joining the two edges, optionally bulging it outward). The tail end
 * (u = 0) is a flat face of width `tailBlock`; the nose (u = 1) closes to a point.
 *
 * The stringer thickness is scaled so the enclosed volume equals `volumeLiters` exactly
 * (`volume` is re-integrated numerically as a check).
 */
import { Vector3 } from 'three';
import { INCH } from './constants';

export interface FinSpec {
  name: string;
  /** Distance from the tail end to the middle of the fin base, m. */
  fromTail: number;
  /** -1 left side fin, 0 centre fin, +1 right side fin. */
  side: -1 | 0 | 1;
  /** Side fins: distance from the rail edge to the fin base, m (ignored for centre fins). */
  railInset: number;
  /** Depth (span) from the bottom to the tip, m. */
  depth: number;
  /** Base chord, m. */
  baseChord: number;
  /** Planform area of one face, m². */
  area: number;
  /** Toe-in, degrees: the leading edge points toward the stringer. */
  toeDeg: number;
  /** Cant, degrees: the tip leans outward toward the rail. */
  cantDeg: number;
  /** kg. */
  mass: number;
}

export interface OutlineSpec {
  /** u of the widest point. */
  widePoint: number;
  /** Width 12" (0.305 m) from the nose tip, m. */
  noseWidth: number;
  /** Width 12" from the tail end, m. */
  tailWidth: number;
  /** Width of the tail end (squash/square tail block), m. */
  tailBlock: number;
  /** Nose curve exponent: 0.5 = round (elliptic), → 1 = pointed. */
  noseShape: number;
  /** Tail curve exponent (~1). */
  tailShape: number;
}

export interface RockerSpec {
  /** Bottom lift at the nose tip above the lowest point, m. */
  nose: number;
  /** Bottom lift at the tail end, m. */
  tail: number;
  /** u of the lowest point of the bottom. */
  lowPoint: number;
  noseExp: number;
  tailExp: number;
}

export interface FoilSpec {
  /** u of the thickest point. */
  thickPoint: number;
  /** Stringer thickness at the nose tip / tail end as a fraction of the maximum. */
  nose: number;
  tail: number;
  noseExp: number;
  tailExp: number;
}

export interface RailSpec {
  /** How far the deck drops toward the rail, fraction of the stringer thickness. */
  deckDrop: number;
  /** Exponent of the deck crown (larger = flatter deck with a sharper roll into the rail). */
  deckExp: number;
  /** How far the bottom tucks up at the rail edge, fraction of the stringer thickness. */
  tuck: number;
  /** Exponent of the tuck (larger = tuck concentrated nearer the edge). */
  tuckExp: number;
}

export interface StanceSpec {
  /** Back foot position measured from the tail end, m (shortboards: over the fins). */
  backFoot: number;
  /** Distance from the back foot to the front foot, m. */
  feetSpread: number;
  /** Prone chest-down COM position relative to the board's centre of volume, m along +X. */
  proneOffset: number;
}

export interface BoardSpec {
  id: string;
  name: string;
  /** m. */
  length: number;
  /** Maximum width, m. */
  width: number;
  /** Enclosed volume, litres. */
  volumeLiters: number;
  /** Total mass including fins, kg. */
  mass: number;
  outline: OutlineSpec;
  rocker: RockerSpec;
  foil: FoilSpec;
  rails: RailSpec;
  /** Areal density of the skin (glass + resin, or soft-top skin), kg/m². */
  skinDensity: number;
  fins: FinSpec[];
  stance: StanceSpec;
}

/** Fin placed on a board: geometry in the board local frame. */
export interface FinGeometry {
  spec: FinSpec;
  /** Middle of the fin base on the bottom, local. */
  base: Vector3;
  /** Tip, local. */
  tip: Vector3;
  /** Unit span direction (base → tip). */
  span: Vector3;
  /** Unit chord direction pointing from the trailing edge to the leading edge (≈ +X, toed in). */
  forward: Vector3;
  /** Unit normal of the fin plane (forward × span). */
  normal: Vector3;
  /** Hydrodynamic centre (≈ 42 % span, quarter chord), local. */
  center: Vector3;
  /** Effective aspect ratio (the board bottom acts as an end plate: 2·depth²/area). */
  aspectRatio: number;
}

const FT = 12 * INCH;
const TWELVE = 12 * INCH;

const thruster = (
  sideFromTail: number,
  centreFromTail: number,
  depth: number,
  chord: number,
  area: number,
  centreScale: number,
  toe: number,
  cant: number,
  mass: number,
  inset = 0.032,
): FinSpec[] => [
  { name: 'left', fromTail: sideFromTail, side: -1, railInset: inset, depth, baseChord: chord, area, toeDeg: toe, cantDeg: cant, mass },
  { name: 'right', fromTail: sideFromTail, side: 1, railInset: inset, depth, baseChord: chord, area, toeDeg: toe, cantDeg: cant, mass },
  {
    name: 'centre', fromTail: centreFromTail, side: 0, railInset: 0, depth: depth * Math.sqrt(centreScale),
    baseChord: chord * Math.sqrt(centreScale), area: area * centreScale, toeDeg: 0, cantDeg: 0, mass,
  },
];

export const BOARD_PRESETS = {
  shortboard: {
    id: 'shortboard',
    name: `Shortboard 6'0"`,
    length: 6 * FT,
    width: 0.49,
    volumeLiters: 28,
    mass: 2.8,
    outline: { widePoint: 0.48, noseWidth: 0.295, tailWidth: 0.37, tailBlock: 0.22, noseShape: 0.72, tailShape: 1 },
    rocker: { nose: 0.12, tail: 0.05, lowPoint: 0.42, noseExp: 2.4, tailExp: 2.0 },
    foil: { thickPoint: 0.47, nose: 0.3, tail: 0.5, noseExp: 1.9, tailExp: 2.4 },
    rails: { deckDrop: 0.45, deckExp: 2.6, tuck: 0.25, tuckExp: 6 },
    skinDensity: 0.9,
    fins: thruster(0.285, 0.09, 0.115, 0.11, 0.0097, 0.95, 3.5, 7, 0.08),
    stance: { backFoot: 0.3, feetSpread: 0.55, proneOffset: -0.06 },
  },
  funboard: {
    id: 'funboard',
    name: `Funboard 7'6"`,
    length: 7.5 * FT,
    width: 0.56,
    volumeLiters: 50,
    mass: 4.5,
    outline: { widePoint: 0.5, noseWidth: 0.4, tailWidth: 0.4, tailBlock: 0.24, noseShape: 0.62, tailShape: 1 },
    rocker: { nose: 0.1, tail: 0.04, lowPoint: 0.45, noseExp: 2.4, tailExp: 2.0 },
    foil: { thickPoint: 0.48, nose: 0.32, tail: 0.5, noseExp: 1.9, tailExp: 2.4 },
    rails: { deckDrop: 0.4, deckExp: 2.6, tuck: 0.25, tuckExp: 6 },
    skinDensity: 1.05,
    fins: thruster(0.3, 0.1, 0.12, 0.115, 0.0105, 1, 3, 6, 0.09),
    stance: { backFoot: 0.5, feetSpread: 0.58, proneOffset: -0.06 },
  },
  longboard: {
    id: 'longboard',
    name: `Longboard 9'2"`,
    length: 9 * FT + 2 * INCH,
    width: 0.58,
    volumeLiters: 72,
    mass: 6.5,
    outline: { widePoint: 0.52, noseWidth: 0.45, tailWidth: 0.38, tailBlock: 0.22, noseShape: 0.5, tailShape: 1 },
    rocker: { nose: 0.085, tail: 0.03, lowPoint: 0.5, noseExp: 2.6, tailExp: 2.0 },
    foil: { thickPoint: 0.5, nose: 0.4, tail: 0.5, noseExp: 2.0, tailExp: 2.4 },
    rails: { deckDrop: 0.35, deckExp: 2.8, tuck: 0.22, tuckExp: 6 },
    skinDensity: 1.15,
    fins: [
      { name: 'single', fromTail: 0.22, side: 0, railInset: 0, depth: 0.23, baseChord: 0.17, area: 0.026, toeDeg: 0, cantDeg: 0, mass: 0.2 },
    ],
    stance: { backFoot: 0.95, feetSpread: 0.6, proneOffset: -0.05 },
  },
  softtop: {
    id: 'softtop',
    name: `Soft-top 8'0"`,
    length: 8 * FT,
    width: 0.58,
    volumeLiters: 86,
    mass: 5.5,
    outline: { widePoint: 0.5, noseWidth: 0.44, tailWidth: 0.42, tailBlock: 0.28, noseShape: 0.55, tailShape: 1 },
    rocker: { nose: 0.075, tail: 0.03, lowPoint: 0.48, noseExp: 2.4, tailExp: 2.0 },
    foil: { thickPoint: 0.48, nose: 0.45, tail: 0.6, noseExp: 2.0, tailExp: 2.4 },
    rails: { deckDrop: 0.25, deckExp: 3.0, tuck: 0.3, tuckExp: 5 },
    skinDensity: 1.5,
    fins: thruster(0.3, 0.1, 0.09, 0.09, 0.0065, 1, 2, 4, 0.05),
    stance: { backFoot: 0.72, feetSpread: 0.6, proneOffset: -0.05 },
  },
} satisfies Record<string, BoardSpec>;

export type BoardPresetId = keyof typeof BOARD_PRESETS;

/** Deep copy of a spec (so a GUI can tweak it without touching the presets). */
export function cloneBoardSpec(spec: BoardSpec): BoardSpec {
  return JSON.parse(JSON.stringify(spec)) as BoardSpec;
}

/** Numerical mass properties of a board (local principal axes assumed = board axes). */
export interface BoardMassProperties {
  /** Centre of mass in the shape frame. */
  comShape: Vector3;
  /** Principal moments about the COM (local x = roll, y = yaw, z = pitch), kg m². */
  inertia: Vector3;
  /** Neglected product of inertia I_xy (from rocker asymmetry), kg m². */
  productXY: number;
  foamMass: number;
  skinMass: number;
  finMass: number;
  /** Foam density implied by the spec, kg/m³. */
  foamDensity: number;
  /** Skin surface area (deck + bottom + rails + tail face), m². */
  skinArea: number;
}

export class BoardShape {
  readonly spec: BoardSpec;
  readonly length: number;
  readonly width: number;
  /** Stringer thickness at the thickest point (from the volume normalisation), m. */
  readonly maxThickness: number;
  /** Numerically re-integrated volume, m³ (equals volumeLiters / 1000 to < 0.1 %). */
  readonly volume: number;
  /** Planform (projected outline) area, m². */
  readonly planformArea: number;
  /** Centre of mass in the shape frame (subtract from shape coordinates to get local ones). */
  readonly comShape = new Vector3();
  /** Centre of volume (buoyancy centroid at full immersion), local frame. */
  readonly centerOfVolume = new Vector3();
  readonly mass: number;
  readonly massProperties: BoardMassProperties;
  readonly fins: FinGeometry[] = [];

  // outline exponents solved from the 12" widths
  private readonly noseA: number;
  private readonly tailA: number;
  /** Fraction of the stringer thickness left at the rail apex. */
  readonly railApex: number;

  constructor(spec: BoardSpec) {
    this.spec = spec;
    this.length = spec.length;
    this.width = spec.width;
    this.mass = spec.mass;
    const o = spec.outline;
    const L = spec.length;
    const sN = (1 - TWELVE / L - o.widePoint) / (1 - o.widePoint);
    const rN = o.noseWidth / spec.width;
    this.noseA = Math.log(1 - Math.pow(rN, 1 / o.noseShape)) / Math.log(sN);
    const tb = o.tailBlock / spec.width;
    const sT = (o.widePoint - TWELVE / L) / o.widePoint;
    const rT = (o.tailWidth / spec.width - tb) / (1 - tb);
    this.tailA = Math.log(1 - Math.pow(rT, 1 / o.tailShape)) / Math.log(sT);
    if (!(this.noseA > 1 && this.tailA > 1)) {
      throw new Error(`Board outline for ${spec.id} is not smooth at the wide point (exponents ${this.noseA}, ${this.tailA})`);
    }
    const r = spec.rails;
    this.railApex = 1 - r.deckDrop - r.tuck;

    // Volume normalisation: V = Tmax · L · ∫ hw(u) foil(u) du · ∫ cross(v) dv.
    const crossInt = 2 * (1 - r.deckDrop / (r.deckExp + 1) - r.tuck / (r.tuckExp + 1));
    const n = 2000;
    let su = 0;
    let area = 0;
    for (let i = 0; i <= n; i++) {
      const u = i / n;
      const wgt = i === 0 || i === n ? 1 : i % 2 === 1 ? 4 : 2;
      su += wgt * this.halfWidth(u) * this.foil(u);
      area += wgt * this.halfWidth(u);
    }
    su /= 3 * n;
    area /= 3 * n;
    this.planformArea = 2 * area * L;
    this.maxThickness = spec.volumeLiters / 1000 / (L * su * crossInt);

    this.massProperties = this.computeMassProperties();
    this.comShape.copy(this.massProperties.comShape);
    this.volume = this.integrateVolume(this.centerOfVolume);

    for (const f of spec.fins) this.fins.push(this.placeFin(f));
  }

  // ---------------------------------------------------------------------------------------------
  // Shape-frame primitives

  /** Half of the outline width at station u, m. */
  halfWidth(u: number): number {
    const o = this.spec.outline;
    const W = this.spec.width;
    if (u >= o.widePoint) {
      const s = Math.min((u - o.widePoint) / (1 - o.widePoint), 1);
      return 0.5 * W * Math.pow(Math.max(1 - Math.pow(s, this.noseA), 0), o.noseShape);
    }
    const s = Math.min((o.widePoint - u) / o.widePoint, 1);
    const tb = o.tailBlock / W;
    return 0.5 * W * (tb + (1 - tb) * Math.pow(Math.max(1 - Math.pow(s, this.tailA), 0), o.tailShape));
  }

  /** Bottom rocker at the stringer: lift above the lowest point, m (shape frame y). */
  rocker(u: number): number {
    const k = this.spec.rocker;
    if (u >= k.lowPoint) return k.nose * Math.pow((u - k.lowPoint) / (1 - k.lowPoint), k.noseExp);
    return k.tail * Math.pow((k.lowPoint - u) / k.lowPoint, k.tailExp);
  }

  /** Stringer thickness as a fraction of the maximum (the foil), 0..1. */
  foil(u: number): number {
    const f = this.spec.foil;
    if (u >= f.thickPoint) return 1 - (1 - f.nose) * Math.pow((u - f.thickPoint) / (1 - f.thickPoint), f.noseExp);
    return 1 - (1 - f.tail) * Math.pow((f.thickPoint - u) / f.thickPoint, f.tailExp);
  }

  /** Stringer thickness at station u, m. */
  stringerThickness(u: number): number {
    return this.maxThickness * this.foil(u);
  }

  /** Deck height above the stringer bottom as a fraction of the stringer thickness. */
  deckProfile(v: number): number {
    const r = this.spec.rails;
    return 1 - r.deckDrop * Math.pow(Math.abs(v), r.deckExp);
  }

  /** Bottom tuck (lift of the bottom toward the rail) as a fraction of the stringer thickness. */
  bottomProfile(v: number): number {
    const r = this.spec.rails;
    return r.tuck * Math.pow(Math.abs(v), r.tuckExp);
  }

  /** Board thickness (deck − bottom) at (u, v), m. */
  thickness(u: number, v: number): number {
    return this.stringerThickness(u) * (this.deckProfile(v) - this.bottomProfile(v));
  }

  /** Bottom height in the shape frame, m. */
  bottomYShape(u: number, v: number): number {
    return this.rocker(u) + this.stringerThickness(u) * this.bottomProfile(v);
  }

  /** Deck height in the shape frame, m. */
  deckYShape(u: number, v: number): number {
    return this.rocker(u) + this.stringerThickness(u) * this.deckProfile(v);
  }

  // ---------------------------------------------------------------------------------------------
  // Board local frame (origin at the centre of mass)

  /** Local x of station u. */
  x(u: number): number {
    return u * this.length - this.comShape.x;
  }

  /** Station u of local x (unclamped). */
  uAtX(x: number): number {
    return (x + this.comShape.x) / this.length;
  }

  /** Local z at (u, v). */
  z(u: number, v: number): number {
    return v * this.halfWidth(u) - this.comShape.z;
  }

  /** Local bottom height at (u, v). */
  bottomY(u: number, v: number): number {
    return this.bottomYShape(u, v) - this.comShape.y;
  }

  /** Local deck height at (u, v). */
  deckY(u: number, v: number): number {
    return this.deckYShape(u, v) - this.comShape.y;
  }

  bottomPoint(u: number, v: number, out: Vector3): Vector3 {
    return out.set(this.x(u), this.bottomY(u, v), this.z(u, v));
  }

  deckPoint(u: number, v: number, out: Vector3): Vector3 {
    return out.set(this.x(u), this.deckY(u, v), this.z(u, v));
  }

  /** Outward (downward-pointing) unit normal of the bottom at (u, v), local frame. */
  bottomNormal(u: number, v: number, out: Vector3): Vector3 {
    const [gx, gz] = this.surfaceGradient(u, v, false);
    return out.set(gx, -1, gz).normalize();
  }

  /** Outward (upward-pointing) unit normal of the deck at (u, v), local frame. */
  deckNormal(u: number, v: number, out: Vector3): Vector3 {
    const [gx, gz] = this.surfaceGradient(u, v, true);
    return out.set(-gx, 1, -gz).normalize();
  }

  /** Unit outward normal of the rail (in the board plane) on side ±1 at station u, local frame. */
  railNormal(u: number, side: number, out: Vector3): Vector3 {
    const du = 1e-4;
    const u0 = Math.max(u - du, 0);
    const u1 = Math.min(u + du, 1);
    const dhw = (this.halfWidth(u1) - this.halfWidth(u0)) / ((u1 - u0) * this.length);
    return out.set(-dhw, 0, side).normalize();
  }

  /** d(y)/dx and d(y)/dz of the bottom or deck surface at fixed z and x respectively. */
  private surfaceGradient(u: number, v: number, deck: boolean): [number, number] {
    const f = deck ? (a: number, b: number) => this.deckYShape(a, b) : (a: number, b: number) => this.bottomYShape(a, b);
    const hw = Math.max(this.halfWidth(u), 1e-3);
    const z = v * hw;
    const dx = 1e-3;
    const xs = u * this.length;
    const at = (xx: number, zz: number): number => {
      const uu = Math.min(Math.max(xx / this.length, 0), 1);
      const h = Math.max(this.halfWidth(uu), 1e-3);
      return f(uu, Math.min(Math.max(zz / h, -1), 1));
    };
    const x0 = Math.max(xs - dx, 0);
    const x1 = Math.min(xs + dx, this.length);
    const gx = (at(x1, z) - at(x0, z)) / (x1 - x0);
    const dz = Math.min(1e-3, 0.5 * hw);
    const z0 = Math.max(z - dz, -hw);
    const z1 = Math.min(z + dz, hw);
    const gz = z1 > z0 ? (at(xs, z1) - at(xs, z0)) / (z1 - z0) : 0;
    return [gx, gz];
  }

  // ---------------------------------------------------------------------------------------------
  // Integrals

  /** Midpoint-rule volume integral; writes the centre of volume (local frame) to `centroid`. */
  integrateVolume(centroid?: Vector3, nu = 400, nv = 80): number {
    let V = 0;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < nu; i++) {
      const u = (i + 0.5) / nu;
      const hw = this.halfWidth(u);
      const dA = (this.length / nu) * ((2 * hw) / nv);
      for (let j = 0; j < nv; j++) {
        const v = -1 + (2 * (j + 0.5)) / nv;
        const yb = this.bottomYShape(u, v);
        const yd = this.deckYShape(u, v);
        const dV = dA * (yd - yb);
        V += dV;
        mx += dV * u * this.length;
        my += dV * 0.5 * (yb + yd);
      }
    }
    if (centroid) centroid.set(mx / V - this.comShape.x, my / V - this.comShape.y, 0);
    return V;
  }

  private computeMassProperties(): BoardMassProperties {
    const spec = this.spec;
    const L = this.length;
    const nu = 240;
    const nv = 48;
    // raw moments about the shape-frame origin, per unit density
    const foam = { m: 0, x: 0, y: 0, z: 0, xx: 0, yy: 0, zz: 0, xy: 0, own: [0, 0, 0] };
    const skin = { m: 0, x: 0, y: 0, z: 0, xx: 0, yy: 0, zz: 0, xy: 0 };
    const addSkin = (dA: number, x: number, y: number, z: number): void => {
      skin.m += dA;
      skin.x += dA * x;
      skin.y += dA * y;
      skin.z += dA * z;
      skin.xx += dA * x * x;
      skin.yy += dA * y * y;
      skin.zz += dA * z * z;
      skin.xy += dA * x * y;
    };
    const dx = L / nu;
    for (let i = 0; i < nu; i++) {
      const u = (i + 0.5) / nu;
      const x = u * L;
      const hw = this.halfWidth(u);
      const dz = (2 * hw) / nv;
      const dA = dx * dz;
      for (let j = 0; j < nv; j++) {
        const v = -1 + (2 * (j + 0.5)) / nv;
        const z = v * hw;
        const yb = this.bottomYShape(u, v);
        const yd = this.deckYShape(u, v);
        const t = yd - yb;
        const dV = dA * t;
        const y = 0.5 * (yb + yd);
        foam.m += dV;
        foam.x += dV * x;
        foam.y += dV * y;
        foam.z += dV * z;
        foam.xx += dV * x * x;
        foam.yy += dV * y * y;
        foam.zz += dV * z * z;
        foam.xy += dV * x * y;
        foam.own[0] += (dV * (dz * dz + t * t)) / 12;
        foam.own[1] += (dV * (dx * dx + dz * dz)) / 12;
        foam.own[2] += (dV * (dx * dx + t * t)) / 12;
        // deck and bottom skin (area elements include the surface slope)
        const [gxd, gzd] = this.surfaceGradient(u, v, true);
        const [gxb, gzb] = this.surfaceGradient(u, v, false);
        addSkin(dA * Math.sqrt(1 + gxd * gxd + gzd * gzd), x, yd, z);
        addSkin(dA * Math.sqrt(1 + gxb * gxb + gzb * gzb), x, yb, z);
      }
      // rails
      const hw1 = this.halfWidth(Math.min(u + 0.5 / nu, 1));
      const hw0 = this.halfWidth(Math.max(u - 0.5 / nu, 0));
      const ds = Math.hypot(dx, hw1 - hw0);
      const tr = this.thickness(u, 1);
      const yr = 0.5 * (this.bottomYShape(u, 1) + this.deckYShape(u, 1));
      addSkin(ds * tr, x, yr, hw);
      addSkin(ds * tr, x, yr, -hw);
    }
    // tail face
    {
      const hw = this.halfWidth(0);
      for (let j = 0; j < nv; j++) {
        const v = -1 + (2 * (j + 0.5)) / nv;
        const t = this.thickness(0, v);
        addSkin(((2 * hw) / nv) * t, 0, this.rocker(0) + this.stringerThickness(0) * (0.5 * (this.deckProfile(v) + this.bottomProfile(v))), v * hw);
      }
    }

    // fins as point masses at their centroids (computed in the shape frame)
    let finMass = 0;
    const fin = { x: 0, y: 0, z: 0, xx: 0, yy: 0, zz: 0, xy: 0 };
    for (const f of spec.fins) {
      const u = f.fromTail / L;
      const hw = this.halfWidth(u);
      const z = f.side === 0 ? 0 : f.side * (hw - f.railInset);
      const y = this.bottomYShape(u, z / hw) - 0.4 * f.depth;
      const x = f.fromTail;
      finMass += f.mass;
      fin.x += f.mass * x;
      fin.y += f.mass * y;
      fin.z += f.mass * z;
      fin.xx += f.mass * x * x;
      fin.yy += f.mass * y * y;
      fin.zz += f.mass * z * z;
      fin.xy += f.mass * x * y;
    }

    const rest = spec.mass - finMass;
    const skinMass = Math.min(spec.skinDensity * skin.m, 0.7 * rest);
    const foamMass = rest - skinMass;
    const fd = foamMass / foam.m;
    const sd = skinMass / skin.m;
    const M = spec.mass;
    const cx = (fd * foam.x + sd * skin.x + fin.x) / M;
    const cy = (fd * foam.y + sd * skin.y + fin.y) / M;
    const cz = (fd * foam.z + sd * skin.z + fin.z) / M;
    const Sxx = fd * foam.xx + sd * skin.xx + fin.xx - M * cx * cx;
    const Syy = fd * foam.yy + sd * skin.yy + fin.yy - M * cy * cy;
    const Szz = fd * foam.zz + sd * skin.zz + fin.zz - M * cz * cz;
    const Sxy = fd * foam.xy + sd * skin.xy + fin.xy - M * cx * cy;
    const inertia = new Vector3(
      Syy + Szz + fd * foam.own[0],
      Sxx + Szz + fd * foam.own[1],
      Sxx + Syy + fd * foam.own[2],
    );
    return {
      comShape: new Vector3(cx, cy, cz),
      inertia,
      productXY: -Sxy,
      foamMass,
      skinMass,
      finMass,
      foamDensity: fd,
      skinArea: skin.m,
    };
  }

  private placeFin(f: FinSpec): FinGeometry {
    const L = this.length;
    const u = f.fromTail / L;
    const hw = this.halfWidth(u);
    const zShape = f.side === 0 ? 0 : f.side * (hw - f.railInset);
    const base = new Vector3(this.x(u), this.bottomY(u, zShape / hw), zShape - this.comShape.z);
    const toe = (f.toeDeg * Math.PI) / 180;
    const cant = (f.cantDeg * Math.PI) / 180;
    const span = new Vector3(0, -Math.cos(cant), f.side * Math.sin(cant)).normalize();
    const forward = new Vector3(Math.cos(toe), 0, -f.side * Math.sin(toe)).normalize();
    const normal = new Vector3().crossVectors(forward, span).normalize();
    const tip = base.clone().addScaledVector(span, f.depth);
    const center = base.clone().addScaledVector(span, 0.42 * f.depth).addScaledVector(forward, -0.12 * f.baseChord);
    return { spec: f, base, tip, span, forward, normal, center, aspectRatio: (2 * f.depth * f.depth) / f.area };
  }
}

const shapeCache = new Map<BoardSpec, BoardShape>();

/** Cached BoardShape for a spec object (keyed by identity; clone the spec to change it). */
export function getBoardShape(spec: BoardSpec): BoardShape {
  let s = shapeCache.get(spec);
  if (!s) {
    s = new BoardShape(spec);
    shapeCache.set(spec, s);
  }
  return s;
}

/**
 * Centre of mass of the board in the shape frame (x from the tail end, y above the rocker low
 * point, z from the stringer). A mesh built in the shape frame is aligned with the physics body
 * by translating it by −comOffset (or use the local-frame functions of BoardShape directly).
 */
export function boardComOffset(spec: BoardSpec, out = new Vector3()): Vector3 {
  return out.copy(getBoardShape(spec).comShape);
}
