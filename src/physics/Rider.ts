/**
 * Rider model (milestone 1): a point mass on stiff "legs" plus buoyant/draggy body segments.
 *
 * - The rider is a point mass (default 75 kg) held by a 3-axis spring-damper (≈3 Hz, ζ≈0.8)
 *   toward a target COM fixed in board space. SurfSim couples it implicitly to the board; the
 *   reaction acts on the board at the target point (= feet force + ankle torque).
 * - While attached (prone/popping/standing) the rider's body rotates with the board: its own
 *   moment of inertia about its COM is added to the board's, and forces on body segments act on
 *   the point mass with their moment about the rider COM transmitted to the board.
 * - Body segments are spheroids (two chest halves, hips, thighs, shins) carrying the whole body
 *   volume m/990 kg/m³ (≈0.076 m³ for 75 kg, wetsuit and lungs included, so a fallen rider just
 *   floats) that provide buoyancy and drag when they dip into the water.
 *
 * Stances
 *   prone:    COM 12 cm above the deck on the stringer near the board's centre of volume.
 *             Paddling: alternating arm strokes, thrust only while the hand is in the water,
 *             hand speed ≤ 3 m/s ⇒ thrust falls with speed (top speed ≈ 2 m/s). Steer = paddle
 *             harder on one side. leanForward slides the body ±15 cm, leanSide ±5 cm.
 *   popping:  0.6 s blend prone → standing (the legs push the board down).
 *   standing: back foot over the fins, front foot `feetSpread` ahead; COM 0.95 m above the feet
 *             (0.6 m crouched). leanForward shifts the COM ±0.22 m (trim); leanSide leans the
 *             stance toward a rail; twist = yaw torque ≤ 40 N·m. balanceAssist (0..1) blends in an
 *             automatic reflex (0.1 s lag): the stance follows the apparent gravity g − a and
 *             steers the board's bank toward leanSide·40° (level by default, i.e. the uphill rail
 *             set into a sloping face) with a lean capped to what the ankles can hold.
 *   fallen:   detached; floats with its segments, tethered by a 2.4 m leash to the tail.
 *
 * Wipeouts: ankle torque needed > what the feet can resist (normal force × 0.13 m in roll,
 * × 0.3 m in pitch) for > 0.35 s; board flipped; nose pearling at speed; board buried deep;
 * legs over-stretched by an impact.
 */
import { Vector3 } from 'three';
import type { BoardShape } from './boardShape';
import { GRAVITY, RHO_WATER } from './constants';
import type { RigidBody } from './RigidBody';

export type Stance = 'prone' | 'popping' | 'standing' | 'fallen';
export type WipeoutReason = 'balance' | 'flipped' | 'pearl' | 'buried' | 'impact';

export interface RiderConfig {
  /** kg. */
  mass: number;
  /** Leg spring natural frequency (with the rider mass), Hz. */
  legFrequency: number;
  legDampingRatio: number;
  /** 0 = raw balance (very hard), 1 = perfect automatic balance reflex. */
  balanceAssist: number;
  /** COM height above the feet standing tall / fully crouched, m. */
  standHeight: number;
  crouchHeight: number;
  /** Prone COM height above the deck, m. */
  proneHeight: number;
  /** Pop-up duration, s. */
  popUpTime: number;
  /** Leash length (slack), m. */
  leashLength: number;
  /** One arm's full stroke cycle at full effort, s. */
  strokePeriod: number;
  /** Peak hand speed relative to the board during the pull, m/s. */
  handSpeed: number;
  /** false = never wipe out (practice / tests). */
  wipeouts: boolean;
}

export const DEFAULT_RIDER_CONFIG: RiderConfig = {
  mass: 75,
  legFrequency: 3,
  legDampingRatio: 0.8,
  balanceAssist: 0.85,
  standHeight: 0.95,
  crouchHeight: 0.6,
  proneHeight: 0.12,
  popUpTime: 0.6,
  leashLength: 2.4,
  strokePeriod: 0.9,
  handSpeed: 3.0,
  wipeouts: true,
};

/** Rider tunables that are not per-rider. */
export const RIDER_MODEL = {
  /** Maximum stance lean beyond the apparent gravity with full assist / no assist, rad. */
  leanAssisted: (6.5 * Math.PI) / 180,
  leanRaw: (35 * Math.PI) / 180,
  /** Assisted steering: lateral acceleration asked for by a full lean (sets the coordinated
   * bank target atan(v·r/g) of the turn), m/s². */
  turnAccel: 6,
  /** Assisted bank control: max bank (rad), gain (rad lean per rad bank error) and damping
   * (rad lean per rad/s of roll rate). */
  bankMax: (50 * Math.PI) / 180,
  /** Max lean of the assisted rider beyond the apparent gravity, rad (ankle limit ≈ 7.8°). */
  balanceMargin: (6.5 * Math.PI) / 180,
  /** On a sloping face the assisted rider rides roughly flat on the water with the uphill rail
   * engaged by this angle (relative to the surface), rad. */
  railEngage: (6 * Math.PI) / 180,
  /** Max bank beyond the apparent gravity the assisted rider asks for, rad. */
  bankLead: (15 * Math.PI) / 180,
  bankGain: 0.2,
  /** Feed-forward lean into the turn at full leanSide, rad. */
  carveLean: (3.5 * Math.PI) / 180,
  bankDamping: 0.15,
  /** Fraction of the apparent-gravity roll the assisted stance follows. */
  rollFollow: 0.3,
  /** COM fore/aft trim range, m. */
  trimRange: 0.22,
  /** Max yaw torque from twisting, N·m. */
  twistTorque: 40,
  /** Stance controller lag, s, and max lean rate, rad/s. */
  stanceLag: 0.18,
  stanceRate: 3,
  /** Feet lever arms for the ankle-torque limit, m (roll: toes/heels, pitch: between the feet). */
  rollLever: 0.13,
  pitchLever: 0.3,
  /** Extra torque the feet can hold regardless of load, N·m. */
  gripTorque: 10,
  /** Time over the ankle limit before falling, s. */
  balanceTime: 0.35,
  /** Leg stiffness across the board relative to the vertical/fore-aft stiffness (knees and
   * hips let the board shift sideways under the rider instead of a rigid lever). */
  lateralStiffness: 1,
  /** Low-pass time constant of the ankle torque / feet load used by the balance check, s. */
  balanceFilter: 0.1,
  /** Mean body density used for the segment volumes (wetsuit, lungs full), kg/m³. */
  bodyDensity: 990,
  /** Body segment drag coefficient (on the projected spheroid area). */
  segmentCd: 0.8,
  /** Prone: drag coefficient for flow along the body, and the factor for segments lying over
   * the board (shielded by the hull). */
  proneAlongCd: 0.2,
  proneShield: 0.5,
  /** Paddling: effective hand + forearm drag area (C_d·A), m², and the drag fraction when the
   * hand moves slower than the water passing the board (paddlers slice the hand in). */
  handCdA: 0.1,
  handBrake: 0.05,
  /** Leash spring (beyond the slack length) N/m and damping N·s/m. */
  leashStiffness: 400,
  leashDamping: 80,
  /** Apparent-gravity filter time constants, s: roll balance (fast) and fore/aft (slow; a fast
   * fore/aft reflex pumps the planing pitch mode into porpoising). */
  accelFilter: 0.08,
  accelFilterPitch: 0.35,
  /** Fraction of the fore/aft apparent-gravity tilt the assisted stance follows. */
  pitchFollow: 0.7,
};

/** A buoyant spheroid body segment (radius ⟂ axis, halfLength along axis). */
export interface BodySegment {
  name: string;
  /** m³. */
  volume: number;
  /** World centre. */
  center: Vector3;
  /** World unit axis. */
  axis: Vector3;
  radius: number;
  halfLength: number;
  /** Submerged fraction of the last step. */
  submerged: number;
}

interface SegmentDef {
  name: string;
  frac: number;
  /** Prone: position relative to the prone COM (x along board, z across), thin axis = deck normal. */
  px: number;
  pz: number;
  pc: number;
  /** Standing radius ⟂ axis (75 kg body). */
  sa: number;
  /** Standing anchor: 'back' / 'front' foot, or 'com'. */
  anchor: 'back' | 'front' | 'com';
  /** Fraction of the way from the anchor foot to the COM, or offset above the COM along up. */
  t: number;
}

const SEGMENTS: SegmentDef[] = [
  { name: 'chestL', frac: 0.205, px: 0.4, pz: -0.11, pc: 0.11, sa: 0.12, anchor: 'com', t: 0.33 },
  { name: 'chestR', frac: 0.205, px: 0.4, pz: 0.11, pc: 0.11, sa: 0.12, anchor: 'com', t: 0.33 },
  { name: 'hips', frac: 0.25, px: -0.02, pz: 0, pc: 0.1, sa: 0.17, anchor: 'com', t: 0 },
  { name: 'thighL', frac: 0.105, px: -0.42, pz: -0.1, pc: 0.075, sa: 0.09, anchor: 'back', t: 0.66 },
  { name: 'thighR', frac: 0.105, px: -0.42, pz: 0.1, pc: 0.075, sa: 0.09, anchor: 'front', t: 0.66 },
  { name: 'shinL', frac: 0.065, px: -0.88, pz: -0.13, pc: 0.055, sa: 0.07, anchor: 'back', t: 0.26 },
  { name: 'shinR', frac: 0.065, px: -0.88, pz: 0.13, pc: 0.055, sa: 0.07, anchor: 'front', t: 0.26 },
];

/** Fraction of a sphere's volume below a plane at signed depth x·r above its centre (x ∈ [-1, 1]). */
export function sphereCapFraction(x: number): number {
  if (x <= -1) return 0;
  if (x >= 1) return 1;
  return 0.25 * (1 + x) * (1 + x) * (2 - x);
}

/** Water plane around a reference point (height, slope, velocity). */
export interface WaterPlane {
  x: number;
  z: number;
  height: number;
  slopeX: number;
  slopeZ: number;
  velX: number;
  velY: number;
  velZ: number;
}

export function createWaterPlane(): WaterPlane {
  return { x: 0, z: 0, height: 0, slopeX: 0, slopeZ: 0, velX: 0, velY: 0, velZ: 0 };
}

/** Pose data for rendering. All world space. */
export interface RiderPose {
  stance: Stance;
  /** 0 prone → 1 standing during the pop-up. */
  popProgress: number;
  com: Vector3;
  /** Body "up": stance up (from feet to COM) when standing, deck normal when prone, world up when fallen. */
  up: Vector3;
  /** Body forward: board nose direction (attached) or head direction (fallen). */
  forward: Vector3;
  backFoot: Vector3;
  frontFoot: Vector3;
  /** Paddling hands (meaningful when prone). */
  handLeft: Vector3;
  handRight: Vector3;
  crouch: number;
  /** Stance lean toward the right rail relative to the deck normal, rad. */
  lean: number;
  segments: BodySegment[];
}

export class Rider {
  config: RiderConfig;
  /** COM position / velocity, world. */
  readonly position = new Vector3();
  readonly velocity = new Vector3();
  stance: Stance = 'prone';
  /** Pop-up progress 0..1. */
  popProgress = 0;
  wipeoutReason: WipeoutReason | null = null;
  /** Time since the wipeout, s. */
  timeSinceWipeout = 0;

  /** Target COM in board local coordinates and its rate of change. */
  readonly targetLocal = new Vector3();
  readonly targetVelLocal = new Vector3();
  /** Stance up (feet → COM) in board local coordinates. */
  readonly upLocal = new Vector3(0, 1, 0);
  readonly backFootLocal = new Vector3();
  readonly frontFootLocal = new Vector3();
  readonly feetMidLocal = new Vector3();

  // stance controller state
  /** Roll / pitch of the apparent up (g − a of the board + rider COM) in board axes, rad. */
  rollApparent = 0;
  pitchApparent = 0;
  lean = 0;
  /** Board bank (world roll of its lateral axis, right rail down +), rad. */
  bankAngle = 0;
  /** Bank the assisted rider is steering toward, rad. */
  bankTarget = 0;
  /** Board turn rate (+ = right), rad/s. */
  turnRate = 0;
  /** Water surface slope under the board (set by SurfSim each step). */
  waterSlopeX = 0;
  waterSlopeZ = 0;
  /** Bank that would lie flat on the water surface, rad. */
  waterBank = 0;
  pitch = 0;
  trim = 0;
  height = 0.95;
  proneShift = 0;
  proneSide = 0;

  // balance
  /** Ankle torque / what the feet can hold, roll and pitch (signed; |x| > 1 = slipping). */
  balanceRoll = 0;
  balancePitch = 0;
  /** Seconds spent over the ankle limit (decays when back in balance). */
  balanceTimer = 0;
  /** Normal force of the feet on the deck (low-passed), N. */
  feetLoad = 0;
  /** Ankle torque about the board's long axis (roll) and lateral axis (pitch), low-passed, N·m. */
  ankleRoll = 0;
  anklePitch = 0;
  /** Leg force on the rider (effective, from the implicit step), world. */
  readonly legForce = new Vector3();
  /** Filtered board + rider COM acceleration (apparent gravity for roll balance), world. */
  readonly accelFiltered = new Vector3();
  /** Slowly filtered acceleration (fore/aft balance), world. */
  readonly accelSlow = new Vector3();
  private flipTimer = 0;
  private pearlTimer = 0;
  private buriedTimer = 0;

  // paddling
  /** Arm stroke phases 0..1 (pull during the first 45 %). */
  armPhaseL = 0.225;
  armPhaseR = 0.725;
  readonly handL = new Vector3();
  readonly handR = new Vector3();
  /** Last total paddle thrust (N, along the board). */
  paddleThrust = 0;
  /** Submerged body volume of the last step, m³. */
  submergedVolume = 0;

  readonly segments: BodySegment[];
  /** Board-local layout of the segments while attached (centre, axis). */
  private readonly segLocal: Vector3[];
  private readonly segAxisLocal: Vector3[];
  private readonly segRadius: Float64Array;
  private readonly segHalf: Float64Array;
  /** Direction of the body when fallen (world, horizontal). */
  private readonly fallenDir = new Vector3(1, 0, 0);

  readonly pose: RiderPose;

  private shape: BoardShape | null = null;
  private proneBase = new Vector3();
  private readonly tmp = new Vector3();
  private readonly tmp2 = new Vector3();
  private readonly tmp3 = new Vector3();
  private readonly prevTarget = new Vector3();
  private hasPrevTarget = false;

  constructor(config: Partial<RiderConfig> = {}) {
    this.config = { ...DEFAULT_RIDER_CONFIG, ...config };
    this.segments = SEGMENTS.map((d) => ({
      name: d.name,
      volume: 0,
      center: new Vector3(),
      axis: new Vector3(0, 1, 0),
      radius: 0.1,
      halfLength: 0.1,
      submerged: 0,
    }));
    this.segLocal = SEGMENTS.map(() => new Vector3());
    this.segAxisLocal = SEGMENTS.map(() => new Vector3(0, 1, 0));
    this.segRadius = new Float64Array(SEGMENTS.length);
    this.segHalf = new Float64Array(SEGMENTS.length);
    this.applyConfig();
    this.pose = {
      stance: 'prone',
      popProgress: 0,
      com: new Vector3(),
      up: new Vector3(0, 1, 0),
      forward: new Vector3(1, 0, 0),
      backFoot: new Vector3(),
      frontFoot: new Vector3(),
      handLeft: new Vector3(),
      handRight: new Vector3(),
      crouch: 0,
      lean: 0,
      segments: this.segments,
    };
  }

  /** Recompute mass-dependent quantities (segment volumes). Call after editing `config`. */
  applyConfig(): void {
    // the segments carry the whole body's volume (head and arms included) at the density of a
    // person in a wetsuit with air in the lungs, so a fallen rider just floats
    const total = this.config.mass / RIDER_MODEL.bodyDensity;
    for (let i = 0; i < SEGMENTS.length; i++) this.segments[i].volume = total * SEGMENTS[i].frac;
  }

  /** Body length scale relative to the 75 kg reference rider. */
  get scale(): number {
    return Math.cbrt(this.config.mass / 75);
  }

  /** Leg spring stiffness (N/m) and damping (N·s/m). */
  legStiffness(): number {
    const w = 2 * Math.PI * this.config.legFrequency;
    return this.config.mass * w * w;
  }

  legDamping(): number {
    return 2 * this.config.legDampingRatio * Math.sqrt(this.legStiffness() * this.config.mass);
  }

  get attached(): boolean {
    return this.stance !== 'fallen';
  }

  /** Set up the stance geometry for a board. */
  setBoard(shape: BoardShape): void {
    this.shape = shape;
    const st = shape.spec.stance;
    const L = shape.length;
    const ub = st.backFoot / L;
    const uf = (st.backFoot + st.feetSpread) / L;
    this.backFootLocal.set(shape.x(ub), shape.deckY(ub, 0), 0);
    this.frontFootLocal.set(shape.x(uf), shape.deckY(uf, 0), 0);
    this.feetMidLocal.addVectors(this.backFootLocal, this.frontFootLocal).multiplyScalar(0.5);
    this.proneBase.set(shape.centerOfVolume.x + st.proneOffset, 0, 0);
  }

  /** Reset the controller state for a stance (does not place the rider; SurfSim does). */
  resetState(stance: Stance): void {
    this.stance = stance;
    this.popProgress = stance === 'standing' ? 1 : 0;
    this.wipeoutReason = null;
    this.timeSinceWipeout = 0;
    this.lean = 0;
    this.pitch = 0;
    this.trim = 0;
    this.height = this.config.standHeight;
    this.proneShift = 0;
    this.proneSide = 0;
    this.balanceRoll = 0;
    this.balancePitch = 0;
    this.balanceTimer = 0;
    this.ankleRoll = 0;
    this.anklePitch = 0;
    this.feetLoad = 0;
    this.flipTimer = 0;
    this.pearlTimer = 0;
    this.buriedTimer = 0;
    // start mid-pull: the first stroke's yaw impulse is then half of a full one and the yaw rate
    // swings symmetrically about zero instead of leaving a ~10° heading offset
    this.armPhaseL = 0.225;
    this.armPhaseR = 0.725;
    this.paddleThrust = 0;
    this.accelFiltered.set(0, 0, 0);
    this.accelSlow.set(0, 0, 0);
    this.legForce.set(0, 0, 0);
    this.hasPrevTarget = false;
    this.targetVelLocal.set(0, 0, 0);
  }

  startPopUp(): void {
    if (this.stance === 'prone') {
      this.stance = 'popping';
      this.popProgress = 0;
    }
  }

  /** Detach the rider (wipeout). */
  fall(reason: WipeoutReason, board: RigidBody): void {
    if (this.stance === 'fallen') return;
    this.stance = 'fallen';
    this.wipeoutReason = reason;
    this.timeSinceWipeout = 0;
    // body lies toward the board's nose direction
    this.fallenDir.set(board.R[0], 0, board.R[6]);
    if (this.fallenDir.lengthSq() < 1e-6) this.fallenDir.set(1, 0, 0);
    this.fallenDir.normalize();
  }

  /** Rider body moment of inertia about its own COM, in board axes, for the current stance. */
  bodyInertia(out: Vector3): Vector3 {
    if (this.stance === 'fallen') return out.set(0, 0, 0);
    const m = this.config.mass;
    const s2 = this.scale * this.scale;
    // prone: body along x (L ≈ 1.75 m) → small roll, large yaw/pitch
    const px = 0.045 * m * s2, py = 0.24 * m * s2, pz = 0.24 * m * s2;
    // standing: body along y → yaw small, roll/pitch ≈ m (0.35 m)²
    const sx = 0.11 * m * s2, sy = 0.025 * m * s2, sz = 0.11 * m * s2;
    const w = this.standWeight();
    return out.set(px + (sx - px) * w, py + (sy - py) * w, pz + (sz - pz) * w);
  }

  /** Blend weight prone (0) → standing (1). */
  standWeight(): number {
    if (this.stance === 'standing') return 1;
    if (this.stance === 'popping') {
      const t = this.popProgress;
      return t * t * (3 - 2 * t);
    }
    return 0;
  }

  /**
   * Advance the stance controller and pop-up, and compute the target COM (board local) and the
   * board-local segment layout. `board` must have its derived matrices up to date.
   */
  updateTargets(dt: number, input: StanceInput, board: RigidBody): void {
    const cfg = this.config;
    const M = RIDER_MODEL;
    if (!this.shape) throw new Error('Rider.setBoard() not called');
    const shape = this.shape;
    const s = this.scale;
    if (this.stance === 'fallen') {
      this.timeSinceWipeout += dt;
      return;
    }
    if (this.stance === 'popping') {
      this.popProgress = Math.min(this.popProgress + dt / cfg.popUpTime, 1);
      if (this.popProgress >= 1) this.stance = 'standing';
    }

    // apparent up in board axes (filtered acceleration of the board + rider COM)
    const af = this.accelFiltered;
    this.tmp.set(af.x, af.y + GRAVITY, af.z);
    if (this.tmp.lengthSq() < 1) this.tmp.set(0, 1, 0);
    this.tmp.normalize();
    board.worldDirToLocal(this.tmp, this.tmp2);
    this.rollApparent = clamp(Math.atan2(this.tmp2.z, this.tmp2.y), -1.2, 1.2);
    const as = this.accelSlow;
    this.tmp.set(as.x, as.y + GRAVITY, as.z);
    if (this.tmp.lengthSq() < 1) this.tmp.set(0, 1, 0);
    this.tmp.normalize();
    board.worldDirToLocal(this.tmp, this.tmp2);
    this.pitchApparent = clamp(Math.atan2(this.tmp2.x, this.tmp2.y), -0.6, 0.6);

    // --- prone target
    const lagK = 1 - Math.exp(-dt / 0.25);
    this.proneShift += (clamp(input.leanForward, -1, 1) * 0.15 - this.proneShift) * lagK;
    // prone balance reflex: shift the body toward the high rail (keeps a sunken board upright)
    const sideTarget = clamp(input.leanSide, -1, 1) * 0.05 + clamp(cfg.balanceAssist, 0, 1) * clamp(0.5 * this.rollApparent, -0.12, 0.12);
    this.proneSide += (sideTarget - this.proneSide) * lagK;
    const pxl = this.proneBase.x + this.proneShift;
    const pu = clamp(shape.uAtX(pxl), 0, 1);
    const pTx = pxl;
    const pTy = shape.deckY(pu, 0) + cfg.proneHeight;
    const pTz = this.proneSide;

    // --- standing target
    const a = clamp(cfg.balanceAssist, 0, 1);
    const rollApp = this.rollApparent;
    const pitchApp = this.pitchApparent;
    // Roll. The assisted rider (a skilled rider's reflexes) steers by banking: leanSide asks for a
    // turn (lateral acceleration turnAccel at full lean); the rider banks the board toward the
    // coordinated angle atan(v·r/g), but never more than bankLead beyond the apparent gravity, and
    // drives the bank there by leaning (weight over the inside rail; a feed-forward lean plus gentle
    // bank feedback) while never leaning more than balanceMargin beyond the apparent
    // gravity (what the ankles can hold). The bank is measured from the water surface under the
    // board: with leanSide = 0 the rider rides flat on the water (level on flat water) with the
    // uphill rail engaged by railEngage on a sloping face.
    // The raw rider (assist 0) just leans relative to the deck.
    const side = clamp(input.leanSide, -1, 1);
    const R = board.R;
    const av = board.angularVelocity;
    const bank = Math.asin(clamp(-R[5], -1, 1));
    const bankRate = av.x * R[0] + av.y * R[3] + av.z * R[6];
    const vh = Math.max(Math.hypot(board.velocity.x, board.velocity.z), 1);
    const turnRate = -av.y; // + = turning right (toward +Z when heading +X)
    const rTarget = (side * M.turnAccel) / Math.max(vh, 2.5);
    this.turnRate = turnRate;
    this.bankAngle = bank;
    const appBank = bank + rollApp;
    // water surface across the board: bank that would lie flat on it, and the uphill side
    const zh = Math.hypot(R[2], R[8]) || 1;
    const sLat = (this.waterSlopeX * R[2] + this.waterSlopeZ * R[8]) / zh;
    const waterBank = -Math.atan(sLat);
    const engage = M.railEngage * clamp(sLat / 0.1, -1, 1);
    this.waterBank = waterBank;
    let bankTarget = waterBank + engage + Math.atan((vh * rTarget) / GRAVITY);
    bankTarget = clamp(bankTarget, -M.bankMax, M.bankMax);
    bankTarget = clamp(bankTarget, appBank - M.bankLead, appBank + M.bankLead);
    this.bankTarget = bankTarget;
    const bankErr = bankTarget - bank;
    // gentle bank feedback plus a feed-forward lean into the commanded turn (initiating a roll
    // needs a real lean; the balance clamp below keeps it within what the ankles can hold)
    const delta = clamp(M.bankGain * bankErr - M.bankDamping * bankRate + side * M.carveLean, -M.leanAssisted, M.leanAssisted);
    // a skilled rider never leans much beyond the apparent gravity (the ankles can't hold it)
    const assisted = clamp(M.rollFollow * rollApp + delta, rollApp - M.balanceMargin, rollApp + M.balanceMargin);
    const raw = (1 - a) * (1 - a);
    const leanTarget = a * assisted + raw * side * M.leanRaw;
    const pitchTarget = a * RIDER_MODEL.pitchFollow * pitchApp;
    const trimTarget = clamp(input.leanForward, -1, 1) * M.trimRange;
    const hTarget = cfg.standHeight + (cfg.crouchHeight - cfg.standHeight) * clamp(input.crouch, 0, 1);
    const k = 1 - Math.exp(-dt / M.stanceLag);
    const maxStep = M.stanceRate * dt;
    this.lean += clamp((leanTarget - this.lean) * k, -maxStep, maxStep);
    this.pitch += clamp((pitchTarget - this.pitch) * k, -maxStep, maxStep);
    this.trim += clamp((trimTarget - this.trim) * k, -1.5 * dt, 1.5 * dt);
    this.height += clamp((hTarget - this.height) * k, -2 * dt, 2 * dt);

    const cp = Math.cos(this.pitch);
    const ux = Math.sin(this.pitch);
    const uy = cp * Math.cos(this.lean);
    const uz = cp * Math.sin(this.lean);
    const h = this.height * s;
    const fm = this.feetMidLocal;
    const sTx = fm.x + this.trim + h * ux;
    const sTy = fm.y + h * uy;
    const sTz = fm.z + h * uz;

    // --- blend
    const w = this.standWeight();
    this.targetLocal.set(pTx + (sTx - pTx) * w, pTy + (sTy - pTy) * w, pTz + (sTz - pTz) * w);
    this.upLocal.set(ux * w, 1 + (uy - 1) * w, uz * w).normalize();
    if (this.hasPrevTarget) this.targetVelLocal.subVectors(this.targetLocal, this.prevTarget).divideScalar(dt);
    else this.targetVelLocal.set(0, 0, 0);
    this.prevTarget.copy(this.targetLocal);
    this.hasPrevTarget = true;

    // --- segment layout (board local)
    for (let i = 0; i < SEGMENTS.length; i++) {
      const d = SEGMENTS[i];
      const vol = this.segments[i].volume;
      // prone spheroid: thin axis = deck normal
      const pc = d.pc * s;
      const pa = Math.sqrt(vol / ((4 / 3) * Math.PI * pc));
      const plx = pTx + d.px * s;
      const ply = pTy - cfg.proneHeight + pc;
      const plz = pTz + d.pz * s;
      // standing spheroid: long axis along the limb / stance up
      const sa = d.sa * s;
      const sc = vol / ((4 / 3) * Math.PI * sa * sa);
      let slx: number, sly: number, slz: number;
      let ax: number, ay: number, az: number;
      if (d.anchor === 'com') {
        slx = sTx + d.t * s * ux;
        sly = sTy + d.t * s * uy;
        slz = sTz + d.t * s * uz + d.pz * 0.9 * s;
        ax = ux; ay = uy; az = uz;
      } else {
        const foot = d.anchor === 'back' ? this.backFootLocal : this.frontFootLocal;
        const dx = sTx - foot.x, dy = sTy - foot.y, dz = sTz - foot.z;
        slx = foot.x + d.t * dx;
        sly = foot.y + d.t * dy;
        slz = foot.z + d.t * dz + (d.name.endsWith('L') ? -0.06 : 0.06) * s;
        const dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
        ax = dx / dl; ay = dy / dl; az = dz / dl;
      }
      this.segLocal[i].set(plx + (slx - plx) * w, ply + (sly - ply) * w, plz + (slz - plz) * w);
      this.segAxisLocal[i].set(ax * w, 1 + (ay - 1) * w, az * w).normalize();
      this.segRadius[i] = pa + (sa - pa) * w;
      this.segHalf[i] = pc + (sc - pc) * w;
    }
  }

  /** Update the filtered board acceleration (call once per step with the board's velocity change). */
  filterAcceleration(ax: number, ay: number, az: number, dt: number): void {
    const k = 1 - Math.exp(-dt / RIDER_MODEL.accelFilter);
    const f = this.accelFiltered;
    // clamp spikes (slams) to ±4 g before filtering
    const lim = 4 * GRAVITY;
    ax = clamp(ax, -lim, lim);
    ay = clamp(ay, -lim, lim);
    az = clamp(az, -lim, lim);
    f.x += (ax - f.x) * k;
    f.y += (ay - f.y) * k;
    f.z += (az - f.z) * k;
    const ks = 1 - Math.exp(-dt / RIDER_MODEL.accelFilterPitch);
    const g = this.accelSlow;
    g.x += (ax - g.x) * ks;
    g.y += (ay - g.y) * ks;
    g.z += (az - g.z) * ks;
  }

  /** Place the segments in the world (attached: rigid with the board; fallen: lying flat). */
  layoutSegments(board: RigidBody): void {
    if (this.stance !== 'fallen') {
      // the layout is relative to the target COM; carry it with the actual rider COM
      for (let i = 0; i < this.segments.length; i++) {
        const sg = this.segments[i];
        this.tmp.subVectors(this.segLocal[i], this.targetLocal);
        board.localDirToWorld(this.tmp, sg.center).add(this.position);
        board.localDirToWorld(this.segAxisLocal[i], sg.axis);
        sg.radius = this.segRadius[i];
        sg.halfLength = this.segHalf[i];
      }
      return;
    }
    // fallen: prone layout, body horizontal along fallenDir, thin axis vertical
    const s = this.scale;
    const d = this.fallenDir;
    for (let i = 0; i < SEGMENTS.length; i++) {
      const def = SEGMENTS[i];
      const sg = this.segments[i];
      const pc = def.pc * s;
      const x = def.px * s;
      const z = def.pz * s;
      // right = d × up = (−d.z, 0, d.x)
      sg.center.set(this.position.x + d.x * x - d.z * z, this.position.y + pc - this.config.proneHeight, this.position.z + d.z * x + d.x * z);
      sg.axis.set(0, 1, 0);
      sg.halfLength = pc;
      sg.radius = Math.sqrt(sg.volume / ((4 / 3) * Math.PI * pc));
    }
  }

  /**
   * Buoyancy and drag on the body segments. Adds the force on the rider to `force` and, when
   * attached, the moment about the rider COM to `torque` (to be applied to the board).
   * Returns the linearised drag damping coefficient (N·s/m, isotropic on the rider).
   */
  segmentForces(water: WaterPlane, board: RigidBody, force: Vector3, torque: Vector3): number {
    const M = RIDER_MODEL;
    const rho = RHO_WATER;
    const attached = this.stance !== 'fallen';
    const prone = this.stance === 'prone';
    const w = board.angularVelocity;
    let damping = 0;
    let subVol = 0;
    const den = 1 / (1 + water.slopeX * water.slopeX + water.slopeZ * water.slopeZ);
    for (let i = 0; i < this.segments.length; i++) {
      const sg = this.segments[i];
      const c = sg.center;
      const hw = water.height + water.slopeX * (c.x - water.x) + water.slopeZ * (c.z - water.z);
      const ay = sg.axis.y;
      const e = Math.sqrt(sg.radius * sg.radius * (1 - ay * ay) + sg.halfLength * sg.halfLength * ay * ay);
      const f = sphereCapFraction((hw - c.y) / e);
      sg.submerged = f;
      if (f <= 0) continue;
      const V = sg.volume * f;
      subVol += V;
      const kb = rho * GRAVITY * V * den;
      let fx = -water.slopeX * kb;
      let fy = kb;
      let fz = -water.slopeZ * kb;
      // drag
      const rx = c.x - this.position.x, ry = c.y - this.position.y, rz = c.z - this.position.z;
      let vx = this.velocity.x - water.velX;
      let vy = this.velocity.y - water.velY;
      let vz = this.velocity.z - water.velZ;
      if (attached) {
        vx += w.y * rz - w.z * ry;
        vy += w.z * rx - w.x * rz;
        vz += w.x * ry - w.y * rx;
      }
      const vm = Math.sqrt(vx * vx + vy * vy + vz * vz);
      // projected area of the spheroid seen by the flow: π a sqrt(a² cos²θ + c² sin²θ)
      const r = sg.radius;
      const ca = vm > 1e-6 ? (vx * sg.axis.x + vy * sg.axis.y + vz * sg.axis.z) / vm : 0;
      const area = Math.PI * r * Math.sqrt(r * r * ca * ca + sg.halfLength * sg.halfLength * (1 - ca * ca));
      // A prone body is elongated along the board (segments in each other's wake): low drag
      // coefficient for flow along the board, bluff across it; the hull shields what lies on it.
      let cdCoef = M.segmentCd;
      let shield = 1;
      if (prone) {
        const bx = board.R[0], by = board.R[3], bz = board.R[6];
        const along = vm > 1e-6 ? (vx * bx + vy * by + vz * bz) / vm : 0;
        cdCoef = M.segmentCd + (M.proneAlongCd - M.segmentCd) * along * along;
        if (this.overBoard(i)) shield = M.proneShield;
      }
      const cd = 0.5 * rho * cdCoef * area * f * shield;
      fx -= cd * vm * vx;
      fy -= cd * vm * vy;
      fz -= cd * vm * vz;
      damping += 2 * cd * vm;
      force.x += fx;
      force.y += fy;
      force.z += fz;
      if (attached) {
        torque.x += ry * fz - rz * fy;
        torque.y += rz * fx - rx * fz;
        torque.z += rx * fy - ry * fx;
      }
    }
    this.submergedVolume = subVol;
    return damping;
  }

  /** Is segment i (board-local layout) above the board's planform? */
  private overBoard(i: number): boolean {
    const shape = this.shape;
    if (!shape) return false;
    const p = this.segLocal[i];
    const u = shape.uAtX(p.x);
    return u > 0 && u < 1 && Math.abs(p.z) < shape.halfWidth(u);
  }

  /**
   * Paddling thrust (prone only). Advances the arm phases and adds the hand forces to the rider
   * (`force`) and their moment about the rider COM to `torque` (board).
   */
  paddleForces(dt: number, input: StanceInput, water: WaterPlane, board: RigidBody, force: Vector3, torque: Vector3): void {
    this.paddleThrust = 0;
    if (this.stance !== 'prone') {
      this.handL.copy(this.position);
      this.handR.copy(this.position);
      return;
    }
    const cfg = this.config;
    const M = RIDER_MODEL;
    const s = this.scale;
    const paddle = clamp(input.paddle, 0, 1);
    const steer = clamp(input.steer, -1, 1);
    // steering: paddle harder on the outside arm (turn right → left arm works harder)
    const pL = clamp(paddle * (1 + 0.5 * steer) + Math.max(steer, 0) * 0.7 * (1 - paddle), 0, 1);
    const pR = clamp(paddle * (1 - 0.5 * steer) + Math.max(-steer, 0) * 0.7 * (1 - paddle), 0, 1);
    const R = board.R;
    const nx = R[0], ny = R[3], nz = R[6]; // board +X in world
    const shoulderX = this.targetLocal.x + 0.45 * s;
    const shoulderY = this.targetLocal.y + 0.03;
    // one stroke clock; the arms are half a cycle apart and pull only when that side has effort
    const effort = Math.max(pL, pR);
    if (effort > 0.02) {
      this.armPhaseL += (dt / cfg.strokePeriod) * (0.7 + 0.3 * effort);
      if (this.armPhaseL >= 1) this.armPhaseL -= 1;
    }
    this.armPhaseR = this.armPhaseL + 0.5 >= 1 ? this.armPhaseL - 0.5 : this.armPhaseL + 0.5;
    for (let arm = 0; arm < 2; arm++) {
      const p = arm === 0 ? pL : pR;
      const side = arm === 0 ? -1 : 1;
      const phase = arm === 0 ? this.armPhaseL : this.armPhaseR;
      const hand = arm === 0 ? this.handL : this.handR;
      const pull = phase < 0.45 && p > 0.02;
      const sPull = pull ? phase / 0.45 : 0;
      const depth = pull ? 0.55 * s * Math.sin(Math.PI * sPull) : -0.15;
      const hx = shoulderX + (pull ? 0.35 - 0.7 * sPull : -0.35 + 0.7 * ((phase - 0.45) / 0.55)) * s;
      this.tmp.set(hx - this.targetLocal.x, shoulderY - depth - this.targetLocal.y, side * 0.33 * s);
      board.localDirToWorld(this.tmp, this.tmp2); // hand relative to the rider COM
      hand.copy(this.tmp2).add(this.position);
      if (!pull) continue;
      const hw = water.height + water.slopeX * (hand.x - water.x) + water.slopeZ * (hand.z - water.z);
      const imm = clamp((hw - hand.y) / 0.2, 0, 1);
      if (imm <= 0) continue;
      // board point velocity at the hand relative to the water, along the board axis
      const w = board.angularVelocity;
      const rx = hand.x - board.position.x, ry = hand.y - board.position.y, rz = hand.z - board.position.z;
      const vpx = board.velocity.x + w.y * rz - w.z * ry - water.velX;
      const vpy = board.velocity.y + w.z * rx - w.x * rz - water.velY;
      const vpz = board.velocity.z + w.x * ry - w.y * rx - water.velZ;
      const vf = vpx * nx + vpy * ny + vpz * nz;
      // the hand reaches stroke speed quickly and holds it through the pull
      const uh = cfg.handSpeed * (0.55 + 0.45 * p) * Math.sqrt(Math.sin(Math.PI * sPull));
      const du = uh - vf;
      const mag = 0.5 * RHO_WATER * M.handCdA * imm * (du > 0 ? du * du : -M.handBrake * du * du);
      const fx = nx * mag, fy = ny * mag, fz = nz * mag;
      force.x += fx;
      force.y += fy;
      force.z += fz;
      const ox = this.tmp2.x, oy = this.tmp2.y, oz = this.tmp2.z;
      torque.x += oy * fz - oz * fy;
      torque.y += oz * fx - ox * fz;
      torque.z += ox * fy - oy * fx;
      this.paddleThrust += mag;
    }
  }

  /**
   * Balance bookkeeping from the effective leg force (on the rider, world): ankle torque about the
   * feet relative to what the feet can resist. Returns true when the rider loses their feet.
   */
  updateBalance(dt: number, board: RigidBody): boolean {
    const M = RIDER_MODEL;
    if (this.stance !== 'standing' && this.stance !== 'popping') {
      this.balanceRoll = 0;
      this.balancePitch = 0;
      this.balanceTimer = Math.max(this.balanceTimer - dt, 0);
      return false;
    }
    // force of the legs on the board (local) and its moment about the feet midpoint
    board.worldDirToLocal(this.legForce, this.tmp3).multiplyScalar(-1);
    const F = this.tmp3;
    const r = this.tmp.subVectors(this.targetLocal, this.feetMidLocal);
    // ankle torques and feet load, low-passed: momentary unweighting over chop (≈ 50 ms) does
    // not make the feet slip
    const kf = 1 - Math.exp(-dt / M.balanceFilter);
    this.ankleRoll += (r.y * F.z - r.z * F.y - this.ankleRoll) * kf;
    this.anklePitch += (r.x * F.y - r.y * F.x - this.anklePitch) * kf;
    this.feetLoad += (Math.max(-F.y, 0) - this.feetLoad) * kf;
    const N = this.feetLoad;
    const w = this.standWeight();
    const rollCap = M.rollLever * N + M.gripTorque;
    const pitchCap = M.pitchLever * N + M.gripTorque;
    this.balanceRoll = this.ankleRoll / rollCap;
    this.balancePitch = this.anklePitch / pitchCap;
    const over = Math.max(Math.abs(this.balanceRoll), Math.abs(this.balancePitch)) > 1 && w > 0.8;
    if (over) this.balanceTimer += dt;
    else this.balanceTimer = Math.max(this.balanceTimer - 2 * dt, 0);
    return this.balanceTimer > M.balanceTime;
  }

  /** Other wipeout criteria; returns the reason or null. Inputs from SurfSim. */
  checkWipeout(
    dt: number,
    boardUpY: number,
    noseDeckDepth: number,
    forwardSpeed: number,
    deckDepth: number,
    legStretch: number,
  ): WipeoutReason | null {
    if (this.stance === 'fallen') return null;
    this.flipTimer = boardUpY < -0.2 ? this.flipTimer + dt : 0;
    if (this.flipTimer > 0.25) return 'flipped';
    this.pearlTimer = noseDeckDepth > 0.06 && forwardSpeed > 2.5 ? this.pearlTimer + dt : Math.max(this.pearlTimer - dt, 0);
    if (this.pearlTimer > 0.2) return 'pearl';
    this.buriedTimer = deckDepth > 0.7 ? this.buriedTimer + dt : 0;
    if (this.buriedTimer > 0.4) return 'buried';
    if (legStretch > 0.45) return 'impact';
    return null;
  }

  /** Refresh `pose` (world space). */
  updatePose(board: RigidBody): void {
    const p = this.pose;
    p.stance = this.stance;
    p.popProgress = this.standWeight();
    p.com.copy(this.position);
    p.crouch = clamp((this.config.standHeight - this.height) / (this.config.standHeight - this.config.crouchHeight), 0, 1);
    p.lean = this.lean * p.popProgress;
    if (this.stance === 'fallen') {
      p.up.set(0, 1, 0);
      p.forward.copy(this.fallenDir);
      p.backFoot.copy(this.position).addScaledVector(this.fallenDir, -0.9 * this.scale);
      p.frontFoot.copy(p.backFoot);
      p.handLeft.copy(this.position);
      p.handRight.copy(this.position);
      return;
    }
    board.localDirToWorld(this.upLocal, p.up);
    p.forward.set(board.R[0], board.R[3], board.R[6]);
    board.localToWorld(this.backFootLocal, p.backFoot);
    board.localToWorld(this.frontFootLocal, p.frontFoot);
    p.handLeft.copy(this.handL);
    p.handRight.copy(this.handR);
  }
}

/** The subset of SurfInput the rider uses. */
export interface StanceInput {
  paddle: number;
  steer: number;
  leanForward: number;
  leanSide: number;
  crouch: number;
}

function clamp(x: number, a: number, b: number): number {
  return x < a ? a : x > b ? b : x;
}
