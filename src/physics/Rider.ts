/**
 * Rider model: a point mass on "legs" plus buoyant/draggy body segments.
 *
 * - The rider is a point mass (default 75 kg). SurfSim couples it implicitly to the board through
 *   the legs. Prone and while popping up it is held by a stiff 3-axis spring-damper toward a
 *   target COM in board space and its own moment of inertia is lumped into the board's (it lies on
 *   / crouches over the board). Standing, it is an inverted pendulum on the feet (below).
 * - Body segments are spheroids (two chest halves, hips, thighs, shins) carrying the whole body
 *   volume m/990 kg/m³ (≈0.076 m³ for 75 kg, wetsuit and lungs included, so a fallen rider just
 *   floats) that provide buoyancy and drag when they dip into the water.
 *
 * Stances
 *   prone:    COM 12 cm above the deck on the stringer near the board's centre of volume.
 *             Paddling: alternating arm strokes, thrust only while the hand is in the water,
 *             hand speed ≤ 3 m/s ⇒ thrust falls with speed (top speed ≈ 2 m/s). Steer = the
 *             outside arm works harder, the inside arm less (back-paddling past zero): steering
 *             without paddling pivots the board on the spot. leanForward slides the body ±15 cm,
 *             leanSide ±5 cm.
 *   popping:  0.6 s blend prone → standing (the legs push the board down).
 *   standing: the feet stand across the stringer (BoardSpec.stance); the body is a linear inverted
 *             pendulum on them (see "Standing balance").
 *   fallen:   detached; floats with its segments (resting on the seabed / beach where it is
 *             shallow or dry), tethered by a 2.4 m leash to the tail.
 *
 * Standing balance (linear inverted pendulum + capture point)
 *   Frame: the apparent up u = (a + g)/|a + g| (a = filtered acceleration of the board + rider COM:
 *   what the rider's balance senses), e_x = the board's long axis and e_z = its lateral axis, both
 *   projected ⟂ u. With the COM r (relative to the feet midpoint) at height h along u, the pendulum
 *   frequency is ω0 = √(|a + g| / h), and the capture point ξ = r⟂ + v⟂/ω0 (v = COM velocity
 *   relative to the feet) is where the centre of pressure (CoP) would have to be to bring the
 *   body to rest over it. The support is the deck under the feet: across ±½ foot length from the
 *   stringer (toes / heels), along from the back of the back foot to the front of the front foot.
 *   The leg force on the rider runs from the CoP p through the COM: F = N u + (N/h)(r⟂ − p) (N =
 *   leg load), so the body accelerates away from the CoP, r̈⟂ = ω0²(r⟂ − p). The reaction acts on
 *   the board at the CoP (SurfSim applies it on the force line through the COM): CoP off the
 *   stringer = toe/heel pressure = the torque that edges the board. No other torque.
 *   The reflex (capture-point control) aims ξ at an aim point inside the support:
 *       p = aim + (1 + k/ω0)(ξ − aim), clamped to the usable support,
 *   which, unsaturated, is a spring-damper on the COM (poles −ω0 and −k). Aim: across, the edge
 *   pressure that drives the board toward the bank the rider wants (leanSide → bank relative to
 *   the water surface, a full lean = bankMax); along, the trim (leanForward). The rider crouches
 *   (input, and a reflex when ξ nears the edge: a lower COM is quicker and recovers more).
 *   balanceAssist (0..1) is the quality of that reflex: capture gain, how much of the feet it
 *   dares to use, how firmly it presses the rail for a bank error, the crouch reflex.
 *   A lean is a bank target: at speed the banked board turns and the turn's apparent gravity
 *   tilts u under the leaning body; without speed it does not turn, the rail sinks and the board
 *   rolls over under the feet (or they slip off a deck tilted past the friction angle).
 *   The rider falls only when the physics leaves no way back: the capture point is beyond the
 *   support (+ a small recovery margin) with the COM already past the edge of the feet, or the
 *   leg force is further from the deck normal than the feet's friction angle (they slip), or the
 *   board flipped / buried / the legs were over-stretched by an impact.
 */
import { Vector3 } from 'three';
import type { BoardShape } from './boardShape';
import { GRAVITY, RHO_WATER } from './constants';
import type { ImplicitSystem } from './implicit';
import type { RigidBody } from './RigidBody';

export type Stance = 'prone' | 'popping' | 'standing' | 'fallen';
export type WipeoutReason = 'balance' | 'flipped' | 'pearl' | 'buried' | 'impact';

export interface RiderConfig {
  /** kg. */
  mass: number;
  /** Leg spring natural frequency (with the rider mass), Hz. */
  legFrequency: number;
  legDampingRatio: number;
  /** Quality of the balance reflex: 0 = clumsy, 1 = very skilled body (most forgiving). */
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
  /** false = never wipe out (practice / tests): when the rider would fall because the body went
   * past recovery, the board flipped or the legs were over-stretched, SurfSim puts them back on
   * their feet in place instead (`SurfSim.recoveries`); a pearl or a buried board is ridden on. */
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


/** Interpolate a [clumsy, skilled] pair by the balance reflex quality. */
export function skill(pair: readonly [number, number], assist: number): number {
  const a = assist < 0 ? 0 : assist > 1 ? 1 : assist;
  return pair[0] + (pair[1] - pair[0]) * a;
}

/** Rider tunables that are not per-rider (plain object: a debug GUI may edit it). */
export const RIDER_MODEL = {
  /** Support across the board: the toes / heels press the deck up to railInset (m) inside the
   * working rail — the feet span the deck and the rider steps onto the rail being weighted. (The
   * hull needs it: rolling a planing funboard through 10–30° of bank before the turn has built up
   * takes ≈ 16 cm of CoP, the wider longboard / soft-top ≈ 24 cm; a foot centred on the stringer
   * reaches ≈ 13 cm.) Foot width along the board, m (75 kg body). */
  railInset: 0.03,
  footWidth: 0.1,
  /** Bank relative to the water surface under the board that a full leanSide edges toward, rad,
   * and how fast that target moves, rad/s. */
  bankMax: (40 * Math.PI) / 180,
  bankRate: (90 * Math.PI) / 180,
  /** Edge pressure (balanceReflex): the pressure that holds a carve, × the half width of the board
   * at the feet, at the forward speed edgeSpeed (∝ v^1.5 below it), reached for a wanted bank of
   * edgeHoldBank, rad; the largest bank-error correction on top of it, × the half width; the
   * correction gains (reflex.edge / edgeDamp) scale as (v / edgeSpeed)^edgeSpeedExp below
   * edgeSpeed, not below edgeSpeedMin. */
  edgeHold: 0.5,
  edgeHoldBank: (15 * Math.PI) / 180,
  edgeFix: 0.6,
  edgeSpeed: 7,
  edgeSpeedExp: 3,
  edgeSpeedMin: 0.15,
  /** Expected turn (balanceReflex): u tilts by atan(turnCoord tan φ) for a bank φ on the water
   * (a conservative share of the ≈ 0.4–0.7 these hulls show), fading in between the forward speeds
   * turnSpeed, m/s, after turnLag, s. */
  turnCoord: 0.3,
  turnSpeed: [1.5, 3.5] as [number, number],
  turnLag: 0.25,
  /** How far beyond the usable support a lean may aim the body, m (the body leans into the turn
   * that will catch it). */
  commit: 0.08,
  /** Felt hold: filter time constant of the CoP that balances the water's roll moment (0 = use
   * the hold model), s. */
  feltHold: 0,
  /** Hip strategy: the trunk (upper body) as a rotor about the COM — its inertia, kg·m² per kg of
   * a 75 kg-scaled body, the hip's torque, N·m per kg (× scale), the trunk's range, rad (the rider
   * falls past twice that), and the frequency of its return to neutral, rad/s. */
  hipInertia: 0.07,
  hipTorque: 1.6,
  hipRange: 0.4,
  hipReturn: 3,
  /** COM aim along the board at a full leanForward, m from the feet midpoint (trim). */
  trimRange: 0.22,
  /** Nose-dive reflex: below this trim relative to the water surface along the board, rad, the
   * rider moves back reflex.nose m per rad (+ noseDamp s × the nose-down pitch rate). */
  noseTrim: (1 * Math.PI) / 180,
  noseDamp: 0.3,
  /** Balance reflex, [clumsy, skilled] by balanceAssist (see the file header):
   *  gain: capture gain k, 1/s (the capture point converges at k, the COM at ω0 and k);
   *  support: usable fraction of the feet (a clumsy rider does not dare the edges);
   *  edge: edge pressure correction per radian of bank error, m/rad, edgeDamp per rad/s of roll rate;
   *  crouch: crouch added as the capture point nears the edge of the feet (0..1);
   *  hip: share of the hip torque used. */
  reflex: {
    gain: [1.5, 6] as [number, number],
    support: [0.6, 0.95] as [number, number],
    edge: [0.45, 0.99] as [number, number],
    edgeDamp: [0.075, 0.165] as [number, number],
    crouch: [0, 0] as [number, number],
    hip: [0.3, 1] as [number, number],
    nose: [0.5, 1.5] as [number, number],
  },
  /** How far the capture point may be beyond the edge of the feet and still be saved (arms, hips:
   * the body's own angular momentum), m, and the body's tilt past the edge of the feet (from the
   * apparent up) beyond which nothing brings it back, rad. */
  recoverMargin: 0.03,
  noReturn: (15 * Math.PI) / 180,
  /** Friction angle of the feet on the waxed deck: a leg force further than this from the deck
   * normal slips, rad (μ ≈ 0.84), for longer than slipTime, s. */
  slipAngle: (40 * Math.PI) / 180,
  slipTime: 0.15,
  /** Floor on the apparent gravity used for the pendulum frequency (unweighted over a crest), × g. */
  minApparentG: 0.3,
  /** Crouch servo: rate of the COM height target, m/s, and its lag, s. */
  crouchRate: 2,
  crouchLag: 0.12,
  /** Twist (standing): the upper body (trunk, arms, head) is a yaw rotor about the deck normal,
   * driven against the board by the trunk muscles (torque ≤ twistTorque, N·m) toward
   * −twist × twistRange (rad) of counter-rotation (+ a stiff stop past the range); twistFreq is
   * the servo's natural frequency, rad/s, with the rotor's yaw inertia twistInertia × m (kg·m² per
   * kg of a 75 kg-scaled body; the rest of the standing body's yaw inertia rides with the board). */
  twistTorque: 40,
  twistInertia: 0.018,
  twistRange: (70 * Math.PI) / 180,
  twistFreq: 18,
  /** Max speed of the prone/pop-up COM target relative to the board, m/s (pop-ups peak ≈ 1.7). */
  targetSpeedMax: 4,
  /** Leg-axis stretch / squash (COM height off its target) that counts as an impact wipeout, m. */
  impactStretch: 0.45,
  /** Low-pass time constant of the feet load (HUD, telemetry), s. */
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
  /** Prone steering: arm effort moved from the inside to the outside arm per unit of steer (the
   * inside arm back-paddles past 0). */
  steerSplit: 1,
  /** A fallen body on the seabed / beach: contact stiffness of the whole body (N/m; ≈ 2 cm into
   * the sand under its weight), damping (N·s/m, ≈ critical) and Coulomb friction (shared by the
   * segments by volume). */
  groundStiffness: 4e4,
  groundDamping: 3500,
  groundFriction: 0.6,
  /** Leash spring (beyond the slack length) N/m and damping N·s/m. */
  leashStiffness: 400,
  leashDamping: 80,
  /** Apparent-gravity filter time constant (what the balance senses), s. */
  accelFilter: 0.08,

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

/** Water plane around a reference point (height, slope, velocity) and the seabed plane under it
 * (height y and slope; on dry land the seabed is above the water height). */
export interface WaterPlane {
  x: number;
  z: number;
  height: number;
  slopeX: number;
  slopeZ: number;
  velX: number;
  velY: number;
  velZ: number;
  seabed: number;
  seabedSlopeX: number;
  seabedSlopeZ: number;
}

export function createWaterPlane(): WaterPlane {
  return { x: 0, z: 0, height: 0, slopeX: 0, slopeZ: 0, velX: 0, velY: 0, velZ: 0, seabed: -100, seabedSlopeX: 0, seabedSlopeZ: 0 };
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

  /** Target COM in board local coordinates and its rate of change (prone / pop-up spring; standing:
   * the aim point at the target height, for reference). */
  readonly targetLocal = new Vector3();
  readonly targetVelLocal = new Vector3();
  /** Stance up (feet → COM) in board local coordinates. */
  readonly upLocal = new Vector3(0, 1, 0);
  readonly backFootLocal = new Vector3();
  readonly frontFootLocal = new Vector3();
  readonly feetMidLocal = new Vector3();
  /** Support half-sizes in the deck plane around the feet midpoint: along the board (x) and
   * across it (z), m. */
  supportX = 0.34;
  supportZ = 0.13;
  /** Half width of the board at the feet (narrower foot), m. */
  railZ = 0.25;

  // --- standing balance (see the file header). Frame vectors are world, refreshed every step.
  /** Apparent up u, the board's long and lateral axes projected ⟂ u. */
  readonly frameU = new Vector3(0, 1, 0);
  readonly frameX = new Vector3(1, 0, 0);
  readonly frameZ = new Vector3(0, 0, 1);
  /** |a + g|, m/s² (filtered), and the pendulum frequency ω0, rad/s. */
  apparentG = GRAVITY;
  omega0 = 3.2;
  /** COM relative to the feet midpoint: height along u, offsets along e_x / e_z (m) and the
   * relative velocity along e_x / e_z / u (m/s). */
  comHeight = 0.95;
  comX = 0;
  comZ = 0;
  comVX = 0;
  comVZ = 0;
  comVU = 0;
  /** Capture point (e_x, e_z), m. */
  captureX = 0;
  captureZ = 0;
  /** Velocity of the aim point relative to the feet along e_x / e_z (it turns with the board's yaw
   * about the water's normal and moves with the lean), m/s. */
  aimVelX = 0;
  aimVelZ = 0;
  /** Hip strategy: the trunk (upper body) as a bounded rotor about the COM. Its angle (rad, + = the
   * upper body toward +e_x / +e_z) and rate, and the moment of the leg force about the COM this
   * step (N·m, + = pushing the COM toward +e_x / +e_z: the trunk swings the other way). */
  trunkX = 0;
  trunkZ = 0;
  /** Moment of the leg force pair about u over the last step (+ = about +u), N·m. */
  legYawMoment = 0;
  trunkVX = 0;
  trunkVZ = 0;
  hipX = 0;
  hipZ = 0;
  /** Centroidal moment pivot the balance law asks for (CoP − hip moment / load), m, and whether
   * it is out of reach (CoP at the edge of the feet and the hip at its limit). */
  cmpX = 0;
  cmpZ = 0;
  /** Tilt of u across the board that the rider expects from the turn (lags the bank), rad. */
  turnTilt = 0;
  /** Reflex aim (where the capture point is driven) and the commanded CoP, m. */
  aimX = 0;
  aimZ = 0;
  copX = 0;
  copZ = 0;
  /** The CoP command is at the edge of the usable support (along / across). */
  copSatX = false;
  copSatZ = false;
  /** Usable support (projected ⟂ u), and the whole support projected, m. */
  usableX = 0.3;
  usableZ = 0.12;
  footX = 0.34;
  footZ = 0.13;
  /** Capture gain k of the reflex this step, 1/s. */
  captureGain = 3;
  /** COM height target along u and its rate, m, m/s. */
  heightTarget = 0.95;
  heightRate = 0;
  /** Bank the rider edges toward (relative to the water) and the target bank (world), rad. */
  bankCmd = 0;
  bankTarget = 0;
  /** Angle between the leg force line (≈ u) and the deck normal, rad. */
  deckAngle = 0;

  /** Board bank (world roll of its lateral axis, right rail down +), rad. */
  bankAngle = 0;
  /** Board turn rate (+ = right), rad/s. */
  turnRate = 0;
  /** Water surface slope under the board (set by SurfSim each step). */
  waterSlopeX = 0;
  waterSlopeZ = 0;
  /** Bank that would lie flat on the water surface, rad. */
  waterBank = 0;
  /** Water velocity under the board (set by SurfSim each step), m/s. */
  waterVelX = 0;
  waterVelY = 0;
  waterVelZ = 0;
  /** Water roll moment on the board about the feet line (+ = right rail down), set by SurfSim each
   * step, N·m. */
  hullRollMoment = 0;
  feltHold = 0;
  /** Body lean toward the right rail relative to the deck normal (pose), rad. */
  lean = 0;
  height = 0.95;
  proneShift = 0;
  proneSide = 0;

  // balance read-out
  /** Capture point relative to the support + recovery margin, along the lateral (roll) and
   * long (pitch) axes: |x| > 1 = past recovery (the HUD's balance dot). */
  balanceRoll = 0;
  balancePitch = 0;
  /** Which way the body went past recovery in the last balance check (null = in balance). */
  tipOver: 'side' | 'forward' | 'back' | null = null;
  /** Seconds the capture point has spent past recovery (decays when back in balance). */
  balanceTimer = 0;
  /** Normal force of the feet on the deck (low-passed), N. */
  feetLoad = 0;
  /** The CoP is at the edge of the usable support this step. */
  ankleSaturated = false;
  /** Upper-body twist relative to the board about the deck normal (+ = toward the right, like
   * turnRate), rad, and the upper body's yaw rate about the deck normal (+ = right), rad/s
   * (standing only; otherwise the upper body turns with the board). */
  twistAngle = 0;
  twistSpin = 0;
  /** Leg force on the rider (effective, from the implicit step), world. */
  readonly legForce = new Vector3();
  /** Filtered board + rider COM acceleration (apparent gravity), world. */
  readonly accelFiltered = new Vector3();
  private flipTimer = 0;
  private pearlTimer = 0;
  private buriedTimer = 0;
  private slipTimer = 0;

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
  private readonly prevTarget = new Vector3();
  private readonly popInput: StanceInput = { paddle: 0, steer: 0, leanForward: 0, leanSide: 0, crouch: 0 };
  private hasPrevTarget = false;
  /** Board-local point the attached segment layout is relative to. */
  private readonly segRefLocal = new Vector3();

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
    if (this.shape) this.setBoard(this.shape);
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
    // support: the deck under the feet (railInset inside the rails across, the feet's own width
    // beyond the feet along)
    const s = this.scale;
    this.supportX = 0.5 * st.feetSpread + 0.5 * RIDER_MODEL.footWidth * s;
    this.railZ = Math.min(shape.halfWidth(ub), shape.halfWidth(uf));
    this.supportZ = Math.max(this.railZ - RIDER_MODEL.railInset, 0.05);
  }

  /** Reset the controller state for a stance (does not place the rider; SurfSim does). */
  resetState(stance: Stance): void {
    this.stance = stance;
    this.popProgress = stance === 'standing' ? 1 : 0;
    this.wipeoutReason = null;
    this.timeSinceWipeout = 0;
    this.lean = 0;
    this.height = this.config.standHeight;
    this.heightTarget = this.config.standHeight * this.scale;
    this.heightRate = 0;
    this.proneShift = 0;
    this.proneSide = 0;
    this.bankCmd = 0;
    this.balanceRoll = 0;
    this.balancePitch = 0;
    this.balanceTimer = 0;
    this.tipOver = null;
    this.feetLoad = 0;
    this.flipTimer = 0;
    this.pearlTimer = 0;
    this.buriedTimer = 0;
    this.slipTimer = 0;
    this.captureX = this.captureZ = 0;
    this.aimX = this.aimZ = 0;
    this.copX = this.copZ = 0;
    this.copSatX = this.copSatZ = false;
    // start mid-pull: the first stroke's yaw impulse is then half of a full one and the yaw rate
    // swings symmetrically about zero instead of leaving a ~10° heading offset
    this.armPhaseL = 0.225;
    this.armPhaseR = 0.725;
    this.paddleThrust = 0;
    this.accelFiltered.set(0, 0, 0);
    this.aimVelX = this.aimVelZ = 0;
    this.trunkX = this.trunkZ = this.trunkVX = this.trunkVZ = 0;
    this.hipX = this.hipZ = 0;
    this.cmpX = this.cmpZ = 0;
    this.turnTilt = 0;
    this.feltHold = 0;
    this.legForce.set(0, 0, 0);
    this.twistAngle = 0;
    this.twistSpin = 0;
    this.hasPrevTarget = false;
    this.targetVelLocal.set(0, 0, 0);
  }

  /**
   * Restart the estimators (apparent gravity) and pre-load the feet with the rider's weight. Call
   * after the rider and board were placed (a spawn).
   */
  restartEstimators(): void {
    this.feetLoad = this.stance === 'standing' ? this.config.mass * GRAVITY : 0;
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

  /** Rider body moment of inertia about its own COM that turns with the board, board axes. Prone
   * (lying on the deck) and popping up it is the whole body; standing only the yaw (the feet hold
   * the body's heading; the upper body's own yaw is the twist DOF) — in roll and pitch the board
   * turns under the feet and the body is the pendulum above them. */
  bodyInertia(out: Vector3): Vector3 {
    if (this.stance === 'fallen') return out.set(0, 0, 0);
    const m = this.config.mass;
    const s2 = this.scale * this.scale;
    // prone: body along x (L ≈ 1.75 m) → small roll, large yaw/pitch
    const px = 0.045 * m * s2, py = 0.24 * m * s2, pz = 0.24 * m * s2;
    // standing: body along y → yaw small
    const sy = 0.025 * m * s2;
    if (this.stance === 'standing') return out.set(0, sy - this.twistInertia(), 0);
    const w = this.standWeight();
    // popping: crouched over the board, still held through hands and feet
    const sx = 0.11 * m * s2, sz = 0.11 * m * s2;
    return out.set(px + (sx - px) * w, py + (sy - py) * w, pz + (sz - pz) * w);
  }

  /** Yaw inertia of the upper body (the twist rotor), kg·m². */
  twistInertia(): number {
    return RIDER_MODEL.twistInertia * this.config.mass * this.scale * this.scale;
  }

  /**
   * Twist (standing): the trunk muscles drive the upper body toward a counter-rotation of
   * −twist × twistRange relative to the board (+ a stiff stop past the range). Advances the upper
   * body's yaw rate and returns the torque on it about the deck normal (+ = right, N·m); the board
   * takes the opposite torque. `boardYawRate`: the board's yaw rate about its deck normal (+ = right).
   */
  twistUpdate(dt: number, twist: number, boardYawRate: number): number {
    const M = RIDER_MODEL;
    const I = this.twistInertia();
    const k = I * M.twistFreq * M.twistFreq;
    const c = 2 * I * M.twistFreq;
    const rel = this.twistSpin - boardYawRate;
    const target = -clamp(twist, -1, 1) * M.twistRange;
    let tau = clamp(k * (target - this.twistAngle) - c * rel, -M.twistTorque, M.twistTorque);
    const over = Math.abs(this.twistAngle) - M.twistRange;
    if (over > 0) tau -= Math.sign(this.twistAngle) * 10 * k * over + c * rel;
    this.twistSpin += (tau / I) * dt;
    return tau;
  }

  /** Advance the twist angle with the board's new yaw rate (after the step's solve). */
  integrateTwist(dt: number, boardYawRate: number): void {
    this.twistAngle += (this.twistSpin - boardYawRate) * dt;
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
   * Advance the pop-up, the stance and the balance reflex, and compute the target COM (board
   * local) and the board-local segment layout. `board` must have its derived matrices up to date.
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
      if (this.popProgress >= 1) {
        this.stance = 'standing';
        // the upper body starts turning with the board
        this.twistAngle = 0;
        this.twistSpin = boardYawRate(board);
      }
    }
    const R = board.R;
    const av = board.angularVelocity;
    this.turnRate = -av.y; // + = turning right (toward +Z when heading +X)
    this.bankAngle = Math.asin(clamp(-R[5], -1, 1));
    // water surface across the board: bank that would lie flat on it
    const zh = Math.hypot(R[2], R[8]) || 1;
    const sLat = (this.waterSlopeX * R[2] + this.waterSlopeZ * R[8]) / zh;
    this.waterBank = -Math.atan(sLat);

    // --- prone target
    const lagK = 1 - Math.exp(-dt / 0.25);
    this.proneShift += (clamp(input.leanForward, -1, 1) * 0.15 - this.proneShift) * lagK;
    // prone balance reflex: shift the body toward the high rail (keeps a sunken board upright)
    {
      const af = this.accelFiltered;
      this.tmp.set(af.x, af.y + GRAVITY, af.z);
      if (this.tmp.lengthSq() < 1) this.tmp.set(0, 1, 0);
      board.worldDirToLocal(this.tmp.normalize(), this.tmp2);
      const rollApparent = clamp(Math.atan2(this.tmp2.z, Math.max(this.tmp2.y, 0.5)), -1.2, 1.2);
      const sideTarget = clamp(input.leanSide, -1, 1) * 0.05 + clamp(cfg.balanceAssist, 0, 1) * clamp(0.5 * rollApparent, -0.12, 0.12);
      this.proneSide += (sideTarget - this.proneSide) * lagK;
    }
    const pxl = this.proneBase.x + this.proneShift;
    const pu = clamp(shape.uAtX(pxl), 0, 1);
    const pTx = pxl;
    const pTy = shape.deckY(pu, 0) + cfg.proneHeight;
    const pTz = this.proneSide;

    // --- standing: balance reflex (also gives the pop-up its end target)
    // (getting up, the rider does not edge yet: hands and feet are still finding the deck)
    if (this.stance === 'popping') {
      const pi = this.popInput;
      pi.paddle = input.paddle;
      pi.steer = input.steer;
      pi.leanForward = input.leanForward;
      pi.leanSide = 0;
      pi.crouch = input.crouch;
      this.balanceReflex(dt, pi, board);
    } else this.balanceReflex(dt, input, board);
    const fm = this.feetMidLocal;
    // aim point at the target height, board local. Getting up (a stiff hands-and-feet push, not
    // yet the balancing body) the rider rises along the water's normal — following the apparent
    // up there would feed the board's own lurches straight back into the push
    if (this.stance === 'popping') {
      const n = this.tmp.set(-this.waterSlopeX, 1, -this.waterSlopeZ).normalize();
      // trim along the board, ⟂ n
      const R = board.R;
      const d = R[0] * n.x + R[3] * n.y + R[6] * n.z;
      const ax = R[0] - d * n.x, ay = R[3] - d * n.y, az = R[6] - d * n.z;
      const al = Math.hypot(ax, ay, az) || 1;
      n.multiplyScalar(this.heightTarget);
      this.tmp.set(n.x + (ax / al) * this.aimX, n.y + (ay / al) * this.aimX, n.z + (az / al) * this.aimX);
    } else this.tmp.copy(this.frameU).multiplyScalar(this.heightTarget).addScaledVector(this.frameX, this.aimX).addScaledVector(this.frameZ, this.aimZ);
    board.worldDirToLocal(this.tmp, this.tmp2);
    const sTx = fm.x + this.tmp2.x;
    const sTy = fm.y + this.tmp2.y;
    const sTz = fm.z + this.tmp2.z;

    // --- blend
    const w = this.standWeight();
    this.targetLocal.set(pTx + (sTx - pTx) * w, pTy + (sTy - pTy) * w, pTz + (sTz - pTz) * w);
    // (bounded to a body-plausible speed: the leg damper pushes c × this)
    if (this.hasPrevTarget) this.targetVelLocal.subVectors(this.targetLocal, this.prevTarget).divideScalar(dt).clampLength(0, M.targetSpeedMax);
    else this.targetVelLocal.set(0, 0, 0);
    this.prevTarget.copy(this.targetLocal);
    this.hasPrevTarget = true;

    // standing body (board local): standing, the actual COM; popping up, the standing target. The
    // segments are laid out around it (layoutSegments carries the layout with the actual COM)
    const S = this.segRefLocal;
    if (this.stance === 'standing') board.worldToLocal(this.position, S);
    else S.set(sTx, sTy, sTz);
    let ux = S.x - fm.x, uy = S.y - fm.y, uz = S.z - fm.z;
    const hh = Math.sqrt(ux * ux + uy * uy + uz * uz);
    if (hh > 1e-3) {
      ux /= hh; uy /= hh; uz /= hh;
    } else {
      ux = 0; uy = 1; uz = 0;
    }
    // pose: stance up blended from the deck normal while popping up
    this.upLocal.set(ux * w, 1 + (uy - 1) * w, uz * w).normalize();
    this.lean = Math.atan2(uz, uy);
    this.height = Math.min(this.heightTarget, hh) / s;
    if (this.stance !== 'standing') S.copy(this.targetLocal);

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
      const cx = fm.x + hh * ux, cy = fm.y + hh * uy, cz = fm.z + hh * uz;
      let slx: number, sly: number, slz: number;
      let ax: number, ay: number, az: number;
      if (d.anchor === 'com') {
        slx = cx + d.t * s * ux;
        sly = cy + d.t * s * uy;
        slz = cz + d.t * s * uz + d.pz * 0.9 * s;
        ax = ux; ay = uy; az = uz;
      } else {
        const foot = d.anchor === 'back' ? this.backFootLocal : this.frontFootLocal;
        const dx = cx - foot.x, dy = cy - foot.y, dz = cz - foot.z;
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

  /**
   * The standing balance reflex (file header): frame, capture point, aim (edge pressure for the
   * wanted bank, trim), CoP command and COM height target. Uses the current rider and board state.
   */
  private balanceReflex(dt: number, input: StanceInput, board: RigidBody): void {
    const M = RIDER_MODEL;
    const cfg = this.config;
    const a = clamp(cfg.balanceAssist, 0, 1);
    const R = board.R;
    const s = this.scale;
    // apparent up and its magnitude
    const af = this.accelFiltered;
    const U = this.frameU.set(af.x, af.y + GRAVITY, af.z);
    let g = U.length();
    if (g < 1e-3) {
      U.set(0, 1, 0);
      g = 0;
    } else U.divideScalar(g);
    this.apparentG = g;
    // board long / lateral axes ⟂ u
    const bxx = R[0], bxy = R[3], bxz = R[6];
    const bzx = R[2], bzy = R[5], bzz = R[8];
    const X = this.frameX;
    let d = bxx * U.x + bxy * U.y + bxz * U.z;
    X.set(bxx - d * U.x, bxy - d * U.y, bxz - d * U.z);
    if (X.lengthSq() < 1e-8) X.set(-bzy * U.z + bzz * U.y, -bzz * U.x + bzx * U.z, -bzx * U.y + bzy * U.x);
    X.normalize();
    const Z = this.frameZ.crossVectors(X, U).normalize();
    // projection of the support onto the plane ⟂ u
    const cx = Math.abs(bxx * X.x + bxy * X.y + bxz * X.z);
    const cz = Math.abs(bzx * Z.x + bzy * Z.y + bzz * Z.z);
    this.footX = this.supportX * cx;
    this.footZ = this.supportZ * cz;
    const use = skill(M.reflex.support, a);
    this.usableX = use * this.footX;
    this.usableZ = use * this.footZ;
    // deck normal vs. the leg force line (≈ u): past the friction angle the feet slip
    this.deckAngle = Math.acos(clamp(R[1] * U.x + R[4] * U.y + R[7] * U.z, -1, 1));

    // COM relative to the feet midpoint (position and velocity)
    const fm = this.feetMidLocal;
    const rfx = R[0] * fm.x + R[1] * fm.y + R[2] * fm.z;
    const rfy = R[3] * fm.x + R[4] * fm.y + R[5] * fm.z;
    const rfz = R[6] * fm.x + R[7] * fm.y + R[8] * fm.z;
    const rx = this.position.x - board.position.x - rfx;
    const ry = this.position.y - board.position.y - rfy;
    const rz = this.position.z - board.position.z - rfz;
    const w = board.angularVelocity;
    const vx = this.velocity.x - (board.velocity.x + w.y * rfz - w.z * rfy);
    const vy = this.velocity.y - (board.velocity.y + w.z * rfx - w.x * rfz);
    const vz = this.velocity.z - (board.velocity.z + w.x * rfy - w.y * rfx);
    this.comHeight = rx * U.x + ry * U.y + rz * U.z;
    this.comX = rx * X.x + ry * X.y + rz * X.z;
    this.comZ = rx * Z.x + ry * Z.y + rz * Z.z;
    const h = Math.max(this.comHeight, 0.3 * s);
    const w0 = Math.sqrt(Math.max(g, M.minApparentG * GRAVITY) / h);
    this.omega0 = w0;
    // the aim turns with the board about the water's normal n (the feet carry the body round the
    // turn; the board's roll and pitch turn under the feet): its velocity relative to the feet,
    // ω_n × r with ω_n = (ω·n) n
    const nInv = 1 / Math.sqrt(1 + this.waterSlopeX * this.waterSlopeX + this.waterSlopeZ * this.waterSlopeZ);
    const nx = -this.waterSlopeX * nInv, ny = nInv, nz = -this.waterSlopeZ * nInv;
    const wn = w.x * nx + w.y * ny + w.z * nz;
    const fvx = wn * (ny * rz - nz * ry), fvy = wn * (nz * rx - nx * rz), fvz = wn * (nx * ry - ny * rx);
    const frameVX = fvx * X.x + fvy * X.y + fvz * X.z;
    const frameVZ = fvx * Z.x + fvy * Z.y + fvz * Z.z;
    let aimVX = frameVX;
    let aimVZ = frameVZ;

    // --- across: edge pressure toward the bank the rider wants (relative to the water surface)
    const side = clamp(input.leanSide, -1, 1);
    {
      const step = M.bankRate * dt;
      this.bankCmd += clamp(side * M.bankMax - this.bankCmd, -step, step);
    }
    this.bankTarget = this.waterBank + this.bankCmd;
    const vf = Math.max(board.velocity.x * bxx + board.velocity.y * bxy + board.velocity.z * bxz, 0);
    const vRel = clamp(vf / M.edgeSpeed, 0, 1);
    // the turn the rider expects: banked φ on the water a planing board turns so that u tilts by
    // about atan(κ tan φ), after a lag. The aim (fixed in the u frame) moves with that tilt — the
    // body leans with the building turn instead of trailing it (no further than the wanted bank)
    {
      const kc = M.turnCoord * clamp((vf - M.turnSpeed[0]) / (M.turnSpeed[1] - M.turnSpeed[0]), 0, 1);
      const cmd = this.bankCmd;
      const phi = clamp(this.bankAngle - this.waterBank, Math.min(0, cmd), Math.max(0, cmd));
      const rate = (Math.atan(kc * Math.tan(phi)) - this.turnTilt) / M.turnLag;
      this.turnTilt += rate * dt;
      aimVZ += h * rate;
    }
    // the pressure that holds a carve (the hull's roll moment is about constant over 15–45° of
    // bank and grows with speed), plus a bounded correction for the bank error and the roll rate:
    // a big error is closed by the hold pressure following the wanted bank, not by throwing the
    // weight across (that would first press the other way). The correction gains fall with speed
    // (a slow hull rolls more for the same pressure), not below edgeSpeedMin (a slow board still
    // answers a hard lean: it tips)
    {
      const sched = Math.max(Math.pow(vRel, M.edgeSpeedExp), M.edgeSpeedMin);
      let hold = M.edgeHold * this.railZ * Math.pow(vRel, 1.5) * clamp(this.bankCmd / M.edgeHoldBank, -1, 1);
      if (M.feltHold > 0) {
        const load = cfg.mass * Math.max(g, M.minApparentG * GRAVITY);
        this.feltHold += (clamp(-this.hullRollMoment / load, -this.footZ, this.footZ) - this.feltHold) * (1 - Math.exp(-dt / M.feltHold));
        hold = this.feltHold;
      }
      const rollRate = w.x * bxx + w.y * bxy + w.z * bxz;
      const fix = skill(M.reflex.edge, a) * sched * (this.bankTarget - this.bankAngle) - skill(M.reflex.edgeDamp, a) * sched * rollRate;
      const pMax = M.edgeFix * this.railZ;
      const lim = this.usableZ + M.commit * s * Math.abs(side);
      this.aimZ = clamp(hold + clamp(fix, -pMax, pMax), -lim, lim);
    }
    // aim along: trim
    // (and, a reflex, back off the front foot when the nose drops toward the water: trim relative to
    // the water surface along the board below noseTrim)
    {
      const ch = Math.hypot(bxx, bxz) || 1;
      const slopeAlong = (this.waterSlopeX * bxx + this.waterSlopeZ * bxz) / ch;
      const trimRel = Math.atan2(bxy, ch) - Math.atan(slopeAlong);
      const pitchRate = w.x * bzx + w.y * bzy + w.z * bzz; // + = nose up (about the lateral axis)
      const nose = Math.max(0, M.noseTrim - trimRel);
      const back = skill(M.reflex.nose, a) * (nose - (nose > 0 ? M.noseDamp * Math.min(pitchRate, 0) : 0));
      this.aimX = clamp(clamp(input.leanForward, -1, 1) * M.trimRange * s - back, -this.usableX, this.usableX);
    }
    // getting up: the body rises over the feet (plus the trim) — no edge or nose reflexes yet (the
    // legs still push the board around under the hands)
    if (this.stance === 'popping') {
      this.aimZ = 0;
      this.aimX = clamp(clamp(input.leanForward, -1, 1) * M.trimRange * s, -this.usableX, this.usableX);
      aimVZ = frameVZ;
    }
    this.aimVelX = aimVX;
    this.aimVelZ = aimVZ;
    // COM velocity relative to the (moving) aim; the capture point (relative to the feet, in the
    // frame turning with the board)
    const relVX = vx * X.x + vy * X.y + vz * X.z;
    const relVZ = vx * Z.x + vy * Z.y + vz * Z.z;
    this.comVX = relVX - aimVX;
    this.comVZ = relVZ - aimVZ;
    this.comVU = vx * U.x + vy * U.y + vz * U.z;
    this.captureX = this.comX + (relVX - frameVX) / w0;
    this.captureZ = this.comZ + (relVZ - frameVZ) / w0;

    // capture-point control → CoP command, clamped to the usable support
    const k = skill(M.reflex.gain, a);
    this.captureGain = k;
    const f = 1 + k / w0;
    // the balance law asks for a centroidal moment pivot (CMP); the CoP (feet) and the hip share it:
    // the CoP goes where the rider wants the pressure (the aim; plus the trunk's return to
    // neutral) as far as the hip can make up the difference, CMP = CoP − hip moment / load
    const cmpX = this.aimX + f * (this.comX + this.comVX / w0 - this.aimX);
    const cmpZ = this.aimZ + f * (this.comZ + this.comVZ / w0 - this.aimZ);
    const load = cfg.mass * Math.max(g, M.minApparentG * GRAVITY);
    const tauMax = skill(M.reflex.hip, a) * M.hipTorque * cfg.mass * s;
    this.allocate(0, cmpX, load, tauMax);
    this.allocate(1, cmpZ, load, tauMax);
    this.ankleSaturated = this.copSatX || this.copSatZ;

    // COM height: crouch input, plus a crouch reflex as the capture point nears the edge
    const near = Math.max(Math.abs(this.captureX) / Math.max(this.footX, 1e-3), Math.abs(this.captureZ) / Math.max(this.footZ, 1e-3));
    const crouch = clamp(clamp(input.crouch, 0, 1) + skill(M.reflex.crouch, a) * clamp((near - 0.5) / 0.5, 0, 1), 0, 1);
    const hT = (cfg.standHeight + (cfg.crouchHeight - cfg.standHeight) * crouch) * s;
    const dh = clamp((hT - this.heightTarget) * (1 - Math.exp(-dt / M.crouchLag)), -M.crouchRate * dt, M.crouchRate * dt);
    this.heightTarget += dh;
    this.heightRate = dh / dt;

    // balance read-out: capture point over (support + recovery margin)
    this.balanceRoll = this.captureZ / (this.footZ + M.recoverMargin);
    this.balancePitch = this.captureX / (this.footX + M.recoverMargin);
  }

  /** Trunk (hip rotor) moment of inertia, kg·m². */
  trunkInertia(): number {
    return RIDER_MODEL.hipInertia * this.config.mass * this.scale * this.scale;
  }

  /**
   * Share the CMP the balance law wants between the CoP and the hip moment along one axis
   * (0 = e_x, 1 = e_z): the CoP takes the aim (+ the trunk's return), within the reach of the hip
   * (its torque, less near the end of the trunk's range) and of the feet.
   */
  private allocate(axis: 0 | 1, cmp: number, load: number, tauMax: number): void {
    const M = RIDER_MODEL;
    const I = this.trunkInertia();
    const th = axis === 0 ? this.trunkX : this.trunkZ;
    const om = axis === 0 ? this.trunkVX : this.trunkVZ;
    const aim = axis === 0 ? this.aimX : this.aimZ;
    const use = axis === 0 ? this.usableX : this.usableZ;
    // hip moment τ pushes the COM toward + and swings the trunk toward −: θ̈ = −τ/I. Room left
    // in each direction, less the distance needed to stop the trunk at full torque
    const range = M.hipRange;
    const acc = tauMax / Math.max(I, 1e-6);
    const brakeNeg = om < 0 ? (om * om) / (2 * acc) : 0;
    const brakePos = om > 0 ? (om * om) / (2 * acc) : 0;
    const soft = 0.3 * range;
    const tauHi = tauMax * clamp((th + range - brakeNeg) / soft, -1, 1);
    const tauLo = -tauMax * clamp((range - th - brakePos) / soft, -1, 1);
    // trunk return to neutral (θ̈ = −ω²θ − 2ωθ̇)
    const wr = M.hipReturn;
    const tauRet = I * (wr * wr * th + 2 * wr * om);
    let p = aim + tauRet / load;
    p = clamp(p, cmp + tauLo / load, cmp + tauHi / load);
    p = clamp(p, -use, use);
    // the CMP within reach of this CoP
    const lo = p - tauHi / load;
    const hi = p - tauLo / load;
    const sat = cmp < lo || cmp > hi;
    const c = clamp(cmp, Math.min(lo, hi), Math.max(lo, hi));
    if (axis === 0) {
      this.copX = p;
      this.cmpX = c;
      this.copSatX = sat;
    } else {
      this.copZ = p;
      this.cmpZ = c;
      this.copSatZ = sat;
    }
  }

  /**
   * After the step's solve: the leg force's moment about the COM (the part of it that does not
   * pass through the COM) turns the trunk. Uses this step's frame and CoP.
   */
  integrateTrunk(dt: number): void {
    const F = this.legForce;
    const U = this.frameU, X = this.frameX, Z = this.frameZ;
    const N = F.x * U.x + F.y * U.y + F.z * U.z;
    const fx = F.x * X.x + F.y * X.y + F.z * X.z;
    const fz = F.x * Z.x + F.y * Z.y + F.z * Z.z;
    const h = this.comHeight;
    // F⟂ = (N/h)(r − p) + τ/h  ⇒  τ = h F⟂ − N (r − p)
    this.hipX = h * fx - N * (this.comX - this.copX);
    this.hipZ = h * fz - N * (this.comZ - this.copZ);
    // the leg force pair's moment about u (feet at the CoP, body at the COM) — no hip rotation
    // takes it: the feet's twisting grip on the deck carries it to the board (SurfSim)
    this.legYawMoment = (this.comZ - this.copZ) * fx - (this.comX - this.copX) * fz;
    const I = this.trunkInertia();
    this.trunkVX -= (this.hipX / I) * dt;
    this.trunkVZ -= (this.hipZ / I) * dt;
    this.trunkX += this.trunkVX * dt;
    this.trunkZ += this.trunkVZ * dt;
  }

  /** Update the filtered board + rider acceleration (call once per step with the system COM's
   * acceleration from the external forces). */
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
  }

  /** Place the segments in the world (attached: rigid with the board; fallen: lying flat). */
  layoutSegments(board: RigidBody): void {
    if (this.stance !== 'fallen') {
      // the layout is relative to the body reference (target COM / standing COM); carry it with
      // the actual rider COM
      for (let i = 0; i < this.segments.length; i++) {
        const sg = this.segments[i];
        this.tmp.subVectors(this.segLocal[i], this.segRefLocal);
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
      const sb = water.seabed + water.seabedSlopeX * (c.x - water.x) + water.seabedSlopeZ * (c.z - water.z);
      const ay = sg.axis.y;
      const e = Math.sqrt(sg.radius * sg.radius * (1 - ay * ay) + sg.halfLength * sg.halfLength * ay * ay);
      // no water where the seabed is above the water height (dry beach)
      const f = hw > sb ? sphereCapFraction((hw - c.y) / e) : 0;
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

  /**
   * Ground contact of a fallen body: a penalty force on each segment's lowest point below the
   * seabed plane (normal stiffness and damping, Coulomb friction regularised below 5 cm/s), added
   * to `force`; the stiffness and the damping (normal and friction) go into the implicit step.
   */
  groundForces(ground: WaterPlane, sys: ImplicitSystem, force: Vector3): void {
    if (this.stance !== 'fallen') return;
    const M = RIDER_MODEL;
    const total = this.config.mass / M.bodyDensity;
    const v = this.velocity;
    let fn = 0, k = 0, c = 0;
    for (let i = 0; i < this.segments.length; i++) {
      const sg = this.segments[i];
      const sc = sg.center;
      const sb = ground.seabed + ground.seabedSlopeX * (sc.x - ground.x) + ground.seabedSlopeZ * (sc.z - ground.z);
      const ay = sg.axis.y;
      const pen = sb - (sc.y - Math.sqrt(sg.radius * sg.radius * (1 - ay * ay) + sg.halfLength * sg.halfLength * ay * ay));
      if (pen <= 0) continue;
      const w = sg.volume / total;
      const f = w * (M.groundStiffness * pen - M.groundDamping * v.y);
      if (f <= 0) continue;
      fn += f;
      k += w * M.groundStiffness;
      c += w * M.groundDamping;
    }
    if (fn <= 0) return;
    // friction: a damper of μ·fn/(|v_t| + 0.05) (implicit: far past the explicit limit at rest)
    const cf = (M.groundFriction * fn) / (Math.hypot(v.x, v.z) + 0.05);
    force.x -= cf * v.x;
    force.y += fn;
    force.z -= cf * v.z;
    sys.addRiderSpring(0, 1, 0, c, k);
    sys.addRiderSpring(1, 0, 0, cf, 0);
    sys.addRiderSpring(0, 0, 1, cf, 0);
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
    // arm efforts −1..1 (negative = back-paddle: the hand sweeps forward and pushes the board
    // back). Steering works the outside arm harder and the inside arm less — back-paddling it
    // for a sharp turn; steering without paddling pivots the board on the spot (outside arm
    // pulls, inside arm back-paddles: a yaw couple with little net thrust, so the fins do not
    // weathervane the board back onto its course). Turn right → left arm forward.
    const pL = clamp(paddle + steer * M.steerSplit, -1, 1);
    const pR = clamp(paddle - steer * M.steerSplit, -1, 1);
    const R = board.R;
    const nx = R[0], ny = R[3], nz = R[6]; // board +X in world
    const shoulderX = this.targetLocal.x + 0.45 * s;
    const shoulderY = this.targetLocal.y + 0.03;
    // one stroke clock; the arms are half a cycle apart and stroke only when that side has effort
    const effort = Math.max(Math.abs(pL), Math.abs(pR));
    if (effort > 0.02) {
      this.armPhaseL += (dt / cfg.strokePeriod) * (0.7 + 0.3 * effort);
      if (this.armPhaseL >= 1) this.armPhaseL -= 1;
    }
    this.armPhaseR = this.armPhaseL + 0.5 >= 1 ? this.armPhaseL - 0.5 : this.armPhaseL + 0.5;
    for (let arm = 0; arm < 2; arm++) {
      const pa = arm === 0 ? pL : pR;
      const p = Math.abs(pa);
      // stroke direction: +1 pull (hand sweeps back), −1 back-paddle (hand sweeps forward)
      const dir = pa < 0 ? -1 : 1;
      const side = arm === 0 ? -1 : 1;
      const phase = arm === 0 ? this.armPhaseL : this.armPhaseR;
      const hand = arm === 0 ? this.handL : this.handR;
      const pull = phase < 0.45 && p > 0.02;
      const sPull = pull ? phase / 0.45 : 0;
      const depth = pull ? 0.55 * s * Math.sin(Math.PI * sPull) : -0.15;
      const hx = shoulderX + dir * (pull ? 0.35 - 0.7 * sPull : -0.35 + 0.7 * ((phase - 0.45) / 0.55)) * s;
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
      // the hand reaches stroke speed quickly and holds it through the stroke
      const uh = cfg.handSpeed * (0.55 + 0.45 * p) * Math.sqrt(Math.sin(Math.PI * sPull));
      const du = uh - dir * vf;
      const mag = dir * 0.5 * RHO_WATER * M.handCdA * imm * (du > 0 ? du * du : -M.handBrake * du * du);
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
   * Balance bookkeeping after the step's solve: feet load and the standing fall checks (file
   * header). Returns true when the rider is past recovery (`tipOver` says which way).
   */
  updateBalance(dt: number, board: RigidBody): boolean {
    const M = RIDER_MODEL;
    this.tipOver = null;
    if (this.stance !== 'standing' && this.stance !== 'popping') {
      this.balanceRoll = 0;
      this.balancePitch = 0;
      this.balanceTimer = Math.max(this.balanceTimer - dt, 0);
      this.slipTimer = 0;
      return false;
    }
    // feet load: the leg force pushing the rider away from the deck, low-passed (HUD)
    const R = board.R;
    const load = Math.max(this.legForce.x * R[1] + this.legForce.y * R[4] + this.legForce.z * R[7], 0);
    this.feetLoad += (load - this.feetLoad) * (1 - Math.exp(-dt / M.balanceFilter));
    if (this.stance !== 'standing') {
      this.slipTimer = 0;
      return false;
    }
    // capture point past the support + recovery margin (still going) with the body tipped past
    // the edge of the feet by more than noReturn: nothing brings it back
    const exX = Math.abs(this.captureX) - (this.footX + M.recoverMargin);
    const exZ = Math.abs(this.captureZ) - (this.footZ + M.recoverMargin);
    const tip = Math.max(this.comHeight, 0.3 * this.scale) * Math.tan(M.noReturn);
    const outX = exX > 0 && Math.abs(this.comX) > this.footX + tip;
    const outZ = exZ > 0 && Math.abs(this.comZ) > this.footZ + tip;
    if (exX > 0 || exZ > 0) this.balanceTimer += dt;
    else this.balanceTimer = Math.max(this.balanceTimer - 2 * dt, 0);
    if (outX || outZ) {
      this.tipOver = outX && (!outZ || exX > exZ) ? (this.captureX > 0 ? 'forward' : 'back') : 'side';
      return true;
    }
    // the upper body thrown far past what the hips can hold (a leg caught in the water)
    const tLim = 2 * M.hipRange;
    if (Math.abs(this.trunkX) > tLim || Math.abs(this.trunkZ) > tLim) {
      this.tipOver = Math.abs(this.trunkX) > Math.abs(this.trunkZ) ? (this.trunkX > 0 ? 'forward' : 'back') : 'side';
      return true;
    }
    // the feet slip off a deck tilted past the friction angle from the leg force
    this.slipTimer = this.deckAngle > M.slipAngle && this.feetLoad > 0.3 * this.config.mass * GRAVITY ? this.slipTimer + dt : 0;
    if (this.slipTimer > M.slipTime) {
      this.tipOver = 'side';
      return true;
    }
    return false;
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
    // the nose driven deep under at speed: the board stops dead under the rider
    this.pearlTimer = noseDeckDepth > 0.15 && forwardSpeed > 2.5 ? this.pearlTimer + dt : Math.max(this.pearlTimer - dt, 0);
    if (this.pearlTimer > 0.3) return 'pearl';
    this.buriedTimer = deckDepth > 0.7 ? this.buriedTimer + dt : 0;
    if (this.buriedTimer > 0.4) return 'buried';
    if (legStretch > RIDER_MODEL.impactStretch) return 'impact';
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

/** Yaw rate of a board about its deck normal (+ = turning right), rad/s. */
export function boardYawRate(board: RigidBody): number {
  const R = board.R, w = board.angularVelocity;
  return -(w.x * R[1] + w.y * R[4] + w.z * R[7]);
}

function clamp(x: number, a: number, b: number): number {
  return x < a ? a : x > b ? b : x;
}
