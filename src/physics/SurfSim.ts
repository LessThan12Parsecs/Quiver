/**
 * SurfSim: board + rider + water, stepped at a fixed 1/240 s.
 *
 *   const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
 *   sim.reset(lineupSpawn(ocean));
 *   sim.step(1 / 240, input);   // sets ocean time, samples water, forces, implicit integration
 *   sim.time; sim.board; sim.rider; sim.telemetry; sim.debugForces
 *
 * One step:
 *  1. ocean.setTime(time); rider stance controller → target COM (board space) + body layout.
 *  2. Water: probe grid under the board (hull.sampleWater), one sample at the board COM (water
 *     state for telemetry and the rider's water plane), plus one at the rider when detached.
 *  3. Forces: gravity, hull + fins (BoardHull), body segments + paddling + ground contact
 *     (Rider), twist (upper body wound against the board: an internal torque pair), legs
 *     (attached: 3-axis spring-damper to the target point) or leash (fallen).
 *  4. Linearly implicit Euler on [v_board, ω_board, v_rider] (ImplicitSystem), implicit
 *     gyroscopic term, then positions/orientation.
 *  5. Balance (ankle torque from the effective leg force), wipeout checks (practice mode, wipeouts
 *     off: recovery in place instead), telemetry, pose.
 */
import { Matrix4, Quaternion, Vector3 } from 'three';
import type { OceanModel, SurfaceSample } from '../ocean/waveModel';
import { createSurfaceSample } from '../ocean/waveModel';
import { BoardHull, createHullDiagnostics, type HullDiagnostics } from './BoardHull';
import { BOARD_PRESETS, getBoardShape, type BoardShape, type BoardSpec } from './boardShape';
import { GRAVITY, PHYSICS_DT } from './constants';
import { ImplicitSystem } from './implicit';
import { RIDER_MODEL, Rider, boardYawRate, createWaterPlane, type RiderConfig, type Stance, type WaterPlane, type WipeoutReason } from './Rider';
import { RigidBody } from './RigidBody';

/** Player input for one step. Axes are −1..1 (0..1 for paddle/crouch); positive = right / forward. */
export interface SurfInput {
  /** Paddle effort 0..1 (prone). */
  paddle: number;
  /** Steer −1 (left) .. 1 (right) while prone: paddle harder on the opposite arm. */
  steer: number;
  /** Weight forward (+) / back (−) while standing (trim); slide on the board while prone. */
  leanForward: number;
  /** Lean toward the left (−) / right (+) rail. */
  leanSide: number;
  /** Crouch 0..1. */
  crouch: number;
  /** Twist (yaw torque) left (−) / right (+). */
  twist: number;
  /** Edge: start the pop-up (prone only). Set for one step. */
  popUp: boolean;
  /** Edge: reset at the lineup (prone, still, outside the peak). Set for one step. */
  reset: boolean;
}

export function createSurfInput(): SurfInput {
  return { paddle: 0, steer: 0, leanForward: 0, leanSide: 0, crouch: 0, twist: 0, popUp: false, reset: false };
}

export interface Spawn {
  x: number;
  z: number;
  /** Direction of the nose: angle from +X toward +Z, rad. */
  headingRad: number;
  stance: 'prone' | 'standing';
  /** Horizontal speed along the heading (ground frame), m/s. Across the heading the board moves
   * with the water (no side-slip through the fins at the start). */
  speed?: number;
  /** Informational: spawned on a wave face. */
  onWave?: boolean;
  /** Simulation time to start at (s); the current time when omitted. */
  time?: number;
}

/** Water state under the board (sampled at the board's centre of mass). */
export interface WaterState {
  height: number;
  slopeX: number;
  slopeZ: number;
  velX: number;
  velY: number;
  velZ: number;
  /** Still-water depth, m. */
  depth: number;
  /** Local swell height (crest to trough), m. */
  waveHeight: number;
  /** Closeness to breaking (0.8–1 steep makeable face). */
  fullness: number;
  /** Whitewater 0..1. */
  breaking: number;
  /** > 1: the wave has broken (here or further out). */
  ratio: number;
  /** Dominant swell waveform phase: 0 = crest, (0, π) = front face. */
  wavePhase: number;
  /** Dominant swell phase speed and direction. */
  phaseSpeed: number;
  dirX: number;
  dirZ: number;
}

export interface SurfTelemetry {
  time: number;
  stance: Stance;
  boardId: string;
  /** |v_board| (ground frame), m/s. */
  speed: number;
  /** Horizontal speed relative to the water under the board, m/s. */
  speedRelWater: number;
  /** Horizontal velocity component along the wave direction, m/s. */
  speedAlongWave: number;
  verticalSpeed: number;
  /** Nose direction angle from +X toward +Z, rad. */
  headingRad: number;
  /** Nose up (+), rad. */
  pitchRad: number;
  /** Right rail down (+), rad. */
  rollRad: number;
  /** Displaced board volume, litres. */
  submergedLiters: number;
  /** Displaced rider body volume, litres. */
  riderSubmergedLiters: number;
  /** Total hydrostatic force magnitude on the board, N. */
  buoyancyN: number;
  /** Vertical (world) part of the planing/impact pressure force, N. */
  planingLiftN: number;
  /** Total water force opposing the board's motion (all hull + fin + body forces), N. */
  dragN: number;
  /** Total fin force magnitude, N. */
  finForceN: number;
  railForceN: number;
  frictionN: number;
  /** Wetted bottom / deck area, m². */
  wettedArea: number;
  deckWettedArea: number;
  wettedLength: number;
  /** Largest fin angle of attack magnitude, deg, and whether any fin is stalled (attached < 0.5). */
  finAlphaDeg: number;
  finStalled: boolean;
  paddleThrustN: number;
  /** Leg force on the board, N, and the feet's normal load, N. */
  legForceN: number;
  feetLoadN: number;
  /** max(|balanceRoll|, |balancePitch|): > 1 means the feet are slipping. */
  balance: number;
  balanceRoll: number;
  balancePitch: number;
  /** Seconds over the ankle limit (wipeout at 0.35 s). */
  balanceTimer: number;
  wipeoutReason: WipeoutReason | null;
  /** Board deck depth below the water at the centre / nose, m (+ = under water). */
  deckDepth: number;
  noseDepth: number;
  water: WaterState;
  /** The board is moving with a wave (on the face/crest at ≥ 60 % of the wave speed). */
  riding: boolean;
  /** Seconds of continuous riding. */
  ridingTime: number;
}

export type DebugForceKind = 'hull' | 'fin' | 'leg' | 'body' | 'paddle' | 'leash' | 'gravity';

export interface DebugForce {
  origin: Vector3;
  /** Force vector, N (world). */
  vector: Vector3;
  kind: DebugForceKind;
}

const MAX_DEBUG_FORCES = 256;

export class SurfSim {
  readonly ocean: OceanModel;
  boardSpec: BoardSpec;
  shape: BoardShape;
  hull: BoardHull;
  readonly board = new RigidBody();
  readonly rider: Rider;
  /** Simulation time, s (also drives the ocean). */
  time = 0;
  readonly telemetry: SurfTelemetry;
  /** Force arrows of the last step (first `debugForceCount` entries; filled when `collectDebug`). */
  readonly debugForces: DebugForce[] = [];
  debugForceCount = 0;
  collectDebug = false;
  /** Hull force totals of the last step. */
  readonly diag: HullDiagnostics;
  /** Test/tool hooks: extra world forces on the rider COM and the board COM (N). */
  readonly extraRiderForce = new Vector3();
  readonly extraBoardForce = new Vector3();
  /** Test/tool hook: extra world torque on the board (N·m). */
  readonly extraBoardTorque = new Vector3();
  /** Water moment on the board about the feet, around its long axis (+ = right rail down), N·m. */
  hullRollMoment = 0;
  /** The trunk's hip-rotation angular momentum (world) at the end of the last step, N·m·s, and
   * the leg force pair's moment about u then (the board takes its opposite the next step). */
  private readonly trunkMomentum = new Vector3();
  private readonly legYawTorque = new Vector3();
  /** The last spawn passed to reset(). */
  lastSpawn: Spawn | null = null;
  /** Practice mode (rider.config.wipeouts off): times the rider was put back on their feet
   * instead of falling, and the fall that was caught. */
  recoveries = 0;
  lastRecovery: WipeoutReason | null = null;

  private readonly sys = new ImplicitSystem();
  private readonly u = new Float64Array(9);
  private readonly underSample: SurfaceSample = createSurfaceSample();
  private readonly riderSample: SurfaceSample = createSurfaceSample();
  private readonly riderPlane: WaterPlane = createWaterPlane();
  private readonly bodyInertia = new Vector3();
  private readonly riderExt = new Vector3();
  private readonly riderTorque = new Vector3();
  private readonly tmp = new Vector3();
  private readonly tmp2 = new Vector3();
  private readonly qTmp = new Quaternion();
  private readonly mTmp = new Matrix4();
  private readonly tmp3 = new Vector3();
  private readonly leashLocal = new Vector3();
  private readonly prevVel = new Vector3();
  /** Leg spring axes (3 × world unit vectors) and their explicit forces (scratch). */
  private readonly legAxes = new Float64Array(9);
  private readonly legF = new Float64Array(3);
  private readonly legD = new Float64Array(3);
  private readonly legV = new Float64Array(3);
  private riderDampingC = 0;
  private centreCells: number[] = [];
  private noseCells: number[] = [];

  constructor(ocean: OceanModel, boardSpec: BoardSpec = BOARD_PRESETS.funboard, riderConfig: Partial<RiderConfig> = {}) {
    this.ocean = ocean;
    this.time = ocean.time;
    this.boardSpec = boardSpec;
    this.shape = getBoardShape(boardSpec);
    this.hull = new BoardHull(this.shape);
    this.diag = createHullDiagnostics(this.shape.fins.length);
    this.rider = new Rider(riderConfig);
    for (let i = 0; i < MAX_DEBUG_FORCES; i++) this.debugForces.push({ origin: new Vector3(), vector: new Vector3(), kind: 'hull' });
    this.telemetry = createTelemetry();
    this.configureBoard();
    this.reset({ x: -75, z: 5, headingRad: 0, stance: 'prone', speed: 0 });
  }

  /** Swap the board (keeps the rider's position/stance; re-settles on the water). */
  setBoard(spec: BoardSpec): void {
    this.boardSpec = spec;
    this.shape = getBoardShape(spec);
    this.hull = new BoardHull(this.shape);
    (this as { diag: HullDiagnostics }).diag = createHullDiagnostics(this.shape.fins.length);
    this.configureBoard();
    const p = this.board.position;
    const heading = Math.atan2(this.board.R[6], this.board.R[0]);
    const stance = this.rider.stance === 'standing' || this.rider.stance === 'popping' ? 'standing' : 'prone';
    this.reset({ x: p.x, z: p.z, headingRad: heading, stance, speed: Math.hypot(this.board.velocity.x, this.board.velocity.z), time: this.time });
  }

  private configureBoard(): void {
    const s = this.shape;
    this.board.setMass(s.mass);
    this.board.setInertia(s.massProperties.inertia.x, s.massProperties.inertia.y, s.massProperties.inertia.z);
    this.rider.setBoard(s);
    this.leashLocal.set(s.x(0.02), s.deckY(0.02, 0), 0);
    // cells used for the deck depth (centre) and pearling (nose) checks
    const h = this.hull;
    this.centreCells = [];
    this.noseCells = [];
    for (let k = 0; k < h.n; k++) {
      const u = h.cu[k];
      const inner = Math.abs(h.cv[k]) < 0.5;
      if (inner && Math.abs(u - 0.5) < 0.5 / h.nu + 1e-9) this.centreCells.push(k);
      if (inner && u > 0.86 && u < 0.95) this.noseCells.push(k);
    }
  }

  /** Board inertia + the attached rider's own body inertia (both diagonal in board axes). */
  private updateBoardInertia(conserve = false): void {
    const I = this.shape.massProperties.inertia;
    const r = this.rider;
    r.bodyInertia(this.bodyInertia);
    const bi = this.bodyInertia;
    const board = this.board;
    const ix = I.x + bi.x, iy = I.y + bi.y, iz = I.z + bi.z;
    const old = board.inertia;
    if (conserve && (ix !== old.x || iy !== old.y || iz !== old.z)) {
      // the body changed shape on the board (getting up): its angular momentum is kept. Standing
      // up, the crouched body's pitch/roll spin goes on as the trunk's hip rotation (its yaw is
      // the twist rotor, which starts turning with the board); otherwise the board + body turn
      // faster or slower for the new inertia
      const w = board.worldDirToLocal(this.tmp.copy(board.angularVelocity), this.tmp2);
      if (r.stance === 'standing') {
        const lx = (old.x - ix) * w.x, lz = (old.z - iz) * w.z;
        const L = board.localDirToWorld(this.tmp.set(lx, 0, lz), this.tmp3);
        const It = r.trunkInertia();
        r.trunkVZ += (L.x * r.frameX.x + L.y * r.frameX.y + L.z * r.frameX.z) / It;
        r.trunkVX -= (L.x * r.frameZ.x + L.y * r.frameZ.y + L.z * r.frameZ.z) / It;
        const X = r.frameX, Z = r.frameZ;
        this.trunkMomentum.set(It * (r.trunkVZ * X.x - r.trunkVX * Z.x), It * (r.trunkVZ * X.y - r.trunkVX * Z.y), It * (r.trunkVZ * X.z - r.trunkVX * Z.z));
      } else {
        w.set((w.x * old.x) / ix, (w.y * old.y) / iy, (w.z * old.z) / iz);
        board.localDirToWorld(w, board.angularVelocity);
      }
    }
    board.setInertia(ix, iy, iz);
  }

  /** Place board and rider for a spawn and clear all transient state. */
  reset(spawn: Spawn): void {
    this.lastSpawn = { ...spawn };
    if (spawn.time !== undefined) this.time = spawn.time;
    const ocean = this.ocean;
    ocean.setTime(this.time);
    const rider = this.rider;
    rider.resetState(spawn.stance);
    this.hull.resetHistory();
    const b = this.board;
    const s = ocean.sample(spawn.x, spawn.z, this.underSample);
    // orientation: flat on the water surface (deck normal = surface normal, so on a face the board
    // starts banked with it), nose along the heading, plus a little nose-up trim when moving
    const speed = spawn.speed ?? 0;
    const ch = Math.cos(spawn.headingRad);
    const sh = Math.sin(spawn.headingRad);
    const ey = this.tmp3.set(-s.slopeX, 1, -s.slopeZ).normalize();
    const hn = ch * ey.x + sh * ey.z;
    const ex = this.tmp.set(ch - hn * ey.x, -hn * ey.y, sh - hn * ey.z).normalize();
    const ez = this.tmp2.crossVectors(ex, ey);
    this.mTmp.makeBasis(ex, ey, ez);
    const q = b.quaternion.setFromRotationMatrix(this.mTmp);
    if (speed > 2) {
      b.updateDerived();
      this.qTmp.setFromAxisAngle(this.tmp.set(b.R[2], b.R[5], b.R[8]), 0.05);
      q.premultiply(this.qTmp);
    }
    b.quaternion.normalize();
    b.position.set(spawn.x, s.height, spawn.z);
    b.updateDerived();
    // across the heading the board drifts with the water (a fin at an angle of attack at the
    // spawn would kick the yaw)
    const wLat = -sh * s.velX + ch * s.velZ;
    const vx = speed * ch - wLat * sh;
    const vz = speed * sh + wLat * ch;
    b.velocity.set(vx, s.dEtaDt + vx * s.slopeX + vz * s.slopeZ, vz);
    b.angularVelocity.set(0, 0, 0);
    this.updateBoardInertia();
    // what the balance senses: a body carried by the water feels the surface pressure gradient,
    // gravity along the surface normal (g cosθ) — start the apparent up there
    {
      const nx = -s.slopeX, nz = -s.slopeZ;
      const inv = 1 / (1 + nx * nx + nz * nz);
      rider.accelFiltered.set(GRAVITY * nx * inv, GRAVITY * inv - GRAVITY, GRAVITY * nz * inv);
    }
    // stance target (twice: the standing body is laid out around the actual COM, so place it at
    // the target first)
    rider.waterSlopeX = s.slopeX;
    rider.waterSlopeZ = s.slopeZ;
    rider.waterVelX = s.velX;
    rider.waterVelY = s.velY;
    rider.waterVelZ = s.velZ;
    const zero = createSurfInput();
    rider.updateTargets(PHYSICS_DT, zero, b);
    b.localToWorld(rider.targetLocal, rider.position);
    b.pointVelocity(rider.position, rider.velocity);
    rider.updateTargets(PHYSICS_DT, zero, b);
    rider.targetVelLocal.set(0, 0, 0);
    // vertical equilibrium (hydrostatics + planing at the spawn velocity) by bisection
    let lo = s.height - 1.6;
    let hi = s.height + 0.6;
    for (let it = 0; it < 32; it++) {
      const mid = 0.5 * (lo + hi);
      if (this.netVerticalForce(mid) > 0) lo = mid;
      else hi = mid;
    }
    b.position.y = 0.5 * (lo + hi);
    // settle what the balance senses: the acceleration the water gives board + rider at the spawn
    // (a fast board on a gentle face decelerates; the rider starts leaning against it), then
    // place the rider for it
    if (rider.stance === 'standing') {
      const fy = this.netVerticalForce(b.position.y);
      const m = b.mass + rider.config.mass;
      const lim = 0.6 * GRAVITY;
      rider.accelFiltered.set(
        Math.max(-lim, Math.min(lim, (this.sys.Q[0] + this.riderExt.x) / m)),
        Math.max(-lim, Math.min(lim, fy / m)),
        Math.max(-lim, Math.min(lim, (this.sys.Q[2] + this.riderExt.z) / m)),
      );
      rider.updateTargets(PHYSICS_DT, zero, b);
      b.localToWorld(rider.targetLocal, rider.position);
      b.pointVelocity(rider.position, rider.velocity);
      rider.updateTargets(PHYSICS_DT, zero, b);
    }
    b.position.y = 0.5 * (lo + hi);
    b.updateDerived();
    b.localToWorld(rider.targetLocal, rider.position);
    b.pointVelocity(rider.position, rider.velocity);
    // start with the legs carrying the rider's weight (static deflection), not from rest length
    rider.position.y -= (rider.config.mass * GRAVITY) / rider.legStiffness();
    this.prevVel.copy(b.velocity);
    rider.restartEstimators();
    this.extraBoardForce.set(0, 0, 0);
    this.extraBoardTorque.set(0, 0, 0);
    this.extraRiderForce.set(0, 0, 0);
    this.telemetry.ridingTime = 0;
    this.telemetry.wipeoutReason = null;
    rider.updatePose(b);
    this.fillTelemetry(0);
  }

  /**
   * Practice mode: instead of falling, the rider gets back up where they are — board upright on
   * the water along its heading at its current speed (board + rider COM), rider standing (prone if
   * they were prone), all transient state cleared. `lastSpawn` is kept (R/T still mean the same).
   */
  private recover(reason: WipeoutReason): void {
    const b = this.board;
    const r = this.rider;
    const mb = b.mass;
    const mr = r.config.mass;
    const hx = b.R[0], hz = b.R[6];
    const hn = Math.hypot(hx, hz) || 1;
    const vAlong = ((mb * b.velocity.x + mr * r.velocity.x) * hx + (mb * b.velocity.z + mr * r.velocity.z) * hz) / ((mb + mr) * hn);
    const spawn = this.lastSpawn;
    this.reset({
      x: b.position.x,
      z: b.position.z,
      headingRad: Math.atan2(hz, hx),
      stance: r.stance === 'prone' ? 'prone' : 'standing',
      speed: vAlong,
      time: this.time,
    });
    this.lastSpawn = spawn;
    this.recoveries++;
    this.lastRecovery = reason;
  }

  /** Net upward force on board + rider (static pose at COM height y, spawn velocity). */
  private netVerticalForce(y: number): number {
    const b = this.board;
    const r = this.rider;
    b.position.y = y;
    b.updateDerived();
    b.localToWorld(r.targetLocal, r.position);
    b.pointVelocity(r.position, r.velocity);
    this.hull.sampleWater(this.ocean, b);
    const sys = this.sys;
    sys.clear();
    this.setBaseMass(b.mass, r.config.mass);
    this.hull.resetWaterHistory();
    this.hull.computeForces(b, sys, this.diag);
    r.layoutSegments(b);
    this.makePlane(this.underSample, b.position.x, b.position.z, this.riderPlane);
    this.riderExt.set(0, 0, 0);
    this.riderTorque.set(0, 0, 0);
    r.segmentForces(this.riderPlane, b, this.riderExt, this.riderTorque);
    return sys.Q[1] + this.riderExt.y - (b.mass + r.config.mass) * GRAVITY;
  }

  private makePlane(s: SurfaceSample, x: number, z: number, out: WaterPlane): void {
    out.x = x;
    out.z = z;
    out.height = s.height;
    out.slopeX = s.slopeX;
    out.slopeZ = s.slopeZ;
    out.velX = s.velX;
    out.velY = s.velY;
    out.velZ = s.velZ;
    out.seabed = -s.depth;
    out.seabedSlopeX = 0;
    out.seabedSlopeZ = 0;
  }

  /** DEBUG: roll torque contributions (about the board's long axis through its COM) of the last step. */
  readonly dbgRoll: Record<string, number> = {};
  private dbgQ(): number {
    const R = this.board.R;
    return this.sys.Q[3] * R[0] + this.sys.Q[4] * R[3] + this.sys.Q[5] * R[6];
  }

  /** Advance the simulation by dt (intended: 1/240 s). */
  step(dt: number, input: SurfInput): void {
    if (input.reset) {
      this.reset(lineupSpawnAt(this.time));
      return;
    }
    const b = this.board;
    const r = this.rider;
    const ocean = this.ocean;
    if (input.popUp) r.startPopUp();
    ocean.setTime(this.time);
    b.updateDerived();
    r.waterSlopeX = this.underSample.slopeX;
    r.waterSlopeZ = this.underSample.slopeZ;
    r.waterVelX = this.underSample.velX;
    r.waterVelY = this.underSample.velY;
    r.waterVelZ = this.underSample.velZ;
    r.updateTargets(dt, input, b);
    this.updateBoardInertia(true);

    // --- water
    this.hull.sampleWater(ocean, b);
    const under = ocean.sample(b.position.x, b.position.z, this.underSample);
    const plane = this.riderPlane;
    if (r.attached) this.makePlane(under, b.position.x, b.position.z, plane);
    else {
      const rx = r.position.x, rz = r.position.z;
      this.makePlane(ocean.sample(rx, rz, this.riderSample), rx, rz, plane);
      // seabed slope under the fallen body (ground contact on the beach face), central over 1 m
      plane.seabedSlopeX = ocean.depthAt(rx - 0.5, rz) - ocean.depthAt(rx + 0.5, rz);
      plane.seabedSlopeZ = ocean.depthAt(rx, rz - 0.5) - ocean.depthAt(rx, rz + 0.5);
    }
    r.layoutSegments(b);

    // --- forces (the hull adds its added mass to M, so M is set first)
    const sys = this.sys;
    sys.clear();
    const mb = b.mass;
    const mr = r.config.mass;
    this.setBaseMass(mb, mr);
    sys.addBoardForce(this.extraBoardForce.x, this.extraBoardForce.y - mb * GRAVITY, this.extraBoardForce.z, 0, 0, 0);
    sys.addBoardTorque(this.extraBoardTorque.x, this.extraBoardTorque.y, this.extraBoardTorque.z);
    this.hull.recordCellForces = this.collectDebug;
    this.hull.waterDt = dt;
    const q0 = sys.Q[0], q1 = sys.Q[1], q2 = sys.Q[2], q3 = sys.Q[3], q4 = sys.Q[4], q5 = sys.Q[5];
    let dq = this.dbgQ();
    this.hull.computeForces(b, sys, this.diag);
    this.dbgRoll.hull = this.dbgQ() - dq;
    {
      // water (hull + fins) moment about the feet line, around the board's long axis (+ = right
      // rail down): what the rider's ankles must hold to keep the board at its bank
      const fx = sys.Q[0] - q0, fy = sys.Q[1] - q1, fz = sys.Q[2] - q2;
      const R = b.R;
      const fm = r.feetMidLocal;
      const rx = R[0] * fm.x + R[1] * fm.y + R[2] * fm.z;
      const ry = R[3] * fm.x + R[4] * fm.y + R[5] * fm.z;
      const rz = R[6] * fm.x + R[7] * fm.y + R[8] * fm.z;
      const tx = sys.Q[3] - q3 - (ry * fz - rz * fy);
      const ty = sys.Q[4] - q4 - (rz * fx - rx * fz);
      const tz = sys.Q[5] - q5 - (rx * fy - ry * fx);
      this.hullRollMoment = tx * R[0] + ty * R[3] + tz * R[6];
      r.hullRollMoment = this.hullRollMoment;
    }
    const ext = this.riderExt.set(this.extraRiderForce.x, this.extraRiderForce.y - mr * GRAVITY, this.extraRiderForce.z);
    const tq = this.riderTorque.set(0, 0, 0);
    this.riderDampingC = r.segmentForces(plane, b, ext, tq);
    const segFx = ext.x - this.extraRiderForce.x, segFy = ext.y - this.extraRiderForce.y + mr * GRAVITY, segFz = ext.z - this.extraRiderForce.z;
    r.paddleForces(dt, input, plane, b, ext, tq);
    r.groundForces(plane, sys, ext);
    sys.addRiderForce(ext.x, ext.y, ext.z);
    sys.addRiderDamping(this.riderDampingC);
    // the water's force on an attached body segment acts on the rider's COM; its moment about the
    // COM goes to the board, which carries the body (the legs stand on it): the pair is exactly
    // the force at the segment, so momentum and angular momentum are those of the external force
    dq = this.dbgQ();
    if (r.attached && !(globalThis as { NO_SEG_TQ?: boolean }).NO_SEG_TQ) sys.addBoardTorque(tq.x, tq.y, tq.z);
    this.dbgRoll.seg = this.dbgQ() - dq;
    dq = this.dbgQ();
    // the trunk's hip rotation is a lean in the balance frame, which turns with the board and the
    // apparent up: turning its angular momentum with the frame takes a torque through the hip, so
    // the board takes the opposite of that change
    if (r.stance === 'standing') {
      const It = r.trunkInertia();
      const X = r.frameX, Z = r.frameZ;
      const lx = It * (r.trunkVZ * X.x - r.trunkVX * Z.x);
      const ly = It * (r.trunkVZ * X.y - r.trunkVX * Z.y);
      const lz = It * (r.trunkVZ * X.z - r.trunkVX * Z.z);
      const p = this.trunkMomentum;
      const y = this.legYawTorque;
      sys.addBoardTorque(-(lx - p.x) / dt - y.x, -(ly - p.y) / dt - y.y, -(lz - p.z) / dt - y.z);
    }
    // twist (standing): the trunk muscles wind the upper body against the board about the deck
    // normal n — an internal torque pair (+ input: upper body left, board nose right). The upper
    // body's yaw momentum turns with the board's tilt: the board carries ω × h of it.
    if (r.stance === 'standing') {
      const R = b.R;
      const tau = r.twistUpdate(dt, input.twist, boardYawRate(b));
      const w = b.angularVelocity;
      const h = -r.twistInertia() * r.twistSpin; // upper body yaw momentum along +n
      sys.addBoardTorque(
        R[1] * tau - h * (w.y * R[7] - w.z * R[4]),
        R[4] * tau - h * (w.z * R[1] - w.x * R[7]),
        R[7] * tau - h * (w.x * R[4] - w.y * R[1]),
      );
    }

    this.dbgRoll.trunkTwist = this.dbgQ() - dq;
    // --- gyroscopic term (implicit, board + attached body inertia), then velocity vector
    b.applyGyroscopic(dt);
    const u = this.u;
    u[0] = b.velocity.x; u[1] = b.velocity.y; u[2] = b.velocity.z;
    u[3] = b.angularVelocity.x; u[4] = b.angularVelocity.y; u[5] = b.angularVelocity.z;
    u[6] = r.velocity.x; u[7] = r.velocity.y; u[8] = r.velocity.z;

    // --- legs (attached) or leash (fallen)
    let legStretch = 0;
    const R = b.R;
    if (r.stance === 'standing') {
      dq = this.dbgQ();
      legStretch = this.standingLegs(dt, u);
      this.dbgRoll.legs = this.dbgQ() - dq;
    } else if (r.attached) {
      const k0 = r.legStiffness();
      const c0 = r.legDamping();
      const kLat = k0;
      const cLat = c0;
      const tl = r.targetLocal;
      const rx = R[0] * tl.x + R[1] * tl.y + R[2] * tl.z;
      const ry = R[3] * tl.x + R[4] * tl.y + R[5] * tl.z;
      const rz = R[6] * tl.x + R[7] * tl.y + R[8] * tl.z;
      const dx = r.position.x - (b.position.x + rx);
      const dy = r.position.y - (b.position.y + ry);
      const dz = r.position.z - (b.position.z + rz);
      legStretch = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const tv = r.targetVelLocal;
      // Spring axes. Standing: along the leg (feet → target COM), across it in the board's lateral
      // direction (roll: toes/heels) and fore/aft (pitch: front/back foot). The two "ankle" axes
      // are torque-limited: the foot pressure centre can only move within the feet, so the
      // moment of the leg force about the feet is at most lever × load (+ grip). Beyond that the
      // force saturates and the body tips over the edge of the feet instead of the board
      // supplying an impossible torque. Prone: the board axes, unlimited (lying on the deck).
      const w = r.standWeight();
      const ax3 = this.legAxes;
      const fm = r.feetMidLocal;
      let hLeg = 0;
      if (w > 0) {
        let lx = tl.x - fm.x, ly = tl.y - fm.y, lz = tl.z - fm.z;
        hLeg = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
        lx /= hLeg; ly /= hLeg; lz /= hLeg;
        // lateral ⊥ leg (from the board z axis), fore/aft = lateral × leg … all in board axes
        let ex = -lz * lx, ey = -lz * ly, ez = 1 - lz * lz;
        const en = Math.sqrt(ex * ex + ey * ey + ez * ez) || 1;
        ex /= en; ey /= en; ez /= en;
        const px = ey * lz - ez * ly, py = ez * lx - ex * lz, pz = ex * ly - ey * lx;
        // world directions: 0 = leg, 1 = fore/aft (pitch ankle), 2 = lateral (roll ankle)
        ax3[0] = R[0] * lx + R[1] * ly + R[2] * lz; ax3[1] = R[3] * lx + R[4] * ly + R[5] * lz; ax3[2] = R[6] * lx + R[7] * ly + R[8] * lz;
        ax3[3] = R[0] * px + R[1] * py + R[2] * pz; ax3[4] = R[3] * px + R[4] * py + R[5] * pz; ax3[5] = R[6] * px + R[7] * py + R[8] * pz;
        ax3[6] = R[0] * ex + R[1] * ey + R[2] * ez; ax3[7] = R[3] * ex + R[4] * ey + R[5] * ez; ax3[8] = R[6] * ex + R[7] * ey + R[8] * ez;
      } else {
        // board axes: 0 = y (normal), 1 = x (along), 2 = z (lateral)
        ax3[0] = R[1]; ax3[1] = R[4]; ax3[2] = R[7];
        ax3[3] = R[0]; ax3[4] = R[3]; ax3[5] = R[6];
        ax3[6] = R[2]; ax3[7] = R[5]; ax3[8] = R[8];
      }
      // target velocity (board local) in world
      const tvx = R[0] * tv.x + R[1] * tv.y + R[2] * tv.z;
      const tvy = R[3] * tv.x + R[4] * tv.y + R[5] * tv.z;
      const tvz = R[6] * tv.x + R[7] * tv.y + R[8] * tv.z;
      const fa = this.legF;
      for (let axis = 0; axis < 3; axis++) {
        const ax = ax3[3 * axis], ay = ax3[3 * axis + 1], az = ax3[3 * axis + 2];
        const d = ax * dx + ay * dy + az * dz;
        const tdot = ax * tvx + ay * tvy + az * tvz;
        const v = ImplicitSystem.linkVelocity(u, ax, ay, az, rx, ry, rz) - tdot;
        const k = axis === 2 ? kLat : k0;
        const c = axis === 2 ? cLat : c0;
        fa[axis] = -k * (d - dt * tdot) - c * v;
        this.legD[axis] = d;
        this.legV[axis] = v;
      }
      let satRoll = false, satPitch = false;
      if (w > 0) {
        // popping up: load on the feet (leg force pushing the rider away from the board) and the
        // torque the feet can hold about their midpoint (CoP within the feet), blended in from the
        // hands-and-feet crouch (unlimited) as the rider gets up
        const load = Math.max(fa[0], 0);
        const capR = r.supportZ * load * w + (1 - w) * 1e5;
        const capP = r.supportX * load * w + (1 - w) * 1e5;
        if (Math.abs(fa[2]) * hLeg > capR) { fa[2] = Math.sign(fa[2]) * capR / hLeg; satRoll = true; }
        if (Math.abs(fa[1]) * hLeg > capP) { fa[1] = Math.sign(fa[1]) * capP / hLeg; satPitch = true; }
      }
      r.ankleSaturated = satRoll || satPitch;
      for (let axis = 0; axis < 3; axis++) {
        const ax = ax3[3 * axis], ay = ax3[3 * axis + 1], az = ax3[3 * axis + 2];
        const k = axis === 2 ? kLat : k0;
        const c = axis === 2 ? cLat : c0;
        const sat = (axis === 2 && satRoll) || (axis === 1 && satPitch);
        // a saturated ankle transmits a constant (capped) force: the foot rolls on its edge, so
        // that axis has no stiffness or damping this step — up to the legs' reach (an impact
        // stretch, where the rider falls or practice mode recovers): past it the spring is back,
        // so a rider hanging off a saturated ankle can never be dragged away from the board
        const d = this.legD[axis];
        const over = sat ? Math.abs(d) - RIDER_MODEL.impactStretch : 0;
        let f = fa[axis];
        if (over > 0) f -= k * (Math.sign(d) * over - dt * (ax * tvx + ay * tvy + az * tvz)) + c * this.legV[axis];
        sys.addRiderForce(ax * f, ay * f, az * f);
        sys.addBoardForce(-ax * f, -ay * f, -az * f, rx, ry, rz);
        if (!sat || over > 0) sys.addLink(ax, ay, az, rx, ry, rz, c, k);
      }
    } else {
      const ll = this.leashLocal;
      const rx = R[0] * ll.x + R[1] * ll.y + R[2] * ll.z;
      const ry = R[3] * ll.x + R[4] * ll.y + R[5] * ll.z;
      const rz = R[6] * ll.x + R[7] * ll.y + R[8] * ll.z;
      const dx = r.position.x - (b.position.x + rx);
      const dy = r.position.y - (b.position.y + ry);
      const dz = r.position.z - (b.position.z + rz);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const slack = r.config.leashLength;
      if (dist > slack) {
        const ax = dx / dist, ay = dy / dist, az = dz / dist;
        const v = ImplicitSystem.linkVelocity(u, ax, ay, az, rx, ry, rz);
        const ks = RIDER_MODEL.leashStiffness;
        const kd = v > 0 ? RIDER_MODEL.leashDamping : 0;
        const f = Math.max(-ks * (dist - slack) - kd * v, -4000);
        sys.addRiderForce(ax * f, ay * f, az * f);
        sys.addBoardForce(-ax * f, -ay * f, -az * f, rx, ry, rz);
        sys.addLink(ax, ay, az, rx, ry, rz, kd, ks);
        if (this.collectDebug) this.pushDebug(r.position, -ax * f, -ay * f, -az * f, 'leash');
      }
    }

    // --- solve
    const du = sys.solve(dt, u);
    b.velocity.set(u[0] + du[0], u[1] + du[1], u[2] + du[2]);
    b.angularVelocity.set(u[3] + du[3], u[4] + du[4], u[5] + du[5]);
    r.velocity.set(u[6] + du[6], u[7] + du[7], u[8] + du[8]);
    // effective leg force on the rider over this step (for the balance model)
    if (r.attached) {
      const cR = this.riderDampingC;
      r.legForce.set(
        (mr * du[6]) / dt - (ext.x - cR * du[6]),
        (mr * du[7]) / dt - (ext.y - cR * du[7]),
        (mr * du[8]) / dt - (ext.z - cR * du[8]),
      );
    } else {
      r.legForce.set(0, 0, 0);
    }
    // apparent gravity for the balance reflex: acceleration of the combined board + rider COM
    // (external forces only; the board alone is dominated by the internal leg forces)
    if (r.attached) {
      const im = 1 / ((mb + mr) * dt);
      r.filterAcceleration((mb * du[0] + mr * du[6]) * im, (mb * du[1] + mr * du[7]) * im, (mb * du[2] + mr * du[8]) * im, dt);
    }

    if (r.stance === 'standing') {
      r.integrateTwist(dt, boardYawRate(b));
      r.integrateTrunk(dt);
      const It = r.trunkInertia();
      const X = r.frameX, Z = r.frameZ;
      this.trunkMomentum.set(It * (r.trunkVZ * X.x - r.trunkVX * Z.x), It * (r.trunkVZ * X.y - r.trunkVX * Z.y), It * (r.trunkVZ * X.z - r.trunkVX * Z.z));
      this.legYawTorque.copy(r.frameU).multiplyScalar(r.legYawMoment);
    } else {
      this.trunkMomentum.set(0, 0, 0);
      this.legYawTorque.set(0, 0, 0);
    }

    // --- positions
    b.integratePosition(dt);
    r.position.addScaledVector(r.velocity, dt);
    this.time += dt;

    // --- balance and wipeouts
    const deckDepth = this.deckDepthOf(this.centreCells);
    const noseDepth = this.deckDepthOf(this.noseCells);
    if (r.attached) {
      const lost = r.updateBalance(dt, b);
      const vrx = b.velocity.x - under.velX, vry = b.velocity.y - under.velY, vrz = b.velocity.z - under.velZ;
      const fwd = vrx * b.R[0] + vry * b.R[3] + vrz * b.R[6];
      // pitched over the front with the nose in the water: the nose caught (pearl)
      const tip: WipeoutReason | null = lost ? (r.tipOver === 'forward' && noseDepth > 0 ? 'pearl' : 'balance') : null;
      const reason = r.checkWipeout(dt, b.R[4], noseDepth, fwd, deckDepth, legStretch) ?? tip;
      if (reason && r.config.wipeouts) r.fall(reason, b);
      else if (reason) {
        // practice (no wipeouts): a body past recovery, a flipped board or over-stretched legs have
        // no attached state to carry on from (the legs would hang the rider off a saturated
        // ankle): put the rider back on their feet in place. A pearl or a buried board is ridden on.
        const caught = reason === 'flipped' ? reason : (tip ?? (legStretch > RIDER_MODEL.impactStretch ? 'impact' : null));
        if (caught) {
          this.recover(caught);
          return;
        }
      }
    }
    this.telemetry.deckDepth = deckDepth;
    this.telemetry.noseDepth = noseDepth;
    r.updatePose(b);
    this.fillTelemetry(dt);

    if (this.collectDebug) this.collectDebugForces(segFx, segFy, segFz);
  }

  /**
   * Standing legs (Rider, "Standing balance"): the force on the rider runs from the centre of
   * pressure p on the deck through the COM. Along the apparent up u a leg spring holds the COM
   * height (the feet only push); across u the load N tilts it, F⟂ = (N/h)(r⟂ − p), with p from the
   * rider's capture-point reflex. While p is inside the usable support that law is a spring-damper
   * on the COM relative to the feet (implicit, linearised at the feet point); at the edge the CoP
   * is fixed there and the body tips over the edge of the feet (explicit, unstable like any
   * pendulum). The reaction acts on the board at the CoP (where the force line meets the deck):
   * equal and opposite, no free torque. The links are linearised at the ankles (feet midpoint on
   * the stringer): the board rolls and pitches under the feet without stretching the legs. Returns
   * the leg-axis stretch (impact check).
   */
  private standingLegs(dt: number, u: Float64Array): number {
    const b = this.board;
    const r = this.rider;
    const sys = this.sys;
    const R = b.R;
    const fm = r.feetMidLocal;
    const U = r.frameU;
    const X = r.frameX;
    const Z = r.frameZ;
    // ankles (feet midpoint on the stringer) and the centre of pressure, relative to the board COM
    const fx = R[0] * fm.x + R[1] * fm.y + R[2] * fm.z;
    const fy = R[3] * fm.x + R[4] * fm.y + R[5] * fm.z;
    const fz = R[6] * fm.x + R[7] * fm.y + R[8] * fm.z;
    const px = fx + r.copX * X.x + r.copZ * Z.x;
    const py = fy + r.copX * X.y + r.copZ * Z.y;
    const pz = fz + r.copX * X.z + r.copZ * Z.z;
    // leg axis: COM height along u toward the (crouch) target; unilateral
    const k0 = r.legStiffness();
    const c0 = r.legDamping();
    const h = r.comHeight;
    const hd = r.heightTarget;
    const hdot = r.heightRate;
    // (every leg force — the explicit part and the implicit increments — acts on the board at the
    // centre of pressure, where the feet press: the moment it has there about the body is what the
    // hip rotation takes up, so the pair stays equal and opposite)
    const vu = ImplicitSystem.linkVelocity(u, U.x, U.y, U.z, px, py, pz) - hdot;
    let N = -k0 * (h - hd - dt * hdot) - c0 * vu;
    const contact = N > 0;
    if (!contact) N = 0;
    // across u: CoP-limited tilt of the leg force
    const nh = N / Math.max(h, 0.3 * r.scale);
    const w0 = r.omega0;
    const k = r.captureGain;
    let Fx = 0;
    let Fz = 0;
    if (contact) {
      const K = (nh * k) / w0;
      const C = (nh * (w0 + k)) / (w0 * w0);
      // (relative to the aim, which moves at aimVel)
      if (!r.copSatX) {
        Fx = -K * (r.comX - r.aimX - dt * r.aimVelX) - C * r.comVX;
        sys.addLink(X.x, X.y, X.z, px, py, pz, C, K);
      } else Fx = nh * (r.comX - r.cmpX);
      if (!r.copSatZ) {
        Fz = -K * (r.comZ - r.aimZ - dt * r.aimVelZ) - C * r.comVZ;
        sys.addLink(Z.x, Z.y, Z.z, px, py, pz, C, K);
      } else Fz = nh * (r.comZ - r.cmpZ);
      sys.addLink(U.x, U.y, U.z, px, py, pz, c0, k0);
    }
    const Fwx = N * U.x + Fx * X.x + Fz * Z.x;
    const Fwy = N * U.y + Fx * X.y + Fz * Z.y;
    const Fwz = N * U.z + Fx * X.z + Fz * Z.z;
    sys.addRiderForce(Fwx, Fwy, Fwz);
    sys.addBoardForce(-Fwx, -Fwy, -Fwz, px, py, pz);
    return Math.abs(h - hd);
  }

  /** Rigid-body mass matrix: board mass, board (+ attached body) inertia, rider mass. */
  private setBaseMass(mb: number, mr: number): void {
    const M = this.sys.M;
    const Iw = this.board.inertiaWorld;
    M.fill(0);
    M[0] = mb; M[10] = mb; M[20] = mb;
    M[30] = Iw[0]; M[31] = Iw[1]; M[32] = Iw[2];
    M[40] = Iw[4]; M[41] = Iw[5];
    M[50] = Iw[8];
    M[60] = mr; M[70] = mr; M[80] = mr;
  }

  /** Maximum deck depth below the water over the given cells, m (+ = submerged). */
  private deckDepthOf(cells: number[]): number {
    const h = this.hull;
    const b = this.board;
    let best = -10;
    for (const k of cells) {
      this.tmp.set(h.cx[k], h.cyd[k], h.cz[k]);
      b.localToWorld(this.tmp, this.tmp2);
      const d = h.waterHeightAtCell(k, this.tmp2.x, this.tmp2.z) - this.tmp2.y;
      if (d > best) best = d;
    }
    return best;
  }

  private fillTelemetry(dt: number): void {
    const t = this.telemetry;
    const b = this.board;
    const r = this.rider;
    const d = this.diag;
    const s = this.underSample;
    const R = b.R;
    t.time = this.time;
    t.stance = r.stance;
    t.boardId = this.boardSpec.id;
    const v = b.velocity;
    t.speed = v.length();
    t.speedRelWater = Math.hypot(v.x - s.velX, v.z - s.velZ);
    t.speedAlongWave = v.x * s.dirX + v.z * s.dirZ;
    t.verticalSpeed = v.y;
    t.headingRad = Math.atan2(R[6], R[0]);
    t.pitchRad = Math.asin(Math.max(-1, Math.min(1, R[3])));
    t.rollRad = Math.atan2(-R[5], R[4]);
    t.submergedLiters = d.submergedVolume * 1000;
    t.riderSubmergedLiters = r.submergedVolume * 1000;
    t.buoyancyN = d.buoyancy.length();
    t.planingLiftN = d.pressure.y;
    // drag: water forces opposing the motion (projected on −v̂)
    const vl = Math.max(t.speed, 1e-6);
    const fx = d.pressure.x + d.friction.x + d.rail.x + d.fin.x;
    const fy = d.pressure.y + d.friction.y + d.rail.y + d.fin.y;
    const fz = d.pressure.z + d.friction.z + d.rail.z + d.fin.z;
    t.dragN = t.speed > 0.05 ? -(fx * v.x + fy * v.y + fz * v.z) / vl : 0;
    t.finForceN = d.fin.length();
    t.railForceN = d.rail.length();
    t.frictionN = d.friction.length();
    t.wettedArea = d.wettedArea;
    t.deckWettedArea = d.deckWettedArea;
    t.wettedLength = d.wettedLength;
    let amax = 0;
    let stalled = false;
    for (let i = 0; i < d.finAlpha.length; i++) {
      if (d.finSubmerged[i] <= 0) continue;
      const a = Math.abs(d.finAlpha[i]);
      const inc = Math.min(a, Math.PI - a);
      if (inc > amax) amax = inc;
      if (d.finAttached[i] < 0.5 && d.finSpeed[i] > 0.5) stalled = true;
    }
    t.finAlphaDeg = (amax * 180) / Math.PI;
    t.finStalled = stalled;
    t.paddleThrustN = r.paddleThrust;
    t.legForceN = r.legForce.length();
    t.feetLoadN = r.feetLoad;
    t.balanceRoll = r.balanceRoll;
    t.balancePitch = r.balancePitch;
    t.balance = Math.max(Math.abs(r.balanceRoll), Math.abs(r.balancePitch));
    t.balanceTimer = r.balanceTimer;
    t.wipeoutReason = r.wipeoutReason;
    const w = t.water;
    w.height = s.height;
    w.slopeX = s.slopeX;
    w.slopeZ = s.slopeZ;
    w.velX = s.velX;
    w.velY = s.velY;
    w.velZ = s.velZ;
    w.depth = s.depth;
    w.waveHeight = s.waveHeight;
    w.fullness = s.fullness;
    w.breaking = s.breaking;
    w.ratio = s.ratio;
    w.wavePhase = s.wavePhase;
    w.phaseSpeed = s.phaseSpeed;
    w.dirX = s.dirX;
    w.dirZ = s.dirZ;
    const onFace = s.wavePhase > -0.35 && s.wavePhase < 2.6;
    t.riding =
      s.waveHeight > 0.3 && t.speed > 2.5 && t.speedAlongWave > 0.6 * s.phaseSpeed && onFace && d.wettedArea > 0;
    t.ridingTime = t.riding ? t.ridingTime + dt : 0;
  }

  private pushDebug(o: Vector3, fx: number, fy: number, fz: number, kind: DebugForceKind): void {
    if (this.debugForceCount >= MAX_DEBUG_FORCES) return;
    const e = this.debugForces[this.debugForceCount++];
    e.origin.copy(o);
    e.vector.set(fx, fy, fz);
    e.kind = kind;
  }

  private collectDebugForces(segFx: number, segFy: number, segFz: number): void {
    // leash arrow (if any) was pushed during the step; keep it, rebuild the rest
    let keepLeash = false;
    if (this.debugForceCount > 0 && this.debugForces[0].kind === 'leash') keepLeash = true;
    this.debugForceCount = keepLeash ? 1 : 0;
    const h = this.hull;
    const b = this.board;
    for (let k = 0; k < h.n; k++) {
      const fx = h.cellForce[3 * k], fy = h.cellForce[3 * k + 1], fz = h.cellForce[3 * k + 2];
      if (fx * fx + fy * fy + fz * fz < 1) continue;
      this.tmp.set(h.cx[k], h.cyb[k], h.cz[k]);
      b.localToWorld(this.tmp, this.tmp2);
      this.pushDebug(this.tmp2, fx, fy, fz, 'hull');
    }
    for (let f = 0; f < h.fins.length; f++) {
      const fx = h.finForceWorld[3 * f], fy = h.finForceWorld[3 * f + 1], fz = h.finForceWorld[3 * f + 2];
      if (fx * fx + fy * fy + fz * fz < 0.01) continue;
      this.tmp2.set(h.finPointWorld[3 * f], h.finPointWorld[3 * f + 1], h.finPointWorld[3 * f + 2]);
      this.pushDebug(this.tmp2, fx, fy, fz, 'fin');
    }
    const r = this.rider;
    if (r.attached) {
      b.localToWorld(r.targetLocal, this.tmp2);
      this.pushDebug(this.tmp2, -r.legForce.x, -r.legForce.y, -r.legForce.z, 'leg');
    }
    this.pushDebug(r.position, segFx, segFy, segFz, 'body');
    if (r.paddleThrust !== 0) {
      this.tmp.set(b.R[0], b.R[3], b.R[6]).multiplyScalar(r.paddleThrust);
      this.pushDebug(r.position, this.tmp.x, this.tmp.y, this.tmp.z, 'paddle');
    }
    this.pushDebug(b.position, 0, -b.mass * GRAVITY, 0, 'gravity');
    this.pushDebug(r.position, 0, -r.config.mass * GRAVITY, 0, 'gravity');
  }
}

/** Lineup spawn (prone, still, outside the peak, facing the beach) at a given time. */
export function lineupSpawnAt(time: number): Spawn {
  return { x: -75, z: 5, headingRad: 0, stance: 'prone', speed: 0, time };
}

function createTelemetry(): SurfTelemetry {
  return {
    time: 0, stance: 'prone', boardId: '', speed: 0, speedRelWater: 0, speedAlongWave: 0, verticalSpeed: 0,
    headingRad: 0, pitchRad: 0, rollRad: 0, submergedLiters: 0, riderSubmergedLiters: 0, buoyancyN: 0,
    planingLiftN: 0, dragN: 0, finForceN: 0, railForceN: 0, frictionN: 0, wettedArea: 0, deckWettedArea: 0,
    wettedLength: 0, finAlphaDeg: 0, finStalled: false, paddleThrustN: 0, legForceN: 0, feetLoadN: 0,
    balance: 0, balanceRoll: 0, balancePitch: 0, balanceTimer: 0, wipeoutReason: null, deckDepth: 0, noseDepth: 0,
    water: {
      height: 0, slopeX: 0, slopeZ: 0, velX: 0, velY: 0, velZ: 0, depth: 0, waveHeight: 0, fullness: 0,
      breaking: 0, ratio: 0, wavePhase: 0, phaseSpeed: 0, dirX: 1, dirZ: 0,
    },
    riding: false,
    ridingTime: 0,
  };
}
