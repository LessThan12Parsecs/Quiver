/**
 * Autopilot surfer: a scripted, skilled rider that only produces SurfInput (the same controls a
 * player has: paddle/steer, pop-up, lean, trim, crouch; it does not twist — an internal torque
 * that only winds the board against the body). No forces, no state tweaks — if it
 * rides well, the physics allows riding well. It is a test tool (rideability metrics) and a game
 * feature (key O): watch a good ride and copy it.
 *
 *   const ap = new Autopilot(ocean);
 *   ap.update(sim, dt, input);  // before every sim.step(dt, input)
 *
 * It reads the board state from the sim and the wave from the ocean model (what a surfer sees:
 * the wave under and around the board, the sets coming in from outside — current time only).
 *
 * Phases
 *   position  prone: paddle to the take-off spot (just outside where the sets break, a little to
 *             one side of the peak); goes for a set wave from here if it is close enough
 *   wait      prone: face the beach, watch for a set wave; let one pass that is already breaking
 *             outside (and sit further out next time)
 *   paddle    prone: paddle hard toward the beach (angled a little to the riding side) once the
 *             crest is within triggerDist; pop up once the board runs with the wave (popSpeed × the
 *             wave speed along the wave direction, at or below the crest) — on these waves that is
 *             the last half second before the crest pitches (late take-off), so if the whitewater
 *             reaches the board while it is still running with it, keep going (stay down, pop as
 *             it slides onto the face); give up if the wave passes under (next time sit further
 *             in) or breaks on the rider and stops the board (further out)
 *   popup     the 0.6 s pop-up, weight a little forward so the board keeps running
 *   ride      standing: drop in angled down the line, then hold a height on the face (or on the
 *             front of the whitewater once the wave has broken over the board): the heading angle
 *             from the wave direction (within [angleMin, angleMax], smoothed) is chosen so that
 *             the speed component along the wave direction moves the board toward the target
 *             phase — lower and faster when the curl is close — which also yields the bottom turns
 *             (low → turn up) and top turns (high → turn down); near the bottom of the wave, fast,
 *             it turns up and keeps the weight off the nose. A cascade (heading error → turn rate →
 *             bank → lean, bounded by what the speed carries: a smooth rider) gives leanSide; trim
 *             (leanForward) manages the speed against the curl behind
 *   kickout   the wave backs off / is lost: turn up and over the back (or straighten out and
 *             settle if too slow to turn)
 *   done      ride over (standing behind the wave, or fallen); optional auto-reset (game)
 */
import type { OceanModel, SwellEval } from '../ocean/waveModel';
import { createSwellEval } from '../ocean/waveModel';
import { GRAVITY } from './constants';
import { RIDER_MODEL } from './Rider';
import type { SurfInput, SurfSim } from './SurfSim';

export type AutopilotPhase = 'position' | 'wait' | 'paddle' | 'popup' | 'ride' | 'kickout' | 'done';

export interface AutopilotConfig {
  /** Take-off spot: x (seaward of the bar crest at the peak) and |z| offset to the riding side.
   * Once a set wave is coming, the rider moves to catchLead m seaward of where that wave will be
   * catchable (where its crest gets to catchRatio of its breaking height, read from its height
   * and the depth ahead: bigger waves break further out). */
  takeoffX: number;
  takeoffZ: number;
  readSets: boolean;
  catchLead: number;
  catchRatio: number;
  /** Preferred riding side: +1 (toward +Z), −1, or 0 = decide per wave (alternate). */
  side: 1 | -1 | 0;
  /** Waiting: drift from the take-off spot (m) after which it paddles back. */
  driftDist: number;
  /** Smallest wave (height seen outside) worth going for, m, and how far seaward its crest is
   * when the rider starts paddling for it, m. */
  minHeight: number;
  triggerDist: number;
  /** Pop up once the board runs with the wave: waveform phase above popPhase (0 = crest) and the
   * speed along the wave direction above popSpeed × the wave speed. */
  popPhase: number;
  popSpeed: number;
  /** Paddling heading angle from the wave direction toward the riding side, rad. */
  paddleAngle: number;
  /** leanForward while getting up and dropping in; the drop ends once the board is below
   * dropPhase on the face or after dropTime, s; heading angle from the wave direction held
   * while dropping, rad (angled down the line: straight down these faces ends in a pearl at the
   * bottom); crouch while dropping. */
  popTrim: number;
  dropPhase: number;
  dropTime: number;
  dropAngle: number;
  dropCrouch: number;
  /** Give up on a wave after paddling this long inside its breaking crest, s (standing up in the
   * foam on top of the wave sinks the board); Infinity = stay with it and pop when it runs. */
  lipTime: number;
  /** Target waveform phase on the face while riding (0 = crest, π/2 ≈ mid-face), and on the front
   * of the whitewater once the wave has broken over the board. */
  trimPhase: number;
  whitewaterPhase: number;
  /** Below this phase (the bottom of the wave), faster than the wave: bottom turn — angle at
   * least bottomAngle, weight off the nose. */
  bottomPhase: number;
  bottomAngle: number;
  /** Preferred distance ahead of the breaking curl, m. */
  pocket: number;
  /** Seconds after the ride ends (kick-out or wipeout) before resetting to the lineup; Infinity =
   * never (tests). */
  autoReset: number;
  /** Heading controller while riding (a cascade, like a surfer: the heading error asks for a turn
   * rate, the turn rate for a bank — u tilts by atan(turnCoord tan bank), so the bank is
   * atan(rate·v / (g·turnCoord)) — and the bank for a lean, leanSide = bank / the rider's full-lean
   * bank): turn rate per rad of heading error, 1/s, and its limit, rad/s; integral of the heading
   * error, bank per rad·s (finds the rail that holds the line across a sloping face). */
  steerGain: number;
  yawMax: number;
  turnCoord: number;
  steerI: number;
  /** The largest leanSide used, by forward speed: maxLean = clamp(leanPerSpeed·(v − leanSpeed0),
   * leanMin, leanMax) (a slow board cannot carry a hard lean: the rail sinks and the turn does
   * not support the body; a hard lean at speed over-banks the board faster than the body follows). */
  leanPerSpeed: number;
  leanSpeed0: number;
  leanMin: number;
  leanMax: number;
  /** Heading angle from the wave direction kept within [angleMin, angleMax] while riding, rad,
   * and the time constant that smooths the target, s. */
  angleMin: number;
  angleMax: number;
  angleTau: number;
}

const TAU = Math.PI * 2;
const DEG = Math.PI / 180;

export const DEFAULT_AUTOPILOT: AutopilotConfig = {
  takeoffX: -37,
  takeoffZ: 5,
  readSets: true,
  catchLead: 2.5,
  catchRatio: 0.88,
  side: 0,
  driftDist: 7,
  minHeight: 1.15,
  triggerDist: 22,
  popPhase: 0,
  popSpeed: 0.92,
  paddleAngle: 35 * DEG,
  popTrim: 0.25,
  dropPhase: 1.0,
  dropTime: 1.2,
  dropAngle: 35 * DEG,
  dropCrouch: 0.5,
  lipTime: Infinity,
  trimPhase: 0.8,
  whitewaterPhase: 1.0,
  bottomPhase: 1.35,
  bottomAngle: 55 * DEG,
  pocket: 7,
  autoReset: 4,
  steerGain: 0.8,
  yawMax: 0.6,
  turnCoord: 0.55,
  steerI: 0.3,
  leanPerSpeed: 0.12,
  leanSpeed0: 1.5,
  leanMin: 0.15,
  leanMax: 0.45,
  angleMin: 20 * DEG,
  angleMax: 75 * DEG,
  angleTau: 0.4,
};

/** Read-only view of what the autopilot is doing (HUD/debug/tests). */
export interface AutopilotStatus {
  phase: AutopilotPhase;
  side: 1 | -1;
  /** Waveform phase under the board and its target, rad. */
  phase0: number;
  phaseTarget: number;
  /** Heading angle from the wave direction toward the riding side and its target, rad. */
  angle: number;
  angleTarget: number;
  /** Distance to the breaking curl behind / closing section ahead along the crest, m (Infinity
   * = none within the scan). */
  curl: number;
  closeout: number;
  /** Waves gone for (paddle phase entered) and caught (popped up on the face). */
  attempts: number;
  catches: number;
  /** Why the last ride ended. */
  endReason: string;
}

function clamp(x: number, a: number, b: number): number {
  return x < a ? a : x > b ? b : x;
}

function wrap(a: number): number {
  a %= TAU;
  return a > Math.PI ? a - TAU : a < -Math.PI ? a + TAU : a;
}

export class Autopilot {
  readonly config: AutopilotConfig;
  readonly status: AutopilotStatus;
  private readonly ocean: OceanModel;
  private readonly se: SwellEval = createSwellEval();
  private readonly se2: SwellEval = createSwellEval();
  private phaseTime = 0;
  private scanTimer = 0;
  private nextSide: 1 | -1 = 1;
  private resetRequested = false;
  // wave state under the board (refreshed every step)
  private psi = 0;
  private c = 4.5;
  private dirX = 1;
  private dirZ = 0;
  private height = 0;
  private fullness = 0;
  private breakingHere = 0;
  /** Heading-controller integral and smoothed angle target while riding. */
  private headI = 0;
  /** Learned shift of the take-off spot toward the beach, m. */
  private spotShift = 0;
  private skipping = false;
  private angF = 0;
  private angInit = false;
  /** Just up after a take-off: still dropping in from the top of the wave. */
  private dropping = false;
  /** Seconds below 1.5 m/s while riding. */
  private slowTime = 0;
  /** Seconds paddling inside the breaking crest. */
  private lipTime = 0;
  /** Where the coming set wave will be catchable (x), NaN = no set in sight. */
  private catchX = NaN;

  constructor(ocean: OceanModel, config: Partial<AutopilotConfig> = {}) {
    this.ocean = ocean;
    this.config = { ...DEFAULT_AUTOPILOT, ...config };
    this.status = {
      phase: 'position',
      side: 1,
      phase0: 0,
      phaseTarget: 0,
      angle: 0,
      angleTarget: 0,
      curl: Infinity,
      closeout: Infinity,
      attempts: 0,
      catches: 0,
      endReason: '',
    };
    this.reset();
  }

  /** Forget the current plan (after a teleport/reset of the sim). */
  reset(): void {
    this.setPhase('position');
    this.status.curl = Infinity;
    this.status.closeout = Infinity;
    this.resetRequested = false;
    const s = this.config.side;
    this.status.side = s === 0 ? this.nextSide : s;
  }

  /** Start in the riding phase (for a standing spawn on a wave face). */
  startRiding(side: 1 | -1): void {
    this.status.side = side;
    this.dropping = false;
    this.headI = 0;
    this.angInit = false;
    this.slowTime = 0;
    this.setPhase('ride');
    this.status.catches++;
    this.status.attempts++;
  }

  private setPhase(p: AutopilotPhase): void {
    this.status.phase = p;
    this.phaseTime = 0;
  }

  /** Compute the controls for the next physics step from the current sim state. */
  update(sim: SurfSim, dt: number, out: SurfInput): void {
    const st = this.status;
    const r = sim.rider;
    const b = sim.board;
    this.ocean.setTime(sim.time);
    this.phaseTime += dt;
    out.paddle = 0;
    out.steer = 0;
    out.leanForward = 0;
    out.leanSide = 0;
    out.crouch = 0;
    out.twist = 0;
    out.popUp = false;
    out.reset = false;

    // wave under the board
    const w = this.ocean.evalSwell(b.position.x, b.position.z, this.se);
    this.psi = w.domPsi;
    this.c = w.domSpeed;
    this.dirX = w.domDirX;
    this.dirZ = w.domDirZ;
    this.height = w.height;
    this.fullness = w.fullness;
    this.breakingHere = w.breaking;
    st.phase0 = this.psi;

    if (r.stance === 'fallen') {
      if (st.phase !== 'done') {
        st.endReason = `wipeout (${r.wipeoutReason ?? '?'})`;
        this.setPhase('done');
      }
    }
    if (this.resetRequested) {
      // the reset edge was consumed by the previous step
      this.resetRequested = false;
      this.reset();
    }

    switch (st.phase) {
      case 'position':
        this.doPosition(sim, out);
        break;
      case 'wait':
        this.doWait(sim, out);
        break;
      case 'paddle':
        this.doPaddle(sim, dt, out);
        break;
      case 'popup':
        this.doRide(sim, dt, out, true);
        if (r.stance === 'standing') {
          this.setPhase('ride');
          this.dropping = true;
        }
        break;
      case 'ride':
        this.doRide(sim, dt, out, false);
        break;
      case 'kickout':
        this.doKickout(sim, dt, out);
        break;
      case 'done':
        out.crouch = r.stance === 'standing' ? 0.6 : 0;
        if (this.phaseTime > this.config.autoReset) {
          out.reset = true;
          this.resetRequested = true;
        }
        break;
    }
  }

  // ------------------------------------------------------------------------------- helpers

  /** Board heading angle (from +X toward +Z). */
  private heading(sim: SurfSim): number {
    return Math.atan2(sim.board.R[6], sim.board.R[0]);
  }

  /** Heading angle from the wave direction toward the riding side, rad. */
  private waveAngle(sim: SurfSim): number {
    const s = this.status.side;
    const dX = this.dirX, dZ = this.dirZ;
    const hX = sim.board.R[0], hZ = sim.board.R[6];
    return Math.atan2((-dZ * hX + dX * hZ) * s, hX * dX + hZ * dZ);
  }

  /** Prone steering toward a world heading (paddle harder on the outside arm; without paddle
   * effort the inside arm back-paddles: a pivot on the spot). Returns the heading error. */
  private steerProne(sim: SurfSim, target: number, out: SurfInput): number {
    const e = wrap(target - this.heading(sim));
    const yaw = sim.rider.turnRate;
    out.steer = clamp(2.2 * e - 0.6 * yaw, -1, 1);
    return e;
  }

  /** Max whitewater over a short cross-shore search around (x, z) (the crest line is curved). */
  private whiteAt(x: number, z: number): number {
    let m = 0;
    for (let k = -2; k <= 2; k++) {
      const e = this.ocean.evalSwell(x + this.dirX * 2.5 * k, z + this.dirZ * 2.5 * k, this.se2);
      if (e.domPsi > -0.6 && e.domPsi < 1.8 && e.breaking > m) m = e.breaking;
    }
    return m;
  }

  /** Scan along the crest line behind (toward the peak) and ahead for breaking water. */
  private scanCrest(px: number, pz: number): void {
    const st = this.status;
    const s = st.side;
    // along-crest unit vector toward the riding side
    const tx = -this.dirZ * s;
    const tz = this.dirX * s;
    st.curl = Infinity;
    st.closeout = Infinity;
    for (let k = 1; k <= 10; k++) {
      const d = 2.5 * k;
      if (st.curl === Infinity && this.whiteAt(px - tx * d, pz - tz * d) > 0.35) st.curl = d;
      if (st.closeout === Infinity && this.whiteAt(px + tx * d, pz + tz * d) > 0.35) st.closeout = d;
      if (st.curl !== Infinity && st.closeout !== Infinity) break;
    }
  }

  /** Slope of the water surface ahead of the nose along the heading (+ = descending), sampled
   * where the board will be in ≈ 0.3 s. */
  private slopeAhead(sim: SurfSim): number {
    const b = sim.board;
    const hx = b.R[0], hz = b.R[6];
    const n = Math.hypot(hx, hz) || 1;
    const ux = hx / n, uz = hz / n;
    const v = Math.max(Math.hypot(b.velocity.x, b.velocity.z), 1);
    const d0 = 1.0, d1 = 1.0 + 0.3 * v;
    const e0 = this.ocean.evalSwell(b.position.x + ux * d0, b.position.z + uz * d0, this.se2).eta;
    const e1 = this.ocean.evalSwell(b.position.x + ux * d1, b.position.z + uz * d1, this.se2).eta;
    return (e0 - e1) / (d1 - d0);
  }

  // -------------------------------------------------------------------------------- phases

  /** Take-off spot, shifted by experience: seaward after a wave broke on the rider, shoreward
   * after one passed under unbroken (too far out to catch). */
  private takeoffSpot(): [number, number] {
    const x = Number.isNaN(this.catchX) ? this.config.takeoffX : this.catchX - this.config.catchLead;
    return [x + this.spotShift, this.config.takeoffZ * this.status.side];
  }

  /** Read the coming set: where its crest will reach catchRatio of the breaking height along the
   * wave direction from (x, z) — its height grows as the water shoals (Green's law, H ∝ h^-1/4)
   * while the depth limit γh falls. */
  private predictCatchX(crestX: number, crestZ: number, height: number): number {
    const gamma = this.ocean.config.breakerIndex;
    const h0 = this.ocean.evalSwell(crestX, crestZ, this.se2).depth;
    for (let d = 0; d < 60; d += 0.5) {
      const x = crestX + this.dirX * d;
      const z = crestZ + this.dirZ * d;
      const h = Math.max(this.ocean.evalSwell(x, z, this.se2).depth, 0.1);
      const H = height * Math.pow(h0 / h, 0.25);
      if (H / (gamma * h) > this.config.catchRatio) return x;
    }
    return NaN;
  }

  /** Watch for the next set wave outside and work out where it will be catchable. */
  private readSet(sim: SurfSim): void {
    const b = sim.board;
    if (!this.config.readSets) return;
    const crest = this.findCrestSeaward(b.position.x, b.position.z, 40);
    if (crest.dist > 0 && crest.height >= this.config.minHeight && crest.breaking < 0.3) {
      const cx = b.position.x - this.dirX * crest.dist;
      const cz = b.position.z - this.dirZ * crest.dist;
      this.catchX = this.predictCatchX(cx, cz, crest.height);
    } else if (crest.dist < 0) this.catchX = NaN;
  }

  private doPosition(sim: SurfSim, out: SurfInput): void {
    const r = sim.rider;
    if (r.stance !== 'prone') return;
    this.readSet(sim);
    const [tx, tz] = this.takeoffSpot();
    const dx = tx - sim.board.position.x;
    const dz = tz - sim.board.position.z;
    const dist = Math.hypot(dx, dz);
    if (dist < (Number.isNaN(this.catchX) ? 3.5 : 1.0)) {
      this.setPhase('wait');
      return;
    }
    // close enough and a set wave is coming: go for it from here
    if (dist < 12 && this.phaseTime > 1 && this.goForWave(sim)) return;
    const e = this.steerProne(sim, Math.atan2(dz, dx), out);
    // facing well off the course: pivot first (paddling while steering only arcs round)
    out.paddle = Math.abs(e) > 0.5 ? 0 : dist > 8 ? 1 : 0.6;
  }

  private doWait(sim: SurfSim, out: SurfInput): void {
    const st = this.status;
    const b = sim.board;
    this.readSet(sim);
    const [tx, tz] = this.takeoffSpot();
    const dist = Math.hypot(tx - b.position.x, tz - b.position.z);
    // face the beach, angled a little toward the riding side
    const waveDir = Math.atan2(this.dirZ, this.dirX);
    this.steerProne(sim, waveDir + st.side * 12 * DEG, out);
    // a set is coming: shuffle in (paddle gently) or let the water carry the board out to the
    // spot where this wave will be catchable (along the wave direction)
    if (!Number.isNaN(this.catchX)) {
      const along = (tx - b.position.x) * this.dirX + (tz - b.position.z) * this.dirZ;
      if (along > 0.7) out.paddle = clamp(0.3 * along, 0.2, 0.6);
      else if (along < -1.5 && this.phaseTime > 1) {
        this.setPhase('position');
        return;
      }
    }
    if (this.phaseTime > 1.5 && this.goForWave(sim)) return;
    // drifted: paddle back before it takes long (the wave train pushes the board shoreward
    // between sets)
    if (dist > this.config.driftDist) this.setPhase('position');
  }

  /** Look outside for an approaching set wave (the next crest seaward within triggerDist) and
   * start paddling for it; let it pass if it is already breaking (and sit further out). */
  private goForWave(sim: SurfSim): boolean {
    const st = this.status;
    const b = sim.board;
    const crest = this.findCrestSeaward(b.position.x, b.position.z);
    if (crest.dist > 0 && crest.dist < this.config.triggerDist && crest.height >= this.config.minHeight) {
      if (crest.breaking > 0.3) {
        if (!this.skipping) this.spotShift = Math.max(this.spotShift - 2.5, -10);
        this.skipping = true;
        return false;
      }
      st.attempts++;
      this.lipTime = 0;
      this.setPhase('paddle');
      return true;
    }
    if (crest.dist < 0 || crest.dist > this.config.triggerDist) this.skipping = false;
    return false;
  }

  /** Distance (along −wave dir) and height of the next crest seaward of (x, z). */
  private readonly crestOut = { dist: -1, height: 0, breaking: 0 };
  private findCrestSeaward(x: number, z: number, maxDist = 30): { dist: number; height: number; breaking: number } {
    const o = this.crestOut;
    o.dist = -1;
    o.height = 0;
    o.breaking = 0;
    let prev = this.ocean.evalSwell(x, z, this.se2).domPsi;
    for (let d = 1; d <= maxDist; d += 1) {
      const e = this.ocean.evalSwell(x - this.dirX * d, z - this.dirZ * d, this.se2);
      // seaward from the trough the waveform phase falls toward 0 at the crest, then turns
      // negative on the back
      if (prev > 0 && e.domPsi <= 0) {
        o.dist = d;
        o.height = e.height;
        o.breaking = e.breaking;
        return o;
      }
      prev = e.domPsi;
    }
    return o;
  }

  private doPaddle(sim: SurfSim, dt: number, out: SurfInput): void {
    const st = this.status;
    const cfg = this.config;
    const b = sim.board;
    const waveDir = Math.atan2(this.dirZ, this.dirX);
    this.steerProne(sim, waveDir + st.side * cfg.paddleAngle, out);
    out.paddle = 1;
    out.leanForward = 0.3; // chest down, weight forward a little: helps the board catch
    const vAlong = b.velocity.x * this.dirX + b.velocity.z * this.dirZ;
    const running = vAlong > cfg.popSpeed * this.c;
    // in the breaking crest (standing up there the board sinks in the foam)
    const inLip = this.breakingHere > 0.35 && this.psi < 0.45;
    this.lipTime = inLip ? this.lipTime + dt : 0;
    // the whitewater stopped the board, or carries it along in the crest: caught inside — next
    // time sit further out
    if ((this.breakingHere > 0.4 && vAlong < 0.7 * this.c) || this.lipTime > cfg.lipTime) {
      st.endReason = 'caught inside';
      this.spotShift = Math.max(this.spotShift - 2.5, -10);
      this.setPhase('position');
      return;
    }
    // the board runs with the wave, at the crest (still unbroken) or on the face: up
    if (running && !(inLip && cfg.lipTime < Infinity) && this.psi > cfg.popPhase && this.psi < 1.6 && this.phaseTime > 0.5) {
      out.popUp = true;
      st.catches++;
      this.headI = 0;
      this.angInit = false;
      this.slowTime = 0;
      this.setPhase('popup');
      return;
    }
    // missed: the crest went past (now on the back of the wave) or took too long
    if ((this.psi < -0.4 && this.phaseTime > 2) || this.phaseTime > 12) {
      st.endReason = 'missed the wave';
      this.spotShift = Math.min(this.spotShift + 2, 8);
      this.nextSide = st.side > 0 ? -1 : 1;
      if (cfg.side === 0) st.side = this.nextSide;
      this.setPhase('position');
    }
  }

  private doRide(sim: SurfSim, dt: number, out: SurfInput, popping: boolean): void {
    const st = this.status;
    const cfg = this.config;
    const b = sim.board;
    const s = st.side;
    // crest scan at ~20 Hz
    this.scanTimer -= dt;
    if (this.scanTimer <= 0) {
      this.scanTimer = 0.05;
      this.scanCrest(b.position.x, b.position.z);
    }
    const angle = this.waveAngle(sim);
    st.angle = angle;
    const v = Math.hypot(b.velocity.x, b.velocity.z);
    const white = this.breakingHere > 0.5;

    // --- target height on the face (phase): nominal; lower (faster) when the curl is close,
    // higher (slower) when far ahead of it; on the front of the whitewater once it has broken
    // over the board
    let psiT = cfg.trimPhase;
    const pocket = cfg.pocket;
    if (st.curl < pocket) psiT += 0.35 * (1 - st.curl / pocket);
    else if (st.curl > 2.5 * pocket) psiT -= 0.2;
    if (white) psiT = Math.max(psiT, cfg.whitewaterPhase);
    st.phaseTarget = psiT;
    // --- along-wave speed that moves the board toward the target phase, and the heading angle
    // that gives it at the current speed
    const ePsi = this.psi - psiT;
    const vAlT = this.c - clamp(2.8 * ePsi, -2.5, 3.5);
    const cosT = clamp(vAlT / Math.max(v, 0.5), -0.6, 0.97);
    let angT = clamp(Math.acos(cosT), cfg.angleMin, cfg.angleMax);
    // dropping in: angled down the line, not yet turning along it
    if (this.dropping && (this.psi > cfg.dropPhase || this.phaseTime > cfg.dropTime)) this.dropping = false;
    if (popping || this.dropping) angT = cfg.dropAngle;
    // the bottom of the wave, fast: bottom turn
    const bottom = !popping && this.psi > cfg.bottomPhase && v > this.c;
    if (bottom) angT = Math.max(angT, cfg.bottomAngle);
    // smooth the target (the phase under the board is noisy over chop)
    if (!this.angInit) {
      this.angF = angT;
      this.angInit = true;
    }
    this.angF += (angT - this.angF) * (1 - Math.exp(-dt / Math.max(cfg.angleTau, 1e-3)));
    angT = this.angF;
    st.angleTarget = angT;

    // --- heading → turn rate → bank → lean (the rider's reflex holds the bank)
    const u = popping ? 0 : this.leanFor(sim, wrap(angT - angle), dt);
    out.leanSide = s * u;

    // --- trim: weight forward for speed when the curl is close or the board is slow for its
    // line, back to slow down / hold the top of the face; off the nose when it points down
    // more steeply than the water ahead (the bottom of the wave, the trough in front of the
    // whitewater); a little forward getting up and dropping in, so the board keeps running
    const needSpeed = (st.curl < pocket ? 1 : 0) + (v < this.c / Math.max(Math.cos(Math.min(angle, 1.3)), 0.3) - 0.3 ? 0.5 : 0);
    let trim = 0.1 + 0.3 * needSpeed;
    if (st.curl > 2.5 * pocket && this.psi < psiT) trim = -0.2;
    if (popping || this.dropping) trim = cfg.popTrim;
    // nose-down attitude relative to the water ahead
    const pitchDown = -Math.asin(clamp(b.R[3], -1, 1));
    const over = pitchDown - Math.atan(Math.max(this.slopeAhead(sim), -0.5));
    if (over > 0.1 && v > 2.5) trim = Math.min(trim, -clamp((over - 0.1) * 3, 0.15, 0.6));
    if (bottom) trim = Math.min(trim, -0.15);
    out.leanForward = clamp(trim, -1, 1);
    out.crouch = clamp(0.3 + 0.6 * Math.abs(u) + (white ? 0.3 : 0), 0, 1);
    if (popping || this.dropping) out.crouch = Math.max(out.crouch, cfg.dropCrouch);

    if (popping) return;
    // --- end of the ride: wave backed off or lost, or the board has stopped
    this.slowTime = v < 1.5 ? this.slowTime + dt : 0;
    let end = '';
    if (this.slowTime > 1) end = 'stopped';
    else if (this.height < 0.45 && this.psi > 1.2) end = 'wave backed off';
    else if (this.fullness < 0.4 && this.psi > 2.0) end = 'wave backed off';
    else if (this.psi > 2.7 || this.psi < -0.6) end = 'lost the wave';
    if (end) {
      st.endReason = white ? `${end} (whitewater)` : end;
      this.setPhase(end === 'stopped' ? 'done' : 'kickout');
    }
  }

  /**
   * Lean (toward the riding side, + = turning so the angle from the wave direction grows) for a
   * heading error e: the turn rate it asks for, the bank that turns the board at that rate, the
   * lean that asks the rider for that bank — bounded by what the speed carries.
   */
  private leanFor(sim: SurfSim, e: number, dt: number): number {
    const cfg = this.config;
    const b = sim.board;
    const v = Math.max(Math.hypot(b.velocity.x, b.velocity.z), 0.5);
    const uMax = clamp(cfg.leanPerSpeed * (v - cfg.leanSpeed0), cfg.leanMin, cfg.leanMax);
    const rate = clamp(cfg.steerGain * e, -cfg.yawMax, cfg.yawMax);
    const bankMax = RIDER_MODEL.bankMax;
    let bank = Math.atan((rate * v) / (GRAVITY * cfg.turnCoord));
    // integral (frozen while saturated)
    const u0 = bank / bankMax;
    if (Math.abs(u0 + this.headI) < uMax) this.headI = clamp(this.headI + (cfg.steerI * e * dt) / bankMax, -uMax, uMax);
    bank = bank + this.headI * bankMax;
    return clamp(bank / bankMax, -uMax, uMax);
  }

  private doKickout(sim: SurfSim, dt: number, out: SurfInput): void {
    const st = this.status;
    const b = sim.board;
    const s = st.side;
    const v = Math.hypot(b.velocity.x, b.velocity.z);
    // turn up the face and over the back of the wave, with no more lean than the speed carries
    // (it should end standing); too slow for that: straighten out and settle
    const angle = this.waveAngle(sim);
    st.angle = angle;
    const target = v > 3 ? 150 * DEG : clamp(angle, 0, 60 * DEG);
    out.leanSide = s * this.leanFor(sim, wrap(target - angle), dt);
    out.leanForward = -0.2;
    out.crouch = 0.5;
    if (this.psi < -0.6 || v < 1.5 || this.phaseTime > 4) this.setPhase('done');
  }
}
