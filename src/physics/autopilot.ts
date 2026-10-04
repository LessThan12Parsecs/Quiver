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
 *   position  prone: paddle to the take-off spot (seaward of the break, a little to one side of
 *             the peak); goes for a set wave from here if it is close enough
 *   wait      prone: face the beach, watch for a set wave; let one pass that is already breaking
 *             outside (and sit further out next time)
 *   paddle    prone: paddle hard toward the beach (angled to the riding side) once the crest is
 *             within triggerDist; pop up when the board runs on the face at popSpeed × the wave
 *             speed; give up if it passes under (next time sit further in) or breaks on the rider
 *             (further out)
 *   popup     the 0.6 s pop-up, weight forward so the board keeps running
 *   ride      standing: angle down the line toward the unbroken shoulder; the heading angle from
 *             the wave direction (kept within [angleMin, angleMax], smoothed) is chosen so that the
 *             speed component along the wave direction keeps the board at the target height on
 *             the face (phase) — lower and faster when the curl is close — which also yields the
 *             bottom turns (low → turn up) and top turns (high → turn down); a PI heading
 *             controller with a bounded lean (a smooth rider) turns it into leanSide; trim
 *             (leanForward) manages the speed against the curl behind
 *   kickout   the section ahead closes out / the wave backs off / the whitewater catches up /
 *             the wave is lost: turn up and over the back
 *   done      ride over (standing behind the wave, or fallen); optional auto-reset (game)
 */
import type { OceanModel, SwellEval } from '../ocean/waveModel';
import { createSwellEval } from '../ocean/waveModel';
import type { SurfInput, SurfSim } from './SurfSim';

export type AutopilotPhase = 'position' | 'wait' | 'paddle' | 'popup' | 'ride' | 'kickout' | 'done';

export interface AutopilotConfig {
  /** Take-off spot: x (seaward of the bar crest at the peak) and |z| offset to the riding side. */
  takeoffX: number;
  takeoffZ: number;
  /** Preferred riding side: +1 (toward +Z), −1, or 0 = decide per wave (alternate). */
  side: 1 | -1 | 0;
  /** Waiting: drift from the take-off spot (m) after which it paddles back. */
  driftDist: number;
  /** Smallest wave (height seen outside) worth going for, m, and how far seaward its crest is
   * when the rider starts paddling for it, m. */
  minHeight: number;
  triggerDist: number;
  /** Pop up once the board runs on the face (waveform phase above popPhase) at popSpeed × the
   * wave speed along the wave direction. */
  popPhase: number;
  popSpeed: number;
  /** leanForward while getting up. */
  popTrim: number;
  /** Target waveform phase on the face while riding (0 = crest, π/2 ≈ mid-face). */
  trimPhase: number;
  /** Preferred distance ahead of the breaking curl, m. */
  pocket: number;
  /** Seconds after the ride ends (kick-out or wipeout) before resetting to the lineup; Infinity =
   * never (tests). */
  autoReset: number;
  /** Heading controller while riding: leanSide per rad of heading error, per rad/s of turn rate
   * (damping) and per rad·s of accumulated error (integral: finds the rail that holds the line
   * across a sloping face); the largest leanSide it uses (a skilled rider carves smoothly — a
   * full lean throws the body over before the board has turned under it). */
  steerGain: number;
  steerDamp: number;
  steerI: number;
  maxLean: number;
  /** Heading angle from the wave direction kept within [angleMin, angleMax] while trimming, rad
   * (never straight down the face: the nose digs in at the bottom), and the time constant that
   * smooths the target, s. */
  angleMin: number;
  angleMax: number;
  angleTau: number;
}

const TAU = Math.PI * 2;
const DEG = Math.PI / 180;

export const DEFAULT_AUTOPILOT: AutopilotConfig = {
  takeoffX: -38,
  takeoffZ: 5,
  side: 0,
  driftDist: 7,
  minHeight: 1.15,
  triggerDist: 22,
  popPhase: 0.15,
  popSpeed: 0.72,
  popTrim: 0.5,
  trimPhase: 0.75,
  pocket: 7,
  autoReset: 4,
  steerGain: 1.0,
  steerDamp: 0.5,
  steerI: 1.5,
  maxLean: 0.6,
  angleMin: 45 * DEG,
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
  private whiteTime = 0;
  /** Heading-controller integral and smoothed angle target while riding. */
  private headI = 0;
  /** Learned shift of the take-off spot toward the beach, m. */
  private spotShift = 0;
  private skipping = false;
  private angF = 0;
  private angInit = false;

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
    this.whiteTime = 0;
    const s = this.config.side;
    this.status.side = s === 0 ? this.nextSide : s;
  }

  /** Start in the riding phase (for a standing spawn on a wave face). */
  startRiding(side: 1 | -1): void {
    this.status.side = side;
    this.headI = 0;
    this.angInit = false;
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
    const px = b.position.x;
    const pz = b.position.z;
    const w = this.ocean.evalSwell(px, pz, this.se);
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
        this.doPaddle(sim, out);
        break;
      case 'popup':
        this.doRide(sim, dt, out, true);
        if (r.stance === 'standing') this.setPhase('ride');
        break;
      case 'ride':
        this.doRide(sim, dt, out, false);
        break;
      case 'kickout':
        this.doKickout(sim, out);
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

  // -------------------------------------------------------------------------------- phases

  /** Take-off spot, shifted by experience: seaward after a wave broke on the rider, shoreward
   * after one passed under unbroken (too far out to catch). */
  private takeoffSpot(): [number, number] {
    return [this.config.takeoffX + this.spotShift, this.config.takeoffZ * this.status.side];
  }

  private doPosition(sim: SurfSim, out: SurfInput): void {
    const r = sim.rider;
    if (r.stance !== 'prone') return;
    const [tx, tz] = this.takeoffSpot();
    const dx = tx - sim.board.position.x;
    const dz = tz - sim.board.position.z;
    const dist = Math.hypot(dx, dz);
    if (dist < 3.5) {
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
    const [tx, tz] = this.takeoffSpot();
    const dist = Math.hypot(tx - b.position.x, tz - b.position.z);
    // face the beach, angled a little toward the riding side
    const waveDir = Math.atan2(this.dirZ, this.dirX);
    this.steerProne(sim, waveDir + st.side * 12 * DEG, out);
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
      this.setPhase('paddle');
      return true;
    }
    if (crest.dist < 0 || crest.dist > this.config.triggerDist) this.skipping = false;
    return false;
  }

  /** Distance (along −wave dir) and height of the next crest seaward of (x, z). */
  private readonly crestOut = { dist: -1, height: 0, breaking: 0 };
  private findCrestSeaward(x: number, z: number): { dist: number; height: number; breaking: number } {
    const o = this.crestOut;
    o.dist = -1;
    o.height = 0;
    o.breaking = 0;
    let prev = this.ocean.evalSwell(x, z, this.se2).domPsi;
    for (let d = 1; d <= 30; d += 1) {
      const e = this.ocean.evalSwell(x - this.dirX * d, z - this.dirZ * d, this.se2);
      // going seaward the phase increases through the front face toward the crest of the next
      // wave: a wrap from +π… no — the waveform phase decreases toward 0 at the crest as we move
      // seaward from the trough, then turns negative on the back
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

  private doPaddle(sim: SurfSim, out: SurfInput): void {
    const st = this.status;
    const b = sim.board;
    const waveDir = Math.atan2(this.dirZ, this.dirX);
    this.steerProne(sim, waveDir + st.side * 15 * DEG, out);
    out.paddle = 1;
    out.leanForward = 0.3; // chest down, weight forward a little: helps the board catch
    const vAlong = b.velocity.x * this.dirX + b.velocity.z * this.dirZ;
    // the board runs: on the face and moving with the wave
    const onFace = this.psi > this.config.popPhase && this.psi < 1.6;
    if (this.breakingHere > 0.4) {
      // the wave broke on the rider: caught inside — next time sit further out
      st.endReason = 'caught inside';
      this.spotShift = Math.max(this.spotShift - 2.5, -10);
      this.setPhase('position');
      return;
    }
    if (onFace && vAlong > this.config.popSpeed * this.c && this.phaseTime > 0.5) {
      out.popUp = true;
      st.catches++;
      this.headI = 0;
      this.angInit = false;
      this.setPhase('popup');
      return;
    }
    // missed: the crest went past (now on the back of the wave) or took too long
    if ((this.psi < -0.4 && this.phaseTime > 2) || this.phaseTime > 12) {
      st.endReason = 'missed the wave';
      this.spotShift = Math.min(this.spotShift + 2, 8);
      this.nextSide = st.side > 0 ? -1 : 1;
      if (this.config.side === 0) st.side = this.nextSide;
      this.setPhase('position');
    }
  }

  private doRide(sim: SurfSim, dt: number, out: SurfInput, popping: boolean): void {
    const st = this.status;
    const cfg = this.config;
    const b = sim.board;
    const r = sim.rider;
    const s = st.side;
    const px = b.position.x;
    const pz = b.position.z;
    // crest scan at ~20 Hz
    this.scanTimer -= dt;
    if (this.scanTimer <= 0) {
      this.scanTimer = 0.05;
      this.scanCrest(px, pz);
    }
    // frame: d = wave direction, n = along the crest toward the riding side
    const dX = this.dirX, dZ = this.dirZ;
    const nX = -dZ * s, nZ = dX * s;
    const hX = b.R[0], hZ = b.R[6];
    const angle = Math.atan2(hX * nX + hZ * nZ, hX * dX + hZ * dZ);
    st.angle = angle;
    const vx = b.velocity.x, vz = b.velocity.z;
    const v = Math.hypot(vx, vz);
    const vAl = vx * dX + vz * dZ;

    // --- target height on the face (phase): nominal; lower (faster) when the curl is close,
    // higher (slower) when far ahead of it
    let psiT = this.config.trimPhase;
    const pocket = this.config.pocket;
    if (st.curl < pocket) psiT += 0.35 * (1 - st.curl / pocket);
    else if (st.curl > 2.5 * pocket) psiT -= 0.2;
    st.phaseTarget = psiT;
    // --- along-wave speed that moves the board toward the target phase, and the heading angle
    // that gives it at the current speed
    const ePsi = this.psi - psiT;
    const vAlT = this.c - clamp(2.8 * ePsi, -2.5, 3.5);
    const cosT = clamp(vAlT / Math.max(v, 0.5), -0.6, 0.97);
    let angT = clamp(Math.acos(cosT), cfg.angleMin, cfg.angleMax);
    // just after the pop-up / drop: do not turn hard yet, angle down the line
    if (popping) angT = clamp(angT, 25 * DEG, 70 * DEG);
    // smooth the target (the phase under the board is noisy over chop)
    if (!this.angInit) {
      this.angF = angT;
      this.angInit = true;
    }
    this.angF += (angT - this.angF) * (1 - Math.exp(-dt / Math.max(cfg.angleTau, 1e-3)));
    angT = this.angF;
    st.angleTarget = angT;

    // --- heading controller → lean (the carve reflex turns the lean into a bank)
    const turnRateAlong = r.turnRate * s; // + = turning toward the riding side (angle grows)
    const e = wrap(angT - angle);
    const uMax = cfg.maxLean;
    const u0 = cfg.steerGain * e - cfg.steerDamp * turnRateAlong;
    // integral of the heading error, frozen while the command is saturated (anti-windup)
    if (Math.abs(u0 + this.headI) < uMax) this.headI = clamp(this.headI + cfg.steerI * e * dt, -uMax, uMax);
    const u = clamp(u0 + this.headI, -uMax, uMax);
    out.leanSide = s * u;

    // --- trim: weight forward for speed when the curl is close or the board is slow for its
    // line, back to slow down / hold the top of the face; crouch through turns and whitewater
    const needSpeed = (st.curl < pocket ? 1 : 0) + (v < this.c / Math.max(Math.cos(Math.min(angle, 1.3)), 0.3) - 0.3 ? 0.5 : 0);
    let trim = 0.15 + 0.35 * needSpeed;
    if (st.curl > 2.5 * pocket && this.psi < psiT) trim = -0.2;
    // steep drop: weight back a touch so the nose does not dig in
    if (this.psi < 0.5 && b.R[3] < -0.25) trim = Math.min(trim, -0.3);
    // take-off: weight over the front foot while getting up, so the board keeps running down the
    // face (standing up over the tail stalls it on the top of the wave)
    if (popping) trim = this.config.popTrim;
    out.leanForward = clamp(trim, -1, 1);
    out.crouch = clamp(0.3 + 0.5 * Math.abs(u) + (this.breakingHere > 0.3 ? 0.3 : 0), 0, 1);

    if (popping) return;
    // --- end of the ride: section closing out ahead, wave backed off, or stuck in the whitewater
    this.whiteTime = this.breakingHere > 0.5 ? this.whiteTime + dt : Math.max(this.whiteTime - dt, 0);
    let end = '';
    if (st.closeout < 6 && st.curl < 6) end = 'closed out';
    else if (this.height < 0.45 && this.psi > 1.2) end = 'wave backed off';
    else if (this.fullness < 0.4 && this.psi > 2.0) end = 'wave backed off';
    else if (this.whiteTime > 1.5) end = 'caught by the whitewater';
    else if (this.psi > 2.6 || this.psi < -0.8) end = 'lost the wave';
    if (end) {
      st.endReason = end;
      this.setPhase('kickout');
    }
    void vAl;
  }

  private doKickout(sim: SurfSim, out: SurfInput): void {
    const st = this.status;
    const b = sim.board;
    const s = st.side;
    // turn up the face and over the back of the wave
    const dX = this.dirX, dZ = this.dirZ;
    const nX = -dZ * s, nZ = dX * s;
    const angle = Math.atan2(b.R[0] * nX + b.R[6] * nZ, b.R[0] * dX + b.R[6] * dZ);
    const e = wrap(150 * DEG - angle);
    out.leanSide = s * clamp(2 * e - 0.3 * sim.rider.turnRate * s, -1, 1);
    out.leanForward = -0.3;
    out.crouch = 0.5;
    if (this.psi < -0.6 || this.phaseTime > 4) this.setPhase('done');
  }
}
