/**
 * The game's notion of a ride (HUD badge + timer, last/best ride, follow-cam framing) and of
 * "caught" (the pop-up cue), updated once per physics step.
 *
 *  - Ride: standing (or popping up) while `telemetry.riding` (the board moves with a wave on its
 *    face/crest). `telemetry.riding` alone also holds for a prone belly ride in the whitewater
 *    or a riderless board carried in after a wipeout — those are not rides. Dropouts shorter
 *    than RIDE_GRACE (a climb to the lip, a turn across the face) don't end a ride; a wipeout
 *    ends it at once. Rides longer than MIN_RIDE are recorded as last/best.
 *  - Caught: prone, on the face and moving with the wave at the speed the autopilot pops up at
 *    (DEFAULT_AUTOPILOT.popPhase / popSpeed: "the board runs"), held 0.15 s against flicker.
 */
import { DEFAULT_AUTOPILOT } from '../physics/autopilot';
import { PHYSICS_DT } from '../physics/constants';
import type { SurfSim } from '../physics/SurfSim';

/** Rides shorter than this (s) are not recorded. */
export const MIN_RIDE = 0.5;
/** Riding dropouts shorter than this (s) don't end a ride. */
export const RIDE_GRACE = 0.6;
const CAUGHT_HOLD = 0.15;

export class RideTracker {
  /** Last finished ride and the best this session, s. */
  lastRide = 0;
  bestRide = 0;
  /** Prone and the board runs with the wave: pop up now. */
  caught = false;
  /** Current ride: sim time it started (−1 = none) and when it was last seen riding. */
  private start = -1;
  private last = 0;
  private caughtUntil = -1;

  /** Riding right now (including a short dropout). */
  get riding(): boolean {
    return this.start >= 0;
  }

  /** Seconds of the current ride (0 = not riding). */
  get time(): number {
    return this.start < 0 ? 0 : this.last - this.start;
  }

  /** After a physics step. */
  update(sim: SurfSim): void {
    const t = sim.telemetry;
    const st = sim.rider.stance;
    const up = st === 'standing' || st === 'popping';
    if (up && t.riding) {
      if (this.start < 0) this.start = sim.time - PHYSICS_DT;
      this.last = sim.time;
    } else if (this.start >= 0 && (!up || sim.time - this.last > RIDE_GRACE)) {
      this.end();
    }
    const w = t.water;
    const ap = DEFAULT_AUTOPILOT;
    const runs =
      st === 'prone' &&
      w.waveHeight > 0.3 &&
      w.breaking < 0.4 &&
      w.wavePhase > ap.popPhase &&
      w.wavePhase < 1.6 &&
      t.speedAlongWave > ap.popSpeed * w.phaseSpeed;
    if (runs) this.caughtUntil = sim.time + CAUGHT_HOLD;
    this.caught = st === 'prone' && sim.time <= this.caughtUntil;
  }

  /** End the current ride (if any) and record it — also on a teleport (R, T, board swap). */
  end(): void {
    const d = this.time;
    if (d > MIN_RIDE) {
      this.lastRide = d;
      this.bestRide = Math.max(this.bestRide, d);
    }
    this.start = -1;
  }

  /** Teleport: record a ride in progress, clear the cue. */
  teleport(): void {
    this.end();
    this.caught = false;
    this.caughtUntil = -1;
  }
}
