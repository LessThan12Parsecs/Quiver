/**
 * Game shell logic that runs in Node (no WebGL):
 *  - URL parameters: unknown and Object.prototype board names fall back to the funboard;
 *  - RideTracker: a ride is standing on a wave — prone belly rides and a riderless board carried
 *    by the whitewater don't count, short dropouts don't end a ride, a teleport records the ride
 *    in progress; the "caught" cue needs the board running on the face;
 *  - Hud: a flash shown after a wipeout (T's "Drop in!") survives the stance change, wipeout
 *    advice follows the direction the rider fell.
 */
import { describe, expect, it } from 'vitest';
import { parseGameParams } from '../src/game/Game';
import { Hud, type HudInfo } from '../src/game/Hud';
import { MIN_RIDE, RIDE_GRACE, RideTracker } from '../src/game/RideTracker';
import { OceanModel } from '../src/ocean/waveModel';
import { DEFAULT_AUTOPILOT } from '../src/physics/autopilot';
import { BOARD_PRESETS } from '../src/physics/boardShape';
import { PHYSICS_DT } from '../src/physics/constants';
import type { Stance } from '../src/physics/Rider';
import { waveSpawn } from '../src/physics/spawn';
import { SurfSim, createSurfInput, type SurfTelemetry } from '../src/physics/SurfSim';

describe('parseGameParams', () => {
  it('rejects Object.prototype keys as board names', () => {
    for (const b of ['toString', 'constructor', '__proto__', 'hasOwnProperty', 'valueOf', 'nope']) {
      expect(parseGameParams(new URLSearchParams(`board=${b}`)).board).toBe('funboard');
    }
    expect(parseGameParams(new URLSearchParams('board=softtop')).board).toBe('softtop');
  });
});

/** Minimal SurfSim stand-in: the fields RideTracker reads. */
function fakeSim(): { sim: SurfSim; set(o: { stance?: Stance; riding?: boolean; psi?: number; vAlong?: number; breaking?: number }): void; step(): void } {
  const telemetry = {
    riding: false,
    speedAlongWave: 0,
    water: { waveHeight: 1.4, breaking: 0, wavePhase: 0, phaseSpeed: 5 },
  } as unknown as SurfTelemetry;
  const state = { time: 0, telemetry, rider: { stance: 'prone' as Stance } };
  return {
    sim: state as unknown as SurfSim,
    set(o) {
      if (o.stance) state.rider.stance = o.stance;
      if (o.riding !== undefined) telemetry.riding = o.riding;
      if (o.psi !== undefined) telemetry.water.wavePhase = o.psi;
      if (o.vAlong !== undefined) telemetry.speedAlongWave = o.vAlong;
      if (o.breaking !== undefined) telemetry.water.breaking = o.breaking;
    },
    step() {
      state.time += PHYSICS_DT;
    },
  };
}

describe('RideTracker', () => {
  const run = (f: ReturnType<typeof fakeSim>, rt: RideTracker, seconds: number): void => {
    for (let i = 0; i < Math.round(seconds / PHYSICS_DT); i++) {
      f.step();
      rt.update(f.sim);
    }
  };

  it('counts standing on a wave, not prone belly rides or a riderless board', () => {
    const f = fakeSim();
    const rt = new RideTracker();
    f.set({ stance: 'prone', riding: true });
    run(f, rt, 3);
    expect(rt.riding).toBe(false);
    f.set({ stance: 'standing' });
    run(f, rt, 2);
    expect(rt.riding).toBe(true);
    expect(rt.time).toBeCloseTo(2, 1);
    // wipeout: the board keeps being carried (telemetry.riding) but the ride is over
    f.set({ stance: 'fallen' });
    run(f, rt, 3);
    expect(rt.riding).toBe(false);
    expect(rt.lastRide).toBeCloseTo(2, 1);
    expect(rt.bestRide).toBeCloseTo(2, 1);
  });

  it('bridges short dropouts, ends on long ones, ignores very short rides', () => {
    const f = fakeSim();
    const rt = new RideTracker();
    f.set({ stance: 'standing', riding: true });
    run(f, rt, 1);
    f.set({ riding: false });
    run(f, rt, RIDE_GRACE * 0.5);
    expect(rt.riding).toBe(true);
    f.set({ riding: true });
    run(f, rt, 1);
    expect(rt.time).toBeGreaterThan(2 + RIDE_GRACE * 0.5 - 0.05);
    f.set({ riding: false });
    run(f, rt, RIDE_GRACE + 0.1);
    expect(rt.riding).toBe(false);
    expect(rt.lastRide).toBeGreaterThan(2);
    const best = rt.bestRide;
    f.set({ riding: true });
    run(f, rt, MIN_RIDE * 0.5);
    f.set({ stance: 'fallen' });
    run(f, rt, 0.1);
    expect(rt.lastRide).toBeCloseTo(best, 6);
  });

  it('a teleport records the ride in progress', () => {
    const f = fakeSim();
    const rt = new RideTracker();
    f.set({ stance: 'standing', riding: true });
    run(f, rt, 1.5);
    rt.teleport();
    expect(rt.riding).toBe(false);
    expect(rt.lastRide).toBeCloseTo(1.5, 1);
  });

  it('caught: prone, on the face, running with the wave', () => {
    const f = fakeSim();
    const rt = new RideTracker();
    const c = 5;
    f.set({ stance: 'prone', psi: 0.5, vAlong: 0.6 * c });
    run(f, rt, 0.5);
    expect(rt.caught).toBe(false); // the 0.6 c "riding" threshold is not a catch
    f.set({ vAlong: (DEFAULT_AUTOPILOT.popSpeed + 0.05) * c });
    run(f, rt, 0.1);
    expect(rt.caught).toBe(true);
    f.set({ psi: -0.3 }); // the crest went past: on the back of the wave
    run(f, rt, 0.3);
    expect(rt.caught).toBe(false);
    f.set({ psi: 0.5, breaking: 0.8 }); // broke on the rider
    run(f, rt, 0.1);
    expect(rt.caught).toBe(false);
  });

  it('real simulation (wave spawns, no input): ride time never exceeds the time standing', () => {
    const ocean = new OceanModel();
    const inp = createSurfInput();
    for (const side of [1, -1] as const) {
      const res = waveSpawn(ocean, { fromTime: 30, side });
      expect(res).not.toBeNull();
      const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
      sim.reset(res!.spawn);
      const rt = new RideTracker();
      let up = 0;
      for (let i = 0; i < Math.round(8 / PHYSICS_DT); i++) {
        sim.step(PHYSICS_DT, inp);
        rt.update(sim);
        const st = sim.rider.stance;
        if (st === 'standing' || st === 'popping') up += PHYSICS_DT;
        else expect(rt.riding).toBe(false);
      }
      rt.end();
      expect(rt.bestRide).toBeLessThanOrEqual(up + 1e-9);
    }
  });
});

/** Tiny DOM stand-in for the HUD's elements. */
function fakeDom(): Map<string, { textContent: string; innerHTML: string; style: Record<string, string>; classList: { set: Set<string> } }> {
  const els = new Map<string, ReturnType<typeof make>>();
  function make() {
    const set = new Set<string>();
    return {
      textContent: '',
      innerHTML: '',
      style: {} as Record<string, string>,
      classList: {
        set,
        add: (...c: string[]) => c.forEach((x) => set.add(x)),
        remove: (...c: string[]) => c.forEach((x) => set.delete(x)),
        toggle: (c: string, on?: boolean) => ((on ?? !set.has(c)) ? set.add(c) : set.delete(c), set.has(c)),
        contains: (c: string) => set.has(c),
      },
    };
  }
  (globalThis as unknown as { document: unknown }).document = {
    getElementById: (id: string) => {
      if (!els.has(id)) els.set(id, make());
      return els.get(id);
    },
  };
  return els;
}

describe('Hud', () => {
  const info = (o: Partial<HudInfo> = {}): HudInfo => ({
    boardName: 'Funboard',
    boardVolumeL: 50,
    fps: 60,
    timeScale: 1,
    paused: false,
    camera: 'follow',
    riding: false,
    rideTime: 0,
    caught: false,
    lastRide: 0,
    bestRide: 0,
    gamepad: false,
    simLoad: 1,
    autopilot: null,
    tipOver: null,
    ...o,
  });
  const telemetry = (stance: Stance, reason: SurfTelemetry['wipeoutReason'] = null): SurfTelemetry => {
    const sim = new SurfSim(new OceanModel(), BOARD_PRESETS.funboard);
    return { ...sim.telemetry, stance, wipeoutReason: reason };
  };

  it("a flash after a wipeout (T: 'Drop in!') is not erased when the rider is back up", () => {
    const els = fakeDom();
    const hud = new Hud();
    const msg = els.get('hud-message')!;
    hud.update(telemetry('standing'), info(), 0);
    hud.update(telemetry('fallen', 'pearl'), info(), 1);
    expect([...msg.classList.set].sort()).toEqual(['bad', 'on']);
    expect(els.get('hud-message-big')!.textContent).toContain('Pearled');
    hud.flash('Drop in!', '1.5 m set wave', 1.6, 2);
    hud.update(telemetry('standing'), info(), 2.05);
    expect(msg.classList.set.has('on')).toBe(true);
    expect(els.get('hud-message-big')!.textContent).toBe('Drop in!');
    hud.update(telemetry('standing'), info(), 4);
    expect(msg.classList.set.has('on')).toBe(false);
    // R after a wipeout (no flash): the wipeout message goes away with the stance change
    hud.update(telemetry('fallen', 'balance'), info({ tipOver: 'side' }), 5);
    expect(msg.classList.set.has('on')).toBe(true);
    hud.update(telemetry('prone'), info(), 6);
    expect(msg.classList.set.has('on')).toBe(false);
  });

  it('balance wipeout advice follows how the rider fell', () => {
    const els = fakeDom();
    const hud = new Hud();
    hud.update(telemetry('standing'), info(), 0);
    hud.update(telemetry('fallen', 'balance'), info({ tipOver: 'back' }), 1);
    expect(els.get('hud-message-big')!.textContent).toContain('back');
    expect(els.get('hud-message-small')!.textContent).not.toContain('leaned');
    hud.update(telemetry('standing'), info(), 2);
    hud.update(telemetry('fallen', 'balance'), info({ tipOver: 'side' }), 3);
    expect(els.get('hud-message-small')!.textContent).toContain('rail');
  });

  it('badge: CAUGHT while prone and running, RIDING while riding', () => {
    const els = fakeDom();
    const hud = new Hud();
    const badge = els.get('hud-ride')!;
    hud.update(telemetry('prone'), info({ caught: true }), 0);
    expect(badge.classList.set.has('on') && badge.classList.set.has('caught')).toBe(true);
    hud.update(telemetry('standing'), info({ riding: true, rideTime: 1.2 }), 1);
    expect(badge.classList.set.has('caught')).toBe(false);
    expect(badge.textContent).toContain('RIDING');
    hud.update(telemetry('fallen', 'pearl'), info(), 2);
    expect(badge.classList.set.has('on')).toBe(false);
  });
});
