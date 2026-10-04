/**
 * Board + rider physics scenarios (flat water on a calm ocean, waves on the default ocean).
 * Key numbers are printed so the calibration can be read from the test log.
 */
import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { DEFAULT_OCEAN_CONFIG, cloneOceanConfig } from '../src/ocean/oceanConfig';
import { OceanModel } from '../src/ocean/waveModel';
import { BoardHull, createHullDiagnostics } from '../src/physics/BoardHull';
import { BOARD_PRESETS, cloneBoardSpec, getBoardShape, type BoardSpec } from '../src/physics/boardShape';
import { GRAVITY, RHO_WATER } from '../src/physics/constants';
import { ImplicitSystem } from '../src/physics/implicit';
import { RIDER_MODEL } from '../src/physics/Rider';
import { RigidBody } from '../src/physics/RigidBody';
import { lineupSpawn, waveSpawn } from '../src/physics/spawn';
import { SurfSim, createSurfInput, type SurfInput } from '../src/physics/SurfSim';
import { Autopilot } from '../src/physics/autopilot';

const DT = 1 / 240;
const calmCfg = cloneOceanConfig(DEFAULT_OCEAN_CONFIG);
calmCfg.swells = [];
calmCfg.wind.count = 0;
const calm = new OceanModel(calmCfg);
const ocean = new OceanModel();
const PRESETS = Object.values(BOARD_PRESETS) as BoardSpec[];

function run(sim: SurfSim, seconds: number, input: SurfInput, each?: (sim: SurfSim, i: number) => void): void {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) {
    each?.(sim, i);
    sim.step(DT, input);
  }
}

/** A free board (no rider) with gravity and the hull's water forces only. */
function floatBoard(spec: BoardSpec, seconds: number, drop: number) {
  const shape = getBoardShape(spec);
  const hull = new BoardHull(shape);
  const body = new RigidBody(shape.mass, shape.massProperties.inertia);
  body.position.set(-75, drop, 5);
  body.updateDerived();
  const sys = new ImplicitSystem();
  const diag = createHullDiagnostics(shape.fins.length);
  const u = new Float64Array(9);
  const ys: number[] = [];
  const vols: number[] = [];
  for (let i = 0; i < seconds / DT; i++) {
    calm.setTime(i * DT);
    hull.sampleWater(calm, body);
    sys.clear();
    const M = sys.M;
    M.fill(0);
    M[0] = M[10] = M[20] = body.mass;
    const I = body.inertiaWorld;
    M[30] = I[0]; M[31] = I[1]; M[32] = I[2]; M[40] = I[4]; M[41] = I[5]; M[50] = I[8];
    M[60] = M[70] = M[80] = 1;
    sys.addBoardForce(0, -body.mass * GRAVITY, 0, 0, 0, 0);
    hull.computeForces(body, sys, diag);
    body.applyGyroscopic(DT);
    u[0] = body.velocity.x; u[1] = body.velocity.y; u[2] = body.velocity.z;
    u[3] = body.angularVelocity.x; u[4] = body.angularVelocity.y; u[5] = body.angularVelocity.z;
    u[6] = u[7] = u[8] = 0;
    const du = sys.solve(DT, u);
    body.velocity.set(u[0] + du[0], u[1] + du[1], u[2] + du[2]);
    body.angularVelocity.set(u[3] + du[3], u[4] + du[4], u[5] + du[5]);
    body.integratePosition(DT);
    ys.push(body.position.y);
    vols.push(diag.submergedVolume);
  }
  return { ys, vols, body };
}

describe('flat water', () => {
  it('1) unloaded boards float with submerged volume = mass/ρ and the bobbing decays', () => {
    for (const spec of PRESETS) {
      const { ys, vols, body } = floatBoard(spec, 10, 0.05);
      const target = spec.mass / RHO_WATER;
      const last = vols.slice(-480);
      const mean = last.reduce((a, b) => a + b, 0) / last.length;
      const swing = (arr: number[]) => Math.max(...arr) - Math.min(...arr);
      const early = swing(ys.slice(0, 120));
      const late = swing(ys.slice(-480));
      console.log(`  ${spec.id}: submerged ${(mean * 1000).toFixed(2)} L (mass/ρ ${(target * 1000).toFixed(2)} L), heave swing ${(early * 100).toFixed(1)} → ${(late * 1000).toFixed(2)} mm`);
      expect(Math.abs(mean - target) / target).toBeLessThan(0.03);
      expect(late).toBeLessThan(early * 0.05);
      expect(Math.abs(body.quaternion.x) + Math.abs(body.quaternion.z)).toBeLessThan(0.05);
    }
  });

  it('2) a prone 75 kg rider floats on every preset; the shortboard rides much lower than the soft-top', () => {
    const deck: Record<string, number> = {};
    for (const spec of PRESETS) {
      const sim = new SurfSim(calm, spec, { wipeouts: false });
      sim.reset({ ...lineupSpawn(calm, 0) });
      let maxSpeed = 0;
      run(sim, 4, createSurfInput(), (s) => (maxSpeed = Math.max(maxSpeed, s.board.velocity.length())));
      const t = sim.telemetry;
      deck[spec.id] = t.deckDepth;
      console.log(
        `  ${spec.id}: deck ${(t.deckDepth * 100).toFixed(1)} cm under, board ${t.submergedLiters.toFixed(1)} L + body ${t.riderSubmergedLiters.toFixed(1)} L, pitch ${((t.pitchRad * 180) / Math.PI).toFixed(1)}°`,
      );
      expect(sim.rider.stance).toBe('prone');
      expect(t.deckDepth).toBeLessThan(0.2); // floats near the surface, not sinking
      expect(sim.board.position.y).toBeGreaterThan(-0.4);
      expect(maxSpeed).toBeLessThan(0.5); // no explosion
      expect(Math.abs(t.rollRad)).toBeLessThan(0.1);
      // buoyancy balances the weight (board + rider)
      const displaced = (t.submergedLiters + t.riderSubmergedLiters) / 1000;
      expect(displaced * RHO_WATER).toBeCloseTo(spec.mass + 75, -1);
    }
    expect(deck.shortboard - deck.softtop).toBeGreaterThan(0.08);
  });

  it('3) standing still: the 28 L shortboard sinks deep, the 86 L soft-top keeps its deck near the surface', () => {
    // feet centred over the centre of volume: this measures volume, not trim (a riding stance
    // near the tail sinks the tail of any board at rest)
    const centred = (spec: BoardSpec) => {
      const s = cloneBoardSpec(spec);
      const sh = getBoardShape(spec);
      s.stance.backFoot = sh.centerOfVolume.x + sh.comShape.x - s.stance.feetSpread / 2;
      return s;
    };
    const depth = (spec: BoardSpec) => {
      const sim = new SurfSim(calm, centred(spec), { wipeouts: false, balanceAssist: 1 });
      sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
      let maxDeck = -1;
      run(sim, 1.5, createSurfInput(), (s) => (maxDeck = Math.max(maxDeck, s.telemetry.deckDepth)));
      return { deck: sim.telemetry.deckDepth, maxDeck, body: sim.telemetry.riderSubmergedLiters };
    };
    const sb = depth(BOARD_PRESETS.shortboard);
    const st = depth(BOARD_PRESETS.softtop);
    console.log(`  shortboard deck ${(sb.deck * 100).toFixed(0)} cm under (body ${sb.body.toFixed(0)} L wet); soft-top deck ${(st.deck * 100).toFixed(1)} cm`);
    expect(sb.deck).toBeGreaterThan(0.4);
    expect(sb.body).toBeGreaterThan(20);
    expect(st.maxDeck).toBeLessThan(0.04);
    // with wipeouts on, standing still on the shortboard buries it
    const sim = new SurfSim(calm, centred(BOARD_PRESETS.shortboard));
    sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
    run(sim, 3, createSurfInput());
    expect(sim.rider.stance).toBe('fallen');
    expect(sim.rider.wipeoutReason).toBe('buried');
  });

  it('4) paddling top speed on flat water is 1.4–2.4 m/s on the funboard (and in range on every board)', () => {
    const input = createSurfInput();
    input.paddle = 1;
    for (const spec of PRESETS) {
      const sim = new SurfSim(calm, spec);
      sim.reset(lineupSpawn(calm, 0));
      let sum = 0;
      let n = 0;
      run(sim, 16, input, (s, i) => {
        if (i * DT > 10) {
          sum += Math.hypot(s.board.velocity.x, s.board.velocity.z);
          n++;
        }
      });
      const v = sum / n;
      console.log(`  ${spec.id}: paddling top speed ${v.toFixed(2)} m/s, thrust ${sim.telemetry.paddleThrustN.toFixed(0)} N (instant)`);
      expect(sim.rider.stance).toBe('prone');
      if (spec.id === 'funboard') {
        expect(v).toBeGreaterThan(1.4);
        expect(v).toBeLessThan(2.4);
      } else {
        expect(v).toBeGreaterThan(1.0);
        expect(v).toBeLessThan(2.6);
      }
    }
  });

  it('paddling harder on one side turns the board (steer)', () => {
    const input = createSurfInput();
    input.paddle = 1;
    input.steer = 1;
    const sim = new SurfSim(calm, BOARD_PRESETS.funboard);
    sim.reset(lineupSpawn(calm, 0));
    run(sim, 6, input);
    console.log(`  heading after 6 s of right steer: ${((sim.telemetry.headingRad * 180) / Math.PI).toFixed(0)}°`);
    expect(sim.telemetry.headingRad).toBeGreaterThan(0.3);
  });

  it('5) planing glide from 6 m/s: smooth deceleration, submerged volume clearly below static', () => {
    const sim = new SurfSim(calm, BOARD_PRESETS.funboard, { wipeouts: false });
    sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 6, time: 0 });
    const speeds: number[] = [];
    const subs: number[] = [];
    run(sim, 2, createSurfInput(), (s) => {
      // speed of the board + rider centre of mass (the light board alone jitters with the legs)
      const m = s.board.mass + s.rider.config.mass;
      speeds.push((s.board.velocity.x * s.board.mass + s.rider.velocity.x * s.rider.config.mass) / m);
      subs.push(s.telemetry.submergedLiters + s.telemetry.riderSubmergedLiters);
    });
    const v0 = speeds[24];
    const v1 = speeds[speeds.length - 1];
    let maxJump = 0;
    for (let i = 25; i < speeds.length; i++) maxJump = Math.max(maxJump, speeds[i] - speeds[i - 1]);
    const meanSub = subs.slice(24, 360).reduce((a, b) => a + b, 0) / 336;
    const atRest = ((75 + 4.5) / RHO_WATER) * 1000; // litres displaced (board + body) at rest
    console.log(`  glide: ${v0.toFixed(2)} → ${v1.toFixed(2)} m/s in 1.9 s (decel ${((v0 - v1) / 1.9).toFixed(2)} m/s²), displaced ${meanSub.toFixed(1)} L vs ${atRest.toFixed(1)} L at rest`);
    expect(v1).toBeLessThan(v0 - 0.8);
    expect(v1).toBeGreaterThan(3);
    expect(maxJump).toBeLessThan(0.05); // no impulses
    expect(meanSub).toBeLessThan(0.45 * atRest);
  });

  it('planing calibration: towed funboard + standing rider is on the plane from ~4 m/s', () => {
    const rows: string[] = [];
    const subAt: Record<number, number> = {};
    for (const v of [4, 5, 6, 8]) {
      const sim = new SurfSim(calm, BOARD_PRESETS.funboard, { wipeouts: false });
      sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: v, time: 0 });
      let integ = 0;
      let sub = 0;
      let lift = 0;
      let trim = 0;
      let n = 0;
      let tow = 0;
      run(sim, 6, createSurfInput(), (s, i) => {
        const e = v - s.board.velocity.x;
        integ += e * DT;
        const f = Math.max(0, 400 * e + 300 * integ);
        s.extraBoardForce.set(f, 0, 0);
        if (i * DT > 3) {
          sub += s.telemetry.submergedLiters;
          lift += s.telemetry.planingLiftN;
          trim += s.telemetry.pitchRad;
          tow += f;
          n++;
        }
      });
      subAt[v] = sub / n;
      rows.push(`${v} m/s: ${(sub / n).toFixed(1)} L, dynamic lift ${((lift / n / ((75 + 4.5) * GRAVITY)) * 100).toFixed(0)} %, trim ${((trim / n) * 57.3).toFixed(1)}°, drag ${(tow / n).toFixed(0)} N`);
    }
    console.log(`  towed planing: ${rows.join(' | ')}`);
    expect(subAt[4]).toBeLessThan(0.5 * 50);
    expect(subAt[8]).toBeLessThan(subAt[5]);
  });

  it('6) directional stability: a 10° sideslip at 5 m/s is corrected by the fins', () => {
    const slip = (spec: BoardSpec) => {
      const sim = new SurfSim(calm, spec, { wipeouts: false });
      sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 5, time: 0 });
      const b = (10 * Math.PI) / 180;
      sim.board.velocity.set(5 * Math.cos(b), sim.board.velocity.y, 5 * Math.sin(b));
      sim.rider.velocity.set(5 * Math.cos(b), sim.rider.velocity.y, 5 * Math.sin(b));
      run(sim, 1, createSurfInput());
      const v = sim.board.velocity;
      const course = Math.atan2(v.z, v.x);
      return ((course - sim.telemetry.headingRad) * 180) / Math.PI;
    };
    const withFins = slip(BOARD_PRESETS.funboard);
    const finless = cloneBoardSpec(BOARD_PRESETS.funboard);
    finless.fins = [];
    const without = slip(finless);
    console.log(`  sideslip after 1 s: ${withFins.toFixed(2)}° with fins, ${without.toFixed(2)}° without`);
    expect(Math.abs(withFins)).toBeLessThan(3);
    expect(Math.abs(without)).toBeGreaterThan(Math.abs(withFins));
  });

  it('7) carving: a full lean at 7 m/s banks the board onto that rail and turns it that way (R < 10 m), rider stays up', () => {
    for (const side of [1, -1]) {
      const sim = new SurfSim(calm, BOARD_PRESETS.funboard);
      sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 7, time: 0 });
      const input = createSurfInput();
      input.leanSide = side;
      let integ = 0;
      let maxBank = 0;
      let bank07 = 0;
      let maxYaw = 0;
      run(sim, 2.5, input, (s, i) => {
        // hold the speed with a tow along the heading (a wave face would provide it)
        const hx = s.board.R[0], hz = s.board.R[6];
        const nn = Math.hypot(hx, hz);
        const vh = (s.board.velocity.x * hx + s.board.velocity.z * hz) / nn;
        const e = 7 - vh;
        integ += e * DT;
        const f = Math.max(0, 400 * e + 300 * integ);
        s.extraBoardForce.set((f * hx) / nn, 0, (f * hz) / nn);
        maxBank = Math.max(maxBank, side * s.rider.bankAngle);
        maxYaw = Math.max(maxYaw, side * s.rider.turnRate);
        if (i === Math.round(0.7 / DT)) bank07 = side * s.rider.bankAngle;
      });
      const t = sim.telemetry;
      const D = 180 / Math.PI;
      console.log(
        `  leanSide ${side}: bank ${(bank07 * D).toFixed(1)}° at 0.7 s (max ${(maxBank * D).toFixed(1)}°), yaw rate max ${(maxYaw * D).toFixed(0)}°/s ` +
          `(radius ${(7 / maxYaw).toFixed(1)} m), heading after 2.5 s ${(t.headingRad * D).toFixed(0)}°, stance ${t.stance}`,
      );
      expect(t.stance).toBe('standing');
      expect(bank07).toBeGreaterThan(20 / D);
      expect(maxBank).toBeGreaterThan(25 / D);
      expect(maxYaw).toBeGreaterThan(45 / D);
      expect(7 / maxYaw).toBeLessThan(10);
      expect(side * t.headingRad).toBeGreaterThan(45 / D);
      expect(Math.abs(t.rollRad)).toBeLessThan(1.2);
    }
  });

  it('a hard lean below 2 m/s tips a floaty board\'s rider over; standing level does not', () => {
    for (const id of ['softtop', 'longboard'] as const) {
      const fallsAt = (lean: number): number => {
        const sim = new SurfSim(calm, BOARD_PRESETS[id]);
        sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 1.5, time: 0 });
        const input = createSurfInput();
        input.leanSide = lean;
        let t = 0;
        for (; t < 4 && sim.rider.stance !== 'fallen'; t += DT) sim.step(DT, input);
        return sim.rider.stance === 'fallen' ? t : Infinity;
      };
      const hard = fallsAt(1);
      const level = fallsAt(0);
      console.log(`  ${id} at 1.5 m/s: full lean falls after ${hard.toFixed(2)} s; no lean: ${level === Infinity ? 'stays up' : `falls ${level.toFixed(2)} s`}`);
      expect(hard).toBeLessThan(3.5);
      expect(level).toBe(Infinity);
    }
  });

  it('pop-up: prone → popping → standing in 0.6 s on a planing-speed board', () => {
    const sim = new SurfSim(calm, BOARD_PRESETS.softtop);
    sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'prone', speed: 5, time: 0 });
    const input = createSurfInput();
    input.popUp = true;
    sim.step(DT, input);
    input.popUp = false;
    expect(sim.rider.stance).toBe('popping');
    run(sim, 0.7, input);
    expect(sim.rider.stance).toBe('standing');
    expect(sim.rider.pose.popProgress).toBe(1);
    // the stance target is ~0.95 m above the feet
    const h = sim.rider.targetLocal.y - sim.rider.feetMidLocal.y;
    expect(h).toBeGreaterThan(0.8);
  });

  it('after a wipeout the rider floats free on the leash and reset returns to the lineup', () => {
    const sim = new SurfSim(calm, BOARD_PRESETS.funboard);
    sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 7, time: 0 });
    const input = createSurfInput();
    input.leanSide = 1;
    sim.collectDebug = true;
    // a raw (unassisted) full lean at speed throws the rider off
    sim.rider.config.balanceAssist = 0;
    let maxDist = 0;
    run(sim, 8, input, (s) => {
      if (s.rider.stance === 'fallen') maxDist = Math.max(maxDist, s.rider.position.distanceTo(s.board.position));
    });
    expect(sim.rider.stance).toBe('fallen');
    expect(sim.rider.wipeoutReason).not.toBeNull();
    console.log(`  wipeout: ${sim.rider.wipeoutReason}, max rider-board distance ${maxDist.toFixed(2)} m`);
    expect(maxDist).toBeLessThan(3.5);
    expect(sim.rider.position.y).toBeGreaterThan(-0.6); // floats
    expect(sim.debugForceCount).toBeGreaterThan(0);
    const reset = createSurfInput();
    reset.reset = true;
    sim.step(DT, reset);
    expect(sim.rider.stance).toBe('prone');
    expect(sim.board.position.x).toBeCloseTo(-75, 6);
  });
});

describe('waves', () => {
  it('waveSpawn finds a steepening unbroken set-wave face down the line from the peak', () => {
    const before = ocean.time;
    const ws = waveSpawn(ocean, { fromTime: 0 });
    expect(ocean.time).toBe(before); // the search restores the ocean time
    expect(ws).not.toBeNull();
    const r = ws!;
    console.log(`  spawn t=${r.time} at (${r.spawn.x}, ${r.spawn.z}) H=${r.waveHeight.toFixed(2)} fullness=${r.fullness.toFixed(2)} c=${r.phaseSpeed.toFixed(2)} slope=${r.slope.toFixed(2)}`);
    expect(r.waveHeight).toBeGreaterThan(1);
    expect(r.fullness).toBeGreaterThanOrEqual(0.5);
    expect(r.fullness).toBeLessThanOrEqual(0.75);
    expect(r.slope).toBeLessThan(0);
    expect(Math.abs(r.spawn.z)).toBeGreaterThanOrEqual(15);
    expect(r.spawn.stance).toBe('standing');
    expect(r.spawn.headingRad).toBeGreaterThan(0); // angled toward +Z, away from the peak
    // the board moves with the water across its heading (no fin kick at the start) and with the
    // wave along the wave direction
    const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
    sim.reset(r.spawn);
    expect(sim.telemetry.speedAlongWave).toBeGreaterThan(0.9 * r.phaseSpeed);
    expect(sim.telemetry.speedAlongWave).toBeLessThan(1.1 * r.phaseSpeed);
  });

  // Scripted rider: hold a line across the face (heading set by the wave phase under the board,
  // opening the angle when low on the face) and trim with leanForward from the phase.
  function scriptedRide(mode: 'trim' | 'back') {
    const side = -1;
    const angle = (60 * Math.PI) / 180;
    const ws = waveSpawn(ocean, { fromTime: 0, side, minFullness: 0.7, angleRad: angle })!;
    const sim = new SurfSim(ocean, BOARD_PRESETS.softtop);
    sim.reset(ws.spawn);
    const input = createSurfInput();
    let ride = 0;
    let best = 0;
    run(sim, 8, input, (s) => {
      const t = s.telemetry;
      const w = t.water;
      if (mode === 'trim') {
        const angT = Math.min(Math.max(angle + 1.2 * (w.wavePhase - 0.85), 0.2), 1.45);
        let err = Math.atan2(w.dirZ, w.dirX) + side * angT - t.headingRad;
        err = Math.atan2(Math.sin(err), Math.cos(err));
        input.leanSide = Math.max(-1, Math.min(1, err - 0.3 * s.rider.turnRate));
        input.leanForward = Math.max(-1, Math.min(1, 1.5 * (0.85 - w.wavePhase) - 0.3 * (t.speedAlongWave - w.phaseSpeed)));
        input.crouch = 0.5;
      } else {
        input.leanForward = -1;
      }
      const riding = t.stance !== 'fallen' && t.speed > 3 && t.riding;
      ride = riding ? ride + DT : 0;
      best = Math.max(best, ride);
    });
    return { best, sim, ws };
  }

  it('8) wave riding is feasible: a scripted trim rides ≥ 4 s at > 3 m/s with the wave; weight fully back loses it', () => {
    const trim = scriptedRide('trim');
    const back = scriptedRide('back');
    console.log(
      `  soft-top from waveSpawn t=${trim.ws.time} H=${trim.ws.waveHeight.toFixed(2)}: trimmed ride ${trim.best.toFixed(2)} s ` +
        `(end: ${trim.sim.rider.wipeoutReason ?? trim.sim.rider.stance}), weight back ${back.best.toFixed(2)} s (end: ${back.sim.rider.wipeoutReason ?? back.sim.rider.stance})`,
    );
    expect(trim.best).toBeGreaterThanOrEqual(4);
    expect(back.best).toBeLessThan(4);
    expect(back.best).toBeLessThan(trim.best - 1);
  });

  it('9) determinism: identical inputs give bit-identical states', () => {
    const go = () => {
      const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
      sim.reset(lineupSpawn(ocean, 40));
      const input = createSurfInput();
      run(sim, 3, input, (_s, i) => {
        input.paddle = i % 400 < 300 ? 1 : 0;
        input.steer = Math.sin(i * 0.01);
        input.popUp = i === 500;
      });
      return [sim.board.position.x, sim.board.position.y, sim.board.position.z, sim.board.quaternion.x, sim.rider.position.y, sim.time];
    };
    expect(go()).toEqual(go());
  });

  it('10) performance: 1 s of simulation (240 steps) takes < 150 ms in Node', () => {
    const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
    sim.reset(waveSpawn(ocean, { fromTime: 0 })!.spawn);
    const input = createSurfInput();
    run(sim, 0.5, input); // warm-up (JIT)
    const t0 = performance.now();
    run(sim, 1, input);
    const ms = performance.now() - t0;
    console.log(`  1 s of sim: ${ms.toFixed(1)} ms (${((ms / 240) * 1000).toFixed(0)} µs/step)`);
    expect(ms).toBeLessThan(150);
  });
});

// long simulations: generous timeouts (the default 5 s is tight on a loaded machine)
describe('rideability (autopilot, failure cases)', { timeout: 30_000 }, () => {
  /** Autopilot ride from a waveSpawn: seconds standing in the ride phase, why it ended, turns. */
  function autopilotRide(from: number, side: 1 | -1, seconds = 12) {
    const ws = waveSpawn(ocean, { fromTime: from, side })!;
    const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
    sim.reset(ws.spawn);
    const ap = new Autopilot(ocean, { autoReset: Infinity });
    ap.startRiding(side);
    const input = createSurfInput();
    let ride = 0;
    let turns = 0;
    let lastSign = 0;
    let headingMin = Infinity;
    let headingMax = -Infinity;
    for (let i = 0; i < seconds / DT; i++) {
      ap.update(sim, DT, input);
      sim.step(DT, input);
      if (ap.status.phase !== 'ride' || sim.rider.stance !== 'standing') break;
      ride = (i + 1) * DT;
      // a turn: the yaw rate swinging past ±25°/s, alternating
      const tr = sim.rider.turnRate;
      const sign = tr > 0.44 ? 1 : tr < -0.44 ? -1 : 0;
      if (sign !== 0 && sign !== lastSign) {
        turns++;
        lastSign = sign;
      }
      headingMin = Math.min(headingMin, ap.status.angle);
      headingMax = Math.max(headingMax, ap.status.angle);
    }
    const end = sim.rider.stance === 'fallen' ? `wipeout: ${sim.rider.wipeoutReason}` : ap.status.endReason || 'still riding';
    return { ride, turns, end, swing: headingMax - headingMin, sim, ap };
  }

  it('autopilot rides waveSpawn waves (3 set waves × both sides): median ≥ 7 s, the best ≥ 10 s, with turns', () => {
    const rides: number[] = [];
    let turns = 0;
    let swing = 0;
    const log: string[] = [];
    for (const from of [0, 40, 50]) {
      for (const side of [-1, 1] as const) {
        const r = autopilotRide(from, side);
        rides.push(r.ride);
        turns += r.turns;
        swing = Math.max(swing, r.swing);
        log.push(`${r.ride.toFixed(1)} s (${r.end}, ${r.turns} turns)`);
      }
    }
    const sorted = [...rides].sort((a, b) => a - b);
    const median = 0.5 * (sorted[2] + sorted[3]);
    console.log(`  autopilot rides: ${log.join(' | ')}; median ${median.toFixed(2)} s`);
    expect(median).toBeGreaterThanOrEqual(7);
    expect(sorted[5]).toBeGreaterThanOrEqual(10);
    expect(turns).toBeGreaterThanOrEqual(4);
    expect(swing).toBeGreaterThan((15 * Math.PI) / 180); // the heading works the face, not one fixed line
  });

  it('no input after waveSpawn: the rider loses the wave or falls within 3 s', () => {
    const log: string[] = [];
    for (const from of [0, 40, 60]) {
      for (const side of [-1, 1] as const) {
        const ws = waveSpawn(ocean, { fromTime: from, side })!;
        const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
        sim.reset(ws.spawn);
        const input = createSurfInput();
        let t = 0;
        let lost = Infinity;
        for (; t < 3 && sim.rider.stance !== 'fallen'; t += DT) {
          sim.step(DT, input);
          if (t > 0.3 && !sim.telemetry.riding && lost === Infinity) lost = t;
        }
        const fell = sim.rider.stance === 'fallen';
        log.push(fell ? `fell (${sim.rider.wipeoutReason}) ${t.toFixed(1)} s` : `lost the wave ${lost.toFixed(1)} s`);
        expect(fell || lost < 3).toBe(true);
      }
    }
    console.log(`  no input: ${log.join(' | ')}`);
  });

  it('weight fully forward on a steep take-off (the critical face next to the peak) pearls', () => {
    let pearls = 0;
    const log: string[] = [];
    for (const from of [0, 40, 60]) {
      for (const side of [-1, 1] as const) {
        const ws = waveSpawn(ocean, { fromTime: from, side, minFullness: 0.75, maxFullness: 0.92, offsets: [8, 10, 6, 12] })!;
        const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
        sim.reset(ws.spawn);
        const input = createSurfInput();
        input.leanForward = 1;
        let t = 0;
        for (; t < 5 && sim.rider.stance !== 'fallen'; t += DT) sim.step(DT, input);
        if (sim.rider.wipeoutReason === 'pearl') pearls++;
        log.push(`${sim.rider.wipeoutReason ?? 'up'} ${t.toFixed(1)} s`);
        expect(sim.rider.stance).toBe('fallen');
      }
    }
    console.log(`  weight fully forward: ${log.join(' | ')}`);
    expect(pearls).toBeGreaterThanOrEqual(4);
  });

  it('paddle-in: from the take-off spot the autopilot gets up on ≥ 50 % of the set waves it goes for', () => {
    let attempts = 0;
    let catches = 0;
    const log: string[] = [];
    for (const t0 of [34, 45, 56, 67]) {
      const ap = new Autopilot(ocean, { autoReset: Infinity, side: -1 });
      const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
      sim.reset({ x: ap.config.takeoffX, z: -ap.config.takeoffZ, headingRad: 0, stance: 'prone', speed: 0, time: t0 });
      const input = createSurfInput();
      for (let i = 0; i < 16 / DT; i++) {
        ap.update(sim, DT, input);
        sim.step(DT, input);
        if (ap.status.catches > 0 || (ap.status.attempts > 0 && ap.status.phase === 'position')) break;
      }
      attempts += ap.status.attempts;
      catches += ap.status.catches;
      log.push(`t${t0}: ${ap.status.catches > 0 ? `up at ${sim.telemetry.speed.toFixed(1)} m/s` : ap.status.attempts > 0 ? ap.status.endReason : 'no wave'}`);
    }
    console.log(`  paddle-in: ${catches}/${attempts} caught (${log.join(', ')})`);
    expect(attempts).toBeGreaterThanOrEqual(3);
    expect(catches / attempts).toBeGreaterThanOrEqual(0.5);
  });

  it('the autopilot is deterministic (same inputs → same ride)', () => {
    const a = autopilotRide(40, -1, 4);
    const b = autopilotRide(40, -1, 4);
    expect([a.sim.board.position.x, a.sim.board.position.z, a.sim.rider.position.y]).toEqual([b.sim.board.position.x, b.sim.board.position.z, b.sim.rider.position.y]);
  });
});

describe('robustness, contact and technique', { timeout: 30_000 }, () => {
  it('practice mode (wipeouts off) stays bounded and puts the rider back on their feet', () => {
    const cases: [string, SurfSim, number][] = [];
    {
      const sim = new SurfSim(calm, BOARD_PRESETS.shortboard, { wipeouts: false });
      sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
      cases.push(['shortboard standing still, flat water', sim, 5]);
    }
    for (const spec of [BOARD_PRESETS.shortboard, BOARD_PRESETS.funboard]) {
      const sim = new SurfSim(ocean, spec, { wipeouts: false });
      sim.reset(lineupSpawn(ocean, 0));
      const pop = createSurfInput();
      pop.popUp = true;
      sim.step(DT, pop);
      cases.push([`${spec.id} pop-up at the lineup, no input`, sim, 8]);
    }
    const log: string[] = [];
    for (const [name, sim, seconds] of cases) {
      let maxV = 0;
      let maxLeg = 0;
      let maxStretch = 0;
      const local = new Vector3();
      run(sim, seconds, createSurfInput(), (s) => {
        maxV = Math.max(maxV, s.board.velocity.length());
        maxLeg = Math.max(maxLeg, s.telemetry.legForceN);
        maxStretch = Math.max(maxStretch, s.board.worldToLocal(s.rider.position, local).distanceTo(s.rider.targetLocal));
      });
      log.push(`${name}: max |v| ${maxV.toFixed(1)} m/s, leg ${maxLeg.toFixed(0)} N, stretch ${maxStretch.toFixed(2)} m, ${sim.recoveries} recoveries (${sim.lastRecovery})`);
      expect(sim.rider.stance).not.toBe('fallen');
      expect(maxV).toBeLessThan(15); // was 1e19 (diverged within 3 s)
      expect(maxLeg).toBeLessThan(5000); // was > 100 kN (stance target flipping with the board upside down)
      expect(maxStretch).toBeLessThan(0.5);
    }
    console.log(`  practice: ${log.join(' | ')}`);
    expect(cases[0][1].recoveries).toBeGreaterThan(0);
  });

  it('the stance target stays continuous with the board upside down (no target-velocity spikes)', () => {
    const sim = new SurfSim(calm, BOARD_PRESETS.funboard, { wipeouts: false });
    sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
    // roll the board (with the rider) slowly through 360° about its long axis, high above the water
    sim.board.position.y += 20;
    sim.board.updateDerived();
    sim.rider.position.y += 20;
    sim.extraBoardForce.set(0, sim.board.mass * GRAVITY, 0);
    sim.extraRiderForce.set(0, sim.rider.config.mass * GRAVITY, 0);
    let maxTv = 0;
    const recover = (sim as unknown as { recover: () => void });
    recover.recover = () => {}; // keep it attached whatever the orientation
    run(sim, 4, createSurfInput(), (s) => {
      s.board.angularVelocity.set(1.6 * s.board.R[0], 1.6 * s.board.R[3], 1.6 * s.board.R[6]);
      maxTv = Math.max(maxTv, s.rider.targetVelLocal.length());
    });
    console.log(`  rolled 360°+: max target speed ${maxTv.toFixed(2)} m/s`);
    expect(maxTv).toBeLessThan(3);
  });

  it('twist is internal: in free space it conserves angular momentum and turns the board only as far as the body winds', () => {
    for (const spec of [BOARD_PRESETS.shortboard, BOARD_PRESETS.longboard]) {
      const sim = new SurfSim(calm, spec, { wipeouts: false });
      sim.reset({ x: 0, z: 0, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
      sim.board.position.y += 50;
      sim.board.updateDerived();
      sim.rider.position.y += 50;
      sim.extraBoardForce.set(0, sim.board.mass * GRAVITY, 0);
      sim.extraRiderForce.set(0, sim.rider.config.mass * GRAVITY, 0);
      const L = (s: SurfSim) => {
        const b = s.board;
        const r = s.rider;
        const l = b.angularMomentum(new Vector3());
        l.add(new Vector3().crossVectors(b.position, b.velocity).multiplyScalar(b.mass));
        l.add(new Vector3().crossVectors(r.position, r.velocity).multiplyScalar(r.config.mass));
        return l.addScaledVector(new Vector3(b.R[1], b.R[4], b.R[7]), -r.twistInertia() * r.twistSpin);
      };
      const L0 = L(sim);
      const input = createSurfInput();
      input.twist = 1;
      let yaw = 0;
      let maxDL = 0;
      run(sim, 2, input, (s) => {
        yaw += s.rider.turnRate * DT;
        maxDL = Math.max(maxDL, L(s).sub(L0).length());
      });
      console.log(`  ${spec.id}: twist held 2 s in free space: board yaw ${((yaw * 180) / Math.PI).toFixed(0)}°, upper body ${((sim.rider.twistAngle * 180) / Math.PI).toFixed(0)}°, max |ΔL| ${maxDL.toFixed(3)} N·m·s`);
      expect(maxDL).toBeLessThan(0.5); // an external 40 N·m twist torque gave 41.8 N·m·s
      expect(yaw).toBeGreaterThan(0.1); // nose right
      expect(yaw).toBeLessThan(RIDER_MODEL.twistRange);
      expect(sim.recoveries).toBe(0);
    }
  });

  it('a grounded board with a standing rider does not chatter on the sand (implicit seabed friction)', () => {
    const sim = new SurfSim(calm, BOARD_PRESETS.funboard, { wipeouts: false });
    sim.reset({ x: 124, z: 5, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
    let prev = sim.board.velocity.x;
    let prevD = 0;
    let maxDv = 0;
    let alternating = 0;
    run(sim, 3, createSurfInput(), (s, i) => {
      const v = s.board.velocity.x;
      const d = v - prev;
      if (i > 24) {
        maxDv = Math.max(maxDv, Math.abs(d));
        if (d * prevD < 0 && Math.abs(d) > 0.05 && Math.abs(prevD) > 0.05) alternating++;
      }
      prev = v;
      prevD = d;
    });
    console.log(`  grounded funboard + rider: max |Δv_x| per step ${maxDv.toFixed(2)} m/s, ${alternating} alternating steps`);
    expect(alternating).toBeLessThan(30); // explicit friction: 187
    expect(maxDv).toBeLessThan(0.5); // explicit friction: 1.1 m/s per step
  });

  it('a fallen rider washed up the beach lies on the sand, not inside it', () => {
    for (const x of [115, 130]) {
      const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
      sim.reset({ x, z: 5, headingRad: 0, stance: 'standing', speed: 0, time: 0 });
      sim.rider.fall('balance', sim.board);
      let worst = -Infinity;
      run(sim, 12, createSurfInput(), (s) => {
        const r = s.rider.position;
        worst = Math.max(worst, ocean.bathymetry.seabedY(r.x, r.z) - r.y);
      });
      const r = sim.rider.position;
      console.log(`  fallen from x=${x}: ends at x=${r.x.toFixed(1)}, COM ${(r.y - ocean.bathymetry.seabedY(r.x, r.z)).toFixed(2)} m above the sand (lowest ${(-worst).toFixed(2)} m)`);
      expect(worst).toBeLessThan(0); // the body COM never goes below the sand (was 0.3–1 m under)
    }
  });

  it('prone: steering without paddling pivots the board 180° on the spot in 3–5 s (back-paddling the inside arm)', () => {
    for (const spec of [BOARD_PRESETS.funboard, BOARD_PRESETS.longboard]) {
      const sim = new SurfSim(calm, spec);
      sim.reset(lineupSpawn(calm, 0));
      const input = createSurfInput();
      input.steer = -1;
      let yaw = 0;
      let t180 = Infinity;
      run(sim, 6, input, (s, i) => {
        yaw += s.rider.turnRate * DT;
        if (t180 === Infinity && yaw < -Math.PI) t180 = i * DT;
      });
      const moved = Math.hypot(sim.board.position.x + 75, sim.board.position.z - 5);
      console.log(`  ${spec.id}: pivot 180° in ${t180.toFixed(1)} s (moved ${moved.toFixed(1)} m)`);
      expect(sim.rider.stance).toBe('prone');
      expect(t180).toBeLessThan(5);
      expect(t180).toBeGreaterThan(2);
      expect(moved).toBeLessThan(2);
    }
  });

  it('the autopilot turns round and paddles back out to its take-off spot quickly', () => {
    const ap = new Autopilot(ocean, { autoReset: Infinity, side: 1 });
    const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
    // inside, facing the beach, in the lull between sets
    sim.reset({ x: ap.config.takeoffX + 18, z: ap.config.takeoffZ, headingRad: 0, stance: 'prone', speed: 0, time: 120 });
    const input = createSurfInput();
    let t = 0;
    for (; t < 30 && ap.status.phase === 'position'; t += DT) {
      ap.update(sim, DT, input);
      sim.step(DT, input);
    }
    console.log(`  back out 18 m from facing the beach: ${ap.status.phase} after ${t.toFixed(1)} s`);
    expect(ap.status.phase).not.toBe('position');
    expect(t).toBeLessThan(25);
  });
});
