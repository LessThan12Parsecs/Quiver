/**
 * Board + rider physics scenarios (flat water on a calm ocean, waves on the default ocean).
 * Key numbers are printed so the calibration can be read from the test log.
 */
import { describe, expect, it } from 'vitest';
import { DEFAULT_OCEAN_CONFIG, cloneOceanConfig } from '../src/ocean/oceanConfig';
import { OceanModel } from '../src/ocean/waveModel';
import { BoardHull, createHullDiagnostics } from '../src/physics/BoardHull';
import { BOARD_PRESETS, cloneBoardSpec, getBoardShape, type BoardSpec } from '../src/physics/boardShape';
import { GRAVITY, RHO_WATER } from '../src/physics/constants';
import { ImplicitSystem } from '../src/physics/implicit';
import { RigidBody } from '../src/physics/RigidBody';
import { lineupSpawn, waveSpawn } from '../src/physics/spawn';
import { SurfSim, createSurfInput, type SurfInput } from '../src/physics/SurfSim';

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

  it('7) carving: leanSide rolls the board onto that rail and turns it that way, without capsizing', () => {
    for (const side of [1, -1]) {
      const sim = new SurfSim(calm, BOARD_PRESETS.funboard);
      sim.reset({ x: -75, z: 5, headingRad: 0, stance: 'standing', speed: 7, time: 0 });
      const input = createSurfInput();
      input.leanSide = side;
      let integ = 0;
      let maxBank = 0;
      run(sim, 2.5, input, (s) => {
        // hold the speed with a tow along the heading (a wave face would provide it)
        const hx = s.board.R[0], hz = s.board.R[6];
        const nn = Math.hypot(hx, hz);
        const vh = (s.board.velocity.x * hx + s.board.velocity.z * hz) / nn;
        const e = 7 - vh;
        integ += e * DT;
        const f = Math.max(0, 400 * e + 300 * integ);
        s.extraBoardForce.set((f * hx) / nn, 0, (f * hz) / nn);
        maxBank = Math.max(maxBank, side * s.rider.bankAngle);
      });
      const t = sim.telemetry;
      console.log(`  leanSide ${side}: heading ${((t.headingRad * 180) / Math.PI).toFixed(1)}°, bank ${((sim.rider.bankAngle * 180) / Math.PI).toFixed(1)}° (max ${((maxBank * 180) / Math.PI).toFixed(1)}°), stance ${t.stance}`);
      expect(t.stance).toBe('standing');
      expect(maxBank).toBeGreaterThan((5 * Math.PI) / 180);
      expect(side * t.headingRad).toBeGreaterThan((10 * Math.PI) / 180);
      expect(Math.abs(t.rollRad)).toBeLessThan(1.2);
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
  it('waveSpawn finds a steep unbroken set-wave face to one side of the peak', () => {
    const before = ocean.time;
    const ws = waveSpawn(ocean, { fromTime: 0 });
    expect(ocean.time).toBe(before); // the search restores the ocean time
    expect(ws).not.toBeNull();
    const r = ws!;
    console.log(`  spawn t=${r.time} at (${r.spawn.x}, ${r.spawn.z}) H=${r.waveHeight.toFixed(2)} fullness=${r.fullness.toFixed(2)} c=${r.phaseSpeed.toFixed(2)} slope=${r.slope.toFixed(2)}`);
    expect(r.waveHeight).toBeGreaterThan(1);
    expect(r.fullness).toBeGreaterThanOrEqual(0.75);
    expect(r.fullness).toBeLessThanOrEqual(0.92);
    expect(r.slope).toBeLessThan(0);
    expect(Math.abs(r.spawn.z)).toBeGreaterThanOrEqual(4);
    expect(r.spawn.stance).toBe('standing');
    expect(r.spawn.headingRad).toBeGreaterThan(0); // angled toward +Z, away from the peak
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
