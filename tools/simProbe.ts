/**
 * Headless physics probe: runs SurfSim and prints a time series for tuning.
 *
 *   npx tsx tools/simProbe.ts                                   # soft-top, scripted trim on a wave
 *   npx tsx tools/simProbe.ts --board shortboard --mode back    # weight fully back (should lose it)
 *   npx tsx tools/simProbe.ts --mode none --side 1 --angle 50   # no input at all
 *   npx tsx tools/simProbe.ts --mode paddle --board funboard    # flat water, paddle then pop up
 *   npx tsx tools/simProbe.ts --mode autopilot --board funboard # autopilot from waveSpawn
 *   npx tsx tools/simProbe.ts --mode autopilot --lineup --seconds 120  # autopilot paddles in
 *
 * Options:
 *   --board <id>         shortboard | funboard | longboard | softtop (default softtop)
 *   --mode <m>           trim (scripted line + trim, as test 8) | back (leanForward −1) |
 *                        none (no input) | paddle (calm water: paddle from rest, pop up at --pop s) |
 *                        autopilot (src/physics/autopilot.ts drives the SurfInput)
 *   --lineup             autopilot: start prone at the lineup and paddle into set waves (auto-reset
 *                        after each ride) instead of standing on a waveSpawn face
 *   --side <±1>          side of the peak for waveSpawn (default −1)
 *   --angle <deg>        heading from the wave direction toward the shoulder (waveSpawn default 55)
 *   --min-fullness <f>   waveSpawn minimum fullness (waveSpawn default 0.5)
 *   --from <s>           waveSpawn search start time (default 0)
 *   --seconds <s>        simulated duration (default 8)
 *   --every <s>          print interval (default 0.1)
 *   --crouch <0..1>      crouch for the trim mode (default 0.5)
 *   --pop <s>            paddle mode: pop-up time (default 4)
 */
import { DEFAULT_OCEAN_CONFIG, cloneOceanConfig } from '../src/ocean/oceanConfig';
import { OceanModel } from '../src/ocean/waveModel';
import { BOARD_PRESETS, type BoardPresetId } from '../src/physics/boardShape';
import { PHYSICS_DT } from '../src/physics/constants';
import { lineupSpawn, waveSpawn } from '../src/physics/spawn';
import { SurfSim, createSurfInput, type Spawn } from '../src/physics/SurfSim';
import { Autopilot } from '../src/physics/autopilot';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const boardId = arg('board', 'softtop') as BoardPresetId;
const mode = arg('mode', 'trim');
const side = (Number(arg('side', '-1')) < 0 ? -1 : 1) as 1 | -1;
const angle = process.argv.includes('--angle') ? (Number(arg('angle', '55')) * Math.PI) / 180 : undefined;
const minFullness = process.argv.includes('--min-fullness') ? Number(arg('min-fullness', '0.5')) : undefined;
const fromTime = Number(arg('from', '0'));
const seconds = Number(arg('seconds', '8'));
const every = Number(arg('every', '0.1'));
const crouch = Number(arg('crouch', '0.5'));
const popAt = Number(arg('pop', '4'));
const lineup = process.argv.includes('--lineup');

const spec = BOARD_PRESETS[boardId];
if (!spec) {
  console.error(`unknown board "${boardId}" (${Object.keys(BOARD_PRESETS).join(', ')})`);
  process.exit(1);
}
if (!['trim', 'back', 'none', 'paddle', 'autopilot'].includes(mode)) {
  console.error(`unknown mode "${mode}" (trim | back | none | paddle | autopilot)`);
  process.exit(1);
}

let ocean: OceanModel;
let spawn: Spawn;
if (mode === 'paddle') {
  const cfg = cloneOceanConfig(DEFAULT_OCEAN_CONFIG);
  cfg.swells = [];
  cfg.wind.count = 0;
  ocean = new OceanModel(cfg);
  spawn = lineupSpawn(ocean, 0);
  console.log(`calm water, ${boardId}, prone from rest; paddle 1, pop-up at ${popAt} s`);
} else if (mode === 'autopilot' && lineup) {
  ocean = new OceanModel();
  spawn = lineupSpawn(ocean, fromTime);
  console.log(`lineup t=${fromTime.toFixed(2)} s, ${boardId}, autopilot paddles into set waves`);
} else {
  ocean = new OceanModel();
  const ws = waveSpawn(ocean, { fromTime, side, minFullness, angleRad: angle });
  if (!ws) {
    console.error('waveSpawn found no qualifying wave face');
    process.exit(1);
  }
  spawn = ws.spawn;
  console.log(
    `waveSpawn t=${ws.time.toFixed(2)} s at (${ws.spawn.x.toFixed(2)}, ${ws.spawn.z.toFixed(2)}) ` +
      `H=${ws.waveHeight.toFixed(2)} m fullness=${ws.fullness.toFixed(2)} c=${ws.phaseSpeed.toFixed(2)} m/s ` +
      `slope=${ws.slope.toFixed(2)} heading=${((ws.spawn.headingRad * 180) / Math.PI).toFixed(1)}° speed=${(ws.spawn.speed ?? 0).toFixed(2)} m/s`,
  );
  console.log(`board ${boardId}, mode ${mode}`);
}

const sim = new SurfSim(ocean, spec);
sim.reset(spawn);
const input = createSurfInput();
const pilot = mode === 'autopilot' ? new Autopilot(ocean, { autoReset: lineup ? 3 : Infinity }) : null;
if (pilot && !lineup) pilot.startRiding(side);
// autopilot ride metrics
const rides: { time: number; turns: number; end: string }[] = [];
let rideTurns = 0;
let lastTurnSign = 0;
let rideStart = -1;
const DEG = 180 / Math.PI;
const f = (v: number, d = 2, w = 7) => v.toFixed(d).padStart(w);

console.log(
  [
    '     t', 'stance  ', '  speed', '  vWave', '      c', '  phase', 'heading', '  pitch', '   roll', '   bank',
    '    lean', ' sub(L)', 'plane(N)', ' drag(N)', '  fin(N)', 'finα', ' bal', ' deck', ' nose', 'ride(s)',
  ].join(' '),
);

const n = Math.round(seconds / PHYSICS_DT);
const printEvery = Math.max(1, Math.round(every / PHYSICS_DT));
let best = 0;
let maxSpeed = 0;
let lastStance = sim.rider.stance;
for (let i = 0; i <= n; i++) {
  const t = sim.telemetry;
  const w = t.water;
  input.popUp = false;
  if (mode === 'trim') {
    // same scripted rider as test 8: hold a line across the face, trim from the wave phase
    const angT = Math.min(Math.max((angle ?? (55 * Math.PI) / 180) + 1.2 * (w.wavePhase - 0.85), 0.2), 1.45);
    let err = Math.atan2(w.dirZ, w.dirX) + side * angT - t.headingRad;
    err = Math.atan2(Math.sin(err), Math.cos(err));
    input.leanSide = Math.max(-1, Math.min(1, err - 0.3 * sim.rider.turnRate));
    input.leanForward = Math.max(-1, Math.min(1, 1.5 * (0.85 - w.wavePhase) - 0.3 * (t.speedAlongWave - w.phaseSpeed)));
    input.crouch = crouch;
  } else if (mode === 'back') {
    input.leanForward = -1;
  } else if (mode === 'paddle') {
    input.paddle = sim.time < popAt ? 1 : 0;
    input.popUp = i === Math.round(popAt / PHYSICS_DT);
  } else if (pilot) {
    const phaseBefore = pilot.status.phase;
    pilot.update(sim, PHYSICS_DT, input);
    const ph = pilot.status.phase;
    if (ph === 'ride' && rideStart < 0) {
      rideStart = sim.time;
      rideTurns = 0;
      lastTurnSign = 0;
    }
    if (ph === 'ride') {
      // count turns: the yaw rate swinging past ±25°/s with alternating sign
      const tr = sim.rider.turnRate;
      const sg = tr > 0.44 ? 1 : tr < -0.44 ? -1 : 0;
      if (sg !== 0 && sg !== lastTurnSign) {
        rideTurns++;
        lastTurnSign = sg;
      }
    }
    if (rideStart >= 0 && ph !== 'ride' && phaseBefore === 'ride') {
      rides.push({ time: sim.time - rideStart, turns: rideTurns, end: pilot.status.endReason });
      rideStart = -1;
    }
  }

  if (i % printEvery === 0 || t.stance !== lastStance) {
    console.log(
      [
        f(sim.time - (spawn.time ?? 0), 2, 6),
        t.stance.padEnd(8),
        f(t.speed), f(t.speedAlongWave), f(w.phaseSpeed), f(w.wavePhase), f(t.headingRad * DEG, 1),
        f(t.pitchRad * DEG, 1), f(t.rollRad * DEG, 1), f(sim.rider.bankAngle * DEG, 1),
        f(sim.rider.lean, 3, 8), f(t.submergedLiters, 1), f(t.planingLiftN, 0, 8), f(t.dragN, 0, 8), f(t.finForceN, 0, 8),
        f(t.finAlphaDeg, 0, 4), f(t.balance, 2, 4), f(t.deckDepth, 2, 5), f(t.noseDepth, 2, 5), f(t.ridingTime, 2),
      ].join(' ') +
        (pilot ? `  AP ${pilot.status.phase} ψ*=${pilot.status.phaseTarget.toFixed(2)} ang ${(pilot.status.angle * DEG).toFixed(0)}/${(pilot.status.angleTarget * DEG).toFixed(0)}° curl ${pilot.status.curl.toFixed(0)}` : '') +
        (t.wipeoutReason && t.stance === 'fallen' ? `  wipeout: ${t.wipeoutReason}` : ''),
    );
    lastStance = t.stance;
  }
  if (i === n) break;
  sim.step(PHYSICS_DT, input);
  best = Math.max(best, sim.telemetry.ridingTime);
  maxSpeed = Math.max(maxSpeed, sim.telemetry.speed);
}

const end = sim.telemetry;
if (pilot) {
  if (rideStart >= 0) rides.push({ time: sim.time - rideStart, turns: rideTurns, end: 'probe ended' });
  const st = pilot.status;
  console.log(`\nautopilot: waves attempted ${st.attempts}, caught ${st.catches}, rides ${rides.length}`);
  for (const r of rides) console.log(`  ride ${r.time.toFixed(2)} s, ${r.turns} turns, end: ${r.end}`);
}
console.log(
  `\nsummary: longest continuous ride ${best.toFixed(2)} s, max speed ${maxSpeed.toFixed(2)} m/s, ` +
    `final stance ${end.stance}${end.wipeoutReason ? ` (${end.wipeoutReason})` : ''}, final speed ${end.speed.toFixed(2)} m/s`,
);
