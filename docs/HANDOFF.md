# Handoff — work in progress

Last stable commit: `a63f4cf` (all checks green). The commit after it is **WIP**: typechecks,
but 11 tests fail (10 in `tests/physicsSim.test.ts`, 1 in `tests/renderEffects.test.ts`).

## What the WIP is doing

Goal: make the core loop (paddle in → catch a set wave → pop up → ride) achievable with skill.
An independent review found that ~1% of human-like attempts produced a 5 s ride and the autopilot
fell on 16/16 pop-ups. The main cause is the standing balance: a hand-tuned reflex controller with
~20 search-fitted gains (`RIDER_MODEL.carve`) plus an ad-hoc tip test.

Done in the WIP (uncommitted owner was stopped mid-way):
- Fall diagnosis.
- Standing balance replaced with a linear-inverted-pendulum + capture-point model
  (`src/physics/Rider.ts`): the rider falls only when a bounded-effort correction can't keep the
  capture point over the feet.
- Crest water speeds up near the breaking limit (`src/ocean/waveModel.ts` + `waveGLSL.ts`).
- New `tests/physicsIntegrity.test.ts`.

Still to do:
- Finish T spawn in a recoverable state + update the autopilot to the new balance model
  (`spawn.ts`, `autopilot.ts`, `SurfSim.ts`).
- Pop-up foot placement (back foot ~1/4–1/3 from the tail, weight forward during the drop).
- Re-tune / update the failing tests to the new model (don't loosen thresholds to pass).
- Docs, HUD hints, then the full check suite.

## Acceptance targets (default ocean, funboard)

1. Autopilot paddle-in from the take-off zone: ≥ 50% of set waves give a ride ≥ 5 s.
2. Autopilot decisions as keyboard presses with 150 ms reaction delay: median T-spawn ride ≥ 5 s.
3. T spawn, no input: ≥ 50% stand ≥ 4 s, but rides don't last forever (median < 12 s).
4. Mistakes still fail: weight forward on a steep take-off pearls, hard lean < 2 m/s falls,
   standing still on 28 L sinks.
5. Balance assist is monotonic (1.0 = most forgiving).
6. Carving kept: 7 m/s full lean → radius ≤ 8 m, bank ≥ 35°.
7. Only physical forces (gravity, water, seabed, leash); internal rider forces equal and opposite.
8. Green: `npx tsc --noEmit`, `npm test`, `npm run check:gpu`, `npm run build`, `npm run smoke`.

Useful tools: `npx tsx tools/simProbe.ts --mode autopilot`, `npm run dev` (O toggles autopilot,
T spawns on a wave). Review scripts from earlier rounds lived in `tools/out/` (git-ignored, not
pushed).
