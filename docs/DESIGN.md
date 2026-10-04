# Quiver — design

A physics-based surfing game on Three.js. The goal is that it *feels real*: buoyancy, weight,
board volume and technique matter, it is hard, and it is fun once you get it right.

Milestone 1 (this document): realistic water + one rideable surf spot, a simple surfboard with a
simple rider model and test controls to exercise buoyancy, paddling, catching waves and carving.
No character animation, scoring, menus or audio yet.

## Conventions

* Three.js, TypeScript (strict), Vite. WebGL2 only (GLSL ES 3.00).
* World axes: **+Y up**, mean sea level `y = 0`. **+X points toward the beach** (swell travels
  roughly +X). Z runs along the shore. Units: metres, seconds, kilograms, radians.
* Board local frame: origin at the board's centre of mass, **+X toward the nose, +Y out of the
  deck, +Z toward the right rail** (X × Y = Z).
* Hot loops (physics, sampling) must not allocate: reuse scratch objects.
* Physics runs at a fixed `1/240 s` step. Rendering interpolates.
* Water density `ρ = 1025 kg/m³`, `g = 9.81 m/s²`, kinematic viscosity `ν = 1.19e-6 m²/s`.

## Module map

```
src/ocean/      DONE — wave model (CPU), GLSL mirror, tables, bathymetry  (do not change the maths
                without updating both sides and re-running `npm run check:gpu`)
src/physics/    rigid body, board shape + hull, hydrodynamics, fins, rider, SurfSim
src/render/     ocean mesh + material, sky/environment, seabed/beach, board & rider meshes, spray
src/game/       game loop, input, camera rig, HUD, debug GUI, debug drawing, spawn helpers
src/main.ts     bootstrap
tests/          vitest (node) — ocean + physics
tools/          headless-browser checks (gpu-check, screenshots)
```

## Ocean (implemented — read `src/ocean/waveModel.ts` header)

`OceanModel` is the single source of truth for the water surface.

* Swell: shoaling over the bathymetry (phase from the dispersion relation, Snell refraction,
  shoaling coefficient), a per-wave **set envelope** (16-wave pattern: sets and lulls), a
  depth-limited **breaking cap with history** (a wave that broke on the bar arrives smaller),
  crest peaking from the Ursell number and a Kepler-equation **forward lean** that makes the face
  steep (≈45–60°) right before breaking. Explicit height field ⇒ no fold-overs, exact CPU queries.
* Wind chop: 12 small Gerstner components (λ 1.2–16 m) on top, Lagrangian.
* Surf spot: a ">"-shaped sandbar (A-frame peak at `z = 0`, crest at `x ≈ -20`, 1.4 m deep) that
  makes set waves (≈1.6 m) break around `x ≈ -35 … -30` and peel both ways at ≈8 m/s; an inner
  trough (≈3 m) where waves reform; a shore break around `x ≈ 85–95`; dry beach from `x ≈ 125`.
* Set waves arrive every ~3 minutes (`SET_LENGTH` 16 × 11 s period).

API you will use:

```ts
const ocean = new OceanModel();            // DEFAULT_OCEAN_CONFIG
ocean.setTime(t);                          // seconds, double precision; physics owns this
ocean.heightAt(x, z);                      // ~3 µs
ocean.sample(x, z, out: SurfaceSample);    // ~10 µs: height, normal, slope, dEta/dt, water
                                           // velocity (vx, vy, vz), depth, waveHeight, ratio,
                                           // fullness, breaking, phaseSpeed, dir, wavePhase
ocean.evalSwell(x, z, out: SwellEval);     // swell-only state (breaking, fullness, psi …)
ocean.depthAt(x, z);                       // still-water depth (negative on land)
ocean.bathymetry.seabedY(x, z);            // analytic seabed (for meshes)
ocean.rebuild(config);                     // after config changes (GUI)
```

`SurfaceSample.velX/velY/velZ` is the water particle velocity at the surface, kinematically
consistent with the moving surface (a particle moving with it stays on the surface). Use it for
all relative-velocity hydrodynamics. `wavePhase` ∈ (-π, π]: 0 = crest, (0, π) = front face.
`fullness` ≈ how close the wave is to breaking (0.8–1 = steep, makeable face), `breaking` = 0–1
whitewater intensity, `ratio` > 1 means the wave has broken (here or further out).

GPU: `OCEAN_GLSL` + `OceanShaderData` (`src/ocean/waveGLSL.ts`). `oceanSurface(restXZ)` returns
the displaced position of a rest grid point plus swell state for shading. Positions agree with the
CPU to < 1 mm (`npm run check:gpu`).

## Physics (`src/physics/`)

Use `three` math classes (`Vector3`, `Quaternion`, `Matrix3`) — they work in Node for tests.

### Rigid body
COM position, orientation quaternion, linear velocity, angular velocity (world), mass, local
diagonal inertia. Force/torque accumulators with `addForceAtPoint`. Semi-implicit Euler with
gyroscopic term, quaternion renormalisation. `pointVelocity(p)`.

### Board shape (shared with the render mesh)
`BoardSpec` presets (outline, rocker, foil, volume, mass, fins). Shape functions of
`u ∈ [0,1]` (tail → nose) and `v ∈ [-1,1]` (left → right rail): half-width, bottom rocker, deck
thickness (foil + rail taper). Thickness scaled so the integrated volume equals `volumeLiters`.

| preset | length | width | volume | mass | fins |
|---|---|---|---|---|---|
| Shortboard 6'0" | 1.83 | 0.49 | 28 L | 2.8 kg | thruster |
| Funboard 7'6" | 2.29 | 0.56 | 50 L | 4.5 kg | thruster |
| Longboard 9'2" | 2.79 | 0.58 | 72 L | 6.5 kg | single |
| Soft-top 8'0" | 2.44 | 0.58 | 86 L | 5.5 kg | thruster (small) |

### Hull + water forces
Planform discretised into cells (≈24 along × 6 across). Water is sampled with `ocean.sample` on a
coarse **probe grid** over the planform (≈8 × 3, every substep) and interpolated to cells (budget:
water sampling ≤ ~1 ms per rendered frame). Per cell:

1. **Hydrostatics**: submerged fraction of the bottom→deck segment (handles any orientation,
   upside down too); buoyancy `ρ g A t f` at the submerged centroid.
2. **Planing / impact pressure** on the wetted bottom: with `v_rel = v_cell − v_water` and the
   bottom's outward normal `n`, `v_n = v_rel·n`; if `v_n > 0`:
   `F = −n · ½ρA_w (C_p1 |v_rel| v_n + C_p2 v_n²)`, `C_p1 ≈ 1`, `C_p2 ≈ 1` (calibrated so a
   7'6" board planes from ≈4–5 m/s); small suction when `v_n < 0`.
3. **Skin friction** on the wetted area: ITTC `C_f = 0.075/(log10 Re − 2)²` × form factor ≈1.3.
4. **Rail side pressure** on submerged rail cells resisting lateral motion.
5. **Fins**: low-aspect lifting surfaces at the tail (toe & cant), lift slope ≈ 2π·AR/(AR+2),
   stall ≈ 14°, profile + induced drag, area scaled by submerged span. They give directional
   stability (weathervaning), drive, and hold in a carve.
6. Small angular damping proportional to wetted area (radiation damping stand-in).

Carving emerges: rolling the board tilts the planing force sideways (centripetal force) and the
fins/rails resist sideslip. Catching a wave emerges: gravity along the face slope + water moving
up the face; you must be paddling near wave speed at the steep part to get in.

### Rider (milestone-1 model, no animation)
A 75 kg point mass connected to the board by a stiff spring-damper ("legs") toward a **target
COM** in board space. Reaction force applied to the board at the target point (= feet force +
ankle torque).

* **Prone**: COM ≈ 12 cm above the deck, lying along the stringer. Body buoyancy/drag modelled
  with a few spheres (chest, hips, legs) that dip into the water — on a 28 L board you lie low in
  the water, on a longboard you float high. **Paddling**: alternating arm strokes (~0.9 s),
  thrust applied when the hand is in the water, efficiency falling with speed (hand-speed limit
  ≈ 3 m/s ⇒ top paddle speed ≈ 2 m/s). Steering by paddling harder on one side.
* **Pop-up**: 0.6 s transition prone → standing.
* **Standing**: feet over the stringer (back foot near the fins). COM ≈ 0.95 m above the deck
  (0.6 m crouched). Inputs move the target COM: forward/back weight (trim: nose down to go,
  weight back to stall/turn) and toe/heel lean (rail to rail). Twist applies a limited yaw torque.
  Standing still on a 28 L board sinks it (realistic). A "balance assist" (0–1) tilts the stance
  toward the apparent gravity.
* **Wipeout** when the ankle torque needed exceeds what the feet can resist for too long, the
  board flips, the nose pearls, or the board is driven deep under. Then the rider floats free,
  tethered by a 2.4 m leash. `R` resets.

### SurfSim
```ts
const sim = new SurfSim(ocean, BOARD_PRESETS.funboard, riderConfig);
sim.reset(spawn);           // spawn = { x, z, headingRad, stance, speed?, onWave? }
sim.step(1/240, input);     // sets ocean time, samples water, forces, integrates
sim.time; sim.board; sim.rider; sim.telemetry; sim.debugForces
```
`SurfInput`: `paddle 0..1, steer −1..1, leanForward −1..1, leanSide −1..1, crouch 0..1,
twist −1..1, popUp (edge), reset (edge)`.

Spawn helpers: *lineup* (prone, outside the peak, facing the beach, still), *on a wave* (find the
next set wave's steep unbroken face near the peak, place the board on it moving at ≈ wave speed,
standing) for testing rides directly.

## Rendering (`src/render/`)

* **Ocean mesh**: camera-centred LOD grid (clipmap/CDLOD style, snapped to avoid swimming,
  morphing to avoid cracks), ≥ 0.25 m spacing near the camera, out to the horizon. Vertex shader
  runs `oceanSurface()`. Detail normals (small waves < 1.2 m) only in the fragment shader, faded
  with distance.
* **Water shading**: Schlick Fresnel (F0 = 0.02), sky reflection from a PMREM environment, GGX
  sun glitter, subsurface glow through thin crests and backlit faces, depth-based absorption
  over the analytic seabed (turquoise over sand, deep blue offshore), whitewater/foam from
  `breaking`, crest foam, lingering surf-zone foam, shore swash; aerial perspective to the
  horizon.
* **Sky**: physical sky (three `Sky`) with a sun; same sun lights board, rider and sand.
* **Beach/seabed** mesh from the bathymetry, sand shading with a wet band at the waterline.
* **Board** mesh generated from the same shape functions as the hull; **rider** = simple
  capsule figure (prone / crouched / standing poses), no animation.
* ACES or AgX tone mapping, sRGB output.

## Game shell (`src/game/`)

Fixed-step accumulator (max ~8 substeps per frame), render interpolation; ocean shader time set to
the interpolated render time. Cameras: follow (default), orbit/free, beach, side-on.
HUD: speed, stance, board, submerged volume, wave height/state under the board, balance meter,
controls help. lil-gui: ocean (swell height/period/direction, wind, breaking, bar → rebuild),
board preset, rider mass, balance assist, time scale/pause/step, sun, debug overlays
(force arrows, probes).

### Controls (keyboard)

| key | prone | standing |
|---|---|---|
| W | paddle | weight forward |
| S | — | weight back |
| A / D | steer (paddle one side) | lean left / right rail |
| Q / E | — | twist (yaw) |
| Shift | — | crouch |
| Space | pop up | — |
| R | reset at lineup | reset at lineup |
| T | spawn on a wave | spawn on a wave |
| C | cycle camera | |
| P / [ / ] | pause / slow-mo / speed up | |
