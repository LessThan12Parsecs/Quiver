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
tests/          vitest (node) — ocean, physics, LOD selection, meshes, game shell (rides, HUD)
tools/          headless-browser checks (gpu-check, screenshots)
```

## Ocean (implemented — read `src/ocean/waveModel.ts` header)

`OceanModel` is the single source of truth for the water surface.

* Swell: shoaling over the bathymetry (phase from the dispersion relation, Snell refraction,
  shoaling coefficient), a per-wave **set envelope** (16-wave pattern: sets and lulls), a
  depth-limited **breaking cap with history** (a wave that broke on the bar arrives smaller),
  crest peaking from the Ursell number and a Kepler-equation **forward lean** that makes the face
  steep (≈45–60°) right before breaking. Explicit height field ⇒ no fold-overs, exact CPU queries
  (the render adds a breaking lip and roller on top, see Rendering — physics never sees them).
* Wind chop: 12 small Gerstner components (λ 1.2–16 m) on top, Lagrangian.
* Surf spot: a ">"-shaped sandbar (A-frame peak at `z = 0`, crest at `x ≈ -20`, 1.4 m deep,
  `sweep` 0.7, `depthSlope` 0.006) that makes set waves (≈1.6 m) break around `x ≈ -35 … -30`
  (peak break of the t ≈ 66.7 s set wave at `x ≈ -33`) and peel both ways at ≈6–6.6 m/s for the
  first 60–70 m, after which the outer section closes out (the earlier sweep 0.45 / 0.008 bar
  peeled 7.5–9 m/s for 80–90 m — faster than a board can stay ahead of: autopilot median ride
  6.3 s there vs 8.3 s now); an inner trough (≈3 m) where waves reform; a shore break around
  `x ≈ 85–95`; dry beach from `x ≈ 125`.
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
7. **Depth regimes**: once the deck is under by half the beam, motion along the normal meets
   cross-flow drag (C_d ≈ 1) instead of the free-surface impact law and the added mass doubles to
   the unbounded-fluid value ρπb²/4; the planing lift fades to `deepLift` (0.4) only once the
   deck is under by half the board length. A board released 3 m deep surfaces at ≈ 5 m/s instead
   of shooting out at 6.6 m/s.
8. **Seabed contact** (shallows, beach): a penalty normal force per cell (its stiffness and
   damping go into the implicit step) and Coulomb friction (μ 0.5, regularised below 5 cm/s). The
   friction is a ≈ 8000 N·s/m damper under a standing rider, so it is linearised into the implicit
   damping too (explicitly it chattered back and forth on a 4.5 kg board).

The hull keeps a little history (water velocity under each cell, smoothed wetted length); `reset`
clears it so a spawn is reproducible whatever ran before.

Carving emerges: rolling the board tilts the planing force sideways (centripetal force) and the
fins/rails resist sideslip. Catching a wave emerges: gravity along the face slope + water moving
up the face; you must be paddling near wave speed at the steep part to get in.

### Rider (milestone-1 model, no animation)
A 75 kg point mass on a 3-axis leg spring-damper (≈3 Hz, ζ 0.8) toward a **target COM** in board
space, integrated implicitly with the board (9 DOF). The reaction acts on the board at the target
point (= feet force + ankle torque). The ankles are **torque-limited**: across the board the feet
hold at most normal force × `rollLever` (0.21 m: toes/heels pressing near the rails — the hull's
own centre of pressure sits 16–22 cm off the stringer at 20–50° of bank) + a little grip, along
it × `pitchLever` (0.3 m); past that the foot rolls on its edge (the spring perpendicular to the
leg saturates) and the body tips. A saturated axis keeps only its capped force up to the legs'
reach (0.45 m from the target, the impact stretch); past it the spring is back, so no mode can drag
the rider off a hanging ankle. The stance target moves continuously whatever the board's
orientation (apparent-up angles that do not wrap when it is upside down, a rate-limited pitch
gimbal, target speed ≤ 4 m/s relative to the board).

* **Prone**: COM ≈ 12 cm above the deck, lying along the stringer. Body buoyancy/drag modelled
  with spheroids (chest, hips, legs). **Paddling**: alternating strokes (~0.9 s), thrust while the
  hand is in the water, hand-speed limit ≈ 3 m/s ⇒ top paddle speed ≈ 2 m/s. Steering works the
  outside arm harder and the inside arm less, back-paddling it past zero: steering without
  paddling pivots the board on the spot (outside arm pulls, inside arm back-paddles — a yaw couple
  with little thrust, so the fins do not weathervane it back), 180° in ≈ 3.5 s on the funboard,
  4.5 s on the longboard; paddling with a full steer is one-arm paddling, a wide arc (≈ 55° in
  6 s).
* **Pop-up**: 0.6 s transition prone → standing.
* **Standing**: back foot over the fins, COM ≈ 0.95 m above the feet (0.6 m crouched); the COM
  target moves like a body (2nd-order servo, bounded acceleration and speed). leanForward = trim
  (±0.22 m), leanSide = carve, twist winds the upper body against the board: a yaw DOF about the
  deck normal (1.35 kg·m², taken out of the body inertia lumped into the board; ±70°, trunk
  torque ≤ 40 N·m, ≈ 3 Hz back to square). It is an internal torque: the board turns only as far
  as the body counter-rotates (in free space ≈ 37° on the shortboard, 17° on the longboard) and
  back when it unwinds — sustained turning has to come from the water (fins, rails, a sunk tail).
  `balanceAssist` (0–1) blends in a skilled rider's reflex:
  * leanSide asks for a **turn rate** (up to the tightest turn the board holds, R = 4.5 m,
    ≤ 110°/s); the bank target (relative to the water surface) is the coordinated angle of that
    turn + a turn-rate correction + a body-tilt correction (capped 45°, less at low speed);
  * the body follows the turn's apparent gravity (`turnTilt` = atan(v·ω/g)), leans toward the
    target bank — the ankle torque that rolls the board — and is never more than `leanMargin`
    (11°, +4.6° at a full lean) beyond that apparent gravity, i.e. about what the feet can hold;
  * fore/aft it follows the slow apparent gravity (70 %), and the ankles absorb the board pitching
    under the body over chop (the body stays steady in the world).
  Whether a carve holds is up to the physics. Funboard at 7 m/s, full lean: ≈ 22° of bank at
  0.7 s, 43° max, yaw ≤ 55°/s (R ≈ 7 m).
* **Wipeouts**: balance — the body (feet → COM) further from the slow apparent up (g − a, 0.35 s
  filter) than the feet hold (asin(lever / leg length)) plus a recoverable tip (10° across, 12°
  along); over the nose with the nose under water counts as a **pearl**; nose pearling at speed;
  board flipped; board buried deep; leg over-stretched by an impact. Then the rider floats free,
  tethered by a 2.4 m leash. `R` resets. A fallen body rests on the seabed / beach (penalty
  contact + friction on its segments, implicit) and gets no buoyancy where the sand is above the
  water. With `wipeouts` off (practice; debug GUI) a body past recovery, a flipped board or
  over-stretched legs put the rider back on their feet in place instead (board upright along its
  heading at its speed; counted in `SurfSim.recoveries`); a pearl or a buried board is ridden on.

### Autopilot (`src/physics/autopilot.ts`, key O)
A scripted skilled rider that only writes `SurfInput` (no forces, no state tweaks): if it rides
well, the physics allows riding well (it does not twist). Phases: *position* (pivot toward the
take-off spot, x ≈ −38, 5 m to one side of the peak, and paddle there), *wait* (watch for a set
wave; let one pass that is already breaking outside and sit further out; paddle back once drifted
7 m), *paddle* (when its crest is ≤ 22 m out: paddle for the
beach angled 15° to the riding side, pop up once the board runs on the face at ≥ 0.72 c; give up
if it passes or breaks on the rider — next time sit further in / out), *popup*, *ride* (heading
angle from the wave direction kept in 45–75°, chosen so the speed along the wave holds a target
height on the face — lower and faster when the curl is close; a PI heading controller with
leanSide ≤ 0.6; trim for speed; crouch through turns), *kickout* (closed out, backed off, caught by
the whitewater or lost the wave: turn up and over the back), *done* (auto-reset in the game).
Measured (funboard, default ocean): from waveSpawn (6 set waves × both sides) median ride 8.3 s,
5/12 ≥ 10 s; from the take-off spot it gets up on 75–100 % of the set waves it goes for, but the
late, steep take-off usually ends in a fall within 1 s; from the lineup (240–300 s from t = 20 /
150) it goes for 9–10 waves and gets up on 2–3, never repositioning for more than ≈ 22 s (with
4°/s prone turns it used to circle for 2 minutes and miss whole sets).
`npx tsx tools/simProbe.ts --mode autopilot [--lineup]`.

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
next set wave's steepening, still unbroken face 20–30 m down the line from the peak (fullness
0.5–0.75), place the board on it standing, angled 55° from the wave direction, moving with the
wave along the wave direction and with the water across its heading) for testing rides directly.
`T` spawns on the side whose wave comes first (alternating), `Shift+T` the other side.

## Rendering (`src/render/`)

* **Ocean mesh**: camera-centred LOD grid (clipmap/CDLOD style, snapped to avoid swimming,
  morphing to avoid cracks), ≥ 0.25 m spacing near the camera, out to the horizon. LOD distances
  are scaled by the field of view (tan(fov/2) / tan 30°, clamped to [0.35, 1]), so the telephoto
  beach camera gets 0.5 m cells under the board it frames instead of 2–4 m ones (≈ 230 k vertices
  at 'high', like the follow camera; the unclamped 0.12 would draw 1.2 M). Vertex shader runs `oceanSurface()`;
  the swell fades out at 40–90 m grid spacing and is skipped beyond. Detail normals (short waves ~2 cm – 5 m) only in the fragment shader:
  two animated spectral ripple tiles (`DetailNormals.ts`, 128 short-crested components each,
  capillary-gravity dispersion) rendered every frame and sampled at 16 m / 4.2 m / 1.1 m scales
  with analytic ray-differential gradients; they store slope moments (LEAN mapping), so the slope
  variance lost to filtering becomes GGX roughness (no moiré, no shimmer). Shading quality knob
  `OceanMesh.setShading('low' | 'medium' | 'high')` (also picked by the quality preset; 'low'
  also drops the gust/slick/domain-warp noise). The game's default pixel-ratio cap follows the
  quality preset (low 1, medium 1.25, otherwise 1.5).
* **Breaking geometry (render only)**, `shaders/breaker.ts`, added to the height-field position
  by the water mesh and the crest spray: as a wave is about to break (fullness → 1, the section
  ahead of the curl of a peeling wave) the upper part of the crest is thrown forward (≤ 1.8 H)
  and droops (0.6 H), folding the mesh into a pitching lip with a cavity under it (a clear sheet
  of water; its underside is shaded with the flipped lip normal); once broken, the whitewater is
  a churning foam mound (≈ 0.25 H, ragged outline) instead of paint on the hump. The physics, CPU
  queries and `check:gpu` keep the exact height field.
* **Water shading**: Schlick Fresnel (F0 = 0.02), sky reflection from a PMREM environment, GGX
  sun glitter, subsurface glow through thin crests and backlit faces, depth-based absorption
  over the analytic seabed (turquoise over sand, deep blue offshore), whitewater/foam from
  `breaking` (billow relief with crease occlusion, thin translucent vs thick opaque foam, chunky
  broken leading edge), trailing foam that opens into lace, shore swash, streaks down steep
  faces, glowing thin lips when backlit; caustics softened and dimmed by surf-zone turbidity;
  aerial perspective to the horizon. Back faces: the surface seen from below only when the camera
  is under the (CPU) surface; above water they are lip undersides or crest-silhouette slivers,
  shaded like the front face.
* **Board–water feedback** (`BoardWake.ts`, render only): spray thrown off the rail cells that
  cut the water line (∝ (relative speed − 2.5 m/s)², mostly the engaged rail in a carve), a foam
  wake from the tail (turbulent strip + spreading edges, 6 s), paddling splashes and a wipeout
  splash. Draw order of the transparent queue: wake 5, board underwater pass 20, rider 30, crest
  spray 40, board spray 41.
* **Sky**: physical sky (three `Sky`) with a sun; same sun lights board, rider and sand.
* **Beach/seabed** mesh from the bathymetry, sand shading with a wet band at the waterline.
* **Board** mesh generated from the same shape functions as the hull; **rider** = simple
  capsule figure (prone / crouched / standing poses), no animation.
* ACES or AgX tone mapping, sRGB output.

## Game shell (`src/game/`)

Fixed-step accumulator (max ~8 substeps per frame), render interpolation; ocean shader time set to
the interpolated render time. Cameras: follow (default), orbit/free, beach, side-on. The follow
camera swings round to face the sea while a set wave's crest approaches the prone rider (≤ 32 m
behind), and while riding drops low (≈ 0.7 m over the water, below the crest) into the flats in
front of the face, looking along the line and up the face.
HUD: speed, stance, board, submerged volume, wave height/state under the board, balance meter,
controls help, a CAUGHT badge (prone, on the face, moving at the autopilot's pop-up speed,
≥ 0.72 c: pop up now), the RIDING badge + ride timer and last/best ride. A ride is the game's
notion (`RideTracker`): standing on a wave (`telemetry.riding` alone also holds for prone belly
rides and riderless boards), dropouts < 0.6 s bridged, recorded when > 0.5 s (also when R/T end
it). Wipeout advice follows how the rider fell (balance: over a rail, off the back of a stalled
board, over the nose). The take-off hint points at the autopilot's take-off spot (x ≈ −38, 5 m
beside the peak, ≈ 5 m outside where the sets break, kept relative to the bar). After a WebGL
context restore the environment map and ripple tiles are regenerated. lil-gui: ocean (swell
height/period/direction, wind, breaking, bar → rebuild), board preset, rider mass, balance reflex
(0.85 tuned — higher over-drives it, it is not "easier"), time scale/pause/step, sun, debug
overlays (force arrows, probes).

### Controls (keyboard)

| key | prone | standing |
|---|---|---|
| W | paddle | weight forward |
| S | — | weight back |
| A / D | steer (outside arm harder; without W: pivot) | lean left / right rail |
| Q / E | — | twist (wind the upper body) |
| Shift | — | crouch |
| Space | pop up | — |
| R | reset at lineup | reset at lineup |
| T / Shift+T | spawn on a wave (Shift: other side) | spawn on a wave |
| O | autopilot on/off | autopilot on/off |
| C | cycle camera | |
| P / [ / ] | pause / slow-mo / speed up | |
