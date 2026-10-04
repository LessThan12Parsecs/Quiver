/**
 * The playable test build: water + sky + beach (src/render), board + rider physics
 * (src/physics SurfSim) and their meshes, input, cameras, HUD and debug tools.
 *
 * Loop (requestAnimationFrame):
 *   1. input → SurfInput (once per frame);
 *   2. fixed physics steps of 1/240 s from a time accumulator (real dt × time scale), at most
 *      MAX_SUBSTEPS per frame (beyond that the game runs slower than real time);
 *   3. render interpolation between the last two physics states (board pose, rider COM and the
 *      ocean time): renderTime = sim.time − (1 − α)·dt;
 *   4. ocean.setTime(renderTime) + shaderData.update(ocean) (SurfSim.step sets its own time on
 *      the shared OceanModel every step, so this always comes after the physics);
 *   5. meshes, camera rig (needs the ocean at the render time), LOD/foam/spray updates, render,
 *      HUD.
 * When paused, frames are rendered only after something changed (cheap when idle, and headless
 * screenshots need an idle compositor); `ready` turns true when no render is pending.
 *
 * Rides (HUD badge + timer, last/best, follow-cam framing) and the "caught" pop-up cue are the
 * game's own notion (RideTracker: standing on a wave; telemetry.riding alone also holds for prone
 * belly rides and riderless boards).
 *
 * WebGL context loss: three.js restores the context, but render-target contents are gone; on
 * 'webglcontextrestored' the environment map and the ripple tiles are regenerated.
 *
 * URL parameters (parseGameParams): spawn=lineup|wave, side=1|-1|0 (wave side, 0 = automatic),
 * cam=follow|orbit|beach|side, board=shortboard|funboard|longboard|softtop, t=<start time s>,
 * paused=1, scale=<time scale>, q=low|medium|high|ultra|<number>, dpr=<max pixel ratio> (default
 * by quality: low 1, medium 1.25, otherwise 1.5),
 * mass=<rider kg>, assist=<0..1> (balance reflex strength, 0.85 = tuned; more is not easier),
 * spray=<intensity, 0 = off>, forces=1, probes=1, gui=0,
 * help=0, clean=1 (no HUD/GUI), shadows=0, xray=0, autopilot=1 (start with the autopilot on).
 */
import * as THREE from 'three';
import { DEFAULT_OCEAN_CONFIG, type OceanConfig } from '../ocean/oceanConfig';
import { OceanModel, createSurfaceSample, createSwellEval } from '../ocean/waveModel';
import { OceanShaderData } from '../ocean/waveGLSL';
import { BOARD_PRESETS, type BoardPresetId } from '../physics/boardShape';
import { PHYSICS_DT } from '../physics/constants';
import { type RiderPose } from '../physics/Rider';
import { Autopilot, DEFAULT_AUTOPILOT } from '../physics/autopilot';
import { lineupSpawn, waveSpawn } from '../physics/spawn';
import { SurfSim, type SurfInput } from '../physics/SurfSim';
import { BeachMesh } from '../render/BeachMesh';
import { BoardMesh } from '../render/BoardMesh';
import { BoardWake } from '../render/BoardWake';
import { Environment } from '../render/Environment';
import { OceanMesh, type OceanQuality } from '../render/OceanMesh';
import { RiderMesh } from '../render/RiderMesh';
import { Spray } from '../render/Spray';
import { CAMERA_MODES, CameraRig, type CameraMode, type CameraTarget } from './CameraRig';
import { DEFAULT_START_TIME, MAX_SUBSTEPS, TIME_SCALES, type SpawnSide } from './constants';
import { DebugDraw } from './DebugDraw';
import { DebugGui } from './DebugGui';
import { Hud } from './Hud';
import { RideTracker } from './RideTracker';
import { Input, type GameAction } from './Input';

export { DEFAULT_START_TIME, MAX_SUBSTEPS, TIME_SCALES, type SpawnSide } from './constants';

export interface GameParams {
  spawn: 'lineup' | 'wave';
  /** Wave spawn side of the peak: +1 (+Z), −1 (−Z), 0 = automatic (next good wave, alternating). */
  side: SpawnSide;
  cam: CameraMode;
  board: BoardPresetId;
  /** Start time (s). */
  t: number;
  paused: boolean;
  timeScale: number;
  quality: OceanQuality | number;
  /** Max device pixel ratio. */
  dpr: number;
  riderMass: number;
  balanceAssist: number | null;
  /** Spray intensity (0 = no spray particles). */
  spray: number;
  forces: boolean;
  probes: boolean;
  gui: boolean;
  help: boolean;
  clean: boolean;
  shadows: boolean;
  xray: boolean;
  autopilot: boolean;
}

export function parseGameParams(q: URLSearchParams): GameParams {
  const num = (k: string, d: number): number => {
    const v = q.get(k);
    const n = v === null || v === '' ? NaN : Number(v);
    return Number.isFinite(n) ? n : d;
  };
  const flag = (k: string, d: boolean): boolean => {
    const v = q.get(k);
    if (v === null) return d;
    return v === '' || v === '1' || v === 'true' || v === 'yes';
  };
  const board = (q.get('board') ?? 'funboard') as BoardPresetId;
  const cam = (q.get('cam') ?? 'follow') as CameraMode;
  const qs = q.get('q');
  let quality: OceanQuality | number = 'high';
  if (qs) quality = Number.isFinite(Number(qs)) ? Number(qs) : ((['low', 'medium', 'high', 'ultra'].includes(qs) ? qs : 'high') as OceanQuality);
  const side = num('side', 0);
  return {
    spawn: q.get('spawn') === 'wave' ? 'wave' : 'lineup',
    side: side > 0 ? 1 : side < 0 ? -1 : 0,
    cam: CAMERA_MODES.includes(cam) ? cam : 'follow',
    board: Object.hasOwn(BOARD_PRESETS, board) ? board : 'funboard',
    t: num('t', DEFAULT_START_TIME),
    paused: flag('paused', false),
    timeScale: Math.min(Math.max(num('scale', 1), 1 / 64), 4),
    quality,
    dpr: Math.min(Math.max(num('dpr', quality === 'low' ? 1 : quality === 'medium' ? 1.25 : 1.5), 0.25), 3),
    riderMass: Math.min(Math.max(num('mass', 75), 40), 130),
    balanceAssist: q.has('assist') ? Math.min(Math.max(num('assist', 0.85), 0), 1) : null,
    spray: Math.max(num('spray', 1), 0),
    forces: flag('forces', false),
    probes: flag('probes', false),
    gui: flag('gui', true),
    help: flag('help', true),
    clean: flag('clean', false),
    shadows: flag('shadows', true),
    xray: flag('xray', true),
    autopilot: flag('autopilot', false),
  };
}

/** JSON-friendly snapshot for automation and tests. */
export interface QuiverState {
  time: number;
  renderTime: number;
  stance: string;
  board: string;
  x: number;
  y: number;
  z: number;
  riderY: number;
  speed: number;
  speedAlongWave: number;
  headingDeg: number;
  pitchDeg: number;
  rollDeg: number;
  submergedLiters: number;
  deckDepth: number;
  /** telemetry.riding: the board moves with a wave (any stance, rider or not). */
  riding: boolean;
  ridingTime: number;
  /** The game's ride: standing on a wave, seconds so far (0 = not riding). */
  ride: number;
  /** Prone and the board runs with the wave (the pop-up cue). */
  caught: boolean;
  wipeoutReason: string | null;
  waveHeight: number;
  fullness: number;
  breaking: number;
  paddleThrustN: number;
  balance: number;
  /** Autopilot phase, or null when the player is in control. */
  autopilot: string | null;
  /** All of the above finite (no NaN/Infinity in the physics state). */
  finite: boolean;
  camera: CameraMode;
  paused: boolean;
  timeScale: number;
}

/** window.__quiver (automation hook for tools/smoke.ts and screenshots). */
export interface QuiverApi {
  /** True when a frame has been rendered and no render is pending. */
  ready: boolean;
  game: Game;
  sim: SurfSim;
  ocean: OceanModel;
  /** Uncaught page errors. */
  errors: string[];
  stats: Record<string, number | string>;
  /** Hold/release virtual keys (e.g. { W: true }); action keys fire their action. */
  setKeys(keys: Record<string, boolean>): void;
  /** Force SurfInput fields (null clears). */
  setInput(o: Partial<SurfInput> | null): void;
  /** Run the physics for `seconds` of simulated time right now (current keys/override). */
  stepSeconds(seconds: number): QuiverState;
  /** Spawn at the lineup or on the next wave. */
  spawn(kind: 'lineup' | 'wave', side?: 1 | -1): QuiverState;
  /** Autopilot on/off. */
  setAutopilot(on: boolean): void;
  setCamera(mode: CameraMode): void;
  setPaused(p: boolean): void;
  state(): QuiverState;
  /** Resolves after the next rendered frame. */
  render(): Promise<void>;
  /**
   * Render a frame now and read the canvas back: mean luminance (0–255) and the fractions of
   * near-black (< 8) and near-white (> 250) pixels — catches black screens / NaN shading.
   */
  pixelStats(): PixelStats;
}

export interface PixelStats {
  width: number;
  height: number;
  mean: number;
  black: number;
  white: number;
}

export class Game {
  readonly params: GameParams;
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly ocean: OceanModel;
  readonly shaderData: OceanShaderData;
  readonly env: Environment;
  readonly oceanMesh: OceanMesh;
  beach: BeachMesh | null;
  readonly spray: Spray | null;
  readonly sim: SurfSim;
  boardMesh: BoardMesh;
  readonly riderMesh: RiderMesh;
  /** Rail spray, wake and splashes of the board/rider. */
  readonly boardWake: BoardWake;
  readonly input: Input;
  readonly cameraRig: CameraRig;
  readonly hud: Hud;
  readonly debugDraw: DebugDraw;
  gui: DebugGui | null = null;

  paused: boolean;
  timeScale: number;
  /** Wave-spawn side of the peak: +1, −1, or 0 = automatic (the side with the next good wave). */
  spawnSide: SpawnSide;
  private lastSpawnSide: 1 | -1 = 1;
  /** The scripted surfer driving the inputs (O), or null when the player is in control. */
  autopilot: Autopilot | null = null;
  /** Standing rides (current, last, best) and the pop-up cue. */
  readonly rides = new RideTracker();
  readonly stats = {
    fps: 0,
    frameMs: 0,
    physicsMs: 0,
    substeps: 0,
    /** Simulated / requested time over the last second (< 1: physics can't keep up). */
    simLoad: 1,
    drawCalls: 0,
    triangles: 0,
    oceanPatches: 0,
    oceanVertices: 0,
  };
  readonly api: QuiverApi;

  // interpolation state: previous physics step (current = the sim itself)
  private readonly prevPos = new THREE.Vector3();
  private readonly prevQuat = new THREE.Quaternion();
  private readonly prevRider = new THREE.Vector3();
  private prevTime = 0;
  /** Interpolated render state. */
  readonly renderPos = new THREE.Vector3();
  readonly renderQuat = new THREE.Quaternion();
  readonly renderRider = new THREE.Vector3();
  renderTime = 0;
  private readonly renderPose: RiderPose;
  private readonly camTarget: CameraTarget;

  private acc = 0;
  private last = -1;
  private dirty = 3;
  private framesRendered = 0;
  private running = false;
  private rafId = 0;
  private fpsAcc = 0;
  private fpsN = 0;
  private loadReq = 0;
  private loadDone = 0;
  /** Height of a set wave approaching from behind (prone), 0 = none (per rendered frame). */
  private setBehind = 0;
  private hintTimer = 0;
  private renderResolvers: Array<() => void> = [];
  private readonly sample = createSurfaceSample();
  private readonly waveProbe = createSwellEval();
  private readonly drawSize = new THREE.Vector2();
  private readonly mA = new THREE.Matrix4();
  private readonly mB = new THREE.Matrix4();
  private readonly one = new THREE.Vector3(1, 1, 1);
  private readonly tmpQ = new THREE.Quaternion();
  private readonly tmpV = new THREE.Vector3();
  private readonly onResize = (): void => this.resize();
  private readonly onContextLost = (): void => {
    console.warn('[quiver] WebGL context lost');
    this.hud.flash('Graphics context lost', 'waiting for the GPU to come back…', 3600, performance.now() / 1000);
  };
  /** three.js re-creates GL objects, but render targets come back empty: regenerate them. */
  private readonly onContextRestored = (): void => {
    this.env.restoreContext(); // PMREM: water sky reflection + scene.environment
    this.oceanMesh.invalidateDetail(); // ripple tiles (not redrawn while the time stands still)
    this.hud.flash('Graphics restored', '', 1.2, performance.now() / 1000);
    this.requestRender(3);
  };

  constructor(container: HTMLElement, params: GameParams) {
    this.params = params;
    this.paused = params.paused;
    this.timeScale = params.timeScale;
    this.spawnSide = params.side;
    if (params.clean) document.body.classList.add('clean');

    // --- renderer, camera
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, params.dpr));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = params.shadows;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.domElement.tabIndex = 0;
    container.appendChild(renderer.domElement);
    this.renderer = renderer;
    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 70000);

    // --- ocean + world
    this.ocean = new OceanModel(DEFAULT_OCEAN_CONFIG);
    this.ocean.setTime(params.t);
    this.shaderData = new OceanShaderData(this.ocean);
    // Environment sets ACES tone mapping, sRGB output and the exposure
    this.env = new Environment(renderer, this.scene);
    const wind = this.ocean.config.wind;
    this.oceanMesh = new OceanMesh(this.shaderData, this.env, {
      quality: params.quality,
      windDirectionDeg: wind.directionDeg,
      windSpeed: wind.speed,
    });
    this.scene.add(this.oceanMesh.object3d);
    this.beach = new BeachMesh(this.ocean, this.env, this.shaderData);
    this.scene.add(this.beach.object3d);
    this.spray =
      params.spray > 0
        ? new Spray(this.shaderData, this.env, { intensity: params.spray, windDirectionDeg: wind.directionDeg, windSpeed: wind.speed })
        : null;
    if (this.spray) this.scene.add(this.spray.object3d);

    // sun shadows (board + rider only; the water/sand shaders don't use shadow maps)
    const sun = this.env.sunLight;
    sun.castShadow = params.shadows;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -3.2;
    sc.right = 3.2;
    sc.top = 3.2;
    sc.bottom = -3.2;
    sc.near = 1;
    sc.far = 120;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.015;
    sun.shadow.radius = 2;
    sc.updateProjectionMatrix();

    // --- physics
    this.sim = new SurfSim(this.ocean, BOARD_PRESETS[params.board], { mass: params.riderMass });
    if (params.balanceAssist !== null) this.sim.rider.config.balanceAssist = params.balanceAssist;

    // --- board + rider meshes
    this.boardMesh = new BoardMesh(this.sim.shape);
    this.boardMesh.underwaterVisible = params.xray;
    this.scene.add(this.boardMesh.object3d);
    this.riderMesh = new RiderMesh();
    this.scene.add(this.riderMesh.object3d);
    this.boardWake = new BoardWake(this.env);
    this.scene.add(this.boardWake.object3d);
    this.renderPose = clonePose(this.sim.rider.pose);
    this.camTarget = {
      position: this.renderPos,
      quaternion: this.renderQuat,
      velocity: this.sim.board.velocity,
      riderCom: this.renderRider,
      stance: 'prone',
      waveDir: new THREE.Vector3(1, 0, 0),
      riding: false,
      lookBack: false,
    };

    // --- game shell
    this.input = new Input(window);
    this.input.onAction = (a, shift) => this.onAction(a, shift);
    this.cameraRig = new CameraRig(this.camera, renderer.domElement, this.ocean);
    this.cameraRig.setMode(params.cam);
    this.hud = new Hud();
    this.hud.helpVisible = params.help;
    if (!params.help) document.body.classList.add('nohelp');
    this.debugDraw = new DebugDraw(this.sim);
    this.debugDraw.forces = params.forces;
    this.debugDraw.probes = params.probes;
    this.scene.add(this.debugDraw.object3d);
    if (params.gui && !params.clean) this.gui = new DebugGui(this);

    window.addEventListener('resize', this.onResize);
    renderer.domElement.addEventListener('pointerdown', () => this.requestRender());
    renderer.domElement.addEventListener('pointermove', (e) => {
      if (e.buttons) this.requestRender();
    });
    renderer.domElement.addEventListener('wheel', () => this.requestRender(), { passive: true });
    renderer.domElement.addEventListener('webglcontextlost', this.onContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', this.onContextRestored);
    window.addEventListener('keydown', () => this.requestRender());
    this.cameraRig.controls.addEventListener('change', () => this.requestRender());

    this.api = this.makeApi();

    // --- initial spawn
    if (params.spawn === 'wave') this.spawnWave();
    else this.resetLineup(params.t);
    if (params.autopilot) this.setAutopilot(true);
  }

  // ------------------------------------------------------------------------------- lifecycle

  start(): void {
    if (this.running) return;
    this.running = true;
    this.last = -1;
    const loop = (now: number): void => {
      if (!this.running) return;
      this.rafId = requestAnimationFrame(loop);
      this.frame(now);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  dispose(): void {
    this.stop();
    window.removeEventListener('resize', this.onResize);
    this.renderer.domElement.removeEventListener('webglcontextlost', this.onContextLost);
    this.renderer.domElement.removeEventListener('webglcontextrestored', this.onContextRestored);
    this.gui?.dispose();
    this.input.dispose();
    this.cameraRig.dispose();
    this.debugDraw.dispose();
    this.boardMesh.dispose();
    this.riderMesh.dispose();
    this.boardWake.dispose();
    this.spray?.dispose();
    this.beach?.dispose();
    this.oceanMesh.dispose();
    this.env.dispose();
    this.shaderData.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  /** Render again (when paused frames are only drawn on demand). */
  requestRender(frames = 2): void {
    this.dirty = Math.max(this.dirty, frames);
    if (this.api) this.api.ready = false;
  }

  // --------------------------------------------------------------------------------- actions

  /** Prone, still, outside the peak, facing the beach. */
  resetLineup(time = this.sim.time): void {
    this.sim.reset(lineupSpawn(this.ocean, time));
    this.afterTeleport();
    this.autopilot?.reset();
  }

  /**
   * Standing on the next set wave's steep unbroken face (searching forward in time, so the
   * clock jumps ahead). Side: explicit, the configured `spawnSide`, or automatic (0): the side
   * whose next qualifying face comes first (the same wave on both sides → the side not used last
   * time). Returns false if none was found (stays where it is).
   */
  spawnWave(side?: 1 | -1): boolean {
    let res: ReturnType<typeof waveSpawn> = null;
    let s: 1 | -1;
    const fixed = side ?? (this.spawnSide === 0 ? 0 : this.spawnSide);
    if (fixed !== 0) {
      s = fixed;
      res = waveSpawn(this.ocean, { fromTime: this.sim.time, side: s });
    } else {
      const preferred: 1 | -1 = this.lastSpawnSide > 0 ? -1 : 1;
      const a = waveSpawn(this.ocean, { fromTime: this.sim.time, side: preferred });
      const b = waveSpawn(this.ocean, { fromTime: this.sim.time, side: preferred > 0 ? -1 : 1, searchSeconds: a ? a.time - this.sim.time - 1 : 420 });
      if (b && (!a || b.time < a.time - 1)) {
        res = b;
        s = preferred > 0 ? -1 : 1;
      } else {
        res = a;
        s = preferred;
      }
    }
    if (!res) {
      console.warn('[quiver] waveSpawn found no wave; staying put');
      return false;
    }
    this.lastSpawnSide = s;
    this.sim.reset(res.spawn);
    this.afterTeleport();
    this.autopilot?.startRiding(s);
    this.hud.flash('Drop in!', `${res.waveHeight.toFixed(1)} m set wave · ${s > 0 ? 'right' : 'left'} of the peak`, 1.6, performance.now() / 1000);
    return true;
  }

  /** Turn the autopilot (scripted surfer) on or off. */
  setAutopilot(on: boolean): void {
    if (on === (this.autopilot !== null)) return;
    if (!on) {
      this.autopilot = null;
      this.hud.flash('You surf', 'autopilot off', 1.2, performance.now() / 1000);
      return;
    }
    const ap = new Autopilot(this.ocean, { autoReset: 5 });
    const st = this.sim.rider.stance;
    if (st === 'standing' || st === 'popping') {
      // already up and riding: decide the side from the board's heading relative to the wave
      const w = this.sim.telemetry.water;
      const R = this.sim.board.R;
      const cross = w.dirX * R[6] - w.dirZ * R[0];
      ap.startRiding(cross >= 0 ? 1 : -1);
    }
    this.autopilot = ap;
    this.hud.flash('Autopilot', 'watch a ride · O to take over', 1.6, performance.now() / 1000);
  }

  setBoard(id: BoardPresetId): void {
    const spec = BOARD_PRESETS[id];
    if (!spec || spec === this.sim.boardSpec) return;
    this.sim.setBoard(spec);
    this.boardMesh.dispose();
    this.boardMesh = new BoardMesh(this.sim.shape);
    this.boardMesh.underwaterVisible = this.params.xray;
    this.scene.add(this.boardMesh.object3d);
    this.afterTeleport();
  }

  setRiderMass(kg: number): void {
    this.sim.rider.config.mass = kg;
    this.sim.rider.applyConfig();
    this.requestRender();
  }

  setTimeScale(s: number): void {
    this.timeScale = s;
    this.requestRender();
  }

  setPaused(p: boolean): void {
    this.paused = p;
    this.acc = 0;
    this.requestRender();
  }

  setCameraMode(m: CameraMode): void {
    this.cameraRig.setMode(m);
    this.requestRender();
  }

  /**
   * Rebuild the ocean from a config (debug GUI). `bathymetryChanged` also regenerates the beach
   * mesh. The simulation keeps its time; the board re-settles naturally.
   */
  applyOceanConfig(cfg: OceanConfig, bathymetryChanged: boolean): void {
    this.ocean.rebuild(cfg);
    this.shaderData.rebuild(this.ocean);
    const w = this.ocean.config.wind;
    const drift = this.oceanMesh.material.uniforms.uWindDrift.value as THREE.Vector2;
    const a = THREE.MathUtils.degToRad(w.directionDeg);
    drift.set(Math.cos(a), Math.sin(a)).multiplyScalar(w.speed * 0.5);
    if (bathymetryChanged && this.beach) {
      this.beach.dispose();
      this.beach = new BeachMesh(this.ocean, this.env, this.shaderData);
      this.scene.add(this.beach.object3d);
    }
    this.requestRender();
  }

  /** Advance exactly n physics steps now (works while paused). */
  stepPhysics(n: number): void {
    for (let i = 0; i < n; i++) {
      this.input.update(PHYSICS_DT, this.sim.rider.stance);
      this.stepOnce();
    }
    this.requestRender();
  }

  /** Advance `seconds` of simulated time now (current keys / override). */
  stepSeconds(seconds: number): void {
    this.stepPhysics(Math.max(0, Math.round(seconds / PHYSICS_DT)));
  }

  /** Snapshot of the simulation state (automation). */
  state(): QuiverState {
    const sim = this.sim;
    const t = sim.telemetry;
    const p = sim.board.position;
    const deg = 180 / Math.PI;
    const s: QuiverState = {
      time: sim.time,
      renderTime: this.renderTime,
      stance: sim.rider.stance,
      board: sim.boardSpec.id,
      x: p.x,
      y: p.y,
      z: p.z,
      riderY: sim.rider.position.y,
      speed: t.speed,
      speedAlongWave: t.speedAlongWave,
      headingDeg: t.headingRad * deg,
      pitchDeg: t.pitchRad * deg,
      rollDeg: t.rollRad * deg,
      submergedLiters: t.submergedLiters,
      deckDepth: t.deckDepth,
      riding: t.riding,
      ridingTime: t.ridingTime,
      ride: this.rides.time,
      caught: this.rides.caught,
      wipeoutReason: t.wipeoutReason,
      waveHeight: t.water.waveHeight,
      fullness: t.water.fullness,
      breaking: t.water.breaking,
      paddleThrustN: t.paddleThrustN,
      balance: t.balance,
      autopilot: this.autopilot ? this.autopilot.status.phase : null,
      finite: true,
      camera: this.cameraRig.mode,
      paused: this.paused,
      timeScale: this.timeScale,
    };
    s.finite = Object.values(s).every((v) => typeof v !== 'number' || Number.isFinite(v)) && this.physicsFinite();
    return s;
  }

  // ------------------------------------------------------------------------------ internals

  private onAction(a: GameAction, shift: boolean): void {
    switch (a) {
      case 'reset':
        this.resetLineup();
        break;
      case 'spawnWave':
        this.spawnWave(shift ? (this.lastSpawnSide > 0 ? -1 : 1) : undefined);
        break;
      case 'autopilot':
        this.setAutopilot(this.autopilot === null);
        break;
      case 'camera':
        this.cameraRig.cycle();
        this.gui?.refresh();
        break;
      case 'pause':
        this.setPaused(!this.paused);
        this.gui?.refresh();
        break;
      case 'slower':
      case 'faster': {
        let i = TIME_SCALES.findIndex((s) => Math.abs(s - this.timeScale) < 1e-9);
        if (i < 0) i = TIME_SCALES.indexOf(1);
        i = Math.min(Math.max(i + (a === 'faster' ? 1 : -1), 0), TIME_SCALES.length - 1);
        this.setTimeScale(TIME_SCALES[i]);
        this.gui?.refresh();
        break;
      }
      case 'help':
        this.hud.helpVisible = !this.hud.helpVisible;
        document.body.classList.remove('nohelp');
        break;
      case 'gui':
        if (!this.gui) this.gui = new DebugGui(this);
        else this.gui.visible = !this.gui.visible;
        break;
      case 'forces': {
        const on = !this.debugDraw.forces;
        this.debugDraw.forces = on;
        this.debugDraw.probes = on;
        this.gui?.refresh();
        break;
      }
      case 'stepFrame':
        this.stepPhysics(shift ? 24 : 4);
        break;
      case 'stepPhysics':
        this.stepPhysics(1);
        break;
    }
    this.requestRender();
  }

  /** After a reset/spawn/board swap: no interpolation across the jump, camera snaps. */
  private afterTeleport(): void {
    const b = this.sim.board;
    this.prevPos.copy(b.position);
    this.prevQuat.copy(b.quaternion);
    this.prevRider.copy(this.sim.rider.position);
    this.prevTime = this.sim.time;
    this.acc = 0;
    // a ride in progress counts (pressing R/T or swapping boards mid-ride)
    this.rides.teleport();
    this.boardWake.reset();
    this.input.consumeEdges();
    this.cameraRig.snap();
    this.requestRender(3);
  }

  private stepOnce(): void {
    const sim = this.sim;
    const b = sim.board;
    this.prevPos.copy(b.position);
    this.prevQuat.copy(b.quaternion);
    this.prevRider.copy(sim.rider.position);
    this.prevTime = sim.time;
    if (this.autopilot) this.autopilot.update(sim, PHYSICS_DT, this.input.surf);
    const resetting = this.input.surf.reset;
    sim.step(PHYSICS_DT, this.input.surf);
    this.input.consumeEdges();
    if (resetting) this.afterTeleport();
    else this.rides.update(sim);
    if (!this.physicsFinite()) {
      console.error('[quiver] non-finite physics state; resetting at the lineup');
      this.resetLineup();
    }
  }

  private physicsFinite(): boolean {
    const b = this.sim.board;
    const r = this.sim.rider;
    return (
      Number.isFinite(b.position.x + b.position.y + b.position.z) &&
      Number.isFinite(b.velocity.x + b.velocity.y + b.velocity.z) &&
      Number.isFinite(b.quaternion.x + b.quaternion.y + b.quaternion.z + b.quaternion.w) &&
      Number.isFinite(r.position.x + r.position.y + r.position.z)
    );
  }

  private frame(now: number): void {
    const realDt = this.last < 0 ? 1 / 60 : Math.min(Math.max((now - this.last) / 1000, 0), 0.25);
    this.last = now;
    const sim = this.sim;
    this.input.update(realDt, sim.rider.stance);

    // --- fixed-step physics
    let n = 0;
    if (!this.paused) {
      this.acc += realDt * this.timeScale;
      const t0 = performance.now();
      while (this.acc >= PHYSICS_DT && n < MAX_SUBSTEPS) {
        this.stepOnce();
        this.acc -= PHYSICS_DT;
        n++;
      }
      // can't keep up: drop the backlog (the game slows down instead of spiralling)
      const want = realDt * this.timeScale;
      if (this.acc >= PHYSICS_DT) this.acc = PHYSICS_DT * 0.999;
      this.stats.physicsMs = performance.now() - t0;
      this.loadReq += want;
      this.loadDone += n * PHYSICS_DT;
      if (this.loadReq > 1) {
        this.stats.simLoad = Math.min(this.loadDone / this.loadReq, 1);
        this.loadReq = 0;
        this.loadDone = 0;
      }
      this.dirty = Math.max(this.dirty, 1);
    }
    this.stats.substeps = n;

    if (this.dirty <= 0) {
      if (!this.api.ready && this.framesRendered > 0) this.markReady();
      return;
    }
    this.dirty--;
    this.fpsAcc += realDt;
    this.fpsN++;
    if (this.fpsAcc > 0.5) {
      this.stats.fps = this.fpsN / this.fpsAcc;
      this.fpsAcc = 0;
      this.fpsN = 0;
    }
    const alpha = this.paused ? 1 : Math.min(Math.max(this.acc / PHYSICS_DT, 0), 1);
    this.renderFrame(alpha, this.paused ? Infinity : realDt, now / 1000);
    if (this.dirty <= 0 || !this.paused) this.markReady();
  }

  private markReady(): void {
    if (this.paused && this.dirty > 0) return;
    this.api.ready = true;
    const r = this.renderResolvers;
    this.renderResolvers = [];
    for (const f of r) f();
  }

  /** Interpolate, update every render component for this camera, draw, update the HUD. */
  private renderFrame(alpha: number, camDt: number, nowSec: number): void {
    const t0 = performance.now();
    const sim = this.sim;
    const b = sim.board;
    const rider = sim.rider;

    // --- interpolation between the previous and the current physics state
    this.renderPos.lerpVectors(this.prevPos, b.position, alpha);
    this.renderQuat.slerpQuaternions(this.prevQuat, b.quaternion, alpha);
    this.renderRider.lerpVectors(this.prevRider, rider.position, alpha);
    this.renderTime = this.prevTime + (sim.time - this.prevTime) * alpha;

    // --- the ocean at the render time (the physics left it at sim.time)
    this.ocean.setTime(this.renderTime);
    this.shaderData.update(this.ocean);

    // --- board + rider
    this.boardMesh.setPose(this.renderPos, this.renderQuat);
    const s = this.ocean.sample(this.renderPos.x, this.renderPos.z, this.sample);
    this.boardMesh.setWaterPlane(this.renderPos.x, this.renderPos.z, s.height, s.slopeX, s.slopeZ);
    this.interpolatePose();
    const surf = this.input.surf;
    const paddling = rider.stance === 'prone' && (surf.paddle > 0.02 || Math.abs(surf.steer) > 0.02);
    const riderWater = rider.stance === 'fallen' ? this.ocean.heightAt(this.renderRider.x, this.renderRider.z) : undefined;
    this.riderMesh.update(this.renderPose, this.renderQuat, rider.scale, paddling, this.renderTime, riderWater);

    // --- camera (needs the ocean at the render time: stays above the surface)
    const ct = this.camTarget;
    ct.stance = rider.stance;
    ct.riding = this.rides.riding;
    this.setBehind = rider.stance === 'prone' ? this.setWaveBehind() : 0;
    ct.lookBack = this.setBehind > 0 && !this.rides.caught;
    ct.waveDir.set(s.dirX, 0, s.dirZ);
    this.cameraRig.update(camDt, ct);

    // sun shadow camera centred on the board
    const sun = this.env.sunLight;
    sun.target.position.copy(this.renderPos);
    sun.position.copy(this.renderPos).addScaledVector(this.env.sunDirection, 60);
    sun.target.updateMatrixWorld();

    // --- world
    this.env.update(this.camera);
    const cp = this.camera.position;
    this.oceanMesh.update(this.camera, this.renderTime, this.ocean.heightAt(cp.x, cp.z));
    this.beach?.update(this.camera, this.renderTime);
    const viewH = this.renderer.getDrawingBufferSize(this.drawSize).y;
    this.spray?.update(this.camera, this.renderTime, viewH);
    this.boardWake.update(sim, this.ocean, this.renderPos, this.renderQuat, s, this.renderTime, this.camera, viewH);
    this.debugDraw.update();

    this.renderer.render(this.scene, this.camera);
    this.framesRendered++;

    const info = this.renderer.info.render;
    const st = this.stats;
    st.drawCalls = info.calls;
    st.triangles = info.triangles;
    st.oceanPatches = this.oceanMesh.stats.patches;
    st.oceanVertices = this.oceanMesh.stats.vertices;
    st.frameMs = performance.now() - t0;

    // --- HUD
    this.hud.update(
      sim.telemetry,
      {
        boardName: sim.boardSpec.name,
        boardVolumeL: sim.boardSpec.volumeLiters,
        fps: st.fps,
        timeScale: this.timeScale,
        paused: this.paused,
        camera: this.cameraRig.mode,
        riding: this.rides.riding,
        rideTime: this.rides.time,
        caught: this.rides.caught,
        lastRide: this.rides.lastRide,
        bestRide: this.rides.bestRide,
        tipOver: rider.tipOver,
        gamepad: this.input.gamepadActive,
        simLoad: st.simLoad,
        autopilot: this.autopilot ? this.autopilot.status.phase : null,
      },
      nowSec,
    );
    this.updateHint(nowSec);
    this.api.stats = {
      time: sim.time,
      fps: Math.round(st.fps * 10) / 10,
      drawCalls: st.drawCalls,
      triangles: st.triangles,
      oceanPatches: st.oceanPatches,
      oceanVertices: st.oceanVertices,
      stance: rider.stance,
      speed: Math.round(sim.telemetry.speed * 100) / 100,
      camera: this.cameraRig.mode,
    };
  }

  /**
   * Render copy of the rider pose: attached → carried rigidly by the interpolated board
   * (D = T_render · T_physics⁻¹), with the interpolated COM; fallen → shifted by the COM delta.
   */
  private interpolatePose(): void {
    const src = this.sim.rider.pose;
    const dst = this.renderPose;
    dst.stance = src.stance;
    dst.popProgress = src.popProgress;
    dst.crouch = src.crouch;
    dst.lean = src.lean;
    dst.com.copy(this.renderRider);
    if (src.stance === 'fallen') {
      const d = this.tmpV.subVectors(this.renderRider, src.com);
      dst.up.copy(src.up);
      dst.forward.copy(src.forward);
      dst.backFoot.copy(src.backFoot).add(d);
      dst.frontFoot.copy(src.frontFoot).add(d);
      dst.handLeft.copy(src.handLeft).add(d);
      dst.handRight.copy(src.handRight).add(d);
      return;
    }
    const b = this.sim.board;
    this.mB.compose(b.position, b.quaternion, this.one).invert();
    this.mA.compose(this.renderPos, this.renderQuat, this.one).multiply(this.mB);
    dst.backFoot.copy(src.backFoot).applyMatrix4(this.mA);
    dst.frontFoot.copy(src.frontFoot).applyMatrix4(this.mA);
    dst.handLeft.copy(src.handLeft).applyMatrix4(this.mA);
    dst.handRight.copy(src.handRight).applyMatrix4(this.mA);
    // directions rotate by the relative rotation only
    this.tmpQ.copy(b.quaternion).invert().premultiply(this.renderQuat);
    dst.up.copy(src.up).applyQuaternion(this.tmpQ);
    dst.forward.copy(src.forward).applyQuaternion(this.tmpQ);
  }

  /**
   * A set wave coming from behind: the height of the next crest seaward of the board if it is
   * within SET_RANGE and a set wave (> the autopilot's minHeight, 1.15 m), else 0. Evaluated at
   * the ocean's current time (call after the render-time setTime).
   */
  private setWaveBehind(): number {
    const p = this.renderPos;
    const se = this.ocean.evalSwell(p.x, p.z, this.waveProbe);
    if (se.height < 0.3 || se.domK <= 0) return 0;
    // waveform phase falls as a wave goes by: the next crest is (psi mod 2π) / k seaward
    const ahead = (((se.domPsi % TWO_PI) + TWO_PI) % TWO_PI) / se.domK;
    if (ahead > SET_RANGE) return 0;
    const dx = se.domDirX;
    const dz = se.domDirZ;
    const h = this.ocean.evalSwell(p.x - dx * ahead, p.z - dz * ahead, this.waveProbe).height;
    return h > DEFAULT_AUTOPILOT.minHeight ? h : 0;
  }

  /**
   * Take-off spot for the hints: where the autopilot sits (x ≈ −38, ≈ 5 m outside where the sets
   * break, 5 m beside the peak), kept relative to the bar crest when the bar is moved (GUI).
   */
  private takeoffSpot(): [number, number] {
    const bar = this.ocean.config.bathymetry.bar;
    const x = bar.x0 + (DEFAULT_AUTOPILOT.takeoffX - DEFAULT_OCEAN_CONFIG.bathymetry.bar.x0);
    const side = this.renderPos.z >= 0 ? 1 : -1;
    return [x, side * DEFAULT_AUTOPILOT.takeoffZ];
  }

  /** Context hints at the bottom (prone: where to wait, when a set is coming, when to pop up). */
  private updateHint(now: number): void {
    if (now - this.hintTimer < 0.25) return;
    this.hintTimer = now;
    const sim = this.sim;
    const st = sim.rider.stance;
    if (this.autopilot) {
      this.hud.setHint(AUTOPILOT_HINT[this.autopilot.status.phase] + ' · O: take over');
      return;
    }
    if (st !== 'prone') {
      this.hud.setHint(st === 'popping' || this.rides.riding ? null : st === 'standing' ? 'Trim with W/S, carve with A/D, crouch with Shift' : null);
      return;
    }
    if (this.rides.caught) {
      this.hud.setHint('The board is running with the wave — pop up now (Space)');
      return;
    }
    const p = this.renderPos;
    const [tx, tz] = this.takeoffSpot();
    const dx = tx - p.x;
    const dz = tz - p.z;
    const inZone = Math.abs(dx) < TAKEOFF_ZONE[0] && Math.abs(p.z) < TAKEOFF_ZONE[1];
    const set = this.setBehind;
    if (set > 0 && (inZone || (dx < 0 && dx > -12 && Math.abs(p.z) < 20)))
      this.hud.setHint(`${set.toFixed(1)} m set wave behind you — point at the beach and paddle hard (W) as it reaches you; pop up (Space) when CAUGHT shows`);
    else if (!inZone) {
      const where = Math.abs(dx) >= TAKEOFF_ZONE[0] ? `${dx > 0 ? 'in' : 'back out'} ≈ ${Math.round(Math.abs(dx))} m` : `≈ ${Math.round(Math.abs(dz))} m along the beach`;
      this.hud.setHint(`Paddle ${where} to the take-off spot, just outside where the sets break, beside the peak (W, steer A/D) · T drops you straight onto a wave`);
    } else this.hud.setHint('In the take-off zone — sit and watch for a set (the camera looks back when one comes) · T drops you onto a wave');
  }

  private resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.params.dpr));
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  /** Render synchronously and read the drawing buffer back (automation; slow). */
  pixelStats(): PixelStats {
    this.renderFrame(this.paused ? 1 : Math.min(Math.max(this.acc / PHYSICS_DT, 0), 1), Infinity, performance.now() / 1000);
    const gl = this.renderer.getContext();
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const buf = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    let sum = 0;
    let black = 0;
    let white = 0;
    const n = w * h;
    for (let i = 0; i < n; i++) {
      const l = 0.2126 * buf[4 * i] + 0.7152 * buf[4 * i + 1] + 0.0722 * buf[4 * i + 2];
      sum += l;
      if (l < 8) black++;
      else if (l > 250) white++;
    }
    return { width: w, height: h, mean: sum / n, black: black / n, white: white / n };
  }

  /** Change the pixel-ratio cap (GUI). */
  setPixelRatioCap(dpr: number): void {
    this.params.dpr = dpr;
    this.resize();
  }

  private makeApi(): QuiverApi {
    const errors: string[] = [];
    window.addEventListener('error', (e) => errors.push(String(e.message)));
    window.addEventListener('unhandledrejection', (e) => errors.push(String(e.reason)));
    const api: QuiverApi = {
      ready: false,
      game: this,
      sim: this.sim,
      ocean: this.ocean,
      errors,
      stats: {},
      setKeys: (k) => {
        this.input.setKeys(k);
        this.requestRender();
      },
      setInput: (o) => {
        this.input.override = o ? { ...o } : null;
      },
      stepSeconds: (sec) => {
        this.stepSeconds(sec);
        return this.state();
      },
      spawn: (kind, side) => {
        if (kind === 'wave') this.spawnWave(side);
        else this.resetLineup();
        return this.state();
      },
      setAutopilot: (on) => this.setAutopilot(on),
      setCamera: (m) => this.setCameraMode(m),
      setPaused: (p) => this.setPaused(p),
      state: () => this.state(),
      render: () =>
        new Promise<void>((resolve) => {
          this.renderResolvers.push(resolve);
          this.requestRender(2);
        }),
      pixelStats: () => this.pixelStats(),
    };
    return api;
  }
}

const TWO_PI = Math.PI * 2;
/** An approaching set wave's crest within this distance behind the board: hint + look-back, m. */
const SET_RANGE = 32;
/** Half-size of the take-off zone around the take-off spot: across the bar (x) and |z|, m. */
const TAKEOFF_ZONE = [5, 14] as const;

const AUTOPILOT_HINT: Record<string, string> = {
  position: 'Autopilot: paddling to the take-off spot near the peak',
  wait: 'Autopilot: sitting in the take-off zone, watching for a set',
  paddle: 'Autopilot: set wave coming — paddling hard, angled toward the shoulder',
  popup: 'Autopilot: popping up as the board starts to run',
  ride: 'Autopilot: riding — weight on the uphill rail, trimming high in the pocket',
  kickout: 'Autopilot: section closing — kicking out over the back',
  done: 'Autopilot: ride over, back to the lineup soon',
};

/** Deep-ish copy of a RiderPose with its own vectors (segments shared: not used for render). */
function clonePose(p: RiderPose): RiderPose {
  return {
    stance: p.stance,
    popProgress: p.popProgress,
    com: p.com.clone(),
    up: p.up.clone(),
    forward: p.forward.clone(),
    backFoot: p.backFoot.clone(),
    frontFoot: p.frontFoot.clone(),
    handLeft: p.handLeft.clone(),
    handRight: p.handRight.clone(),
    crouch: p.crouch,
    lean: p.lean,
    segments: p.segments,
  };
}
