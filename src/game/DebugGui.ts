/**
 * lil-gui debug panel (G toggles): session (board, rider, spawn side, time scale, pause/step),
 * ocean (swell, sets, wind, breaking, sandbar → OceanModel.rebuild + OceanShaderData.rebuild
 * (+ BeachMesh rebuild for bathymetry changes)), sun & rendering, debug overlays, and the
 * physics tunables (HYDRO, FIN_MODEL, RIDER_MODEL are plain objects read every step).
 */
import GUI, { type Controller } from 'lil-gui';
import * as THREE from 'three';
import { DEFAULT_OCEAN_CONFIG, PRIMARY_SET_PATTERN, cloneOceanConfig, type OceanConfig } from '../ocean/oceanConfig';
import { HYDRO } from '../physics/BoardHull';
import { BOARD_PRESETS, type BoardPresetId } from '../physics/boardShape';
import { FIN_MODEL } from '../physics/fins';
import { RIDER_MODEL } from '../physics/Rider';
import type { OceanQuality } from '../render/OceanMesh';
import { CAMERA_MODES, type CameraMode } from './CameraRig';
import { TIME_SCALES, type SpawnSide } from './constants';
import type { Game } from './Game';

const DEG = 180 / Math.PI;

export class DebugGui {
  readonly gui: GUI;
  private readonly game: Game;
  private readonly controllers: Controller[] = [];
  /** Working copy of the ocean config edited by the panel. */
  private cfg: OceanConfig;
  private readonly ui: {
    board: BoardPresetId;
    mass: number;
    assist: number;
    wipeouts: boolean;
    side: SpawnSide;
    paused: boolean;
    timeScale: number;
    camera: CameraMode;
    swellHeight: number;
    swellPeriod: number;
    swellDirection: number;
    setStrength: number;
    secondaryHeight: number;
    windSpeed: number;
    windDirection: number;
    chop: number;
    breakerIndex: number;
    peakingMax: number;
    skewMax: number;
    barEnabled: boolean;
    barX: number;
    barDepth: number;
    barSweep: number;
    barWidth: number;
    barDepthSlope: number;
    sunElevation: number;
    sunAzimuth: number;
    exposure: number;
    toneMapping: string;
    quality: string;
    dpr: number;
    spray: number;
    shadows: boolean;
    xray: boolean;
    forces: boolean;
    probes: boolean;
    arrowScale: number;
    oceanDebug: number;
    wireframe: boolean;
    bankMaxDeg: number;
    slipDeg: number;
    railInsetCm: number;
    stallDeg: number;
  };

  constructor(game: Game) {
    this.game = game;
    this.cfg = cloneOceanConfig(game.ocean.config);
    const sim = game.sim;
    const env = game.env;
    const c = this.cfg;
    const sw0 = c.swells[0];
    this.ui = {
      board: sim.boardSpec.id as BoardPresetId,
      mass: sim.rider.config.mass,
      assist: sim.rider.config.balanceAssist,
      wipeouts: sim.rider.config.wipeouts,
      side: game.spawnSide,
      paused: game.paused,
      timeScale: game.timeScale,
      camera: game.cameraRig.mode,
      swellHeight: sw0?.height ?? 1,
      swellPeriod: sw0?.period ?? 11,
      swellDirection: sw0?.directionDeg ?? 0,
      setStrength: 1,
      secondaryHeight: c.swells[1]?.height ?? 0,
      windSpeed: c.wind.speed,
      windDirection: c.wind.directionDeg,
      chop: c.wind.amplitudeScale,
      breakerIndex: c.breakerIndex,
      peakingMax: c.peakingMax,
      skewMax: c.skewMax,
      barEnabled: c.bathymetry.bar.enabled,
      barX: c.bathymetry.bar.x0,
      barDepth: c.bathymetry.bar.peakDepth,
      barSweep: c.bathymetry.bar.sweep,
      barWidth: c.bathymetry.bar.width,
      barDepthSlope: c.bathymetry.bar.depthSlope,
      sunElevation: env.elevationDeg,
      sunAzimuth: env.azimuthDeg,
      exposure: env.exposure,
      toneMapping: 'ACES',
      quality: String(game.params.quality),
      dpr: game.params.dpr,
      spray: game.spray?.intensity ?? 0,
      shadows: game.params.shadows,
      xray: game.params.xray,
      forces: game.debugDraw.forces,
      probes: game.debugDraw.probes,
      arrowScale: game.debugDraw.scale * 1000,
      oceanDebug: game.oceanMesh.debugView,
      wireframe: false,
      bankMaxDeg: RIDER_MODEL.bankMax * DEG,
      slipDeg: RIDER_MODEL.slipAngle * DEG,
      railInsetCm: RIDER_MODEL.railInset * 100,
      stallDeg: FIN_MODEL.stallAngle * DEG,
    };
    const ui = this.ui;
    const gui = new GUI({ title: 'Quiver debug  (G)', width: 300 });
    this.gui = gui;
    gui.onChange(() => game.requestRender());
    const add = <T extends Controller>(ctl: T): T => {
      this.controllers.push(ctl);
      return ctl;
    };

    // --- session
    const fs = gui.addFolder('Session');
    add(fs.add(ui, 'board', Object.keys(BOARD_PRESETS)).name('board (keeps pose)')).onChange((v: BoardPresetId) => game.setBoard(v));
    add(fs.add(ui, 'mass', 45, 120, 1).name('rider mass kg')).onChange((v: number) => game.setRiderMass(v));
    // the quality of the rider's balance reflex: 0 clumsy … 1 very skilled (most forgiving)
    add(fs.add(ui, 'assist', 0, 1, 0.01).name('balance skill (1 = easiest)')).onChange((v: number) => (sim.rider.config.balanceAssist = v));
    add(fs.add(ui, 'wipeouts')).onChange((v: boolean) => (sim.rider.config.wipeouts = v));
    add(fs.add(ui, 'side', { 'left (−Z)': -1, 'right (+Z)': 1, auto: 0 }).name('wave spawn side')).onChange((v: SpawnSide) => (game.spawnSide = Number(v) as SpawnSide));
    add(fs.add(ui, 'camera', CAMERA_MODES).name('camera (C)')).onChange((v: CameraMode) => game.setCameraMode(v));
    add(fs.add(ui, 'paused').name('paused (P)')).onChange((v: boolean) => game.setPaused(v));
    const scales: Record<string, number> = {};
    for (const s of TIME_SCALES) scales[s >= 1 ? `×${s}` : `×1/${Math.round(1 / s)}`] = s;
    add(fs.add(ui, 'timeScale', scales).name('time scale ([ ])')).onChange((v: number) => game.setTimeScale(Number(v)));
    fs.add({ step: () => game.stepPhysics(4) }, 'step').name('step 1/60 s (.)');
    fs.add({ step1: () => game.stepPhysics(1) }, 'step1').name('step 1/240 s (,)');
    fs.add({ reset: () => game.resetLineup() }, 'reset').name('reset at lineup (R)');
    fs.add({ wave: () => game.spawnWave() }, 'wave').name('spawn on a wave (T)');

    // --- ocean
    const fo = gui.addFolder('Ocean (rebuild on release)');
    const apply = (bathy = false): void => this.applyOcean(bathy);
    add(fo.add(ui, 'swellHeight', 0.1, 3, 0.05).name('swell height m')).onFinishChange(() => apply());
    add(fo.add(ui, 'swellPeriod', 5, 18, 0.5).name('swell period s')).onFinishChange(() => apply());
    add(fo.add(ui, 'swellDirection', -35, 35, 1).name('swell direction °')).onFinishChange(() => apply());
    add(fo.add(ui, 'setStrength', 0, 1.5, 0.05).name('set strength (lulls)')).onFinishChange(() => apply());
    add(fo.add(ui, 'secondaryHeight', 0, 1.5, 0.05).name('2nd swell height m')).onFinishChange(() => apply());
    add(fo.add(ui, 'windSpeed', 0, 15, 0.5).name('wind m/s')).onFinishChange(() => apply());
    add(fo.add(ui, 'windDirection', -180, 180, 5).name('wind direction °')).onFinishChange(() => apply());
    add(fo.add(ui, 'chop', 0, 2, 0.05).name('chop scale')).onFinishChange(() => apply());
    add(fo.add(ui, 'breakerIndex', 0.5, 1.0, 0.01).name('breaker index γ')).onFinishChange(() => apply());
    add(fo.add(ui, 'peakingMax', 1, 4, 0.05).name('crest peaking')).onFinishChange(() => apply());
    add(fo.add(ui, 'skewMax', 0, 0.97, 0.01).name('face skew (steepness)')).onFinishChange(() => apply());
    const fb = fo.addFolder('Sandbar (rebuilds the beach)');
    add(fb.add(ui, 'barEnabled').name('bar')).onFinishChange(() => apply(true));
    add(fb.add(ui, 'barX', -80, 30, 1).name('crest x m')).onFinishChange(() => apply(true));
    add(fb.add(ui, 'barDepth', 0.5, 4, 0.05).name('depth at peak m')).onFinishChange(() => apply(true));
    add(fb.add(ui, 'barSweep', 0, 1.2, 0.01).name('sweep (A-frame)')).onFinishChange(() => apply(true));
    add(fb.add(ui, 'barWidth', 6, 60, 1).name('width m')).onFinishChange(() => apply(true));
    add(fb.add(ui, 'barDepthSlope', 0, 0.03, 0.001).name('depth slope (peel)')).onFinishChange(() => apply(true));
    fb.close();
    fo.add({ defaults: () => this.resetOcean() }, 'defaults').name('reset ocean to defaults');
    fo.close();

    // --- sun & render
    const fr = gui.addFolder('Sun & render');
    add(fr.add(ui, 'sunElevation', -2, 89, 0.5).name('sun elevation °')).onFinishChange(() => env.setSun(ui.sunElevation, ui.sunAzimuth));
    add(fr.add(ui, 'sunAzimuth', 0, 360, 1).name('sun azimuth °')).onFinishChange(() => env.setSun(ui.sunElevation, ui.sunAzimuth));
    add(fr.add(ui, 'exposure', 0.05, 1.5, 0.01)).onChange((v: number) => (env.exposure = v));
    add(fr.add(ui, 'toneMapping', ['ACES', 'AgX', 'Neutral'])).onChange((v: string) => {
      game.renderer.toneMapping = v === 'AgX' ? THREE.AgXToneMapping : v === 'Neutral' ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping;
    });
    add(fr.add(ui, 'quality', ['low', 'medium', 'high', 'ultra']).name('ocean quality')).onChange((v: string) => game.oceanMesh.setQuality(v as OceanQuality));
    add(fr.add(ui, 'dpr', 0.5, 3, 0.25).name('max pixel ratio')).onFinishChange((v: number) => game.setPixelRatioCap(v));
    if (game.spray) add(fr.add(ui, 'spray', 0, 3, 0.05).name('spray')).onChange((v: number) => (game.spray!.intensity = v));
    add(fr.add(ui, 'shadows').name('board/rider shadows')).onChange((v: boolean) => {
      game.env.sunLight.castShadow = v;
      game.renderer.shadowMap.enabled = v;
      game.params.shadows = v;
      game.renderer.shadowMap.needsUpdate = true;
    });
    add(fr.add(ui, 'xray').name('see board under water')).onChange((v: boolean) => {
      game.params.xray = v;
      game.boardMesh.underwaterVisible = v;
    });
    const look = game.oceanMesh.look;
    fr.add(look, 'turbidity', 0, 5, 0.05).name('water turbidity');
    fr.add(look, 'foamIntensity', 0, 2, 0.01).name('foam');
    fr.close();

    // --- debug
    const fd = gui.addFolder('Debug');
    add(fd.add(ui, 'forces').name('force arrows (F)')).onChange((v: boolean) => (game.debugDraw.forces = v));
    add(fd.add(ui, 'probes').name('water probes')).onChange((v: boolean) => (game.debugDraw.probes = v));
    add(fd.add(ui, 'arrowScale', 0.2, 10, 0.1).name('arrow mm / N')).onChange((v: number) => (game.debugDraw.scale = v / 1000));
    add(fd.add(ui, 'oceanDebug', { off: 0, lod: 1, normals: 2, foam: 3, breaking: 4, 'water body': 5 }).name('ocean view')).onChange((v: number) => (game.oceanMesh.debugView = Number(v)));
    add(fd.add(ui, 'wireframe').name('ocean wireframe')).onChange((v: boolean) => (game.oceanMesh.wireframe = v));
    fd.close();

    // --- physics tunables
    const ft = gui.addFolder('Physics tunables');
    ft.add(HYDRO, 'cp1', 0.3, 2, 0.01).name('planing cp1');
    ft.add(HYDRO, 'cp2', 0.3, 2, 0.01).name('impact cp2');
    ft.add(HYDRO, 'formFactor', 1, 2, 0.01).name('friction form factor');
    ft.add(HYDRO, 'railCp1', 0.2, 2, 0.01).name('rail cp1');
    ft.add(HYDRO, 'angularDamping', 0, 40, 0.5).name('angular damping');
    ft.add(HYDRO, 'radiation', 0, 1500, 10).name('wave-making damping');
    ft.add(HYDRO, 'addedMass', 0, 2, 0.05).name('added mass');
    add(ft.add(ui, 'stallDeg', 8, 30, 0.5).name('fin stall °')).onChange((v: number) => (FIN_MODEL.stallAngle = v / DEG));
    add(ft.add(ui, 'bankMaxDeg', 10, 60, 1).name('full-lean bank °')).onChange((v: number) => (RIDER_MODEL.bankMax = v / DEG));
    add(ft.add(ui, 'railInsetCm', 0, 12, 0.5).name('support: inset from rail cm')).onChange((v: number) => {
      RIDER_MODEL.railInset = v / 100;
      sim.rider.applyConfig();
    });
    ft.add(RIDER_MODEL, 'hipTorque', 0, 3, 0.05).name('hip torque N·m/kg');
    ft.add(RIDER_MODEL, 'recoverMargin', 0, 0.1, 0.005).name('recovery margin m');
    add(ft.add(ui, 'slipDeg', 20, 70, 1).name('feet friction angle °')).onChange((v: number) => (RIDER_MODEL.slipAngle = v / DEG));
    ft.add(RIDER_MODEL, 'twistTorque', 0, 150, 1).name('twist torque N·m');
    ft.close();

    gui.close();
  }

  get visible(): boolean {
    return this.gui.domElement.style.display !== 'none';
  }

  set visible(v: boolean) {
    this.gui.show(v);
  }

  /** Re-read values changed outside the panel (keys). */
  refresh(): void {
    const g = this.game;
    this.ui.paused = g.paused;
    this.ui.timeScale = g.timeScale;
    this.ui.camera = g.cameraRig.mode;
    this.ui.board = g.sim.boardSpec.id as BoardPresetId;
    this.ui.forces = g.debugDraw.forces;
    this.ui.probes = g.debugDraw.probes;
    for (const c of this.controllers) c.updateDisplay();
  }

  dispose(): void {
    this.gui.destroy();
  }

  private applyOcean(bathymetryChanged: boolean): void {
    const ui = this.ui;
    const c = this.cfg;
    if (c.swells[0]) {
      c.swells[0].height = ui.swellHeight;
      c.swells[0].period = ui.swellPeriod;
      c.swells[0].directionDeg = ui.swellDirection;
      // set strength: 0 = every wave is set-sized, 1 = default lulls, > 1 deeper lulls
      c.swells[0].setPattern = PRIMARY_SET_PATTERN.map((p) => Math.max(1 - ui.setStrength * (1 - p), 0.05));
    }
    if (c.swells[1]) c.swells[1].height = ui.secondaryHeight;
    c.wind.speed = ui.windSpeed;
    c.wind.directionDeg = ui.windDirection;
    c.wind.amplitudeScale = ui.chop;
    c.breakerIndex = ui.breakerIndex;
    c.peakingMax = ui.peakingMax;
    c.skewMax = ui.skewMax;
    const bar = c.bathymetry.bar;
    bar.enabled = ui.barEnabled;
    bar.x0 = ui.barX;
    bar.peakDepth = ui.barDepth;
    bar.sweep = ui.barSweep;
    bar.width = ui.barWidth;
    bar.depthSlope = ui.barDepthSlope;
    this.game.applyOceanConfig(c, bathymetryChanged);
  }

  private resetOcean(): void {
    this.cfg = cloneOceanConfig(DEFAULT_OCEAN_CONFIG);
    const c = this.cfg;
    const ui = this.ui;
    ui.swellHeight = c.swells[0].height;
    ui.swellPeriod = c.swells[0].period;
    ui.swellDirection = c.swells[0].directionDeg;
    ui.setStrength = 1;
    ui.secondaryHeight = c.swells[1]?.height ?? 0;
    ui.windSpeed = c.wind.speed;
    ui.windDirection = c.wind.directionDeg;
    ui.chop = c.wind.amplitudeScale;
    ui.breakerIndex = c.breakerIndex;
    ui.peakingMax = c.peakingMax;
    ui.skewMax = c.skewMax;
    const bar = c.bathymetry.bar;
    ui.barEnabled = bar.enabled;
    ui.barX = bar.x0;
    ui.barDepth = bar.peakDepth;
    ui.barSweep = bar.sweep;
    ui.barWidth = bar.width;
    ui.barDepthSlope = bar.depthSlope;
    this.game.applyOceanConfig(c, true);
    for (const ctl of this.controllers) ctl.updateDisplay();
  }
}
