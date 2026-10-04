/**
 * Render-only ocean viewer (no physics): water, sky, beach, spray (src/render OceanScene).
 *
 * URL params:
 *   cam=<lineup|beach|side|face|aerial|horizon>   camera preset (default beach)
 *   t=<seconds>     freeze the ocean at this time (otherwise time runs from 60 s)
 *   q=<low|medium|high|ultra|number>               ocean vertex density (default high)
 *   sun=<elev,azim> sun elevation/azimuth in degrees
 *   exposure=<n>    tone-mapping exposure
 *   fov=<deg>       vertical field of view (default 55)
 *   debug=<0..8>    ocean debug view (1 LOD, 2 normals, 3 foam, 4 breaking, 5 body, 6 black,
 *                   7 roughness, 8 pixel footprint)
 *   shading=<low|medium|high>  water fragment shading level (default: follows q)
 *   hide=<ocean,beach,spray>   hide components (debugging)
 *   spray=<n>       spray intensity (0 disables)
 *   beachDebug=<0..2>  beach debug view (1 vegetation masks, 2 swash)
 *   pos=x,y,z&look=x,y,z   custom camera (overrides the preset)
 *   clean=1         hide HUD and GUI (screenshots)
 * Keys: space pause, left/right step 0.5 s (shift: 5 s), 1-6 presets, L LOD view, G wireframe.
 * Automation: window.__oceanView = { ready, setView({ cam, t, q, pos, look }) -> Promise, stats, errors }.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';
import { OceanModel } from '../../src/ocean/waveModel';
import { OceanShaderData } from '../../src/ocean/waveGLSL';
import { OceanScene, type OceanQuality } from '../../src/render';

interface Preset {
  pos: [number, number, number];
  target: [number, number, number];
}

const PRESETS: Record<string, Preset> = {
  lineup: { pos: [-75, 2, 5], target: [40, 0, 0] },
  beach: { pos: [125, 3.5, 35], target: [-35, 0.5, 0] },
  side: { pos: [-24, 1.6, 38], target: [-34, 0.6, 0] },
  face: { pos: [-27, 1.0, 9], target: [-33, 0.6, 0] },
  aerial: { pos: [-80, 45, 90], target: [-10, 0, 0] },
  horizon: { pos: [-40, 2.5, 0], target: [-3000, 0, 0] },
};

interface ViewState {
  cam?: string;
  t?: number;
  q?: string;
  /** Custom camera position / look-at target (override `cam`). */
  pos?: [number, number, number];
  look?: [number, number, number];
}

interface OceanViewApi {
  ready: boolean;
  setView(v: ViewState): Promise<void>;
  /**
   * Render `frames` frames synchronously (waiting for the GPU after each) and return the mean ms per frame;
   * `animate` advances the ocean time by 1/60 s per frame. For relative cost comparisons.
   */
  bench(frames: number, animate?: boolean): number;
  /** The render components (debugging / benchmarks from automation). */
  world?: OceanScene;
  stats: Record<string, number | string>;
  errors: string[];
}

const params = new URLSearchParams(location.search);
const api: OceanViewApi = { ready: false, setView, bench: (n, a) => bench(n, a), stats: {}, errors: [] };
(window as unknown as { __oceanView: OceanViewApi }).__oceanView = api;
window.addEventListener('error', (e) => api.errors.push(String(e.message)));
if (params.get('clean') === '1') document.body.classList.add('clean');

function parseQuality(q: string | null): OceanQuality | number {
  if (!q) return 'high';
  const n = Number(q);
  if (Number.isFinite(n)) return n;
  return (['low', 'medium', 'high', 'ultra'].includes(q) ? q : 'high') as OceanQuality;
}

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(Number(params.get('fov') ?? 55), window.innerWidth / window.innerHeight, 0.1, 70000);

const sunParam = (params.get('sun') ?? '').split(',').map(Number);
const model = new OceanModel();
const shaderData = new OceanShaderData(model);
const world = new OceanScene(renderer, scene, model, shaderData, {
  quality: parseQuality(params.get('q')),
  environment: {
    sunElevationDeg: sunParam.length === 2 && Number.isFinite(sunParam[0]) ? sunParam[0] : undefined,
    sunAzimuthDeg: sunParam.length === 2 && Number.isFinite(sunParam[1]) ? sunParam[1] : undefined,
    exposure: params.has('exposure') ? Number(params.get('exposure')) : undefined,
  },
  spray: { intensity: params.has('spray') ? Number(params.get('spray')) : undefined },
});
const { env, ocean } = world;
api.world = world;
const beach = world.beach!;
const spray = world.spray!;
ocean.debugView = Number(params.get('debug') ?? 0);
beach.debugView = Number(params.get('beachDebug') ?? 0);
const shadingParam = params.get('shading');
if (shadingParam === 'low' || shadingParam === 'medium' || shadingParam === 'high') ocean.setShading(shadingParam);
for (const h of (params.get('hide') ?? '').split(',')) {
  if (h === 'ocean') ocean.object3d.visible = false;
  else if (h === 'beach') beach.object3d.visible = false;
  else if (h === 'spray') spray.object3d.visible = false;
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxPolarAngle = Math.PI * 0.499;

let time = 60;
let paused = false;
let speed = 1;
let presetName = 'beach';

function applyPreset(name: string): void {
  const p = PRESETS[name] ?? PRESETS.beach;
  presetName = PRESETS[name] ? name : 'beach';
  camera.position.set(...p.pos);
  controls.target.set(...p.target);
  camera.lookAt(controls.target);
  controls.update();
}

function parseVec(v: string | null): [number, number, number] | undefined {
  const a = (v ?? '').split(',').map(Number);
  return a.length === 3 && a.every(Number.isFinite) ? (a as [number, number, number]) : undefined;
}

function applyCustom(pos?: [number, number, number], look?: [number, number, number]): void {
  if (pos) camera.position.set(...pos);
  if (look) controls.target.set(...look);
  camera.lookAt(controls.target);
  controls.update();
  presetName = 'custom';
}

applyPreset(params.get('cam') ?? 'beach');
applyCustom(parseVec(params.get('pos')), parseVec(params.get('look')));
if (params.has('t')) {
  time = Number(params.get('t'));
  paused = true;
}

// ---------------------------------------------------------------------------- GUI + keys
const gui = new GUI({ title: 'Ocean view' });
const ui = {
  time,
  paused,
  speed,
  preset: presetName,
  quality: String(params.get('q') ?? 'high'),
  sunElevation: env.elevationDeg,
  sunAzimuth: env.azimuthDeg,
  exposure: env.exposure,
  haze: env.hazeDensity * 1e4,
  debug: ocean.debugView,
  shading: ocean.shading as string,
  wireframe: false,
};
gui.add(ui, 'time', 0, 400, 0.1).listen().onChange((v: number) => (time = v));
gui.add(ui, 'paused').listen().onChange((v: boolean) => (paused = v));
gui.add(ui, 'speed', 0, 3, 0.05).onChange((v: number) => (speed = v));
gui.add(ui, 'preset', Object.keys(PRESETS)).listen().onChange((v: string) => applyPreset(v));
gui.add(ui, 'quality', ['low', 'medium', 'high', 'ultra']).onChange((v: string) => {
  ocean.setQuality(parseQuality(v));
  ui.shading = ocean.shading;
});
const fSun = gui.addFolder('Sun & sky');
fSun.add(ui, 'sunElevation', -2, 89, 0.5).onFinishChange(() => env.setSun(ui.sunElevation, ui.sunAzimuth));
fSun.add(ui, 'sunAzimuth', 0, 360, 1).onFinishChange(() => env.setSun(ui.sunElevation, ui.sunAzimuth));
fSun.add(ui, 'exposure', 0.05, 2, 0.01).onChange((v: number) => (env.exposure = v));
fSun.add(ui, 'haze', 0, 10, 0.1).name('haze (1e-4/m)').onChange((v: number) => (env.hazeDensity = v * 1e-4));
const fWater = gui.addFolder('Water');
const look = ocean.look;
fWater.add(look, 'turbidity', 0, 5, 0.05);
fWater.add(look, 'bottomAlbedo', 0, 1.5, 0.01);
fWater.add(look, 'foamIntensity', 0, 2, 0.01);
fWater.add(look, 'sssIntensity', 0, 3, 0.01);
fWater.add(look, 'causticsIntensity', 0, 3, 0.01);
fWater.add(look, 'roughness', 0.005, 0.2, 0.001);
fWater.add(spray, 'intensity', 0, 3, 0.01).name('spray');
fWater.add(look, 'faceStreaks', 0, 0.15, 0.005).name('face streaks');
fWater.add(ui, 'shading', ['low', 'medium', 'high']).onChange((v: 'low' | 'medium' | 'high') => ocean.setShading(v));
fWater.add(ui, 'debug', { off: 0, lod: 1, normals: 2, foam: 3, breaking: 4, body: 5, roughness: 7, footprint: 8 }).onChange((v: number) => (ocean.debugView = v));
fWater.add(ui, 'wireframe').onChange((v: boolean) => (ocean.wireframe = v));
fSun.close();
fWater.close();

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.code === 'Space') paused = !paused;
  else if (e.code === 'ArrowRight') time += e.shiftKey ? 5 : 0.5;
  else if (e.code === 'ArrowLeft') time -= e.shiftKey ? 5 : 0.5;
  else if (e.code === 'KeyL') ocean.debugView = ocean.debugView === 1 ? 0 : 1;
  else if (e.code === 'KeyG') ocean.wireframe = !ocean.material.wireframe;
  else if (/^Digit[1-6]$/.test(e.code)) applyPreset(Object.keys(PRESETS)[Number(e.code.slice(5)) - 1]);
  ui.paused = paused;
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------------------- loop
// When paused, frames are rendered only after a change (SwiftShader frames take seconds, and
// screenshots need an idle compositor). `ready` turns true one frame after the last render.
const hud = document.getElementById('hud')!;
let dirty = 3;
let readyResolvers: Array<() => void> = [];
let last = performance.now();
let fpsAcc = 0;
let fpsN = 0;
let fps = 0;

function invalidate(frames = 2): void {
  dirty = Math.max(dirty, frames);
  api.ready = false;
}
controls.addEventListener('change', () => invalidate());
gui.onChange(() => invalidate());
window.addEventListener('keydown', () => invalidate());
window.addEventListener('resize', () => invalidate());

function setView(v: ViewState): Promise<void> {
  if (v.cam) applyPreset(v.cam);
  if (v.pos || v.look) applyCustom(v.pos, v.look);
  if (v.t !== undefined) {
    time = v.t;
    paused = true;
  }
  if (v.q) ocean.setQuality(parseQuality(v.q));
  invalidate(2);
  return new Promise((resolve) => readyResolvers.push(resolve));
}

function bench(frames: number, animate = false): number {
  // (a 1-pixel readPixels forces the GPU work to complete; gl.finish does not in Chromium)
  const gl = renderer.getContext();
  const px = new Uint8Array(4);
  const t0 = performance.now();
  for (let i = 0; i < frames; i++) {
    if (animate) time += 1 / 60;
    model.setTime(time);
    shaderData.update(model);
    world.update(camera, time);
    renderer.render(scene, camera);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  }
  invalidate(1);
  return (performance.now() - t0) / Math.max(frames, 1);
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  controls.update();
  if (!paused) {
    time += dt * speed;
    dirty = Math.max(dirty, 1);
  }
  if (dirty <= 0) {
    if (!api.ready) {
      api.ready = true;
      const r = readyResolvers;
      readyResolvers = [];
      r.forEach((f) => f());
    }
    return;
  }
  dirty--;
  fpsAcc += dt;
  fpsN++;
  if (fpsAcc > 0.5) {
    fps = fpsN / fpsAcc;
    fpsAcc = 0;
    fpsN = 0;
  }
  ui.time = time;
  model.setTime(time);
  shaderData.update(model);
  world.update(camera, time);
  renderer.render(scene, camera);

  const s = ocean.stats;
  api.stats = {
    time,
    preset: presetName,
    patches: s.patches,
    vertices: s.vertices,
    triangles: s.triangles,
    levels: s.levels,
    beachVertices: beach.vertexCount,
    drawCalls: renderer.info.render.calls,
    fps,
  };
  hud.textContent =
    `t ${time.toFixed(2)} s ${paused ? '(paused)' : ''}  ${fps.toFixed(0)} fps\n` +
    `ocean ${s.patches} patches, ${(s.vertices / 1000).toFixed(0)}k verts, ${s.levels} levels\n` +
    `cam ${camera.position.toArray().map((v) => v.toFixed(1)).join(', ')}`;
}
requestAnimationFrame(frame);
