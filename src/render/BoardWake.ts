/**
 * Board–water feedback (render only, driven by the simulation state; nothing here feeds back):
 *
 *  - Rail spray: droplets (and a little mist) thrown off the rail cells that cut the water line.
 *    Rate ∝ (relative speed − 2.5 m/s)² × how much of the rail is at the water line, strongly
 *    up for the engaged (lower) rail in a carve — a turn that bites throws a fan off its rail.
 *    Thrown outward (rail side normal) and up, carrying part of the board's velocity.
 *  - Wake: a foam trail from the tail (turbulent centre strip + two spreading edge lines),
 *    sampled every WAKE_STEP s while the hull is wetted and moving through the water, fading
 *    over WAKE_LIFE s; its vertices follow the (CPU) water surface every frame.
 *  - Splashes: paddling hands entering the water, and a burst when the rider wipes out.
 *
 * Particles are simulated on the CPU (pool of MAX_PARTICLES, per rendered frame with the render
 * time step: paused = frozen). Droplets fall under gravity with drag and die at the water plane
 * under the board. Lighting matches Spray (sun forward-scattering lobe + sky, per vertex).
 * Draw order: wake 5 (after the water, before the board's underwater pass, 20), particles 41
 * (after the rider, 30, and the crest spray, 40).
 */
import * as THREE from 'three';
import type { OceanModel, SurfaceSample } from '../ocean/waveModel';
import type { SurfSim } from '../physics/SurfSim';
import type { Environment } from './Environment';
import { ENV_GLSL, NOISE_GLSL } from './shaders/common';

const MAX_PARTICLES = 2048;
/** Wake: seconds between trail samples, lifetime of a sample (s), ring size. */
const WAKE_STEP = 0.08;
const WAKE_LIFE = 6;
const WAKE_SAMPLES = Math.ceil(WAKE_LIFE / WAKE_STEP) + 2;
/** Height of the wake over the CPU surface, m: the drawn water drops wind chop finer than its
 * LOD grid, so it can sit a few cm above the exact surface. */
const WAKE_LIFT = 0.07;
const G = 9.81;

const PARTICLE_VERTEX = /* glsl */ `
${NOISE_GLSL}
${ENV_GLSL}
attribute vec4 aData;        // age / life, size (m), alpha, kind (0 droplet, 1 mist)
uniform float uPixelScale;
varying float vAlpha;
varying float vKind;
varying vec3 vColor;
void main() {
  float u = aData.x;
  vKind = aData.w;
  vec4 mv = viewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float px = aData.y * uPixelScale / max(-mv.z, 0.1);
  float fade = smoothstep(0.0, 0.08, u) * (1.0 - smoothstep(0.55, 1.0, u));
  vAlpha = aData.z * fade * clamp(px / 1.5, 0.0, 1.0);
  gl_PointSize = clamp(px, 1.5, 64.0);
  if (vAlpha < 0.002 || u >= 1.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vColor = vec3(0.0);
    return;
  }
  vec3 V = normalize(cameraPosition - position);
  float cosT = dot(-V, uSunDirection);
  float g = 0.75;
  float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosT, 1.5) / (4.0 * 3.14159265);
  vColor = applyHaze(uSunColor * (0.1 + hg * 1.1) * 0.9 + uSkyIrradiance * 0.32, position, cameraPosition);
}
`;

const PARTICLE_FRAGMENT = /* glsl */ `
varying float vAlpha;
varying float vKind;
varying vec3 vColor;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  float soft = vKind > 0.5 ? exp(-3.5 * r2) - 0.03 : smoothstep(1.0, 0.35, r2);
  gl_FragColor = vec4(vColor, vAlpha * soft);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const WAKE_VERTEX = /* glsl */ `
attribute vec4 aWake;        // age / life, strength, across (-1..1), along (m)
varying vec4 vWake;
varying vec3 vWorld;
void main() {
  vWake = aWake;
  vWorld = position;
  gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
}
`;

const WAKE_FRAGMENT = /* glsl */ `
${NOISE_GLSL}
${ENV_GLSL}
varying vec4 vWake;
varying vec3 vWorld;
void main() {
  float u = vWake.x;
  float s = vWake.z;
  // turbulent strip behind the tail + the two edge lines spreading out
  float sc = s / 0.38;
  float se = (abs(s) - 0.82) / 0.13;
  float centre = exp(-sc * sc);
  float edges = exp(-se * se) * smoothstep(0.0, 0.15, u);
  float prof = 0.75 * centre * (1.0 - 0.6 * u) + 0.5 * edges;
  // bubbly, patchy foam that opens up as it ages
  vec2 p = vWorld.xz;
  float n = vnoise(p * 2.2) * 0.6 + vnoise(p * 5.3 + 3.1) * 0.4;
  float pat = smoothstep(0.25 + 0.45 * u, 0.75 + 0.2 * u, n + 0.25 * centre);
  float a = vWake.y * prof * mix(0.55, 1.0, pat) * pow(max(1.0 - u, 0.0), 1.6);
  if (a < 0.003) discard;
  vec3 alb = vec3(0.86, 0.9, 0.93);
  vec3 col = alb / 3.14159265 * (uSunColor * max(uSunDirection.y, 0.0) * 0.9 + uSkyIrradiance);
  col = applyHaze(col, vWorld, cameraPosition);
  gl_FragColor = vec4(col, clamp(a, 0.0, 0.85));
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

export class BoardWake {
  /** Add this to the scene. */
  readonly object3d = new THREE.Group();
  /** Live particles (stats/tests). */
  liveParticles = 0;

  // particles
  private readonly points: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  private readonly pPos: Float32Array;
  private readonly pData: Float32Array;
  private readonly pVel = new Float32Array(MAX_PARTICLES * 3);
  private readonly pLife = new Float32Array(MAX_PARTICLES);
  private readonly pAge = new Float32Array(MAX_PARTICLES);
  private next = 0;
  private emitCarry = 0;

  // wake ring buffer: x, z, lateral x, z, half width, strength, birth time, along (m)
  private readonly wake = new Float64Array(WAKE_SAMPLES * 8);
  private wakeHead = 0;
  private wakeCount = 0;
  private wakeAlong = 0;
  private lastWakeTime = -Infinity;
  private wakeOn = false;
  private readonly wakeMesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  private readonly wPos: Float32Array;
  private readonly wData: Float32Array;

  private lastTime = Number.NaN;
  private lastStance = '';
  private rng = 12345;
  private tailX = -1;
  private hullRef: unknown = null;
  private readonly v = new THREE.Vector3();
  private readonly n = new THREE.Vector3();

  constructor(env: Environment) {
    this.object3d.name = 'board-wake';
    // --- particles
    const pg = new THREE.BufferGeometry();
    this.pPos = new Float32Array(MAX_PARTICLES * 3);
    this.pData = new Float32Array(MAX_PARTICLES * 4);
    for (let i = 0; i < MAX_PARTICLES; i++) this.pData[i * 4] = 1; // dead
    pg.setAttribute('position', new THREE.BufferAttribute(this.pPos, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('aData', new THREE.BufferAttribute(this.pData, 4).setUsage(THREE.DynamicDrawUsage));
    const pm = new THREE.ShaderMaterial({
      name: 'BoardSpray',
      uniforms: { ...env.uniforms, uPixelScale: { value: 600 } },
      defines: { ...env.envDefines },
      vertexShader: PARTICLE_VERTEX,
      fragmentShader: PARTICLE_FRAGMENT,
      transparent: true,
      depthWrite: false,
    });
    this.points = new THREE.Points(pg, pm);
    this.points.frustumCulled = false;
    this.points.renderOrder = 41;
    this.object3d.add(this.points);

    // --- wake ribbon (2 vertices per sample, chronological order rebuilt each frame)
    const wg = new THREE.BufferGeometry();
    this.wPos = new Float32Array(WAKE_SAMPLES * 2 * 3);
    this.wData = new Float32Array(WAKE_SAMPLES * 2 * 4);
    wg.setAttribute('position', new THREE.BufferAttribute(this.wPos, 3).setUsage(THREE.DynamicDrawUsage));
    wg.setAttribute('aWake', new THREE.BufferAttribute(this.wData, 4).setUsage(THREE.DynamicDrawUsage));
    const idx = new Uint16Array((WAKE_SAMPLES - 1) * 6);
    for (let i = 0; i < WAKE_SAMPLES - 1; i++) {
      const a = 2 * i;
      idx.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], i * 6);
    }
    wg.setIndex(new THREE.BufferAttribute(idx, 1));
    wg.setDrawRange(0, 0);
    const wm = new THREE.ShaderMaterial({
      name: 'BoardWake',
      uniforms: { ...env.uniforms },
      defines: { ...env.envDefines },
      vertexShader: WAKE_VERTEX,
      fragmentShader: WAKE_FRAGMENT,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -8,
    });
    this.wakeMesh = new THREE.Mesh(wg, wm);
    this.wakeMesh.frustumCulled = false;
    this.wakeMesh.renderOrder = 5;
    this.object3d.add(this.wakeMesh);
  }

  /** Forget the trail and the particles (after a teleport). */
  reset(): void {
    this.wakeCount = 0;
    this.lastWakeTime = -Infinity;
    this.wakeOn = false;
    for (let i = 0; i < MAX_PARTICLES; i++) this.pData[i * 4] = 1;
    this.liveParticles = 0;
    this.lastTime = Number.NaN;
    (this.wakeMesh.geometry as THREE.BufferGeometry).setDrawRange(0, 0);
  }

  /**
   * Per rendered frame, after the ocean is set to `time` (the render time). `position` /
   * `quaternion` = the interpolated board pose, `water` = ocean.sample at the board.
   */
  update(
    sim: SurfSim,
    ocean: OceanModel,
    position: THREE.Vector3,
    quaternion: THREE.Quaternion,
    water: SurfaceSample,
    time: number,
    camera: THREE.PerspectiveCamera,
    viewportHeightPx: number,
  ): void {
    const dt = Number.isFinite(this.lastTime) ? Math.min(Math.max(time - this.lastTime, 0), 0.1) : 0;
    this.lastTime = time;
    this.points.material.uniforms.uPixelScale.value = (camera.projectionMatrix.elements[5] * viewportHeightPx) / 2;
    if (sim.hull !== this.hullRef) {
      this.hullRef = sim.hull;
      let m = 0;
      for (let k = 0; k < sim.hull.n; k++) m = Math.min(m, sim.hull.cx[k]);
      this.tailX = m;
    }
    if (dt > 0) {
      this.emit(sim, position, quaternion, water, dt);
      this.integrate(water, position, dt);
      this.sampleWake(sim, position, quaternion, time);
    }
    this.buildWake(ocean, time);
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.points.material.dispose();
    this.wakeMesh.geometry.dispose();
    this.wakeMesh.material.dispose();
    this.object3d.removeFromParent();
  }

  // -------------------------------------------------------------------------------------------

  private rand(): number {
    // xorshift32 (deterministic, allocation-free)
    let x = this.rng;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rng = x;
    return (x >>> 0) / 4294967296;
  }

  private spawn(x: number, y: number, z: number, vx: number, vy: number, vz: number, size: number, alpha: number, kind: number, life: number): void {
    const i = this.next;
    this.next = (this.next + 1) % MAX_PARTICLES;
    this.pPos[i * 3] = x;
    this.pPos[i * 3 + 1] = y;
    this.pPos[i * 3 + 2] = z;
    this.pVel[i * 3] = vx;
    this.pVel[i * 3 + 1] = vy;
    this.pVel[i * 3 + 2] = vz;
    this.pAge[i] = 0;
    this.pLife[i] = life;
    this.pData[i * 4] = 0;
    this.pData[i * 4 + 1] = size;
    this.pData[i * 4 + 2] = alpha;
    this.pData[i * 4 + 3] = kind;
  }

  private emit(sim: SurfSim, pos: THREE.Vector3, q: THREE.Quaternion, water: SurfaceSample, dt: number): void {
    const t = sim.telemetry;
    const st = sim.rider.stance;
    const bv = sim.board.velocity;
    const vrel = t.speedRelWater;
    const hull = sim.hull;
    // --- rail spray
    const base = Math.max(vrel - 2.5, 0);
    if (base > 0 && t.wettedArea > 0.02) {
      const bank = Math.min(Math.abs(Math.sin(t.rollRad)), 0.8);
      const lowRail = t.rollRad > 0 ? 1 : -1; // right rail down for positive roll
      for (let k = 0; k < hull.n; k++) {
        const side = hull.rail[k];
        if (side === 0) continue;
        const f = hull.cellSubmerged[k];
        if (f <= 0.03 || f >= 0.97) continue; // the spray root is where the rail cuts the water line
        const engaged = side === lowRail ? 0.15 + 2.2 * bank : 0.1;
        const rate = 5 * 4 * f * (1 - f) * base * base * engaged; // particles / s from this cell
        this.emitCarry += rate * dt;
        while (this.emitCarry >= 1) {
          this.emitCarry -= 1;
          this.v.set(hull.rx[k], hull.ryb[k], hull.rz[k]).applyQuaternion(q).add(pos);
          this.n.set(hull.sn[3 * k], 0.3, hull.sn[3 * k + 2]).applyQuaternion(q);
          this.n.y = Math.abs(this.n.y);
          const out = (0.25 + 0.35 * this.rand()) * vrel * (0.4 + bank);
          const up = (0.15 + 0.3 * this.rand()) * vrel * (0.5 + bank);
          const mist = this.rand() < 0.2;
          this.spawn(
            this.v.x,
            this.v.y + 0.02,
            this.v.z,
            bv.x * 0.55 + this.n.x * out + (this.rand() - 0.5) * 0.6,
            up,
            bv.z * 0.55 + this.n.z * out + (this.rand() - 0.5) * 0.6,
            mist ? 0.15 + 0.3 * this.rand() : 0.025 + 0.05 * this.rand(),
            mist ? 0.18 : 0.75,
            mist ? 1 : 0,
            mist ? 0.9 + 0.8 * this.rand() : 0.5 + 0.6 * this.rand(),
          );
        }
      }
    }
    // --- paddling: hands entering the water
    if (st === 'prone' && t.paddleThrustN > 5) {
      this.handSplash(sim.rider.pose.handLeft, sim, water, dt);
      this.handSplash(sim.rider.pose.handRight, sim, water, dt);
    }
    // --- wipeout splash
    if (st === 'fallen' && this.lastStance !== 'fallen' && this.lastStance !== '') {
      const r = sim.rider.position;
      const sp = Math.min(Math.hypot(bv.x, bv.z), 8);
      for (let j = 0; j < 140; j++) {
        const a = this.rand() * Math.PI * 2;
        const out = (0.5 + this.rand()) * (0.6 + 0.25 * sp);
        const mist = j % 7 === 0;
        this.spawn(
          r.x + Math.cos(a) * 0.3,
          water.height + 0.05,
          r.z + Math.sin(a) * 0.3,
          Math.cos(a) * out + bv.x * 0.3,
          (1.2 + 2.5 * this.rand()) * (0.6 + 0.08 * sp),
          Math.sin(a) * out + bv.z * 0.3,
          mist ? 0.25 + 0.3 * this.rand() : 0.025 + 0.05 * this.rand(),
          mist ? 0.2 : 0.8,
          mist ? 1 : 0,
          mist ? 1.2 + this.rand() : 0.6 + 0.6 * this.rand(),
        );
      }
    }
    this.lastStance = st;
  }

  private handSplash(h: THREE.Vector3, sim: SurfSim, water: SurfaceSample, dt: number): void {
    if (h.y > water.height + 0.02 || this.rand() > 18 * dt) return;
    const bv = sim.board.velocity;
    for (let j = 0; j < 3; j++) {
      const vx = bv.x * 0.3 + (this.rand() - 0.5);
      const vz = bv.z * 0.3 + (this.rand() - 0.5);
      this.spawn(h.x, water.height + 0.03, h.z, vx, 0.8 + 1.2 * this.rand(), vz, 0.015 + 0.02 * this.rand(), 0.6, 0, 0.4 + 0.3 * this.rand());
    }
  }

  private integrate(water: SurfaceSample, pos: THREE.Vector3, dt: number): void {
    let live = 0;
    for (let i = 0; i < MAX_PARTICLES; i++) {
      const d = i * 4;
      if (this.pData[d] >= 1) continue;
      const age = this.pAge[i] + dt;
      this.pAge[i] = age;
      const mist = this.pData[d + 3] > 0.5;
      const k = i * 3;
      const drag = Math.exp(-(mist ? 2.5 : 0.6) * dt);
      this.pVel[k] *= drag;
      this.pVel[k + 2] *= drag;
      this.pVel[k + 1] = this.pVel[k + 1] * drag - (mist ? 0.15 : 1) * G * dt;
      this.pPos[k] += this.pVel[k] * dt;
      this.pPos[k + 1] += this.pVel[k + 1] * dt;
      this.pPos[k + 2] += this.pVel[k + 2] * dt;
      // droplets end in the water (the plane of the surface under the board)
      const wy = water.height + water.slopeX * (this.pPos[k] - pos.x) + water.slopeZ * (this.pPos[k + 2] - pos.z);
      const dead = age >= this.pLife[i] || (!mist && this.pVel[k + 1] < 0 && this.pPos[k + 1] < wy);
      this.pData[d] = dead ? 1 : age / this.pLife[i];
      if (!dead) live++;
    }
    this.liveParticles = live;
    const g = this.points.geometry;
    g.getAttribute('position').needsUpdate = true;
    g.getAttribute('aData').needsUpdate = true;
  }

  /** Push a trail sample at the tail while the hull moves through the water. */
  private sampleWake(sim: SurfSim, pos: THREE.Vector3, q: THREE.Quaternion, time: number): void {
    if (time - this.lastWakeTime < WAKE_STEP) return;
    this.lastWakeTime = time;
    const t = sim.telemetry;
    let strength = sim.rider.stance === 'fallen' ? 0 : smooth(0.8, 4.5, t.speedRelWater) * Math.min(t.wettedArea / 0.25, 1);
    // no trail while out of the water / stopped: a strength-0 sample ends the ribbon, and a new
    // one starts with a strength-0 sample (the segment bridging the gap stays invisible)
    if (strength <= 0) {
      if (!this.wakeOn) return;
      this.wakeOn = false;
    } else if (!this.wakeOn) {
      this.wakeOn = true;
      strength = 0;
    }
    const prev = this.wakeCount > 0 ? ((this.wakeHead - 1 + WAKE_SAMPLES) % WAKE_SAMPLES) * 8 : -1;
    this.v.set(this.tailX, 0, 0).applyQuaternion(q).add(pos);
    this.n.set(0, 0, 1).applyQuaternion(q);
    const nl = Math.hypot(this.n.x, this.n.z) || 1;
    const o = this.wakeHead * 8;
    if (prev >= 0) this.wakeAlong += Math.hypot(this.v.x - this.wake[prev], this.v.z - this.wake[prev + 1]);
    this.wake[o] = this.v.x;
    this.wake[o + 1] = this.v.z;
    this.wake[o + 2] = this.n.x / nl;
    this.wake[o + 3] = this.n.z / nl;
    this.wake[o + 4] = 0.22 + 0.04 * Math.min(t.speedRelWater, 8);
    this.wake[o + 5] = strength;
    this.wake[o + 6] = time;
    this.wake[o + 7] = this.wakeAlong;
    this.wakeHead = (this.wakeHead + 1) % WAKE_SAMPLES;
    this.wakeCount = Math.min(this.wakeCount + 1, WAKE_SAMPLES);
  }

  /** Ribbon vertices (oldest → newest) on the current water surface. */
  private buildWake(ocean: OceanModel, time: number): void {
    let n = 0;
    for (let j = this.wakeCount; j >= 1; j--) {
      const o = ((this.wakeHead - j + WAKE_SAMPLES) % WAKE_SAMPLES) * 8;
      const age = time - this.wake[o + 6];
      if (age >= WAKE_LIFE || age < 0) continue;
      const u = age / WAKE_LIFE;
      const hw = this.wake[o + 4] + 0.3 * age;
      for (let side = -1; side <= 1; side += 2) {
        const x = this.wake[o] + side * hw * this.wake[o + 2];
        const z = this.wake[o + 1] + side * hw * this.wake[o + 3];
        const k = n * 2 + (side + 1) / 2;
        this.wPos[k * 3] = x;
        this.wPos[k * 3 + 1] = ocean.heightAt(x, z) + WAKE_LIFT;
        this.wPos[k * 3 + 2] = z;
        this.wData[k * 4] = u;
        this.wData[k * 4 + 1] = this.wake[o + 5];
        this.wData[k * 4 + 2] = side;
        this.wData[k * 4 + 3] = this.wake[o + 7];
      }
      n++;
    }
    const g = this.wakeMesh.geometry;
    g.setDrawRange(0, Math.max(n - 1, 0) * 6);
    g.getAttribute('position').needsUpdate = true;
    g.getAttribute('aWake').needsUpdate = true;
  }
}

function smooth(a: number, b: number, x: number): number {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
}
