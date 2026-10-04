/**
 * Seabed + beach + hinterland terrain.
 *
 * A static tensor-product grid with graded spacing: fine (0.75 m) across the swash zone and the
 * dry beach, coarser offshore and far along the coast. Heights come straight from the analytic
 * bathymetry (ocean.bathymetry.seabedY) up to the dune toe; behind the beach procedural dunes and
 * low hills are added so the view toward land looks natural. Covers x in [-400, 4000],
 * z in [-6000, 6000].
 *
 * The underwater part is normally hidden by the (opaque) water, whose shader fakes the seabed with
 * the same sand albedo; it is drawn after the ocean (renderOrder) so early-z rejects it.
 */
import * as THREE from 'three';
import type { OceanModel } from '../ocean/waveModel';
import type { OceanShaderData } from '../ocean/waveGLSL';
import type { Environment } from './Environment';
import { BEACH_FRAGMENT, BEACH_VERTEX } from './shaders/beach';

export interface BeachMeshOptions {
  /** Spacing multiplier (1 = default; smaller = denser). */
  resolution?: number;
  /** Top of the permanently wet band, metres above still water. */
  wetLevel?: number;
  /** Add procedural dunes / hills behind the beach (default true). */
  hinterland?: boolean;
}

/** Graded sample positions: [start, end, step] segments; steps grow geometrically if growth > 1. */
function samples(segments: Array<[number, number, number, number?]>, scale: number): number[] {
  const out: number[] = [];
  for (const [a, b, step0, growth = 1] of segments) {
    let x = a;
    let step = step0 * scale;
    if (out.length && Math.abs(out[out.length - 1] - a) < 1e-6) x = a;
    while (x < b - 1e-6) {
      if (!out.length || x > out[out.length - 1] + 1e-6) out.push(x);
      x += step;
      step *= growth;
    }
  }
  const last = segments[segments.length - 1][1];
  if (out[out.length - 1] < last - 1e-6) out.push(last);
  return out;
}

function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

/** Cheap deterministic 2D value-noise fbm for terrain shaping (CPU). */
function hash(ix: number, iz: number): number {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iz, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x: number, z: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  const a = hash(ix, iz);
  const b = hash(ix + 1, iz);
  const c = hash(ix, iz + 1);
  const d = hash(ix + 1, iz + 1);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}
function fbm(x: number, z: number, oct: number): number {
  let s = 0;
  let a = 0.5;
  let w = 0;
  for (let i = 0; i < oct; i++) {
    s += a * vnoise(x, z);
    w += a;
    const nx = 0.8 * x - 0.6 * z;
    z = 0.6 * x + 0.8 * z;
    x = nx * 2.03 + 11.7;
    z = z * 2.03 + 5.3;
    a *= 0.5;
  }
  return s / w;
}

export class BeachMesh {
  readonly object3d: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly material: THREE.ShaderMaterial;
  readonly vertexCount: number;
  private readonly ocean: OceanModel;
  private readonly camPos = new THREE.Vector3();

  constructor(ocean: OceanModel, env: Environment, shaderData: OceanShaderData, opts: BeachMeshOptions = {}) {
    this.ocean = ocean;
    const s = opts.resolution ?? 1;
    const hinterland = opts.hinterland !== false;
    const xs = samples(
      [
        [-400, 40, 10],
        [40, 95, 2.5],
        [95, 190, 0.75],
        [190, 320, 2],
        [320, 4000, 3, 1.075],
      ],
      s,
    );
    const zHalf = samples(
      [
        [0, 260, 2],
        [260, 6000, 2, 1.07],
      ],
      s,
    );
    const zs = [...zHalf.slice(1).map((z) => -z).reverse(), ...zHalf];
    const nx = xs.length;
    const nz = zs.length;
    const bathy = ocean.bathymetry;

    const heightAt = (x: number, z: number): number => {
      let y = bathy.seabedY(x, z);
      if (!hinterland) return y;
      // Foredune ridge parallel to the shore (steep seaward face), back dunes, then low hills.
      const crest = 214 + 14 * (fbm(z / 260, 1.3, 3) - 0.5);
      const d = x - crest;
      const fh = 6 + 5 * fbm(z / 110, 5.7, 3);
      const fore = fh * (d < 0 ? Math.exp(-((d / 13) ** 2)) : Math.exp(-((d / 34) ** 2)));
      const back = smooth(225, 320, x) * (1 - smooth(900, 1500, x)) * (2 + 9 * fbm(x / 55, z / 75, 4) ** 1.5);
      const hills =
        smooth(450, 2600, x) * (60 * fbm(x / 900, z / 1300, 5) + 28 * fbm(x / 260, z / 330, 4) - 22) +
        smooth(1300, 3800, x) * Math.max(520 * fbm(x / 1400 + 9.1, z / 1900, 5) - 150, 0);
      y += fore * smooth(150, 185, x) + back + Math.max(hills, -4);
      return y;
    };

    const pos = new Float32Array(nx * nz * 3);
    const veg = new Float32Array(nx * nz);
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const x = xs[i];
        const z = zs[j];
        pos[k * 3] = x;
        pos[k * 3 + 1] = heightAt(x, z);
        pos[k * 3 + 2] = z;
        veg[k] = hinterland ? smooth(192, 222, x) + smooth(290, 620, x) : 0;
      }
    }
    const nrm = new Float32Array(nx * nz * 3);
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const i0 = Math.max(i - 1, 0);
        const i1 = Math.min(i + 1, nx - 1);
        const j0 = Math.max(j - 1, 0);
        const j1 = Math.min(j + 1, nz - 1);
        const dydx = (pos[(j * nx + i1) * 3 + 1] - pos[(j * nx + i0) * 3 + 1]) / (xs[i1] - xs[i0]);
        const dydz = (pos[(j1 * nx + i) * 3 + 1] - pos[(j0 * nx + i) * 3 + 1]) / (zs[j1] - zs[j0]);
        const inv = 1 / Math.hypot(dydx, 1, dydz);
        nrm[k * 3] = -dydx * inv;
        nrm[k * 3 + 1] = inv;
        nrm[k * 3 + 2] = -dydz * inv;
      }
    }
    const idx = new Uint32Array((nx - 1) * (nz - 1) * 6);
    let o = 0;
    for (let j = 0; j < nz - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i;
        const b = a + 1;
        const c = a + nx;
        const d = c + 1;
        idx[o++] = a;
        idx[o++] = c;
        idx[o++] = b;
        idx[o++] = b;
        idx[o++] = c;
        idx[o++] = d;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    geo.setAttribute('aVeg', new THREE.BufferAttribute(veg, 1));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeBoundingSphere();
    this.vertexCount = nx * nz;

    const wind = ocean.config.wind.directionDeg * (Math.PI / 180);
    this.material = new THREE.ShaderMaterial({
      name: 'BeachSand',
      uniforms: {
        ...shaderData.uniforms,
        ...env.uniforms,
        uTime: { value: 0 },
        uSwashX: { value: new THREE.Vector2(100, 121) },
        uWetLevel: { value: opts.wetLevel ?? 0.7 },
        uWindDir: { value: new THREE.Vector2(Math.cos(wind), Math.sin(wind)) },
        uDebug: { value: 0 },
        uUnderwater: { value: 0 },
      },
      defines: { ...env.envDefines },
      vertexShader: BEACH_VERTEX,
      fragmentShader: BEACH_FRAGMENT,
    });
    this.object3d = new THREE.Mesh(geo, this.material);
    this.object3d.name = 'beach';
    this.object3d.renderOrder = 1;
    this.object3d.matrixAutoUpdate = false;
  }

  /** 0 = off, 1 = vegetation masks (dunes, inland, cover), 2 = swash (sheet, foam, wet). */
  set debugView(mode: number) {
    this.material.uniforms.uDebug.value = mode;
  }

  get debugView(): number {
    return this.material.uniforms.uDebug.value as number;
  }

  /**
   * Per frame (after ocean.setTime): animation time (s) for the swash foam, and underwater
   * detection for the camera (terrain gets in-water attenuation when the camera is submerged).
   */
  update(camera: THREE.Camera, time: number): void {
    const u = this.material.uniforms;
    u.uTime.value = time;
    camera.getWorldPosition(this.camPos);
    u.uUnderwater.value = this.camPos.y < this.ocean.heightAt(this.camPos.x, this.camPos.z) ? 1 : 0;
  }

  dispose(): void {
    this.object3d.geometry.dispose();
    this.material.dispose();
    this.object3d.removeFromParent();
  }
}
