/**
 * Camera-centred CDLOD ocean surface.
 *
 * Geometry: one instanced square grid patch of P x P cells (plus a P/2 x P/2 "quarter" patch for
 * areas CDLOD renders at the parent's resolution). Every frame `update(camera)` walks a world-aligned
 * quadtree (cdlod.ts; leaf = P * finestSpacing metres, top level reaching `maxDistance`), selects
 * nodes by 3D distance to the camera and the view frustum, and uploads them as instances (origin,
 * grid spacing, level), sorted front to back.
 *
 * Crack-free / swim-free: patches sit on a fixed world grid (no swimming); near the end of each
 * level's range the vertex shader morphs odd vertices onto their even neighbours (CDLOD), so at
 * a level boundary the finer patch exactly matches the coarser one (verified in tests/cdlod.test.ts).
 *
 * Shading: see shaders/ocean.ts. The material's uniforms include the shared objects of
 * OceanShaderData.uniforms and Environment.uniforms (spread, so their per-frame updates apply).
 */
import * as THREE from 'three';
import { DEFAULT_OCEAN_CONFIG } from '../ocean/oceanConfig';
import { mulberry32 } from '../ocean/waveModel';
import type { OceanShaderData } from '../ocean/waveGLSL';
import type { Environment } from './Environment';
import { CdlodSelector } from './cdlod';
import { MAX_LODS, OCEAN_FRAGMENT, OCEAN_VERTEX, RIPPLE_COUNT } from './shaders/ocean';

export type OceanQuality = 'low' | 'medium' | 'high' | 'ultra';

/** Range-to-node-size ratio of the near levels per quality preset (vertex density knob). */
export const OCEAN_QUALITY_RATIO: Record<OceanQuality, number> = {
  low: 2.6,
  medium: 3.6,
  high: 4.8,
  ultra: 6.4,
};

export interface OceanMeshOptions {
  /** Preset name or the near-level range/size ratio directly (>= 2.2; higher = denser). */
  quality?: OceanQuality | number;
  /** Finest grid spacing in metres (<= 0.25 resolves steep faces). */
  finestSpacing?: number;
  /** Cells per patch side (power of two, >= 8). */
  patchResolution?: number;
  /** Number of fine levels that use the quality ratio (the rest use the minimum safe ratio). */
  nearLevels?: number;
  /** Surface is rendered out to this distance (m). Keep camera.far a bit larger. */
  maxDistance?: number;
  /** Vertical/horizontal culling margin for displaced waves (m). */
  maxWaveHeight?: number;
  /** Wind direction (deg from +X toward +Z) and speed (m/s) for capillary ripples and gusts. */
  windDirectionDeg?: number;
  windSpeed?: number;
  /** Total mean-square slope of the capillary ripples (Cox-Munk share below 1.2 m). */
  rippleSlopeVariance?: number;
}

export interface OceanMeshStats {
  /** Patches drawn this frame (full + quarter). */
  patches: number;
  /** Vertices processed this frame (sum over drawn patches). */
  vertices: number;
  triangles: number;
  levels: number;
}

/**
 * Water optics / look parameters, bound to the material's uniforms (edit live, e.g. from a GUI).
 * Optics follow Lee et al. (1999): absorption a and backscattering bb per RGB channel (1/m) for
 * clear offshore water, plus extras in the surf zone scaled by `turbidity`.
 */
export class WaterLook {
  constructor(private readonly u: Record<string, THREE.IUniform>) {}
  get absorption(): THREE.Vector3 { return this.u.uAbsorption.value as THREE.Vector3; }
  get backscatter(): THREE.Vector3 { return this.u.uBackscatter.value as THREE.Vector3; }
  get surfAbsorption(): THREE.Vector3 { return this.u.uSurfAbsorption.value as THREE.Vector3; }
  get surfBackscatter(): THREE.Vector3 { return this.u.uSurfBackscatter.value as THREE.Vector3; }
  get bubbleBackscatter(): THREE.Vector3 { return this.u.uBubbleBackscatter.value as THREE.Vector3; }
  get turbidity(): number { return this.u.uTurbidity.value as number; }
  set turbidity(v: number) { this.u.uTurbidity.value = v; }
  get bottomAlbedo(): number { return this.u.uBottomAlbedo.value as number; }
  set bottomAlbedo(v: number) { this.u.uBottomAlbedo.value = v; }
  get foamIntensity(): number { return this.u.uFoamIntensity.value as number; }
  set foamIntensity(v: number) { this.u.uFoamIntensity.value = v; }
  get sssIntensity(): number { return this.u.uSssIntensity.value as number; }
  set sssIntensity(v: number) { this.u.uSssIntensity.value = v; }
  get causticsIntensity(): number { return this.u.uCausticsIntensity.value as number; }
  set causticsIntensity(v: number) { this.u.uCausticsIntensity.value = v; }
  /** Base GGX alpha of the surface (before footprint-filtered slope variance). */
  get roughness(): number { return this.u.uRoughness.value as number; }
  set roughness(v: number) { this.u.uRoughness.value = v; }
}

const MAX_INSTANCES = 8192;

function makePatchGeometry(cells: number): THREE.InstancedBufferGeometry {
  const n = cells + 1;
  const pos = new Float32Array(n * n * 3);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = (j * n + i) * 3;
      pos[k] = i;
      pos[k + 1] = 0;
      pos[k + 2] = j;
    }
  }
  const idx = new Uint32Array(cells * cells * 6);
  let o = 0;
  for (let j = 0; j < cells; j++) {
    for (let i = 0; i < cells; i++) {
      const a = j * n + i;
      const b = a + 1;
      const c = a + n;
      const d = c + 1;
      // counter-clockwise seen from +Y
      idx[o++] = a;
      idx[o++] = c;
      idx[o++] = b;
      idx[o++] = b;
      idx[o++] = c;
      idx[o++] = d;
    }
  }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  const inst = new THREE.InstancedBufferAttribute(new Float32Array(MAX_INSTANCES * 4), 4);
  inst.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute('aNode', inst);
  g.instanceCount = 0;
  return g;
}

export class OceanMesh {
  /** Add this to the scene. */
  readonly object3d = new THREE.Group();
  readonly material: THREE.ShaderMaterial;
  /** Live water look parameters (uniform-backed). */
  readonly look: WaterLook;
  readonly stats: OceanMeshStats = { patches: 0, vertices: 0, triangles: 0, levels: 0 };
  /** The LOD selector (ranges, levels, last selection). */
  readonly lod: CdlodSelector;

  private readonly P: number;
  private readonly margin: number;

  private readonly fullGeo: THREE.InstancedBufferGeometry;
  private readonly quarterGeo: THREE.InstancedBufferGeometry;
  private readonly fullMesh: THREE.Mesh;
  private readonly quarterMesh: THREE.Mesh;

  // selection scratch (no allocation per frame)
  private readonly frustum = new THREE.Frustum();
  private readonly projView = new THREE.Matrix4();
  private readonly box = new THREE.Box3();
  private readonly camPos = new THREE.Vector3();
  private readonly order = [new Uint32Array(MAX_INSTANCES), new Uint32Array(MAX_INSTANCES)];
  private sortStage: Float32Array;
  private readonly sortFn = (a: number, b: number): number => this.sortStage[a * 5 + 4] - this.sortStage[b * 5 + 4];
  private readonly visibleFn = (x: number, z: number, s: number): boolean => {
    const m = this.margin;
    this.box.min.set(x - m, -m, z - m);
    this.box.max.set(x + s + m, m, z + s + m);
    return this.frustum.intersectsBox(this.box);
  };

  // ripples
  private readonly ripK = new Float64Array(RIPPLE_COUNT);
  private readonly ripOmega = new Float64Array(RIPPLE_COUNT);
  private readonly ripPhase0 = new Float64Array(RIPPLE_COUNT);
  private readonly ripDir = new Float64Array(RIPPLE_COUNT * 2);
  private readonly windDir = new THREE.Vector2();

  constructor(shaderData: OceanShaderData, env: Environment, opts: OceanMeshOptions = {}) {
    this.lod = new CdlodSelector(
      {
        patchResolution: opts.patchResolution ?? 32,
        finestSpacing: opts.finestSpacing ?? 0.25,
        nearLevels: opts.nearLevels ?? 3,
        maxDistance: opts.maxDistance ?? 60000,
        maxLevels: MAX_LODS,
      },
      MAX_INSTANCES,
    );
    this.P = this.lod.patchResolution;
    this.sortStage = this.lod.data[0];
    this.margin = opts.maxWaveHeight ?? 4;
    this.object3d.name = 'ocean';

    const windDeg = opts.windDirectionDeg ?? DEFAULT_OCEAN_CONFIG.wind.directionDeg;
    const windSpeed = opts.windSpeed ?? DEFAULT_OCEAN_CONFIG.wind.speed;
    this.windDir.set(Math.cos(THREE.MathUtils.degToRad(windDeg)), Math.sin(THREE.MathUtils.degToRad(windDeg)));
    this.buildRipples(opts.rippleSlopeVariance ?? 0.012 * Math.min(Math.max(windSpeed / 4, 0.3), 2.5));

    const lodMorph = Array.from({ length: MAX_LODS }, () => new THREE.Vector2(1e9, 1));
    const ripA = Array.from({ length: RIPPLE_COUNT }, () => new THREE.Vector4());
    const ripAmp = Array.from({ length: RIPPLE_COUNT / 4 }, () => new THREE.Vector4());
    for (let i = 0; i < RIPPLE_COUNT; i++) {
      ripA[i].set(this.ripDir[i * 2], this.ripDir[i * 2 + 1], this.ripK[i], 0);
      ripAmp[i >> 2].setComponent(i & 3, this.ripAmp[i]);
    }

    this.material = new THREE.ShaderMaterial({
      name: 'OceanSurface',
      uniforms: {
        ...shaderData.uniforms,
        ...env.uniforms,
        uLodOrigin: { value: new THREE.Vector3() },
        uLodMorph: { value: lodMorph },
        uTime: { value: 0 },
        uRipA: { value: ripA },
        uRipAmp: { value: ripAmp },
        uRipOrigin: { value: new THREE.Vector2() },
        uWindDrift: { value: this.windDir.clone().multiplyScalar(windSpeed * 0.5) },
        uAbsorption: { value: new THREE.Vector3(0.32, 0.062, 0.03) },
        uBackscatter: { value: new THREE.Vector3(0.0012, 0.0018, 0.0034) },
        uSurfAbsorption: { value: new THREE.Vector3(0.01, 0.02, 0.06) },
        uSurfBackscatter: { value: new THREE.Vector3(0.01, 0.012, 0.013) },
        uTurbidity: { value: 1 },
        uBubbleBackscatter: { value: new THREE.Vector3(0.08, 0.085, 0.09) },
        uBottomAlbedo: { value: 0.62 },
        uFoamIntensity: { value: 1 },
        uSssIntensity: { value: 1 },
        uCausticsIntensity: { value: 1 },
        uRoughness: { value: 0.03 },
        uDebug: { value: 0 },
      },
      defines: { ...env.envDefines },
      vertexShader: OCEAN_VERTEX,
      fragmentShader: OCEAN_FRAGMENT,
      // back faces = the surface seen from below (underwater camera)
      side: THREE.DoubleSide,
    });
    this.look = new WaterLook(this.material.uniforms);

    this.fullGeo = makePatchGeometry(this.P);
    this.quarterGeo = makePatchGeometry(this.P / 2);
    this.fullMesh = new THREE.Mesh(this.fullGeo, this.material);
    this.quarterMesh = new THREE.Mesh(this.quarterGeo, this.material);
    for (const m of [this.fullMesh, this.quarterMesh]) {
      m.frustumCulled = false;
      m.matrixAutoUpdate = false;
      this.object3d.add(m);
    }
    this.setQuality(opts.quality ?? 'high');
  }

  /** Change the vertex density (preset or near range/size ratio). */
  setQuality(q: OceanQuality | number): void {
    const lod = this.lod;
    lod.setNearRatio(typeof q === 'number' ? q : OCEAN_QUALITY_RATIO[q]);
    const morph = this.material.uniforms.uLodMorph.value as THREE.Vector2[];
    for (let i = 0; i < MAX_LODS; i++) {
      const start = i < lod.levels ? lod.morphStart[i] : Infinity;
      if (Number.isFinite(start)) morph[i].set(start, 1 / (lod.ranges[i] - start));
      else morph[i].set(1e9, 1);
    }
    this.stats.levels = lod.levels;
  }

  /** LOD ranges (3D distance from the camera, m) per level. */
  get ranges(): readonly number[] {
    return this.lod.ranges;
  }

  /** 0 = off, 1 = LOD levels, 2 = normals, 3 = foam masks, 4 = breaking/fullness/ratio, 5 = body colour, 6 = black (crack test). */
  set debugView(mode: number) {
    this.material.uniforms.uDebug.value = mode;
  }

  get debugView(): number {
    return this.material.uniforms.uDebug.value as number;
  }

  set wireframe(v: boolean) {
    this.material.wireframe = v;
  }

  /**
   * Per frame, before rendering with `camera`: select LOD patches and set animation time.
   * `time` is the ocean (simulation) time in seconds used for ripples/foam animation; the wave
   * geometry itself follows OceanShaderData (model.setTime + shaderData.update).
   */
  update(camera: THREE.Camera, time: number): void {
    camera.updateMatrixWorld();
    camera.getWorldPosition(this.camPos);
    const u = this.material.uniforms;
    const cam = this.camPos;
    (u.uLodOrigin.value as THREE.Vector3).set(cam.x, Math.abs(cam.y), cam.z);
    u.uTime.value = time;
    this.updateRipples(cam, time);

    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView, camera.coordinateSystem, camera.reversedDepth);
    this.lod.select(cam, this.visibleFn);
    this.upload(0, this.fullGeo);
    this.upload(1, this.quarterGeo);
    const [nf, nq] = this.lod.count;
    const vf = (this.P + 1) * (this.P + 1);
    const vq = (this.P / 2 + 1) * (this.P / 2 + 1);
    this.stats.patches = nf + nq;
    this.stats.vertices = nf * vf + nq * vq;
    this.stats.triangles = nf * this.P * this.P * 2 + (nq * this.P * this.P) / 2;
  }

  dispose(): void {
    this.fullGeo.dispose();
    this.quarterGeo.dispose();
    this.material.dispose();
    this.object3d.removeFromParent();
  }

  // ---------------------------------------------------------------------------------------

  private readonly ripAmp = new Float64Array(RIPPLE_COUNT);

  private buildRipples(mss: number): void {
    const rng = mulberry32(90210);
    const lMin = 0.06;
    const lMax = 1.15;
    const g = 9.81;
    const sigma = 0.074 / 1025;
    const base = Math.atan2(this.windDir.y, this.windDir.x);
    const ak = Math.sqrt((2 * mss) / RIPPLE_COUNT);
    for (let i = 0; i < RIPPLE_COUNT; i++) {
      const f = (i + 0.5 + (rng() - 0.5) * 0.9) / RIPPLE_COUNT;
      const lambda = lMin * Math.pow(lMax / lMin, f);
      const k = (2 * Math.PI) / lambda;
      // wide spread for short waves, some against the wind
      const spread = (rng() - 0.5) * 2 * (rng() < 0.15 ? Math.PI : 1.25);
      const th = base + spread;
      this.ripK[i] = k;
      this.ripOmega[i] = Math.sqrt(g * k + sigma * k * k * k);
      this.ripPhase0[i] = rng() * Math.PI * 2;
      this.ripDir[i * 2] = Math.cos(th);
      this.ripDir[i * 2 + 1] = Math.sin(th);
      this.ripAmp[i] = ak * (0.7 + 0.6 * rng());
    }
  }

  private updateRipples(cam: THREE.Vector3, time: number): void {
    const u = this.material.uniforms;
    const ox = Math.round(cam.x / 64) * 64;
    const oz = Math.round(cam.z / 64) * 64;
    (u.uRipOrigin.value as THREE.Vector2).set(ox, oz);
    const A = u.uRipA.value as THREE.Vector4[];
    const TWO_PI = Math.PI * 2;
    for (let i = 0; i < RIPPLE_COUNT; i++) {
      const k = this.ripK[i];
      let ph = this.ripPhase0[i] - this.ripOmega[i] * time + k * (this.ripDir[i * 2] * ox + this.ripDir[i * 2 + 1] * oz);
      ph -= TWO_PI * Math.floor(ph / TWO_PI);
      A[i].w = ph;
    }
  }

  /** Upload the selected patches of one kind as instances, sorted front to back (early-z). */
  private upload(kind: 0 | 1, geo: THREE.InstancedBufferGeometry): void {
    const n = this.lod.count[kind];
    const s = this.lod.data[kind];
    const ord = this.order[kind];
    for (let i = 0; i < n; i++) ord[i] = i;
    this.sortStage = s;
    ord.subarray(0, n).sort(this.sortFn);
    const attr = geo.getAttribute('aNode') as THREE.InstancedBufferAttribute;
    const dst = attr.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const k = ord[i] * 5;
      dst[i * 4] = s[k];
      dst[i * 4 + 1] = s[k + 1];
      dst[i * 4 + 2] = s[k + 2];
      dst[i * 4 + 3] = s[k + 3];
    }
    attr.clearUpdateRanges();
    attr.addUpdateRange(0, Math.max(n, 1) * 4);
    attr.needsUpdate = true;
    geo.instanceCount = n;
  }
}
