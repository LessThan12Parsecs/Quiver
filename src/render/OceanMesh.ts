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
 * Short-wave detail normals come from two animated spectral ripple tiles (DetailNormals.ts),
 * regenerated in `update` (offscreen pass on the environment's renderer) and sampled at three
 * scales: 16 m (tile A), 4.2 m and 1.1 m (tile B), each rotated relative to the wind.
 *
 * Quality: `quality` sets the vertex density (CDLOD ranges); `shading` sets the fragment cost
 * (detail layers, foam relief, face streaks, tile size / anisotropy). By default a quality preset
 * also picks the matching shading level (low -> low, medium -> medium, high/ultra -> high).
 */
import * as THREE from 'three';
import { DEFAULT_OCEAN_CONFIG } from '../ocean/oceanConfig';
import type { OceanShaderData } from '../ocean/waveGLSL';
import { RippleTile } from './DetailNormals';
import type { Environment } from './Environment';
import { CdlodSelector } from './cdlod';
import { MAX_LODS, OCEAN_FRAGMENT, OCEAN_VERTEX } from './shaders/ocean';

export type OceanQuality = 'low' | 'medium' | 'high' | 'ultra';

/** Fragment shading level of the water. */
export type OceanShading = 'low' | 'medium' | 'high';

export interface OceanShadingPreset {
  /** Detail-normal layers sampled (1-3); missing layers only add their slope variance to roughness. */
  layers: 1 | 2 | 3;
  /** Whitewater micro-relief (lumps, bubble domes). */
  foamDetail: boolean;
  /** Streaks on steep wave faces. */
  faceDetail: boolean;
  /** Ripple tile B (4.2 m; layers 1, 2, face streaks) resolution (texels) and wave components. */
  tileSize: number;
  components: number;
  /** Tile A (16 m, layer 0; only with layers > 1): resolution and components. Its finest waves
   * (16 m / (size / 4)) only need to reach into tile B's range. */
  tileSizeA: number;
  componentsA: number;
  /** Anisotropic filtering of the tiles. */
  anisotropy: number;
}

export const OCEAN_SHADING: Record<OceanShading, OceanShadingPreset> = {
  low: { layers: 1, foamDetail: false, faceDetail: false, tileSize: 128, components: 64, tileSizeA: 0, componentsA: 0, anisotropy: 2 },
  medium: { layers: 2, foamDetail: false, faceDetail: true, tileSize: 256, components: 96, tileSizeA: 128, componentsA: 64, anisotropy: 2 },
  high: { layers: 3, foamDetail: true, faceDetail: true, tileSize: 256, components: 128, tileSizeA: 128, componentsA: 64, anisotropy: 4 },
};

const SHADING_FOR_QUALITY: Record<OceanQuality, OceanShading> = { low: 'low', medium: 'medium', high: 'high', ultra: 'high' };

/**
 * Detail-normal layers: which tile, world size (m), rotation from the wind direction (rad) and
 * share of the total short-wave slope variance. Tile A spans 16 m (lambda 5.3 m - 50 cm at 128
 * texels), tile B 4.2 m (1.4 m - 6.6 cm at 256 texels) and is reused at 1.1 m (37 cm - 1.7 cm)
 * near the camera.
 */
const DETAIL_LAYERS = [
  { tile: 0, size: 16, rotation: 0, share: 0.3 },
  { tile: 1, size: 4.2, rotation: 0.41, share: 0.37 },
  { tile: 1, size: 1.1, rotation: -0.67, share: 0.33 },
] as const;
const TILE_WORLD_SIZE = [16, 4.2];

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
  /** Fragment shading level (default: follows the quality preset; 'high' for numeric quality). */
  shading?: OceanShading;
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
  /** Wind direction (deg from +X toward +Z) and speed (m/s) for detail ripples and gusts. */
  windDirectionDeg?: number;
  windSpeed?: number;
  /**
   * Total mean-square slope of the short-wave detail normals (Cox-Munk share not carried by the
   * geometric wind chop). Default scales with the wind speed (from uWindDrift, so live wind changes
   * apply): 0.016 at 4 m/s.
   */
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
  /** Slope rms of the streaks running down steep wave faces. */
  get faceStreaks(): number { return this.u.uFaceStreaks.value as number; }
  set faceStreaks(v: number) { this.u.uFaceStreaks.value = v; }
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

  // detail normals
  private readonly renderer: THREE.WebGLRenderer;
  private tiles: RippleTile[] = [];
  private shadingLevel: OceanShading = 'high';
  private readonly rippleMss: number | undefined;
  private windAngle: number;
  private readonly bufSize = new THREE.Vector2();

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
    this.windAngle = THREE.MathUtils.degToRad(windDeg);
    const windDir = new THREE.Vector2(Math.cos(this.windAngle), Math.sin(this.windAngle));
    this.rippleMss = opts.rippleSlopeVariance;
    this.renderer = env.renderer;

    const lodMorph = Array.from({ length: MAX_LODS }, () => new THREE.Vector2(1e9, 1));

    this.material = new THREE.ShaderMaterial({
      name: 'OceanSurface',
      uniforms: {
        ...shaderData.uniforms,
        ...env.uniforms,
        uLodOrigin: { value: new THREE.Vector3() },
        uLodMorph: { value: lodMorph },
        uTime: { value: 0 },
        uWindDrift: { value: windDir.clone().multiplyScalar(windSpeed * 0.5) },
        uPixelAngle: { value: 0.0015 },
        uDetail0: { value: null },
        uDetail1: { value: null },
        uDetailDecS: { value: new THREE.Vector4(1, 1, 1, 1) },
        uDetailDecB: { value: new THREE.Vector4(0, 0, 0, 0) },
        uDetailOrigin: { value: new THREE.Vector2() },
        uDetailL0: { value: new THREE.Vector4(1, 0, 1 / 16, 0) },
        uDetailL1: { value: new THREE.Vector4(1, 0, 1 / 4.2, 0) },
        uDetailL2: { value: new THREE.Vector4(1, 0, 1 / 1.1, 0) },
        uDetailOff01: { value: new THREE.Vector4() },
        uDetailOff2: { value: new THREE.Vector2() },
        uFaceStreaks: { value: 0.035 },
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
    const q = opts.quality ?? 'high';
    this.setShading(opts.shading ?? (typeof q === 'number' ? 'high' : SHADING_FOR_QUALITY[q]));

    this.fullGeo = makePatchGeometry(this.P);
    this.quarterGeo = makePatchGeometry(this.P / 2);
    this.fullMesh = new THREE.Mesh(this.fullGeo, this.material);
    this.quarterMesh = new THREE.Mesh(this.quarterGeo, this.material);
    for (const m of [this.fullMesh, this.quarterMesh]) {
      m.frustumCulled = false;
      m.matrixAutoUpdate = false;
      this.object3d.add(m);
    }
    this.setQuality(q, false);
  }

  /**
   * Change the vertex density (preset or near range/size ratio). With `withShading` (default) a
   * preset name also selects the matching shading level.
   */
  setQuality(q: OceanQuality | number, withShading = true): void {
    if (withShading && typeof q === 'string' && SHADING_FOR_QUALITY[q]) this.setShading(SHADING_FOR_QUALITY[q]);
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

  /** Current fragment shading level. */
  get shading(): OceanShading {
    return this.shadingLevel;
  }

  /** Change the fragment shading level (recompiles the material, regenerates the ripple tiles). */
  setShading(level: OceanShading): void {
    const p = OCEAN_SHADING[level];
    if (!p) return;
    const first = this.tiles.length === 0;
    if (!first && level === this.shadingLevel) return;
    this.shadingLevel = level;
    for (const t of this.tiles) t.dispose();
    // tile B (4.2 m: layers 1, 2 and the face streaks) always; tile A (16 m) from 2 layers up
    const af = p.anisotropy;
    const tileB = new RippleTile(this.renderer, { size: p.tileSize, components: p.components, worldSize: TILE_WORLD_SIZE[1], anisotropy: af, seed: 23 });
    const tileA =
      p.layers > 1
        ? new RippleTile(this.renderer, { size: p.tileSizeA, components: p.componentsA, worldSize: TILE_WORLD_SIZE[0], anisotropy: af, seed: 11 })
        : tileB;
    this.tiles = tileA === tileB ? [tileB] : [tileA, tileB];
    const u = this.material.uniforms;
    u.uDetail0.value = tileA.texture;
    u.uDetail1.value = tileB.texture;
    (u.uDetailDecS.value as THREE.Vector4).copy(tileB.decodeScale);
    (u.uDetailDecB.value as THREE.Vector4).copy(tileB.decodeBias);
    const d = this.material.defines;
    d.DETAIL_LAYERS = p.layers;
    d.FOAM_DETAIL = p.foamDetail ? 1 : 0;
    d.FACE_DETAIL = p.faceDetail ? 1 : 0;
    this.material.needsUpdate = true;
  }

  /** LOD ranges (3D distance from the camera, m) per level. */
  get ranges(): readonly number[] {
    return this.lod.ranges;
  }

  /**
   * 0 = off, 1 = LOD levels, 2 = normals, 3 = foam masks, 4 = breaking/fullness/ratio, 5 = body
   * colour, 6 = black (crack test), 7 = roughness (r: total alpha, g: detail, b: chop; x4),
   * 8 = pixel footprint (log2 colour bands).
   */
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
    this.renderer.getDrawingBufferSize(this.bufSize);
    const p5 = camera.projectionMatrix.elements[5];
    u.uPixelAngle.value = 2 / (Math.max(Math.abs(p5), 1e-6) * Math.max(this.bufSize.y, 1));
    this.updateDetail(cam, time);

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
    for (const t of this.tiles) t.dispose();
    this.tiles = [];
    this.fullGeo.dispose();
    this.quarterGeo.dispose();
    this.material.dispose();
    this.object3d.removeFromParent();
  }

  // ---------------------------------------------------------------------------------------

  /** Per frame: detail layer orientation/scale/strength from the live wind, tiles at `time`. */
  private updateDetail(cam: THREE.Vector3, time: number): void {
    const u = this.material.uniforms;
    const drift = u.uWindDrift.value as THREE.Vector2;
    const dl = drift.length();
    if (dl > 1e-4) this.windAngle = Math.atan2(drift.y, drift.x);
    const windSpeed = dl / 0.5;
    const mss = this.rippleMss ?? 0.016 * Math.min(Math.max(windSpeed / 4, 0.3), 2.5);
    // camera-snapped origin keeps the detail coordinates small (float precision far from 0)
    const ox = Math.round(cam.x / 64) * 64;
    const oz = Math.round(cam.z / 64) * 64;
    (u.uDetailOrigin.value as THREE.Vector2).set(ox, oz);
    const vecs = [u.uDetailL0.value, u.uDetailL1.value, u.uDetailL2.value] as THREE.Vector4[];
    const off01 = u.uDetailOff01.value as THREE.Vector4;
    const off2 = u.uDetailOff2.value as THREE.Vector2;
    const fract = (v: number): number => v - Math.floor(v);
    DETAIL_LAYERS.forEach((layer, i) => {
      const th = this.windAngle + layer.rotation;
      const ax = Math.cos(th);
      const az = Math.sin(th);
      const inv = 1 / layer.size;
      vecs[i].set(ax, az, inv, Math.sqrt(mss * layer.share));
      // uv phase of the origin (exact in double precision)
      const ou = fract((ox * ax + oz * az) * inv);
      const ov = fract((-ox * az + oz * ax) * inv);
      if (i === 0) off01.set(ou, ov, off01.z, off01.w);
      else if (i === 1) off01.set(off01.x, off01.y, ou, ov);
      else off2.set(ou, ov);
    });
    for (const t of this.tiles) t.update(this.renderer, time);
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
