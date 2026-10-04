/**
 * CDLOD quadtree selection for the ocean surface (Strugar 2009), independent of WebGL so it can
 * be unit tested.
 *
 * The world is tiled by a quadtree whose leaves are `leafSize = P * finestSpacing` metres; level L
 * nodes are leafSize * 2^L. A node of level L is drawn with a P x P grid (spacing size / P) if it
 * intersects the sphere of radius ranges[L] around the camera but not ranges[L-1]; otherwise its
 * children are visited, and children out of the finer range are drawn as "quarter" patches
 * (P/2 x P/2 cells) at the parent's spacing. Distances are 3D from the camera to the node's square
 * at y = 0 (the shader uses the same metric for morphing, see `morphFactor`).
 *
 * Crack-free guarantee: vertices of level L morph toward the level L+1 grid between
 * morphStart[L] and ranges[L] (odd grid coordinates collapse onto the lower even neighbour). With
 * ranges[L] - ranges[L-1] >= 1.05 * size[L] (enforced), every boundary between a level L and an
 * L+1 area is at distance >= ranges[L] (fine side fully morphed) and < morphStart[L+1] (coarse side
 * not yet morphing), so shared edges coincide exactly.
 */

/** Fraction of a level's range [ranges[L-1], ranges[L]] after which morphing starts. */
export const CDLOD_MORPH_START = 0.7;

export interface CdlodOptions {
  /** Cells per patch side (power of two, >= 8). */
  patchResolution?: number;
  /** Grid spacing of level 0, metres. */
  finestSpacing?: number;
  /** Levels 0..nearLevels use `nearRatio` (range / node size); farther levels the minimum safe ratio. */
  nearLevels?: number;
  /** Range / node-size ratio of the near levels (>= 2.2): the vertex density knob. */
  nearRatio?: number;
  /** The top level's range reaches at least this distance (m). */
  maxDistance?: number;
  /** Maximum number of levels. */
  maxLevels?: number;
}

/** Minimal frustum interface (THREE.Frustum + a box test function). */
export type NodeVisibility = (x: number, z: number, size: number) => boolean;

export interface CameraLike {
  x: number;
  y: number;
  z: number;
}

/** Patch kinds in the selection output. */
export const PatchKind = { Full: 0, Quarter: 1 } as const;

export class CdlodSelector {
  readonly patchResolution: number;
  readonly finestSpacing: number;
  readonly leafSize: number;
  readonly nearLevels: number;
  readonly maxDistance: number;
  readonly maxLevels: number;
  nearRatio = 4.8;
  /** Range (3D distance from the camera) of every level. */
  ranges: number[] = [];
  /** Distance where level L starts morphing toward L+1 (Infinity for the top level). */
  morphStart: number[] = [];
  levels = 0;

  /**
   * Selected patches per kind: stride 5 = [originX, originZ, gridSpacing, level, sortKey]
   * (sortKey = squared horizontal distance of the patch centre). Valid entries: count[kind].
   */
  readonly data: [Float32Array, Float32Array];
  readonly count: [number, number] = [0, 0];
  readonly capacity: number;

  private cx = 0;
  private cy = 0;
  private cz = 0;
  private visible: NodeVisibility | null = null;

  constructor(opts: CdlodOptions = {}, capacity = 8192) {
    this.patchResolution = opts.patchResolution ?? 32;
    const P = this.patchResolution;
    if (P < 8 || (P & (P - 1)) !== 0) throw new Error('patchResolution must be a power of two >= 8');
    this.finestSpacing = opts.finestSpacing ?? 0.25;
    this.leafSize = P * this.finestSpacing;
    this.nearLevels = opts.nearLevels ?? 3;
    this.maxDistance = opts.maxDistance ?? 60000;
    this.maxLevels = opts.maxLevels ?? 16;
    this.capacity = capacity;
    this.data = [new Float32Array(capacity * 5), new Float32Array(capacity * 5)];
    this.setNearRatio(opts.nearRatio ?? 4.8);
  }

  /** Node size of a level, metres. */
  nodeSize(level: number): number {
    return this.leafSize * Math.pow(2, level);
  }

  /** Recompute ranges for a new near range/size ratio. */
  setNearRatio(ratio: number): void {
    this.nearRatio = Math.max(ratio, 2.2);
    const r: number[] = [];
    for (let L = 0; L < this.maxLevels; L++) {
      const size = this.nodeSize(L);
      const prev = L > 0 ? r[L - 1] : 0;
      const want = L <= this.nearLevels ? this.nearRatio * size : 0;
      r.push(Math.max(want, prev + 1.05 * size));
      if (r[L] >= this.maxDistance) break;
    }
    this.ranges = r;
    this.levels = r.length;
    this.morphStart = r.map((end, L) => {
      if (L === r.length - 1) return Infinity;
      const prev = L > 0 ? r[L - 1] : 0;
      return prev + CDLOD_MORPH_START * (end - prev);
    });
  }

  /** Morph factor of a vertex at 3D distance d in a level-L patch (same formula as the shader). */
  morphFactor(level: number, d: number): number {
    const s = this.morphStart[level];
    if (!Number.isFinite(s)) return 0;
    return Math.min(Math.max((d - s) / (this.ranges[level] - s), 0), 1);
  }

  /**
   * Select patches for a camera at `cam` (y = height above sea level). `visible` culls nodes
   * (e.g. a frustum test of the node's padded box); null = no culling. Results in data/count.
   */
  select(cam: CameraLike, visible: NodeVisibility | null): void {
    this.cx = cam.x;
    this.cy = Math.abs(cam.y);
    this.cz = cam.z;
    this.visible = visible;
    this.count[0] = 0;
    this.count[1] = 0;
    const top = this.levels - 1;
    const size = this.nodeSize(top);
    const R = this.ranges[top];
    const x0 = Math.floor((cam.x - R) / size);
    const x1 = Math.floor((cam.x + R) / size);
    const z0 = Math.floor((cam.z - R) / size);
    const z1 = Math.floor((cam.z + R) / size);
    for (let iz = z0; iz <= z1; iz++) {
      for (let ix = x0; ix <= x1; ix++) this.node(ix * size, iz * size, top);
    }
  }

  /** Squared 3D distance from the camera to the square [x, x+s] x [z, z+s] at y = 0. */
  dist2(x: number, z: number, s: number): number {
    const dx = Math.max(x - this.cx, 0, this.cx - (x + s));
    const dz = Math.max(z - this.cz, 0, this.cz - (z + s));
    return dx * dx + dz * dz + this.cy * this.cy;
  }

  private isVisible(x: number, z: number, s: number): boolean {
    return this.visible === null || this.visible(x, z, s);
  }

  /** Returns false if the node is out of its level's range (the parent then covers it). */
  private node(x: number, z: number, level: number): boolean {
    const size = this.nodeSize(level);
    const r = this.ranges[level];
    const d2 = this.dist2(x, z, size);
    if (d2 >= r * r) return false;
    if (!this.isVisible(x, z, size)) return true;
    const cell = size / this.patchResolution;
    if (level === 0) {
      this.push(0, x, z, cell, 0, size);
      return true;
    }
    const rc = this.ranges[level - 1];
    if (d2 >= rc * rc) {
      this.push(0, x, z, cell, level, size);
      return true;
    }
    const h = size / 2;
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        const qx = x + i * h;
        const qz = z + j * h;
        if (!this.node(qx, qz, level - 1) && this.isVisible(qx, qz, h)) this.push(1, qx, qz, cell, level, h);
      }
    }
    return true;
  }

  private push(kind: 0 | 1, x: number, z: number, cell: number, level: number, size: number): void {
    const n = this.count[kind];
    if (n >= this.capacity) return;
    const s = this.data[kind];
    const dx = x + size / 2 - this.cx;
    const dz = z + size / 2 - this.cz;
    s[n * 5] = x;
    s[n * 5 + 1] = z;
    s[n * 5 + 2] = cell;
    s[n * 5 + 3] = level;
    s[n * 5 + 4] = dx * dx + dz * dz;
    this.count[kind] = n + 1;
  }
}
