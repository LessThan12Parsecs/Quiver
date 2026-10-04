/**
 * Debug overlays drawn on top of everything (no depth test):
 *  - force arrows from `sim.debugForces` (hull cells, fins, legs, body, paddling, leash,
 *    gravity), coloured by kind, length = force × `scale` (m/N);
 *  - water probes: the hull's probe points on the board bottom (cyan = under water, orange =
 *    in the air) and the sampled water surface above/below each one (white).
 * Positions are those of the last physics step (not render-interpolated: at 10 m/s they can
 * trail the board by up to 4 cm). Enabling sets `sim.collectDebug`.
 */
import * as THREE from 'three';
import type { DebugForceKind, SurfSim } from '../physics/SurfSim';

const KIND_COLORS: Record<DebugForceKind, THREE.Color> = {
  hull: new THREE.Color(0x4fc3ff),
  fin: new THREE.Color(0xff5cf0),
  leg: new THREE.Color(0xffa040),
  body: new THREE.Color(0x6dff7a),
  paddle: new THREE.Color(0xfff35c),
  leash: new THREE.Color(0xffffff),
  gravity: new THREE.Color(0xff4040),
};

const MAX_ARROWS = 256;
const MAX_PROBES = 64;

export class DebugDraw {
  readonly object3d = new THREE.Group();
  /** Arrow length per newton, m/N. */
  scale = 1 / 350;
  private readonly lines: THREE.LineSegments;
  private readonly linePos: Float32Array;
  private readonly lineCol: Float32Array;
  private readonly points: THREE.Points;
  private readonly pointPos: Float32Array;
  private readonly pointCol: Float32Array;
  private _forces = false;
  private _probes = false;
  private readonly sim: SurfSim;
  private readonly a = new THREE.Vector3();
  private readonly b = new THREE.Vector3();
  private readonly s1 = new THREE.Vector3();
  private readonly s2 = new THREE.Vector3();

  constructor(sim: SurfSim) {
    this.sim = sim;
    this.object3d.name = 'debug';
    // 3 segments (shaft + 2 head strokes) × 2 vertices per arrow
    const nv = MAX_ARROWS * 6;
    this.linePos = new Float32Array(nv * 3);
    this.lineCol = new Float32Array(nv * 3);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(this.linePos, 3).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('color', new THREE.BufferAttribute(this.lineCol, 3).setUsage(THREE.DynamicDrawUsage));
    lg.setDrawRange(0, 0);
    this.lines = new THREE.LineSegments(
      lg,
      new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, depthWrite: false, transparent: true, toneMapped: false, fog: false }),
    );
    this.lines.frustumCulled = false;
    this.lines.renderOrder = 999;
    this.object3d.add(this.lines);

    this.pointPos = new Float32Array(MAX_PROBES * 2 * 3);
    this.pointCol = new Float32Array(MAX_PROBES * 2 * 3);
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(this.pointPos, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('color', new THREE.BufferAttribute(this.pointCol, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setDrawRange(0, 0);
    this.points = new THREE.Points(
      pg,
      new THREE.PointsMaterial({ size: 6, sizeAttenuation: false, vertexColors: true, depthTest: false, depthWrite: false, transparent: true, toneMapped: false, fog: false }),
    );
    this.points.frustumCulled = false;
    this.points.renderOrder = 999;
    this.object3d.add(this.points);
    this.object3d.visible = false;
  }

  get forces(): boolean {
    return this._forces;
  }

  set forces(v: boolean) {
    this._forces = v;
    this.lines.visible = v;
    this.sync();
  }

  get probes(): boolean {
    return this._probes;
  }

  set probes(v: boolean) {
    this._probes = v;
    this.points.visible = v;
    this.sync();
  }

  /** Refresh from the simulation's last step. */
  update(): void {
    if (this._forces) this.updateArrows();
    if (this._probes) this.updateProbes();
  }

  dispose(): void {
    this.object3d.removeFromParent();
    this.lines.geometry.dispose();
    (this.lines.material as THREE.Material).dispose();
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
  }

  private sync(): void {
    this.object3d.visible = this._forces || this._probes;
    this.sim.collectDebug = this._forces;
  }

  private updateArrows(): void {
    const sim = this.sim;
    const n = Math.min(sim.debugForceCount, MAX_ARROWS);
    const P = this.linePos;
    const C = this.lineCol;
    let v = 0;
    const put = (p: THREE.Vector3, c: THREE.Color): void => {
      P[v * 3] = p.x;
      P[v * 3 + 1] = p.y;
      P[v * 3 + 2] = p.z;
      C[v * 3] = c.r;
      C[v * 3 + 1] = c.g;
      C[v * 3 + 2] = c.b;
      v++;
    };
    for (let i = 0; i < n; i++) {
      const f = sim.debugForces[i];
      const len = f.vector.length() * this.scale;
      if (len < 1e-3) continue;
      const c = KIND_COLORS[f.kind];
      this.a.copy(f.origin);
      this.b.copy(f.origin).addScaledVector(f.vector, this.scale);
      put(this.a, c);
      put(this.b, c);
      // arrow head: two strokes back from the tip in a plane containing the shaft
      const dir = this.s1.copy(f.vector).normalize();
      const side = this.s2.set(0, 1, 0).cross(dir);
      if (side.lengthSq() < 1e-6) side.set(1, 0, 0).cross(dir);
      side.normalize();
      const hl = Math.min(0.25 * len, 0.12);
      for (const sgn of [-1, 1]) {
        put(this.b, c);
        this.a.copy(this.b).addScaledVector(dir, -hl).addScaledVector(side, sgn * 0.5 * hl);
        put(this.a, c);
      }
    }
    const g = this.lines.geometry;
    g.setDrawRange(0, v);
    (g.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (g.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
  }

  private updateProbes(): void {
    const h = this.sim.hull;
    const n = Math.min(h.np, MAX_PROBES);
    const P = this.pointPos;
    const C = this.pointCol;
    for (let i = 0; i < n; i++) {
      const x = h.probeWorld[3 * i];
      const y = h.probeWorld[3 * i + 1];
      const z = h.probeWorld[3 * i + 2];
      const wh = h.probeSamples[i].height;
      const under = y < wh;
      const k = 6 * i;
      P[k] = x;
      P[k + 1] = y;
      P[k + 2] = z;
      C[k] = under ? 0.2 : 1;
      C[k + 1] = under ? 0.9 : 0.55;
      C[k + 2] = under ? 1 : 0.1;
      P[k + 3] = x;
      P[k + 4] = wh;
      P[k + 5] = z;
      C[k + 3] = 1;
      C[k + 4] = 1;
      C[k + 5] = 1;
    }
    const g = this.points.geometry;
    g.setDrawRange(0, 2 * n);
    (g.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (g.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
  }
}
