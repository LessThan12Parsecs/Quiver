import { describe, expect, it } from 'vitest';
import { CdlodSelector } from '../src/render/cdlod';
import { mulberry32 } from '../src/ocean/waveModel';

interface Patch {
  x: number;
  z: number;
  cell: number;
  level: number;
  cells: number;
}

function patches(sel: CdlodSelector): Patch[] {
  const out: Patch[] = [];
  for (const kind of [0, 1] as const) {
    const d = sel.data[kind];
    for (let i = 0; i < sel.count[kind]; i++) {
      out.push({
        x: d[i * 5],
        z: d[i * 5 + 1],
        cell: d[i * 5 + 2],
        level: d[i * 5 + 3],
        cells: kind === 0 ? sel.patchResolution : sel.patchResolution / 2,
      });
    }
  }
  return out;
}

/** Morphed rest position of grid vertex (i, j) of a patch: same maths as the vertex shader. */
function morphed(sel: CdlodSelector, p: Patch, i: number, j: number, cam: { x: number; y: number; z: number }): [number, number] {
  const x0 = p.x + i * p.cell;
  const z0 = p.z + j * p.cell;
  const d = Math.hypot(x0 - cam.x, cam.y, z0 - cam.z);
  const k = sel.morphFactor(p.level, d);
  return [x0 - (i % 2) * k * p.cell, z0 - (j % 2) * k * p.cell];
}

describe('CDLOD selection', () => {
  const sel = new CdlodSelector({ nearRatio: 4.8 });

  it('ranges satisfy the crack-free spacing constraint and reach the horizon', () => {
    for (let L = 1; L < sel.levels; L++) {
      expect(sel.ranges[L] - sel.ranges[L - 1]).toBeGreaterThanOrEqual(1.05 * sel.nodeSize(L) - 1e-9);
      // a level L-1 node intersecting its range ends before level L starts morphing
      expect(sel.ranges[L - 1] + Math.SQRT2 * sel.nodeSize(L - 1)).toBeLessThan(sel.morphStart[L]);
    }
    expect(sel.ranges[sel.levels - 1]).toBeGreaterThanOrEqual(6000);
    expect(sel.finestSpacing).toBeLessThanOrEqual(0.25);
  });

  const rng = mulberry32(7);
  const cams = Array.from({ length: 12 }, (_, i) => ({
    x: (rng() - 0.5) * 400,
    y: [0.6, 2, 3.5, 12, 45, 180][i % 6],
    z: (rng() - 0.5) * 400,
  }));

  it('covers the disc around the camera exactly once (no holes, no overlaps)', () => {
    for (const cam of cams) {
      sel.select(cam, null);
      const ps = patches(sel);
      const R = Math.sqrt(sel.ranges[sel.levels - 1] ** 2 - cam.y * cam.y) * 0.98;
      for (let n = 0; n < 4000; n++) {
        // denser sampling near the camera
        const r = R * Math.pow(rng(), 3);
        const a = rng() * Math.PI * 2;
        const px = cam.x + r * Math.cos(a);
        const pz = cam.z + r * Math.sin(a);
        let hits = 0;
        for (const p of ps) {
          const s = p.cell * p.cells;
          if (px >= p.x && px < p.x + s && pz >= p.z && pz < p.z + s) hits++;
        }
        expect(hits).toBe(1);
      }
    }
  });

  it('neighbouring patches share identical (morphed) edge vertices', () => {
    for (const cam of cams) {
      sel.select(cam, null);
      const ps = patches(sel);
      // Index edge vertices by their exact morphed position per edge line.
      const key = (axis: number, c: number): string => `${axis}:${c}`;
      const edges = new Map<string, Array<{ p: Patch; lo: number; hi: number; pts: number[] }>>();
      for (const p of ps) {
        const n = p.cells;
        const s = p.cell * n;
        const lines: Array<[number, number, (t: number) => [number, number]]> = [
          [0, p.x, (t) => [0, t]],
          [0, p.x + s, (t) => [n, t]],
          [1, p.z, (t) => [t, 0]],
          [1, p.z + s, (t) => [t, n]],
        ];
        for (const [axis, c, ij] of lines) {
          const pts: number[] = [];
          for (let t = 0; t <= n; t++) {
            const [i, j] = ij(t);
            const m = morphed(sel, p, i, j, cam);
            // the morphed point must stay on the edge line
            expect(Math.abs(m[axis] - c)).toBeLessThan(1e-9);
            pts.push(m[1 - axis]);
          }
          const lo = axis === 0 ? p.z : p.x;
          const k = key(axis, c);
          if (!edges.has(k)) edges.set(k, []);
          edges.get(k)!.push({ p, lo, hi: lo + s, pts });
        }
      }
      let shared = 0;
      for (const list of edges.values()) {
        for (let a = 0; a < list.length; a++) {
          for (let b = a + 1; b < list.length; b++) {
            const A = list[a];
            const B = list[b];
            const lo = Math.max(A.lo, B.lo);
            const hi = Math.min(A.hi, B.hi);
            if (hi - lo <= 1e-9) continue;
            shared++;
            const inA = new Set(A.pts.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9).map((v) => v.toFixed(6)));
            const inB = new Set(B.pts.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9).map((v) => v.toFixed(6)));
            expect([...inA].sort()).toEqual([...inB].sort());
            expect(Math.abs(A.p.level - B.p.level)).toBeLessThanOrEqual(1);
          }
        }
      }
      expect(shared).toBeGreaterThan(50);
    }
  });

  it('keeps the vertex count in budget for a typical view', () => {
    // 60 deg vertical / 16:9 view from 2 m: emulate frustum culling with a horizontal wedge.
    const cam = { x: 0, y: 2, z: 0 };
    const half = (92 / 2) * (Math.PI / 180);
    sel.select(cam, (x, z, s) => {
      // node visible if any corner or the centre lies within the view wedge looking along +X
      const pts = [[x, z], [x + s, z], [x, z + s], [x + s, z + s], [x + s / 2, z + s / 2]];
      if (x <= cam.x && cam.x <= x + s && z <= cam.z && cam.z <= z + s) return true;
      return pts.some(([px, pz]) => Math.abs(Math.atan2(pz - cam.z, px - cam.x)) < half + 0.15);
    });
    const P = sel.patchResolution;
    const verts = sel.count[0] * (P + 1) ** 2 + sel.count[1] * (P / 2 + 1) ** 2;
    expect(verts).toBeGreaterThan(80_000);
    expect(verts).toBeLessThan(260_000);
  });
});
