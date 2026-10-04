/**
 * Board/rider meshes and game input (Node: geometry only, no WebGL).
 *  - the board hull mesh is closed, faces outward, sits exactly in the physics body frame and
 *    encloses the physics volume (divergence theorem) for every preset;
 *  - fins span their spec depth from the physics fin base;
 *  - two-bone IK keeps bone lengths; the rider figure stands on the physics feet;
 *  - Input maps keys per stance and latches the pop-up edge.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { Input } from '../src/game/Input';
import { OceanModel } from '../src/ocean/waveModel';
import { BOARD_PRESETS, getBoardShape } from '../src/physics/boardShape';
import { PHYSICS_DT } from '../src/physics/constants';
import { waveSpawn } from '../src/physics/spawn';
import { SurfSim, createSurfInput } from '../src/physics/SurfSim';
import { buildFinGeometry, buildHullGeometry } from '../src/render/BoardMesh';
import { RiderMesh, solveTwoBone } from '../src/render/RiderMesh';

function meshVolume(g: THREE.BufferGeometry): number {
  const p = g.getAttribute('position');
  const idx = g.getIndex()!;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  let v = 0;
  for (let i = 0; i < idx.count; i += 3) {
    a.fromBufferAttribute(p, idx.getX(i));
    b.fromBufferAttribute(p, idx.getX(i + 1));
    c.fromBufferAttribute(p, idx.getX(i + 2));
    v += a.dot(b.clone().cross(c)) / 6;
  }
  return v;
}

describe('BoardMesh geometry', () => {
  for (const id of Object.keys(BOARD_PRESETS) as Array<keyof typeof BOARD_PRESETS>) {
    it(`${id}: closed, outward, encloses the physics volume`, () => {
      const shape = getBoardShape(BOARD_PRESETS[id]);
      const g = buildHullGeometry(shape);
      const vol = meshVolume(g);
      // positive = outward winding; the rounded rails bulge a few mm outside the physics rail
      expect(vol).toBeGreaterThan(0);
      const rel = vol / shape.volume - 1;
      expect(rel).toBeGreaterThan(-0.01);
      expect(rel).toBeLessThan(0.04);
      // bounding box = physics dimensions in the local (COM) frame
      g.computeBoundingBox();
      const bb = g.boundingBox!;
      // tail block / nose tip at the physics stations (the rail bulge adds a few mm at the tail)
      expect(bb.min.x).toBeGreaterThan(shape.x(0) - 0.006);
      expect(bb.min.x).toBeLessThan(shape.x(0) + 1e-6);
      expect(bb.max.x).toBeCloseTo(shape.x(1), 6);
      expect(bb.max.z - bb.min.z).toBeGreaterThan(shape.width - 1e-6);
      expect(bb.max.z - bb.min.z).toBeLessThan(shape.width + 0.03);
    });
  }

  it('deck/bottom vertices are the physics surface points, normals point out', () => {
    const shape = getBoardShape(BOARD_PRESETS.funboard);
    const g = buildHullGeometry(shape, 150, 28, 10);
    const pos = g.getAttribute('position');
    const nrm = g.getAttribute('normal');
    const uv = g.getAttribute('uv');
    const p = new THREE.Vector3();
    const q = new THREE.Vector3();
    let deckChecked = 0;
    let bottomChecked = 0;
    for (let i = 0; i < pos.count; i++) {
      p.fromBufferAttribute(pos, i);
      const u = uv.getX(i);
      const v = uv.getY(i) * 2 - 1;
      if (u < 0.1 || u > 0.9 || Math.abs(v) > 0.6) continue;
      // which surface is it on? compare with both
      shape.deckPoint(u, v, q);
      if (q.distanceTo(p) < 1e-6) {
        expect(nrm.getY(i)).toBeGreaterThan(0.8);
        deckChecked++;
        continue;
      }
      shape.bottomPoint(u, v, q);
      if (q.distanceTo(p) < 1e-6) {
        expect(nrm.getY(i)).toBeLessThan(-0.8);
        bottomChecked++;
      }
    }
    expect(deckChecked).toBeGreaterThan(500);
    expect(bottomChecked).toBeGreaterThan(500);
  });

  it('fins hang from the physics fin base to their depth', () => {
    const shape = getBoardShape(BOARD_PRESETS.funboard);
    for (const fin of shape.fins) {
      const g = buildFinGeometry(fin.base, fin.forward, fin.span, fin.spec.depth, fin.spec.baseChord, 0.007);
      const pos = g.getAttribute('position');
      const p = new THREE.Vector3();
      let maxSpan = -1;
      let minSpan = 1;
      for (let i = 0; i < pos.count; i++) {
        p.fromBufferAttribute(pos, i).sub(fin.base);
        const s = p.dot(fin.span);
        maxSpan = Math.max(maxSpan, s);
        minSpan = Math.min(minSpan, s);
      }
      expect(maxSpan).toBeGreaterThan(fin.spec.depth * 0.97);
      expect(maxSpan).toBeLessThan(fin.spec.depth * 1.03 + 0.003);
      expect(minSpan).toBeLessThan(0); // root inside the board bottom
    }
  });
});

describe('RiderMesh', () => {
  it('two-bone IK keeps the bone lengths', () => {
    const a = new THREE.Vector3();
    const c = new THREE.Vector3();
    const pole = new THREE.Vector3(0, 0, 1);
    const b = new THREE.Vector3();
    const e = new THREE.Vector3();
    for (let i = 0; i < 50; i++) {
      a.set(Math.sin(i), Math.cos(i * 1.3), 0.2 * i);
      c.copy(a).add(new THREE.Vector3(Math.sin(i * 2.1), Math.cos(i * 0.7), Math.sin(i * 0.3)).multiplyScalar(0.3 + (i % 7) * 0.12));
      solveTwoBone(a, c, 0.44, 0.43, pole, b, e);
      expect(a.distanceTo(b)).toBeCloseTo(0.44, 5);
      expect(b.distanceTo(e)).toBeCloseTo(0.43, 5);
      // the end lies on the ray a → c, at c when reachable
      if (a.distanceTo(c) < 0.86) expect(e.distanceTo(c)).toBeLessThan(1e-3);
    }
  });

  it('lays out prone, standing and fallen poses from the simulation without NaNs', () => {
    const ocean = new OceanModel();
    const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
    const rider = new RiderMesh({ castShadow: false });
    const check = (): void => {
      rider.update(sim.rider.pose, sim.board.quaternion, sim.rider.scale, true, sim.time);
      rider.object3d.updateMatrixWorld(true);
      rider.object3d.traverse((o) => {
        expect(Number.isFinite(o.position.x + o.position.y + o.position.z)).toBe(true);
        expect(Number.isFinite(o.quaternion.x + o.quaternion.w)).toBe(true);
        expect(Number.isFinite(o.scale.x + o.scale.y)).toBe(true);
      });
    };
    // prone at the lineup: every body part within ~0.45 m of the deck plane
    check();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(sim.board.quaternion);
    rider.object3d.children.forEach((m) => {
      const h = m.position.clone().sub(sim.board.position).dot(up);
      expect(h).toBeGreaterThan(-0.2);
      expect(h).toBeLessThan(0.45);
    });
    // standing on a wave: the lowest point of the figure is at the feet on the deck
    const res = waveSpawn(ocean, { fromTime: 30 });
    expect(res).not.toBeNull();
    sim.reset(res!.spawn);
    const input = createSurfInput();
    for (let i = 0; i < 60; i++) sim.step(PHYSICS_DT, input);
    check();
    const pose = sim.rider.pose;
    const box = new THREE.Box3().setFromObject(rider.object3d);
    const upW = new THREE.Vector3(0, 1, 0).applyQuaternion(sim.board.quaternion);
    const feetH = pose.frontFoot.dot(upW);
    expect(box.max.y - box.min.y).toBeGreaterThan(1.3); // standing tall-ish
    let lowest = Infinity;
    rider.object3d.children.forEach((m) => (lowest = Math.min(lowest, m.position.dot(upW))));
    expect(lowest).toBeGreaterThan(feetH - 0.02);
    expect(lowest).toBeLessThan(feetH + 0.12);
    // fallen
    sim.rider.fall('balance', sim.board);
    sim.step(PHYSICS_DT, input);
    check();
    rider.dispose();
  });
});

describe('Input', () => {
  const fakeWindow = (): Window =>
    ({ addEventListener: () => undefined, removeEventListener: () => undefined, navigator: undefined }) as unknown as Window;

  it('maps keys per stance and latches the pop-up', () => {
    const inp = new Input(fakeWindow());
    inp.setKeys({ W: true, A: true });
    let s = inp.update(Infinity, 'prone');
    expect(s.paddle).toBe(1);
    expect(s.steer).toBe(-1);
    expect(s.leanForward).toBe(0);
    expect(s.leanSide).toBe(0);
    s = inp.update(Infinity, 'standing');
    expect(s.paddle).toBe(0);
    expect(s.leanForward).toBe(1);
    expect(s.leanSide).toBe(-1);
    inp.setKeys({ W: false, A: false, D: true, Shift: true, E: true });
    s = inp.update(Infinity, 'standing');
    expect(s.leanSide).toBe(1);
    expect(s.crouch).toBe(1);
    expect(s.twist).toBe(1);
    // ramps: a short tap gives a partial lean
    inp.setKeys({ D: false });
    inp.update(Infinity, 'standing');
    inp.setKeys({ D: true });
    s = inp.update(0.05, 'standing');
    expect(s.leanSide).toBeGreaterThan(0.2);
    expect(s.leanSide).toBeLessThan(0.6);
    // pop-up edge survives frames until a physics step consumes it
    inp.setKeys({ Space: true });
    expect(inp.update(0.016, 'prone').popUp).toBe(true);
    expect(inp.update(0.016, 'prone').popUp).toBe(true);
    inp.consumeEdges();
    expect(inp.update(0.016, 'prone').popUp).toBe(false);
    // action keys fire callbacks
    const actions: string[] = [];
    inp.onAction = (a) => actions.push(a);
    inp.setKeys({ T: true, R: true });
    expect(actions).toEqual(['spawnWave', 'reset']);
  });
});
