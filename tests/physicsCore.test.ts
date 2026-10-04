import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { BOARD_PRESETS, BoardShape, boardComOffset, getBoardShape } from '../src/physics/boardShape';
import { RHO_WATER } from '../src/physics/constants';
import { FIN_MODEL, finCoefficients, finForce, finLiftSlope } from '../src/physics/fins';
import { ImplicitSystem } from '../src/physics/implicit';
import { RigidBody } from '../src/physics/RigidBody';

describe('RigidBody', () => {
  it('integrates a constant force exactly (semi-implicit Euler)', () => {
    const b = new RigidBody(2);
    const f = new Vector3(4, 0, 0);
    for (let i = 0; i < 100; i++) {
      b.clearForces();
      b.addForce(f);
      b.integrate(0.01);
    }
    expect(b.velocity.x).toBeCloseTo(2, 9); // a = 2 m/s² for 1 s
    expect(b.position.x).toBeCloseTo(1.01, 9); // symplectic Euler: Σ v_i dt
  });

  it('addForceAtPoint produces r × F and pointVelocity is v + ω × r', () => {
    const b = new RigidBody(1, new Vector3(1, 1, 1));
    b.position.set(1, 2, 3);
    b.addForceAtPoint(new Vector3(0, 0, 10), new Vector3(2, 2, 3));
    expect(b.torque.toArray()).toEqual([0, -10, 0]);
    b.velocity.set(1, 0, 0);
    b.angularVelocity.set(0, 0, 2);
    const v = b.pointVelocity(new Vector3(1, 3, 3), new Vector3());
    // ω × r = (0,0,2) × (0,1,0) = (-2, 0, 0)
    expect(v.x).toBeCloseTo(-1, 12);
    expect(v.y).toBeCloseTo(0, 12);
  });

  it('local/world transforms round-trip', () => {
    const b = new RigidBody(1);
    b.position.set(-3, 1, 7);
    b.quaternion.setFromAxisAngle(new Vector3(1, 2, 3).normalize(), 0.7);
    b.updateDerived();
    const p = new Vector3(0.3, -0.2, 0.9);
    const w = b.localToWorld(p, new Vector3());
    const back = b.worldToLocal(w, new Vector3());
    expect(back.distanceTo(p)).toBeLessThan(1e-12);
    const ref = p.clone().applyQuaternion(b.quaternion).add(b.position);
    expect(ref.distanceTo(w)).toBeLessThan(1e-12);
  });

  it('torque-free spin conserves angular momentum and does not gain energy (board-like inertia)', () => {
    const b = new RigidBody(4.5, new Vector3(0.09, 1.6, 1.5));
    b.angularVelocity.set(3, 0.4, 0.2);
    const L0 = b.angularMomentum(new Vector3());
    const E0 = b.kineticEnergy();
    for (let i = 0; i < 240 * 10; i++) {
      b.clearForces();
      b.integrate(1 / 240);
    }
    const L1 = b.angularMomentum(new Vector3());
    const E1 = b.kineticEnergy();
    console.log(`  spin: |L| ${L0.length().toFixed(4)} -> ${L1.length().toFixed(4)}, E ${E0.toFixed(4)} -> ${E1.toFixed(4)}`);
    expect(L1.distanceTo(L0) / L0.length()).toBeLessThan(0.02);
    expect(E1).toBeLessThanOrEqual(E0 * 1.0001);
    expect(E1).toBeGreaterThan(E0 * 0.9);
    expect(Math.abs(b.quaternion.length() - 1)).toBeLessThan(1e-12);
  });

  it('spinning about the intermediate axis stays bounded (implicit gyroscopic term)', () => {
    const b = new RigidBody(1, new Vector3(1, 2, 3));
    b.angularVelocity.set(0.01, 5, 0.01);
    const E0 = b.kineticEnergy();
    let maxE = E0;
    for (let i = 0; i < 240 * 20; i++) {
      b.clearForces();
      b.integrate(1 / 240);
      maxE = Math.max(maxE, b.kineticEnergy());
    }
    expect(maxE).toBeLessThanOrEqual(E0 * 1.0001);
  });
});

describe('ImplicitSystem', () => {
  it('solves the damped linear system exactly (compare to a direct computation)', () => {
    const sys = new ImplicitSystem();
    sys.clear();
    const M = sys.M;
    M.fill(0);
    for (let i = 0; i < 9; i++) M[i * 9 + i] = i < 3 ? 4 : i < 6 ? 0.5 : 75;
    sys.addBoardForce(10, -40, 3, 0.2, 0.1, -0.3);
    sys.addRiderForce(0, -735, 0);
    sys.addBoardDamping(0, -1, 0, 0.5, -0.05, 0.1, 3000);
    sys.addLink(0, 1, 0, 0.1, 1, 0, 2000, 26000);
    const u = new Float64Array([1, 0.2, 0, 0.1, -0.3, 0.05, 1, 0, 0]);
    const h = 1 / 240;
    const du = Float64Array.from(sys.solve(h, u));
    // A du should equal rhs: build A and rhs directly from the symmetric matrices
    const full = (X: Float64Array, i: number, j: number) => (i <= j ? X[i * 9 + j] : X[j * 9 + i]);
    for (let i = 0; i < 9; i++) {
      let lhs = 0;
      let ku = 0;
      for (let j = 0; j < 9; j++) {
        lhs += (full(M, i, j) + h * full(sys.C, i, j) + h * h * full(sys.K, i, j)) * du[j];
        ku += full(sys.K, i, j) * u[j];
      }
      expect(lhs).toBeCloseTo(h * (sys.Q[i] - h * ku), 9);
    }
  });

  it('a very stiff spring between the light board and the heavy rider stays stable at 240 Hz', () => {
    // board 3 kg, rider 75 kg, k = 26.6 kN/m, applied 1 m above the board COM (roll lever)
    const sys = new ImplicitSystem();
    const u = new Float64Array(9);
    let d = 0.05; // initial lateral stretch
    let maxD = 0;
    for (let step = 0; step < 2400; step++) {
      sys.clear();
      const M = sys.M;
      M.fill(0);
      M[0] = M[10] = M[20] = 3;
      M[30] = 0.05; M[40] = 0.6; M[50] = 0.6;
      M[60] = M[70] = M[80] = 75;
      const v = ImplicitSystem.linkVelocity(u, 0, 0, 1, 0, 1, 0);
      const f = -26600 * d - 2260 * v;
      sys.addRiderForce(0, 0, f);
      sys.addBoardForce(0, 0, -f, 0, 1, 0);
      sys.addLink(0, 0, 1, 0, 1, 0, 2260, 26600);
      const du = sys.solve(1 / 240, u);
      for (let i = 0; i < 9; i++) u[i] += du[i];
      d += ImplicitSystem.linkVelocity(u, 0, 0, 1, 0, 1, 0) / 240;
      maxD = Math.max(maxD, Math.abs(d));
    }
    expect(Number.isFinite(d)).toBe(true);
    expect(maxD).toBeLessThan(0.06);
    expect(Math.abs(d)).toBeLessThan(1e-3);
  });
});

describe('board shape', () => {
  it('integrates to the specified volume for every preset and has sane dimensions', () => {
    for (const spec of Object.values(BOARD_PRESETS)) {
      const s = new BoardShape(spec);
      const V = s.integrateVolume(undefined, 600, 120);
      console.log(
        `  ${spec.id}: T=${(s.maxThickness * 100).toFixed(2)} cm, V=${(V * 1000).toFixed(2)} L, planform=${s.planformArea.toFixed(3)} m², ` +
          `foam ${s.massProperties.foamDensity.toFixed(0)} kg/m³, I=${s.massProperties.inertia.toArray().map((x) => x.toFixed(3))}`,
      );
      expect(Math.abs(V * 1000 - spec.volumeLiters) / spec.volumeLiters).toBeLessThan(0.003);
      expect(Math.abs(s.volume * 1000 - spec.volumeLiters) / spec.volumeLiters).toBeLessThan(0.003);
      expect(s.maxThickness).toBeGreaterThan(0.05);
      expect(s.maxThickness).toBeLessThan(0.1);
      expect(2 * s.halfWidth(spec.outline.widePoint)).toBeCloseTo(spec.width, 6);
      expect(2 * s.halfWidth(1 - 0.3048 / spec.length)).toBeCloseTo(spec.outline.noseWidth, 6);
      expect(2 * s.halfWidth(0.3048 / spec.length)).toBeCloseTo(spec.outline.tailWidth, 6);
      expect(s.halfWidth(1)).toBe(0);
      expect(s.massProperties.foamDensity).toBeGreaterThan(12);
      expect(s.massProperties.foamDensity).toBeLessThan(80);
    }
  });

  it('rocker matches the spec (nose 11–13 cm and tail 4–6 cm on the shortboard, less on longboards)', () => {
    const sb = getBoardShape(BOARD_PRESETS.shortboard);
    const lb = getBoardShape(BOARD_PRESETS.longboard);
    expect(sb.rocker(1)).toBeGreaterThanOrEqual(0.11);
    expect(sb.rocker(1)).toBeLessThanOrEqual(0.13);
    expect(sb.rocker(0)).toBeGreaterThanOrEqual(0.04);
    expect(sb.rocker(0)).toBeLessThanOrEqual(0.06);
    expect(sb.rocker(BOARD_PRESETS.shortboard.rocker.lowPoint)).toBe(0);
    expect(lb.rocker(1) / lb.length).toBeLessThan(sb.rocker(1) / sb.length);
  });

  it('local frame: the centre of mass is the origin and the mesh functions agree with the hull frame', () => {
    for (const spec of Object.values(BOARD_PRESETS)) {
      const s = getBoardShape(spec);
      expect(boardComOffset(spec).distanceTo(s.comShape)).toBe(0);
      // deck above bottom everywhere, thickness consistent
      for (const u of [0.02, 0.3, 0.5, 0.8, 0.97]) {
        for (const v of [-1, -0.5, 0, 0.7, 1]) {
          const t = s.deckY(u, v) - s.bottomY(u, v);
          expect(t).toBeGreaterThan(0);
          expect(t).toBeCloseTo(s.thickness(u, v), 12);
        }
      }
      expect(Math.abs(s.comShape.z)).toBeLessThan(1e-9);
      // COM between bottom and deck near the middle
      expect(s.deckY(0.45, 0)).toBeGreaterThan(0);
      expect(s.bottomY(0.45, 0)).toBeLessThan(0);
      // outward normals
      const n = new Vector3();
      expect(s.bottomNormal(0.5, 0, n).y).toBeLessThan(-0.99);
      expect(s.bottomNormal(0.95, 0, n).x).toBeGreaterThan(0.05); // nose rocker faces forward-down
      expect(s.deckNormal(0.5, 0, n).y).toBeGreaterThan(0.99);
      expect(s.bottomNormal(0.5, 0.95, n).z).toBeGreaterThan(0.05); // rail tuck tilts outward
    }
  });

  it('places fins under the tail with toe-in and cant', () => {
    const s = getBoardShape(BOARD_PRESETS.shortboard);
    const [left, right, centre] = s.fins;
    expect(left.base.z).toBeLessThan(0);
    expect(right.base.z).toBeGreaterThan(0);
    expect(centre.base.z).toBeCloseTo(0, 9);
    expect(right.tip.y).toBeLessThan(right.base.y);
    expect(right.tip.z).toBeGreaterThan(right.base.z); // cant: tip leans toward the rail
    expect(right.forward.z).toBeLessThan(0); // toe-in: leading edge toward the stringer
    expect(left.forward.z).toBeGreaterThan(0);
    expect(centre.base.x).toBeLessThan(left.base.x);
    for (const f of s.fins) expect(f.base.x).toBeLessThan(-0.4);
  });
});

describe('fins', () => {
  const span = new Vector3(0, -1, 0);
  const forward = new Vector3(1, 0, 0);
  const normal = new Vector3().crossVectors(forward, span); // (0, 0, -1)
  const res = { fx: 0, fy: 0, fz: 0, alpha: 0, speed: 0, damping: 0, attached: 1 };

  it('lift slope follows 2π·AR/(AR+2) and stalls around 14°', () => {
    const c = { cl: 0, cd: 0, attached: 1, slope: 0 };
    const AR = 2.5;
    const a = finLiftSlope(AR);
    expect(a).toBeCloseTo((2 * Math.PI * 2.5) / 4.5, 12);
    finCoefficients(0.02, AR, c);
    expect(c.cl / 0.02).toBeCloseTo(a, 1);
    const cls: number[] = [];
    for (let deg = 0; deg <= 40; deg += 2) cls.push(finCoefficients((deg * Math.PI) / 180, AR, c).cl);
    const peak = cls.indexOf(Math.max(...cls)) * 2;
    expect(peak).toBeGreaterThanOrEqual(10);
    expect(peak).toBeLessThanOrEqual(16);
    expect(finCoefficients((30 * Math.PI) / 180, AR, c).cl).toBeLessThan(0.75 * Math.max(...cls));
    // odd symmetry, drag even and positive
    expect(finCoefficients(-0.1, AR, c).cl).toBeCloseTo(-finCoefficients(0.1, AR, c).cl, 12);
    expect(finCoefficients(-0.1, AR, c).cd).toBeCloseTo(finCoefficients(0.1, AR, c).cd, 12);
    expect(finCoefficients(Math.PI / 2, AR, c).cd).toBeGreaterThan(FIN_MODEL.cnMax * 0.95);
  });

  it('straight flow gives pure drag along the flow', () => {
    finForce(5, 0, 0, span, forward, normal, 0.01, 2.5, res);
    expect(res.alpha).toBeCloseTo(0, 12);
    expect(res.fx).toBeLessThan(0);
    expect(Math.abs(res.fz)).toBeLessThan(1e-9);
    expect(-res.fx).toBeCloseTo(0.5 * RHO_WATER * 25 * 0.01 * FIN_MODEL.cd0, 6);
  });

  it('sideslip produces a force opposing the slip (independent of the normal sign) with a small thrust part', () => {
    const beta = (8 * Math.PI) / 180;
    // fin moving forward and to the right (+Z)
    finForce(5 * Math.cos(beta), 0, 5 * Math.sin(beta), span, forward, normal, 0.01, 2.5, res);
    expect(res.fz).toBeLessThan(0);
    const fz1 = res.fz;
    const lift = Math.hypot(res.fx, res.fz);
    expect(lift).toBeGreaterThan(0.5 * RHO_WATER * 25 * 0.01 * 0.5 * finLiftSlope(2.5) * beta);
    finForce(5 * Math.cos(beta), 0, 5 * Math.sin(beta), span, forward, normal.clone().negate(), 0.01, 2.5, res);
    expect(res.fz).toBeCloseTo(fz1, 9);
    // moving left → force to the right
    finForce(5 * Math.cos(beta), 0, -5 * Math.sin(beta), span, forward, normal, 0.01, 2.5, res);
    expect(res.fz).toBeGreaterThan(0);
    // spanwise flow is ignored
    finForce(5, 3, 0, span, forward, normal, 0.01, 2.5, res);
    expect(Math.abs(res.fy)).toBeLessThan(1e-9);
  });

  it('a toed-in right fin pushes inward in straight flow (and the left fin outward-symmetric)', () => {
    const s = getBoardShape(BOARD_PRESETS.shortboard);
    const right = s.fins[1];
    const left = s.fins[0];
    finForce(5, 0, 0, right.span, right.forward, right.normal, 0.01, 2.5, res);
    const fr = res.fz;
    finForce(5, 0, 0, left.span, left.forward, left.normal, 0.01, 2.5, res);
    expect(fr).toBeLessThan(0);
    expect(res.fz).toBeCloseTo(-fr, 9);
  });
});
