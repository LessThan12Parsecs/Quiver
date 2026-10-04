/**
 * Physics integrity (A8): the only external forces on board + rider are gravity, the water (hull,
 * fins, body segments, paddling hands), the seabed and the leash; everything the rider does
 * (legs, ankles/feet pressure, hip, twist, pop-up) is internal — equal and opposite.
 *
 * Check: high above the water (no water, no seabed) the total linear momentum changes only by the
 * gravity impulse, and the total angular momentum about the system COM (board spin, both bodies'
 * motion about the COM, the upper-body twist rotor and the trunk's hip rotation) stays constant,
 * whatever the rider does.
 */
import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { DEFAULT_OCEAN_CONFIG, cloneOceanConfig } from '../src/ocean/oceanConfig';
import { OceanModel } from '../src/ocean/waveModel';
import { BOARD_PRESETS } from '../src/physics/boardShape';
import { GRAVITY } from '../src/physics/constants';
import { SurfSim, createSurfInput } from '../src/physics/SurfSim';

const DT = 1 / 240;
const calmCfg = cloneOceanConfig(DEFAULT_OCEAN_CONFIG);
calmCfg.swells = [];
calmCfg.wind.count = 0;
const calm = new OceanModel(calmCfg);

/** Linear momentum of board + rider, N·s. */
function momentum(sim: SurfSim): Vector3 {
  const b = sim.board;
  const r = sim.rider;
  return new Vector3().copy(b.velocity).multiplyScalar(b.mass).addScaledVector(r.velocity, r.config.mass);
}

/** Angular momentum about the system COM, N·m·s: board spin + orbital terms + internal rotors. */
function angularMomentum(sim: SurfSim): Vector3 {
  const b = sim.board;
  const r = sim.rider;
  const mb = b.mass;
  const mr = r.config.mass;
  const m = mb + mr;
  const c = new Vector3().copy(b.position).multiplyScalar(mb).addScaledVector(r.position, mr).divideScalar(m);
  const vc = momentum(sim).divideScalar(m);
  const L = b.angularMomentum(new Vector3());
  const d = new Vector3();
  const dv = new Vector3();
  L.add(new Vector3().crossVectors(d.subVectors(b.position, c), dv.subVectors(b.velocity, vc)).multiplyScalar(mb));
  L.add(new Vector3().crossVectors(d.subVectors(r.position, c), dv.subVectors(r.velocity, vc)).multiplyScalar(mr));
  // upper-body twist about the deck normal (taken out of the board's inertia)
  L.addScaledVector(new Vector3(b.R[1], b.R[4], b.R[7]), -r.twistInertia() * r.twistSpin);
  // trunk hip rotation (pitch about −Z for trunkX, roll about +X for trunkZ, in the balance frame)
  const It = r.trunkInertia();
  L.addScaledVector(r.frameX, It * r.trunkVZ).addScaledVector(r.frameZ, -It * r.trunkVX);
  return L;
}

describe('physics integrity (internal forces only)', () => {
  it('high above the water, held up at the board COM: momentum changes only by the external impulses, angular momentum about the COM only by their torque', () => {
    for (const spec of [BOARD_PRESETS.funboard, BOARD_PRESETS.longboard]) {
      const sim = new SurfSim(calm, spec, { wipeouts: false });
      sim.reset({ x: 0, z: 0, headingRad: 0.3, stance: 'prone', speed: 0, time: 0 });
      // lift everything 60 m (no water, no seabed: the hull, the body segments and the hands are
      // all out of the water) and hold the board up at its COM with a force that carries the
      // weight of board + rider, and level with a torque (a gimbal; yaw stays free), so the legs
      // are loaded as on the water. Both are external: their impulse and torque are accounted for.
      sim.board.position.y += 60;
      sim.board.updateDerived();
      sim.rider.position.y += 60;
      const mb = sim.board.mass;
      const m = mb + sim.rider.config.mass;
      const support = new Vector3(0, m * GRAVITY, 0);
      const input = createSurfInput();
      const p0 = momentum(sim);
      let L0: Vector3 | null = null;
      const tauInt = new Vector3(); // ∫ external torque about the COM dt since L0
      const tauB = new Vector3();
      const up = new Vector3();
      const Y = new Vector3(0, 1, 0);
      let maxDp = 0;
      let maxDL = 0;
      let maxL = 0;
      let stood = 0;
      const c = new Vector3();
      const arm = new Vector3();
      const n = Math.round(3 / DT);
      for (let i = 0; i < n; i++) {
        const t = i * DT;
        // prone: paddle and steer (hands in the air), then pop up and work the body: lean, trim,
        // crouch, twist
        input.paddle = t < 0.5 ? 1 : 0;
        input.steer = t < 0.5 ? Math.sin(9 * t) : 0;
        input.popUp = i === Math.round(0.5 / DT);
        input.leanSide = t > 1.2 ? 0.6 * Math.sin(4 * t) : 0;
        input.leanForward = t > 1.2 ? 0.6 * Math.cos(3 * t) : 0;
        input.crouch = t > 1.2 ? 0.5 + 0.5 * Math.sin(2 * t) : 0;
        input.twist = t > 2 ? Math.sin(5 * t) : 0;
        // the support's torque about the system COM over this step (midpoint-free: forces are
        // applied for the whole step at the start-of-step positions)
        c.copy(sim.board.position).multiplyScalar(mb).addScaledVector(sim.rider.position, sim.rider.config.mass).divideScalar(m);
        arm.subVectors(sim.board.position, c);
        sim.extraBoardForce.copy(support);
        const b = sim.board;
        up.set(b.R[1], b.R[4], b.R[7]);
        const w = b.angularVelocity;
        const wy = w.y;
        tauB.crossVectors(up, Y).multiplyScalar(600).addScaledVector(w, -10).addScaledVector(Y, 10 * wy);
        sim.extraBoardTorque.copy(tauB);
        sim.step(DT, input);
        if (L0) tauInt.add(new Vector3().crossVectors(arm, support).add(tauB).multiplyScalar(DT));
        if (sim.rider.stance === 'standing') stood++;
        maxDp = Math.max(maxDp, momentum(sim).sub(p0).length());
        // angular momentum, prone, getting up and standing (the hip/twist rotors hold some once
        // standing)
        {
          const L = angularMomentum(sim);
          if (!L0) L0 = L;
          else {
            maxDL = Math.max(maxDL, L.clone().sub(L0).sub(tauInt).length());
            maxL = Math.max(maxL, L.clone().sub(L0).length());
          }
        }
      }
      console.log(`  ${spec.id}: |Δp| max ${maxDp.toExponential(2)} N·s, |ΔL − ∫τ_support| about the COM max ${maxDL.toFixed(4)} N·m·s (|ΔL| up to ${maxL.toFixed(2)}) (${(stood * DT).toFixed(1)} s of it standing), ${sim.recoveries} recoveries`);
      expect(stood * DT).toBeGreaterThan(1.5);
      expect(sim.recoveries).toBe(0);
      expect(maxDp).toBeLessThan(1e-6 * m * GRAVITY * 3);
      // (a first-order residue of the explicit rotor couplings: < 1 % of what the body exchanges)
      expect(maxDL).toBeLessThan(0.1);
    }
  });
});
