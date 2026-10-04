/**
 * Render-side effects that can be checked without WebGL:
 *  - transparent draw order: crest spray after the rider and the board's underwater pass;
 *  - BoardWake (rail spray, wake, splashes) driven by a real SurfSim: a planing board leaves a
 *    wake, a carve throws spray off the rail, a paused frame changes nothing, reset clears.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { OceanModel, createSurfaceSample } from '../src/ocean/waveModel';
import { BOARD_PRESETS } from '../src/physics/boardShape';
import { PHYSICS_DT } from '../src/physics/constants';
import { waveSpawn } from '../src/physics/spawn';
import { SurfSim, createSurfInput } from '../src/physics/SurfSim';
import { BoardWake } from '../src/render/BoardWake';
import type { Environment } from '../src/render/Environment';
import { RIDER_RENDER_ORDER } from '../src/render/RiderMesh';
import { SPRAY_RENDER_ORDER } from '../src/render/Spray';

describe('transparent draw order', () => {
  it('crest spray blends after the board underwater pass (20) and the rider', () => {
    expect(RIDER_RENDER_ORDER).toBeGreaterThan(20);
    expect(SPRAY_RENDER_ORDER).toBeGreaterThan(RIDER_RENDER_ORDER);
  });
});

describe('BoardWake', () => {
  // only uniforms / defines are read from the environment (no renderer needed)
  const env = { uniforms: {}, envDefines: {} } as unknown as Environment;
  const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000);
  camera.updateProjectionMatrix();

  function ride(lean: number): { sim: SurfSim; wake: BoardWake; ocean: OceanModel; frames: (n: number) => void } {
    const ocean = new OceanModel();
    const res = waveSpawn(ocean, { fromTime: 30, side: 1 });
    expect(res).not.toBeNull();
    const sim = new SurfSim(ocean, BOARD_PRESETS.funboard);
    sim.reset(res!.spawn);
    const wake = new BoardWake(env);
    const inp = createSurfInput();
    inp.leanSide = lean;
    const water = createSurfaceSample();
    const frames = (n: number): void => {
      for (let f = 0; f < n; f++) {
        for (let i = 0; i < 4; i++) sim.step(PHYSICS_DT, inp); // 60 fps
        const p = sim.board.position;
        ocean.sample(p.x, p.z, water);
        wake.update(sim, ocean, p, sim.board.quaternion, water, sim.time, camera, 720);
      }
    };
    return { sim, wake, ocean, frames };
  }

  const drawn = (w: BoardWake): number => (w.object3d.children[1] as THREE.Mesh).geometry.drawRange.count;

  it('a planing board leaves a wake; reset clears it', () => {
    const { sim, wake, frames } = ride(0);
    frames(45);
    expect(sim.rider.stance).not.toBe('prone');
    expect(drawn(wake)).toBeGreaterThan(6 * 4);
    // paused (same render time): nothing changes
    const before = drawn(wake);
    const water = createSurfaceSample();
    wake.update(sim, sim.ocean, sim.board.position, sim.board.quaternion, water, sim.time, camera, 720);
    expect(drawn(wake)).toBe(before);
    wake.reset();
    expect(drawn(wake)).toBe(0);
    expect(wake.liveParticles).toBe(0);
  });

  it('a carve throws spray off the rail', () => {
    const straight = ride(0);
    straight.frames(40);
    const carve = ride(1);
    carve.frames(40);
    expect(Math.abs(carve.sim.telemetry.rollRad)).toBeGreaterThan(0.2);
    expect(carve.wake.liveParticles).toBeGreaterThan(20);
    expect(carve.wake.liveParticles).toBeGreaterThan(straight.wake.liveParticles);
  });
});
