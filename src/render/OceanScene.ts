/**
 * Convenience bundle of the render components for a surf spot: sky/sun/IBL (Environment), the
 * CDLOD water surface (OceanMesh), the beach/seabed terrain (BeachMesh) and crest spray (Spray).
 *
 *   const model = new OceanModel();
 *   const shaderData = new OceanShaderData(model);
 *   const oceanScene = new OceanScene(renderer, scene, model, shaderData, { quality: 'high' });
 *   // every frame:
 *   model.setTime(t); shaderData.update(model);
 *   oceanScene.update(camera, t);
 *   renderer.render(scene, camera);
 *   // after model.rebuild(config): shaderData.rebuild(model) (the beach mesh is static: recreate it
 *   // if the bathymetry changed).
 */
import * as THREE from 'three';
import type { OceanModel } from '../ocean/waveModel';
import type { OceanShaderData } from '../ocean/waveGLSL';
import { BeachMesh, type BeachMeshOptions } from './BeachMesh';
import { Environment, type EnvironmentOptions } from './Environment';
import { OceanMesh, type OceanMeshOptions, type OceanQuality } from './OceanMesh';
import { Spray, type SprayOptions } from './Spray';

export interface OceanSceneOptions {
  /** Ocean vertex density (preset or near range/size ratio); overrides ocean.quality. */
  quality?: OceanQuality | number;
  environment?: EnvironmentOptions;
  ocean?: OceanMeshOptions;
  /** false = no terrain. */
  beach?: BeachMeshOptions | false;
  /** false = no spray particles. */
  spray?: SprayOptions | false;
}

export class OceanScene {
  readonly env: Environment;
  readonly ocean: OceanMesh;
  readonly beach: BeachMesh | null;
  readonly spray: Spray | null;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly model: OceanModel;
  private readonly drawSize = new THREE.Vector2();

  constructor(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    model: OceanModel,
    shaderData: OceanShaderData,
    opts: OceanSceneOptions = {},
  ) {
    this.renderer = renderer;
    this.model = model;
    this.env = new Environment(renderer, scene, opts.environment);
    const wind = model.config.wind;
    this.ocean = new OceanMesh(shaderData, this.env, {
      windDirectionDeg: wind.directionDeg,
      windSpeed: wind.speed,
      ...opts.ocean,
      quality: opts.quality ?? opts.ocean?.quality ?? 'high',
    });
    scene.add(this.ocean.object3d);
    this.beach = opts.beach === false ? null : new BeachMesh(model, this.env, shaderData, opts.beach);
    if (this.beach) scene.add(this.beach.object3d);
    this.spray =
      opts.spray === false
        ? null
        : new Spray(shaderData, this.env, { windDirectionDeg: wind.directionDeg, windSpeed: wind.speed, ...opts.spray });
    if (this.spray) scene.add(this.spray.object3d);
  }

  /**
   * Per frame, after `model.setTime(time)` and `shaderData.update(model)`, before rendering with
   * `camera`. `time` is the ocean time (s) used for detail animation (ripples, foam, spray).
   */
  update(camera: THREE.PerspectiveCamera, time: number): void {
    this.env.update(camera);
    this.ocean.update(camera, time, this.model.heightAt(camera.position.x, camera.position.z));
    this.beach?.update(camera, time);
    this.spray?.update(camera, time, this.renderer.getDrawingBufferSize(this.drawSize).y);
  }

  dispose(): void {
    this.spray?.dispose();
    this.beach?.dispose();
    this.ocean.dispose();
    this.env.dispose();
  }
}
