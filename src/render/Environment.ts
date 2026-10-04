/**
 * Sky, sun and image-based lighting shared by every render component.
 *
 *  - Visible physical sky: three's Preetham `Sky` (with sun disc and a few clouds), kept
 *    centred on the camera.
 *  - Sun: DirectionalLight whose colour/intensity come from the same atmosphere model
 *    (transmittance along the sun ray, calibrated against the sky's irradiance), plus a weak
 *    HemisphereLight fill. Board/rider PBR materials get sky reflections from
 *    `scene.environment` (a PMREM of the sky without the sun disc over a dark sea).
 *  - Shared shader uniforms (`uniforms`) and PMREM defines (`envDefines`) for the custom
 *    ShaderMaterials (water, sand), so a sun change propagates everywhere.
 *
 * Units are the sky shader's linear output ("sky units"); the renderer's ACES tone mapping
 * and exposure (set here) map them to the display.
 */
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { DEFAULT_SKY_PARAMS, SkyModel, luminance, type SkyParams } from './skyModel';

export interface CloudOptions {
  coverage: number;
  density: number;
  elevation: number;
  scale: number;
}

export interface EnvironmentOptions {
  /** Sun elevation above the horizon, degrees. */
  sunElevationDeg?: number;
  /** Sun azimuth, degrees from +X toward +Z (180 = over the ocean, -X). */
  sunAzimuthDeg?: number;
  /** Renderer tone-mapping exposure. */
  exposure?: number;
  sky?: Partial<SkyParams>;
  clouds?: Partial<CloudOptions>;
  /** PMREM cube face size. */
  envMapSize?: number;
  /** Aerial-perspective extinction coefficient (1/m); visibility ~ 3.9 / density. */
  hazeDensity?: number;
  /** Ratio of direct-sun normal irradiance to the sky's horizontal irradiance. */
  sunToSkyRatio?: number;
  /** HemisphereLight intensity as a fraction of the sky irradiance (fill; IBL does the rest). */
  hemiFill?: number;
  /** Set renderer tone mapping (ACES), output colour space (sRGB) and exposure. Default true. */
  configureRenderer?: boolean;
}

/** Sun azimuth convention used throughout: direction to the sun from elevation/azimuth. */
export function sunDirectionFromAngles(elevDeg: number, azimDeg: number, out = new THREE.Vector3()): THREE.Vector3 {
  const e = THREE.MathUtils.degToRad(elevDeg);
  const a = THREE.MathUtils.degToRad(azimDeg);
  return out.set(Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a));
}

const GROUND_VS = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const GROUND_FS = /* glsl */ `
uniform vec3 uHorizon;
uniform vec3 uDeep;
varying vec3 vDir;
void main() {
  float t = smoothstep(0.0, 0.25, -normalize(vDir).y);
  gl_FragColor = vec4(mix(uHorizon, uDeep, t), 1.0);
}`;

export class Environment {
  readonly sky: Sky;
  readonly sunLight: THREE.DirectionalLight;
  readonly hemiLight: THREE.HemisphereLight;
  /** Unit vector toward the sun. */
  readonly sunDirection = new THREE.Vector3();
  /** Sun irradiance at normal incidence (linear sky units, = sunLight.color * intensity). */
  readonly sunIrradiance = new THREE.Color();
  /** Sky irradiance on a horizontal surface. */
  readonly skyIrradiance = new THREE.Color();
  /** Sky radiance just above the horizon, averaged over azimuth (fog colour). */
  readonly horizonColor = new THREE.Color();
  /** Shared uniforms; spread into ShaderMaterials (`{ ...env.uniforms }`). */
  readonly uniforms: {
    uSunDirection: THREE.IUniform<THREE.Vector3>;
    uSunColor: THREE.IUniform<THREE.Vector3>;
    uSkyIrradiance: THREE.IUniform<THREE.Vector3>;
    uEnvMap: THREE.IUniform<THREE.Texture | null>;
    uHazeDensity: THREE.IUniform<number>;
    uEnvIntensity: THREE.IUniform<number>;
  };
  elevationDeg: number;
  azimuthDeg: number;
  skyParams: SkyParams;
  sunToSkyRatio: number;
  hemiFill: number;

  /** The renderer this environment was created for (also used for offscreen passes). */
  readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly model: SkyModel;
  private readonly pmrem: THREE.PMREMGenerator;
  private readonly envScene = new THREE.Scene();
  private readonly envSky: Sky;
  private readonly ground: THREE.Mesh<THREE.SphereGeometry, THREE.ShaderMaterial>;
  private envTarget: THREE.WebGLRenderTarget | null = null;
  private readonly envSize: number;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, opts: EnvironmentOptions = {}) {
    this.renderer = renderer;
    this.scene = scene;
    this.elevationDeg = opts.sunElevationDeg ?? 24;
    this.azimuthDeg = opts.sunAzimuthDeg ?? 215;
    this.skyParams = { ...DEFAULT_SKY_PARAMS, ...opts.sky };
    this.sunToSkyRatio = opts.sunToSkyRatio ?? 9;
    this.hemiFill = opts.hemiFill ?? 0.25;
    this.envSize = opts.envMapSize ?? 256;
    if (opts.configureRenderer !== false) {
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = opts.exposure ?? 0.32;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
    }

    this.uniforms = {
      uSunDirection: { value: new THREE.Vector3(0, 1, 0) },
      uSunColor: { value: new THREE.Vector3(1, 1, 1) },
      uSkyIrradiance: { value: new THREE.Vector3(1, 1, 1) },
      uEnvMap: { value: null },
      uHazeDensity: { value: opts.hazeDensity ?? 1.6e-4 },
      uEnvIntensity: { value: 1 },
    };

    const clouds: CloudOptions = { coverage: 0.22, density: 0.45, elevation: 0.55, scale: 0.00022, ...opts.clouds };
    const makeSky = (sunDisc: boolean): Sky => {
      const s = new Sky();
      const u = s.material.uniforms;
      u.showSunDisc.value = sunDisc ? 1 : 0;
      u.cloudCoverage.value = clouds.coverage;
      u.cloudDensity.value = clouds.density;
      u.cloudElevation.value = clouds.elevation;
      u.cloudScale.value = clouds.scale;
      u.time.value = 0;
      return s;
    };
    this.sky = makeSky(true);
    this.sky.scale.setScalar(1000);
    this.sky.frustumCulled = false;
    this.sky.renderOrder = 1000; // drawn last: early-z skips covered pixels
    scene.add(this.sky);

    this.envSky = makeSky(false);
    this.envSky.scale.setScalar(200);
    this.envScene.add(this.envSky);
    this.ground = new THREE.Mesh(
      new THREE.SphereGeometry(80, 48, 16, 0, Math.PI * 2, Math.PI / 2 - 0.001, Math.PI / 2 + 0.001),
      new THREE.ShaderMaterial({
        uniforms: { uHorizon: { value: new THREE.Color() }, uDeep: { value: new THREE.Color() } },
        vertexShader: GROUND_VS,
        fragmentShader: GROUND_FS,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    );
    this.envScene.add(this.ground);

    this.sunLight = new THREE.DirectionalLight(0xffffff, 1);
    this.sunLight.name = 'sun';
    scene.add(this.sunLight);
    scene.add(this.sunLight.target);
    this.hemiLight = new THREE.HemisphereLight(0xffffff, 0x000000, 0);
    this.hemiLight.name = 'skyFill';
    scene.add(this.hemiLight);

    this.model = new SkyModel(new THREE.Vector3(0, 1, 0), this.skyParams);
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.setSun(this.elevationDeg, this.azimuthDeg);
  }

  /** PMREM environment texture (also assigned to scene.environment). */
  get envMap(): THREE.Texture {
    return this.envTarget!.texture;
  }

  /** Defines a ShaderMaterial needs to sample `uEnvMap` with textureCubeUV (see ENV_GLSL). */
  get envDefines(): Record<string, string | number> {
    const h = this.envTarget!.height;
    const maxMip = Math.log2(h) - 2;
    const fmt = (v: number): string => (Number.isInteger(v) ? v.toFixed(1) : String(v));
    return {
      ENVMAP_TYPE_CUBE_UV: '',
      CUBEUV_TEXEL_WIDTH: fmt(1 / (3 * Math.max(Math.pow(2, maxMip), 7 * 16))),
      CUBEUV_TEXEL_HEIGHT: fmt(1 / h),
      CUBEUV_MAX_MIP: fmt(maxMip),
    };
  }

  /** Move the sun; recomputes light colours and regenerates the environment map. */
  setSun(elevationDeg: number, azimuthDeg: number): void {
    this.elevationDeg = elevationDeg;
    this.azimuthDeg = azimuthDeg;
    sunDirectionFromAngles(elevationDeg, azimuthDeg, this.sunDirection);
    this.refresh();
  }

  /** Change atmosphere parameters (turbidity etc.) and refresh. */
  setSkyParams(p: Partial<SkyParams>): void {
    Object.assign(this.skyParams, p);
    this.refresh();
  }

  get exposure(): number {
    return this.renderer.toneMappingExposure;
  }

  set exposure(v: number) {
    this.renderer.toneMappingExposure = v;
  }

  get hazeDensity(): number {
    return this.uniforms.uHazeDensity.value;
  }

  set hazeDensity(v: number) {
    this.uniforms.uHazeDensity.value = v;
    this.updateFog();
  }

  /** Per frame: keep the sky dome around the camera; optional cloud animation time (s). */
  update(camera: THREE.Camera, time?: number): void {
    this.sky.position.copy(camera.position);
    if (time !== undefined) this.sky.material.uniforms.time.value = time;
  }

  dispose(): void {
    this.scene.remove(this.sky, this.sunLight, this.sunLight.target, this.hemiLight);
    this.sky.geometry.dispose();
    this.sky.material.dispose();
    this.envSky.geometry.dispose();
    this.envSky.material.dispose();
    this.ground.geometry.dispose();
    this.ground.material.dispose();
    this.envTarget?.dispose();
    this.pmrem.dispose();
    if (this.scene.environment === this.envTarget?.texture) this.scene.environment = null;
  }

  private refresh(): void {
    const sun = this.sunDirection;
    const p = this.skyParams;
    for (const s of [this.sky, this.envSky]) {
      const u = s.material.uniforms;
      (u.sunPosition.value as THREE.Vector3).copy(sun);
      u.turbidity.value = p.turbidity;
      u.rayleigh.value = p.rayleigh;
      u.mieCoefficient.value = p.mieCoefficient;
      u.mieDirectionalG.value = p.mieDirectionalG;
    }
    const m = this.model;
    m.set(sun, p);

    // Horizon colour (azimuth average) and a dark-sea lower hemisphere for the env map.
    const c = new THREE.Color();
    const d = new THREE.Vector3();
    this.horizonColor.setRGB(0, 0, 0);
    const n = 32;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      d.set(Math.cos(a), 0.035, Math.sin(a)).normalize();
      m.radiance(d, c);
      this.horizonColor.r += c.r / n;
      this.horizonColor.g += c.g / n;
      this.horizonColor.b += c.b / n;
    }
    const deep = this.horizonColor.clone().multiplyScalar(0.06).add(new THREE.Color(0.002, 0.012, 0.025));
    const ground = this.ground.material.uniforms;
    (ground.uHorizon.value as THREE.Color).copy(this.horizonColor).multiplyScalar(0.55);
    (ground.uDeep.value as THREE.Color).copy(deep);

    // Sky irradiance on a horizontal surface, sun irradiance from the transmittance along the
    // sun ray, scaled so direct-normal / sky-horizontal = sunToSkyRatio (clear-sky typical ~9).
    m.irradiance(new THREE.Vector3(0, 1, 0), deep, this.skyIrradiance);
    const fex = m.transmittance(sun, new THREE.Color());
    const chroma = fex.clone().multiplyScalar(1 / Math.max(luminance(fex), 1e-6));
    const fadeBelow = THREE.MathUtils.smoothstep(sun.y, -0.02, 0.06);
    const eSun = this.sunToSkyRatio * luminance(this.skyIrradiance) * fadeBelow;
    this.sunIrradiance.copy(chroma).multiplyScalar(eSun);

    this.sunLight.color.copy(chroma);
    this.sunLight.intensity = eSun;
    this.sunLight.position.copy(sun).multiplyScalar(1000);
    this.sunLight.target.position.set(0, 0, 0);
    const skyAvg = this.skyIrradiance.clone();
    this.hemiLight.color.copy(skyAvg).multiplyScalar(1 / Math.max(luminance(skyAvg), 1e-6));
    this.hemiLight.groundColor.copy(deep).multiplyScalar(1 / Math.max(luminance(skyAvg), 1e-6));
    this.hemiLight.intensity = luminance(skyAvg) * this.hemiFill;

    const u = this.uniforms;
    u.uSunDirection.value.copy(sun);
    u.uSunColor.value.set(this.sunIrradiance.r, this.sunIrradiance.g, this.sunIrradiance.b);
    u.uSkyIrradiance.value.set(this.skyIrradiance.r, this.skyIrradiance.g, this.skyIrradiance.b);

    // Environment map: sky without the sun disc (the sun is added analytically) over the sea.
    const old = this.envTarget;
    this.envTarget = this.pmrem.fromScene(this.envScene, 0, 0.1, 1000, { size: this.envSize });
    u.uEnvMap.value = this.envTarget.texture;
    this.scene.environment = this.envTarget.texture;
    old?.dispose();
    this.updateFog();
  }

  private updateFog(): void {
    // FogExp2 (used by built-in materials) is exp(-(density d)^2); match exp(-sigma d) at ~3 km.
    const sigma = this.uniforms.uHazeDensity.value;
    const d = 3000;
    const density = Math.sqrt(sigma * d) / d;
    if (this.scene.fog instanceof THREE.FogExp2) {
      this.scene.fog.color.copy(this.horizonColor);
      this.scene.fog.density = density;
    } else {
      this.scene.fog = new THREE.FogExp2(this.horizonColor.getHex(THREE.LinearSRGBColorSpace), density);
      this.scene.fog.color.copy(this.horizonColor);
    }
  }
}
