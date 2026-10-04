/**
 * CPU port of the Preetham sky in three/addons/objects/Sky.js (cloud-free, no sun disc).
 *
 * Used to calibrate the lighting so that everything is consistent with what the sky shader
 * draws: the sun colour (atmospheric transmittance along the sun ray), the sky irradiance on a
 * horizontal surface and the horizon colour. Values are in the sky shader's linear output units
 * ("sky units"), i.e. the same units the PMREM environment map ends up in.
 */
import * as THREE from 'three';

export interface SkyParams {
  turbidity: number;
  rayleigh: number;
  mieCoefficient: number;
  mieDirectionalG: number;
}

export const DEFAULT_SKY_PARAMS: SkyParams = {
  turbidity: 2.0,
  rayleigh: 1.2,
  mieCoefficient: 0.003,
  mieDirectionalG: 0.86,
};

const TOTAL_RAYLEIGH = [5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5];
const MIE_CONST = [1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14];
const CUTOFF_ANGLE = 1.6110731556870734;
const STEEPNESS = 1.5;
const EE = 1000;
const RAYLEIGH_ZENITH_LENGTH = 8.4e3;
const MIE_ZENITH_LENGTH = 1.25e3;
const THREE_OVER_SIXTEENPI = 0.05968310365946075;
const ONE_OVER_FOURPI = 0.07957747154594767;

/** Evaluates the sky shader on the CPU for a fixed sun direction. */
export class SkyModel {
  private readonly sun = new THREE.Vector3(0, 1, 0);
  private sunE = 0;
  private readonly betaR = [0, 0, 0];
  private readonly betaM = [0, 0, 0];
  private params: SkyParams = { ...DEFAULT_SKY_PARAMS };

  constructor(sunDirection: THREE.Vector3, params: SkyParams = DEFAULT_SKY_PARAMS) {
    this.set(sunDirection, params);
  }

  set(sunDirection: THREE.Vector3, params: SkyParams = this.params): void {
    this.params = { ...params };
    this.sun.copy(sunDirection).normalize();
    const zc = Math.min(Math.max(this.sun.y, -1), 1);
    this.sunE = EE * Math.max(0, 1 - Math.exp(-(CUTOFF_ANGLE - Math.acos(zc)) / STEEPNESS));
    // vSunfade is 1 for a unit-length sun position, so the rayleigh coefficient is used as is.
    const c = 0.2 * params.turbidity * 10e-18;
    for (let i = 0; i < 3; i++) {
      this.betaR[i] = TOTAL_RAYLEIGH[i] * params.rayleigh;
      this.betaM[i] = 0.434 * c * MIE_CONST[i] * params.mieCoefficient;
    }
  }

  /** Atmospheric transmittance along direction `dir` (Fex in the shader). */
  transmittance(dir: THREE.Vector3, out: THREE.Color): THREE.Color {
    const zenith = Math.acos(Math.max(0, dir.y));
    const inv = 1 / (Math.cos(zenith) + 0.15 * Math.pow(93.885 - (zenith * 180) / Math.PI, -1.253));
    const sR = RAYLEIGH_ZENITH_LENGTH * inv;
    const sM = MIE_ZENITH_LENGTH * inv;
    out.r = Math.exp(-(this.betaR[0] * sR + this.betaM[0] * sM));
    out.g = Math.exp(-(this.betaR[1] * sR + this.betaM[1] * sM));
    out.b = Math.exp(-(this.betaR[2] * sR + this.betaM[2] * sM));
    return out;
  }

  /** Sky radiance (shader output before tone mapping, without the sun disc and clouds). */
  radiance(dir: THREE.Vector3, out: THREE.Color): THREE.Color {
    const fex = this.transmittance(dir, out);
    const fx = [fex.r, fex.g, fex.b];
    const cosTheta = dir.x * this.sun.x + dir.y * this.sun.y + dir.z * this.sun.z;
    const rc = cosTheta * 0.5 + 0.5;
    const rPhase = THREE_OVER_SIXTEENPI * (1 + rc * rc);
    const g = this.params.mieDirectionalG;
    const g2 = g * g;
    const mPhase = ONE_OVER_FOURPI * ((1 - g2) / Math.pow(1 - 2 * g * cosTheta + g2, 1.5));
    const mixK = Math.min(Math.max(Math.pow(1 - this.sun.y, 5), 0), 1);
    const res = [0, 0, 0];
    const add = [0, 0.0003, 0.00075];
    for (let i = 0; i < 3; i++) {
      const ratio = (this.betaR[i] * rPhase + this.betaM[i] * mPhase) / (this.betaR[i] + this.betaM[i]);
      let lin = Math.pow(this.sunE * ratio * (1 - fx[i]), 1.5);
      lin *= 1 + (Math.pow(this.sunE * ratio * fx[i], 0.5) - 1) * mixK;
      res[i] = (lin + 0.1 * fx[i]) * 0.04 + add[i];
    }
    return out.setRGB(res[0], res[1], res[2]);
  }

  /**
   * Cosine-weighted irradiance from the sky hemisphere around `normal` (sky below the horizon is
   * replaced by `groundRadiance`). Numerical quadrature, ~2k evaluations.
   */
  irradiance(normal: THREE.Vector3, groundRadiance: THREE.Color, out: THREE.Color): THREE.Color {
    const nTheta = 24;
    const nPhi = 48;
    const d = new THREE.Vector3();
    const c = new THREE.Color();
    let r = 0;
    let gg = 0;
    let b = 0;
    // Integrate over the full sphere with weight max(dot(n, d), 0).
    for (let i = 0; i < nTheta; i++) {
      const th = ((i + 0.5) / nTheta) * Math.PI; // polar from +Y
      const st = Math.sin(th);
      const ct = Math.cos(th);
      const dw = (Math.PI / nTheta) * ((2 * Math.PI) / nPhi) * st;
      for (let j = 0; j < nPhi; j++) {
        const ph = ((j + 0.5) / nPhi) * 2 * Math.PI;
        d.set(st * Math.cos(ph), ct, st * Math.sin(ph));
        const w = d.dot(normal);
        if (w <= 0) continue;
        if (d.y >= 0) this.radiance(d, c);
        else c.copy(groundRadiance);
        r += c.r * w * dw;
        gg += c.g * w * dw;
        b += c.b * w * dw;
      }
    }
    return out.setRGB(r, gg, b);
  }
}

/** Rec. 709 luminance of a linear colour. */
export function luminance(c: THREE.Color): number {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}
