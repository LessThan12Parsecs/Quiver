/**
 * Renders OCEAN_GLSL's oceanSurface() for a set of rest points into a float render target and
 * compares the read-back values with the CPU OceanModel. Result is exposed on window.__gpuCheck
 * (consumed by tools/runGpuCheck.ts) and printed into the page.
 */
import * as THREE from 'three';
import { OceanModel, createSwellEval, createWindEval, wrapPi } from '../../src/ocean/waveModel';
import { OCEAN_GLSL, OceanShaderData } from '../../src/ocean/waveGLSL';

interface Check {
  ok: boolean;
  samples: number;
  maxErr: Record<string, number>;
  worst: Record<string, { x0: number; z0: number; t: number; cpu: number; gpu: number }>;
  error?: string;
}

const W = 64;

function restPoints(): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  // surf zone (dense), bar edges, offshore, shore, beyond the field, far ocean
  for (let i = 0; i < 1200; i++) pts.push([-80 + (i % 60) * 2.3, -140 + Math.floor(i / 60) * 14.1]);
  for (let i = 0; i < 400; i++) pts.push([-60 + (i % 40) * 0.37, -5 + Math.floor(i / 40) * 1.13]);
  for (let i = 0; i < 200; i++) pts.push([-1500 + i * 9.3, -400 + i * 4.1]);
  for (let i = 0; i < 200; i++) pts.push([80 + (i % 20) * 3.1, -300 + Math.floor(i / 20) * 61]);
  for (let i = 0; i < 100; i++) pts.push([-3000 + i * 61, 500 + i * 37]);
  return pts;
}

async function run(): Promise<Check> {
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setSize(W, W);
  const model = new OceanModel();
  const data = new OceanShaderData(model);
  const pts = restPoints();
  const n = pts.length;
  const H = Math.ceil(n / W);
  const rest = new Float32Array(n * 2);
  const index = new Float32Array(n);
  pts.forEach(([x, z], i) => {
    rest[i * 2] = x;
    rest[i * 2 + 1] = z;
    index[i] = i;
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geo.setAttribute('rest', new THREE.BufferAttribute(rest, 2));
  geo.setAttribute('pindex', new THREE.BufferAttribute(index, 1));

  const makeMat = (body: string): THREE.ShaderMaterial =>
    new THREE.ShaderMaterial({
      uniforms: { ...data.uniforms },
      vertexShader: `${OCEAN_GLSL}
        attribute vec2 rest;
        attribute float pindex;
        varying vec4 vOut;
        void main() {
          OceanPoint o = oceanSurface(rest);
          ${body}
          float px = mod(pindex, ${W}.0);
          float py = floor(pindex / ${W}.0);
          gl_Position = vec4((px + 0.5) / ${W}.0 * 2.0 - 1.0, (py + 0.5) / ${H}.0 * 2.0 - 1.0, 0.0, 1.0);
          gl_PointSize = 1.0;
        }`,
      fragmentShader: `varying vec4 vOut; void main() { gl_FragColor = vOut; }`,
    });
  const passes = [
    makeMat('vOut = vec4(o.position, o.swell.breaking);'),
    makeMat('vOut = vec4(o.swell.vel, o.swell.ratio, o.swell.psi);'),
    makeMat('vOut = vec4(o.swell.depth, o.swell.height, o.swell.skew, o.swell.peaking);'),
  ];
  const rt = new THREE.WebGLRenderTarget(W, H, { type: THREE.FloatType, format: THREE.RGBAFormat });
  const scene = new THREE.Scene();
  const points = new THREE.Points(geo, passes[0]);
  points.frustumCulled = false;
  scene.add(points);
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const buf = new Float32Array(W * H * 4);

  const keys = ['x', 'y', 'z', 'breaking', 'velX', 'velZ', 'ratio', 'psi', 'depth', 'height', 'skew', 'peaking'];
  const maxErr: Record<string, number> = {};
  const worst: Check['worst'] = {};
  keys.forEach((k) => (maxErr[k] = 0));
  const se = createSwellEval();
  const we = createWindEval();
  let samples = 0;

  for (const t of [0, 37.3, 67.4, 123.9, 1000.25, 4321.7]) {
    model.setTime(t);
    data.update(model);
    const gpu: Float32Array[] = [];
    for (const mat of passes) {
      points.material = mat;
      renderer.setRenderTarget(rt);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      renderer.render(scene, cam);
      renderer.readRenderTargetPixels(rt, 0, 0, W, H, buf);
      gpu.push(buf.slice());
    }
    renderer.setRenderTarget(null);
    for (let i = 0; i < n; i++) {
      const [x0, z0] = pts[i];
      model.evalWind(x0, z0, we);
      const px = x0 + we.dx;
      const pz = z0 + we.dz;
      model.evalSwell(px, pz, se);
      const cpu: Record<string, number> = {
        x: px, y: we.dy + se.eta, z: pz, breaking: se.breaking,
        velX: se.ux, velZ: se.uz, ratio: Math.min(se.ratio, 50), psi: se.domPsi,
        depth: se.depth, height: se.height, skew: se.skew, peaking: se.peaking,
      };
      const g = (p: number, c: number): number => gpu[p][i * 4 + c];
      const gv: Record<string, number> = {
        x: g(0, 0), y: g(0, 1), z: g(0, 2), breaking: g(0, 3),
        velX: g(1, 0), velZ: g(1, 1), ratio: Math.min(g(1, 2), 50), psi: g(1, 3),
        depth: g(2, 0), height: g(2, 1), skew: g(2, 2), peaking: g(2, 3),
      };
      for (const k of keys) {
        let e = Math.abs(cpu[k] - gv[k]);
        if (k === 'psi') e = Math.abs(wrapPi(cpu[k] - gv[k])) * Math.min(1, se.height); // phase only matters with height
        if (k === 'ratio') e /= Math.max(1, Math.abs(cpu[k]));
        if (!(e <= maxErr[k])) {
          maxErr[k] = Number.isFinite(e) ? e : Infinity;
          worst[k] = { x0, z0, t, cpu: cpu[k], gpu: gv[k] };
        }
      }
      samples++;
    }
  }
  const tol: Record<string, number> = {
    x: 2e-3, y: 5e-3, z: 2e-3, breaking: 2e-2, velX: 5e-2, velZ: 5e-2, ratio: 1e-3,
    psi: 2e-2, depth: 1e-3, height: 2e-3, skew: 2e-2, peaking: 2e-2,
  };
  const ok = keys.every((k) => maxErr[k] <= tol[k]);
  return { ok, samples, maxErr, worst };
}

run()
  .then((r) => {
    (window as unknown as { __gpuCheck: Check }).__gpuCheck = r;
    document.getElementById('out')!.textContent = JSON.stringify(r, null, 2);
  })
  .catch((e: unknown) => {
    const r: Check = { ok: false, samples: 0, maxErr: {}, worst: {}, error: String((e as Error)?.stack ?? e) };
    (window as unknown as { __gpuCheck: Check }).__gpuCheck = r;
    document.getElementById('out')!.textContent = r.error!;
  });
