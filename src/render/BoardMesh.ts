/**
 * Surfboard mesh built from the physics BoardShape (src/physics/boardShape.ts), so the visible
 * board is exactly the simulated hull.
 *
 * Geometry is in the board LOCAL frame (origin at the centre of mass, +X nose, +Y deck normal,
 * +Z right rail): copy `sim.board.position` / `sim.board.quaternion` into `object3d` (or call
 * `setPose`). Construction: per station u (tail → nose) a closed cross-section ring — deck
 * (v = −1 → +1), rounded right rail, bottom (v = +1 → −1), rounded left rail — swept along the
 * board; the tail block is capped, the nose closes to a point. Rails bulge a few millimetres
 * outside the physics outline (the physics rail is a flat side face of thickness
 * `thickness(u, ±1)`). Fins come from `shape.fins` (base, span, forward, chord) as raked,
 * bevelled foils.
 *
 * Materials: glassed deck/rail/bottom (clear-coated PBR, canvas textures with stringer, logo and
 * traction pad) or a soft-top look, lit by the scene's sun + `scene.environment`.
 *
 * Underwater pass ("x-ray"): the water surface is opaque, so a submerged deck would vanish. A
 * second, transparent copy of the hull is drawn after the ocean with `depthFunc = GreaterDepth`
 * (only where the water hides the board) and faded/tinted by the depth below a local water
 * plane (`setWaterPlane`, one ocean sample per frame). Fins are excluded (they would show
 * through the board).
 */
import * as THREE from 'three';
import type { BoardShape } from '../physics/boardShape';

/** Colours/finish of a board. */
export interface BoardLook {
  deck: number;
  rail: number;
  bottom: number;
  stringer: number;
  /** Traction pad colour, or null for none. */
  pad: number | null;
  fin: number;
  /** Soft-top foam deck (matte, no clear coat). */
  soft: boolean;
  logo: string;
}

export const BOARD_LOOKS: Record<string, BoardLook> = {
  shortboard: { deck: 0xf4f1ea, rail: 0xe9604a, bottom: 0xf1eee6, stringer: 0xb48a58, pad: 0x24272b, fin: 0x1b1e23, soft: false, logo: 'QUIVER' },
  funboard: { deck: 0xf3ecd9, rail: 0x2a9fb0, bottom: 0x8fd2d6, stringer: 0xa97c4c, pad: 0x2b2f35, fin: 0x23262b, soft: false, logo: 'QUIVER' },
  longboard: { deck: 0xf0dfb4, rail: 0xd9a441, bottom: 0xe8c77d, stringer: 0x7a5530, pad: null, fin: 0x3a2a1c, soft: false, logo: 'QUIVER  LOG' },
  softtop: { deck: 0x2f74c4, rail: 0xf3f3f0, bottom: 0xf5f5f2, stringer: 0x2f74c4, pad: null, fin: 0x30343a, soft: true, logo: 'QUIVER  SOFT' },
};

export interface BoardMeshOptions {
  /** Stations along the board. */
  segmentsU?: number;
  /** Deck/bottom samples across the board. */
  segmentsV?: number;
  /** Samples around each rail. */
  railSegments?: number;
  /** Overrides the preset look (keyed by spec id). */
  look?: Partial<BoardLook>;
  /** Draw the underwater (x-ray) pass. Default true. */
  underwaterPass?: boolean;
  /** Cast/receive shadows. Default receive only. */
  castShadow?: boolean;
  receiveShadow?: boolean;
}

/** Uniforms shared by the underwater-pass materials. */
interface XrayUniforms {
  /** (slopeX, slopeZ, offset, enabled): water height = offset + slopeX·x + slopeZ·z. */
  uWaterPlane: THREE.IUniform<THREE.Vector4>;
  /** In-scattered water colour (linear, pre-exposure). */
  uWaterTint: THREE.IUniform<THREE.Color>;
  /** Extinction per metre of path through the water (RGB). */
  uWaterExt: THREE.IUniform<THREE.Vector3>;
  /** Surface transmission × light loss on the way down (RGB). */
  uWaterTransmit: THREE.IUniform<THREE.Vector3>;
}

const DECK = 0;
const RAIL = 1;
const BOTTOM = 2;

export class BoardMesh {
  /** Add to the scene; its transform is the physics body pose (local frame = board frame). */
  readonly object3d = new THREE.Group();
  readonly shape: BoardShape;
  readonly look: BoardLook;
  /** Hull geometry (deck/rail/bottom groups) and fins geometry. */
  readonly hullGeometry: THREE.BufferGeometry;
  readonly finGeometries: THREE.BufferGeometry[] = [];
  readonly materials: THREE.Material[] = [];
  private readonly textures: THREE.Texture[] = [];
  private readonly xray: THREE.Mesh | null = null;
  private readonly xrayUniforms: XrayUniforms = {
    uWaterPlane: { value: new THREE.Vector4(0, 0, -1e4, 0) },
    uWaterTint: { value: new THREE.Color(0.012, 0.09, 0.1) },
    uWaterExt: { value: new THREE.Vector3(1.6, 0.42, 0.36) },
    uWaterTransmit: { value: new THREE.Vector3(0.42, 0.62, 0.62) },
  };

  constructor(shape: BoardShape, opts: BoardMeshOptions = {}) {
    this.shape = shape;
    this.look = { ...(BOARD_LOOKS[shape.spec.id] ?? BOARD_LOOKS.funboard), ...opts.look };
    this.object3d.name = `board:${shape.spec.id}`;

    this.hullGeometry = buildHullGeometry(shape, opts.segmentsU ?? 150, opts.segmentsV ?? 28, opts.railSegments ?? 10);
    const mats = this.makeHullMaterials();
    const hull = new THREE.Mesh(this.hullGeometry, mats);
    hull.name = 'hull';
    hull.castShadow = opts.castShadow ?? false;
    hull.receiveShadow = opts.receiveShadow ?? true;
    this.object3d.add(hull);

    const finMat = new THREE.MeshPhysicalMaterial({
      name: 'fin',
      color: this.look.fin,
      roughness: 0.28,
      clearcoat: 1,
      clearcoatRoughness: 0.08,
    });
    this.materials.push(finMat);
    for (const fin of shape.fins) {
      const g = buildFinGeometry(fin.base, fin.forward, fin.span, fin.spec.depth, fin.spec.baseChord, fin.spec.side === 0 ? 0.008 : 0.0065);
      this.finGeometries.push(g);
      const m = new THREE.Mesh(g, finMat);
      m.name = `fin:${fin.spec.name}`;
      m.castShadow = opts.castShadow ?? false;
      m.receiveShadow = opts.receiveShadow ?? true;
      this.object3d.add(m);
    }

    if (opts.underwaterPass !== false) {
      const xm = mats.map((m) => this.makeXrayMaterial(m as THREE.MeshStandardMaterial));
      this.xray = new THREE.Mesh(this.hullGeometry, xm);
      this.xray.name = 'hull-underwater';
      this.xray.renderOrder = 20;
      this.object3d.add(this.xray);
    }
  }

  /** Copy a physics pose (board COM position + local→world quaternion). */
  setPose(position: THREE.Vector3, quaternion: THREE.Quaternion): void {
    this.object3d.position.copy(position);
    this.object3d.quaternion.copy(quaternion);
  }

  /**
   * Local water plane for the underwater pass: surface height `height` at world (x, z) with
   * gradient (slopeX, slopeZ). Call once per frame (e.g. from one ocean.sample at the board).
   */
  setWaterPlane(x: number, z: number, height: number, slopeX: number, slopeZ: number): void {
    // plane: y = height + slopeX (X − x) + slopeZ (Z − z) = a + slopeX X + slopeZ Z
    this.xrayUniforms.uWaterPlane.value.set(slopeX, slopeZ, height - slopeX * x - slopeZ * z, 1);
  }

  /** Underwater-pass optics: in-scattered colour, extinction per metre, surface/light loss. */
  setWaterOptics(tint: THREE.Color, extinction: THREE.Vector3, transmit?: THREE.Vector3): void {
    this.xrayUniforms.uWaterTint.value.copy(tint);
    this.xrayUniforms.uWaterExt.value.copy(extinction);
    if (transmit) this.xrayUniforms.uWaterTransmit.value.copy(transmit);
  }

  set underwaterVisible(v: boolean) {
    if (this.xray) this.xray.visible = v;
  }

  dispose(): void {
    this.object3d.removeFromParent();
    this.hullGeometry.dispose();
    for (const g of this.finGeometries) g.dispose();
    for (const m of this.materials) m.dispose();
    for (const t of this.textures) t.dispose();
  }

  // -------------------------------------------------------------------------------------------

  private makeHullMaterials(): THREE.Material[] {
    const L = this.look;
    const deckTex = this.makeTexture(false);
    const bottomTex = this.makeTexture(true);
    let deck: THREE.Material;
    let rail: THREE.Material;
    let bottom: THREE.Material;
    if (L.soft) {
      deck = new THREE.MeshStandardMaterial({ name: 'deck', map: deckTex, roughness: 0.88, metalness: 0 });
      rail = new THREE.MeshStandardMaterial({ name: 'rail', color: L.rail, roughness: 0.75 });
      bottom = new THREE.MeshPhysicalMaterial({ name: 'bottom', map: bottomTex, roughness: 0.3, clearcoat: 0.6, clearcoatRoughness: 0.15 });
    } else {
      deck = new THREE.MeshPhysicalMaterial({ name: 'deck', map: deckTex, roughness: 0.42, clearcoat: 1, clearcoatRoughness: 0.14 });
      rail = new THREE.MeshPhysicalMaterial({ name: 'rail', color: L.rail, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.06 });
      bottom = new THREE.MeshPhysicalMaterial({ name: 'bottom', map: bottomTex, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.05 });
    }
    const out: THREE.Material[] = [];
    out[DECK] = deck;
    out[RAIL] = rail;
    out[BOTTOM] = bottom;
    this.materials.push(deck, rail, bottom);
    return out;
  }

  /** Canvas texture in (u, v) space: x = u (tail → nose), y = (v + 1) / 2 (left → right rail). */
  private makeTexture(bottom: boolean): THREE.CanvasTexture {
    const W = 1024;
    const H = 256;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const g = canvas.getContext('2d')!;
    const L = this.look;
    const hex = (c: number): string => `#${c.toString(16).padStart(6, '0')}`;
    const shape = this.shape;
    const len = shape.length;
    g.fillStyle = hex(bottom ? L.bottom : L.deck);
    g.fillRect(0, 0, W, H);
    if (L.soft && !bottom) {
      // soft-top: moulded ribs across the deck and a lighter centre band
      g.fillStyle = 'rgba(255,255,255,0.08)';
      g.fillRect(0, H * 0.3, W, H * 0.4);
      g.strokeStyle = 'rgba(0,0,0,0.18)';
      g.lineWidth = 2;
      for (let x = 30; x < W * 0.62; x += 16) {
        g.beginPath();
        g.moveTo(x, H * 0.12);
        g.lineTo(x, H * 0.88);
        g.stroke();
      }
    } else {
      // stringer (≈ 6 mm wide) along the centre line
      g.fillStyle = hex(L.stringer);
      g.fillRect(0, H / 2 - 1.5, W, 3);
    }
    if (!bottom) {
      if (L.pad !== null) {
        // traction pad over the tail: grooved, with a kick at the tail end
        const u0 = 0.008 * W;
        const u1 = (0.33 / len) * W;
        g.fillStyle = hex(L.pad);
        roundRect(g, u0, H * 0.1, u1 - u0, H * 0.8, 10);
        g.fill();
        g.strokeStyle = 'rgba(255,255,255,0.07)';
        g.lineWidth = 2;
        for (let y = H * 0.16; y < H * 0.86; y += 9) {
          g.beginPath();
          g.moveTo(u0 + 12, y);
          g.lineTo(u1 - 6, y);
          g.stroke();
        }
        g.fillStyle = 'rgba(255,255,255,0.12)';
        g.fillRect(u0, H * 0.1, 14, H * 0.8);
      } else if (!L.soft) {
        // longboard: patchy wax over the standing area
        const rnd = mulberry(7);
        g.fillStyle = 'rgba(255,255,255,0.22)';
        for (let i = 0; i < 900; i++) {
          const x = (0.12 + rnd() * 0.55) * W;
          const y = (0.15 + rnd() * 0.7) * H;
          g.beginPath();
          g.arc(x, y, 1 + rnd() * 3, 0, Math.PI * 2);
          g.fill();
        }
      }
      // leash plug at the physics leash point (u = 0.02)
      g.fillStyle = '#16181b';
      g.beginPath();
      g.arc(0.02 * W + 4, H / 2, 6, 0, Math.PI * 2);
      g.fill();
    }
    // logo near the wide point, reading tail → nose
    g.save();
    g.translate(W * 0.6, H / 2);
    if (bottom) g.scale(1, -1); // seen from below the canvas is mirrored
    g.font = `bold ${L.soft && !bottom ? 34 : 30}px sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillStyle = L.soft && !bottom ? 'rgba(255,255,255,0.85)' : 'rgba(20,24,30,0.72)';
    g.fillText(L.logo, 0, bottom ? 0 : H * 0.18);
    g.restore();

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.flipY = false; // canvas y ↔ v as seen from above (+Z = right rail = down on screen)
    tex.anisotropy = 8;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    this.textures.push(tex);
    return tex;
  }

  /** Transparent copy of a hull material drawn only where the water surface hides the board. */
  private makeXrayMaterial(src: THREE.MeshStandardMaterial): THREE.Material {
    const m = src.clone();
    m.name = `${src.name}-underwater`;
    m.transparent = true;
    m.depthWrite = false;
    m.depthFunc = THREE.GreaterDepth;
    m.side = THREE.FrontSide;
    const u = this.xrayUniforms;
    m.onBeforeCompile = (shader) => {
      shader.uniforms.uWaterPlane = u.uWaterPlane;
      shader.uniforms.uWaterTint = u.uWaterTint;
      shader.uniforms.uWaterExt = u.uWaterExt;
      shader.uniforms.uWaterTransmit = u.uWaterTransmit;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vXrayWorld;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvXrayWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          '#include <common>\nvarying vec3 vXrayWorld;\nuniform vec4 uWaterPlane;\nuniform vec3 uWaterTint;\nuniform vec3 uWaterExt;\nuniform vec3 uWaterTransmit;',
        )
        .replace(
          '#include <tonemapping_fragment>',
          /* glsl */ `
  {
    // depth below the local water plane; light path = down to the board + back up to the eye
    float surf = uWaterPlane.z + uWaterPlane.x * vXrayWorld.x + uWaterPlane.y * vXrayWorld.z;
    float d = surf - vXrayWorld.y;
    vec3 V = normalize(cameraPosition - vXrayWorld);
    float path = max(d, 0.0) * (1.0 + 1.0 / max(abs(V.y), 0.2));
    vec3 T = exp(-uWaterExt * path);
    gl_FragColor.rgb = gl_FragColor.rgb * uWaterTransmit * T + uWaterTint * (1.0 - T);
    // above the plane the occluder is not the water right above the board (e.g. a crest
    // between camera and board): draw nothing there
    gl_FragColor.a = smoothstep(0.0, 0.015, d) * 0.78 * exp(-0.9 * path) * uWaterPlane.w;
  }
  #include <tonemapping_fragment>`,
        );
    };
    m.customProgramCacheKey = () => 'board-xray';
    this.materials.push(m);
    return m;
  }
}

// ---------------------------------------------------------------------------------------------
// Geometry

/** Station distribution: denser toward the nose tip and the tail. */
function stationU(t: number): number {
  const c = 0.5 - 0.5 * Math.cos(Math.PI * t);
  return 0.45 * t + 0.55 * c;
}

export function buildHullGeometry(shape: BoardShape, nU = 150, nV = 28, nR = 10): THREE.BufferGeometry {
  // ring layout: deck v −1 → +1 (nV + 1 points), right rail interior (nR − 1), bottom v +1 → −1
  // (nV + 1), left rail interior (nR − 1); closed (wraps)
  const ringDeck = nV + 1;
  const M = 2 * ringDeck + 2 * (nR - 1);
  const rows = nU + 1;
  const pos = new Float32Array((rows * M + (M + 1)) * 3);
  const uv = new Float32Array((rows * M + (M + 1)) * 2);
  const p = new THREE.Vector3();
  const rn = new THREE.Vector3();
  const vs: number[] = [];
  for (let j = 0; j <= nV; j++) vs.push(Math.sin((Math.PI / 2) * (-1 + (2 * j) / nV)));

  const put = (k: number, x: number, y: number, z: number, a: number, b: number): void => {
    pos[k * 3] = x;
    pos[k * 3 + 1] = y;
    pos[k * 3 + 2] = z;
    uv[k * 2] = a;
    uv[k * 2 + 1] = b;
  };

  for (let i = 0; i < rows; i++) {
    const u = stationU(i / nU);
    const base = i * M;
    let k = base;
    // deck, left → right
    for (let j = 0; j <= nV; j++) {
      const v = vs[j];
      shape.deckPoint(u, v, p);
      put(k++, p.x, p.y, p.z, u, (v + 1) / 2);
    }
    // right rail: deck edge → bottom edge, bulging outward
    const railFor = (side: number, reverse: boolean): void => {
      const yd = shape.deckY(u, side);
      const yb = shape.bottomY(u, side);
      const z0 = shape.z(u, side);
      const x0 = shape.x(u);
      shape.railNormal(u, side, rn);
      const h = 0.5 * (yd - yb);
      const ym = 0.5 * (yd + yb);
      const bulge = Math.min(0.55 * h, 0.5 * shape.halfWidth(u));
      for (let r = 1; r < nR; r++) {
        const s = reverse ? nR - r : r;
        const th = (Math.PI * s) / nR; // 0 deck edge → π bottom edge
        // slightly "down" rail: the apex sits below the middle
        const yy = ym + h * Math.cos(th) - 0.18 * h * Math.sin(th);
        const o = bulge * Math.pow(Math.sin(th), 0.75);
        put(k++, x0 + rn.x * o, yy, z0 + rn.z * o, u, side > 0 ? 1 : 0);
      }
    };
    railFor(1, false);
    // bottom, right → left
    for (let j = nV; j >= 0; j--) {
      const v = vs[j];
      shape.bottomPoint(u, v, p);
      put(k++, p.x, p.y, p.z, u, (v + 1) / 2);
    }
    // left rail: bottom edge → deck edge
    railFor(-1, true);
  }

  // tail cap (separate vertices: hard edge); vertex rows*M is the centre
  const capBase = rows * M;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let m = 0; m < M; m++) {
    cx += pos[m * 3];
    cy += pos[m * 3 + 1];
    cz += pos[m * 3 + 2];
  }
  put(capBase, cx / M, cy / M, cz / M, 0, 0.5);
  for (let m = 0; m < M; m++) put(capBase + 1 + m, pos[m * 3], pos[m * 3 + 1], pos[m * 3 + 2], 0, uv[m * 2 + 1]);

  // indices grouped by material: deck, rail, bottom
  const deckIdx: number[] = [];
  const railIdx: number[] = [];
  const bottomIdx: number[] = [];
  const segKind = (m: number): number => {
    if (m < nV) return DECK;
    if (m < nV + nR) return RAIL;
    if (m < nV + nR + nV) return BOTTOM;
    return RAIL;
  };
  for (let i = 0; i < nU; i++) {
    for (let m = 0; m < M; m++) {
      const a = i * M + m;
      const b = i * M + ((m + 1) % M);
      const c = (i + 1) * M + m;
      const d = (i + 1) * M + ((m + 1) % M);
      const list = segKind(m) === DECK ? deckIdx : segKind(m) === RAIL ? railIdx : bottomIdx;
      // ring order (deck → right rail → bottom → left rail) with the next station toward the
      // nose makes (a, b, c) face outward (deck: +Z × +X = +Y)
      list.push(a, b, c, b, d, c);
    }
  }
  // cap faces −X (tail end)
  for (let m = 0; m < M; m++) {
    const a = capBase + 1 + m;
    const b = capBase + 1 + ((m + 1) % M);
    railIdx.push(capBase, b, a);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  const index = [...deckIdx, ...railIdx, ...bottomIdx];
  geo.setIndex(index);
  geo.addGroup(0, deckIdx.length, DECK);
  geo.addGroup(deckIdx.length, railIdx.length, RAIL);
  geo.addGroup(deckIdx.length + railIdx.length, bottomIdx.length, BOTTOM);
  geo.computeVertexNormals();
  // the tail cap is flat: exact normal (computeVertexNormals averages it with nothing else, but
  // the centre fan can be uneven)
  const nrm = geo.getAttribute('normal') as THREE.BufferAttribute;
  for (let m = 0; m <= M; m++) nrm.setXYZ(capBase + m, -1, 0, 0);
  geo.computeBoundingSphere();
  return geo;
}

/**
 * A raked fin: outline in (chord c along `forward`, span s along `span`), extruded along the fin
 * normal with a bevel; the root starts inside the board bottom.
 */
export function buildFinGeometry(
  base: THREE.Vector3,
  forward: THREE.Vector3,
  span: THREE.Vector3,
  depth: number,
  chord: number,
  thickness: number,
): THREE.BufferGeometry {
  const s = new THREE.Shape();
  const root = -0.012;
  s.moveTo(-0.5 * chord, root);
  s.lineTo(0.5 * chord, root);
  s.lineTo(0.5 * chord, 0);
  // leading edge sweeping back to a rounded tip
  s.bezierCurveTo(0.42 * chord, 0.45 * depth, 0.05 * chord, 0.85 * depth, -0.38 * chord, depth);
  s.quadraticCurveTo(-0.5 * chord, 1.0 * depth, -0.5 * chord, 0.93 * depth);
  // trailing edge with a slight hollow
  s.quadraticCurveTo(-0.2 * chord, 0.45 * depth, -0.5 * chord, 0);
  s.lineTo(-0.5 * chord, root);
  const bevel = Math.min(0.0018, thickness * 0.3);
  const g = new THREE.ExtrudeGeometry(s, {
    depth: thickness - 2 * bevel,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
    curveSegments: 14,
  });
  g.translate(0, 0, -0.5 * (thickness - 2 * bevel));
  const normal = new THREE.Vector3().crossVectors(forward, span).normalize();
  const m = new THREE.Matrix4().makeBasis(forward, span, normal).setPosition(base);
  g.applyMatrix4(m);
  g.computeVertexNormals();
  return g;
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  g.beginPath();
  g.moveTo(x + r, y);
  g.lineTo(x + w - r, y);
  g.quadraticCurveTo(x + w, y, x + w, y + r);
  g.lineTo(x + w, y + h - r);
  g.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  g.lineTo(x + r, y + h);
  g.quadraticCurveTo(x, y + h, x, y + h - r);
  g.lineTo(x, y + r);
  g.quadraticCurveTo(x, y, x + r, y);
  g.closePath();
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
