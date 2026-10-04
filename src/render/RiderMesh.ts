/**
 * Capsule-figure rider (no animation rig): the body is laid out every frame from the physics
 * RiderPose (src/physics/Rider.ts) so it always shows what the simulation is doing.
 *
 *  - prone:    lying along the stringer, chest up, head raised; hands follow the physics paddling
 *              hands (`pose.handLeft/Right`), or rest on the rails when not paddling.
 *  - popping:  key joints blend prone → standing with `pose.popProgress`.
 *  - standing: regular-footed (left foot forward): feet on the physics foot points, body sideways
 *              facing the right rail (+Z local), pelvis at the rider COM and the spine along the
 *              stance up axis (feet → COM, includes the lean); knees from two-bone IK, so the
 *              crouch (COM height) bends the legs; arms out for balance.
 *  - fallen:   treading water around the physics COM, head toward `pose.forward`; when the
 *              water height at the rider is given, the figure is lifted (visual only, ≤ 0.6 m)
 *              so the head sits just above the surface — the physics body (mean density
 *              990 kg/m³) floats ~97 % submerged and would be hidden by the opaque water.
 *
 * Limbs are capsules between joints with fixed bone lengths (scaled by `rider.scale`), so a
 * pose that cannot be reached shows a straightened limb instead of a stretched one.
 * All inputs are world space; `object3d` stays at the origin.
 *
 * Draw order: the materials are in the transparent queue (opacity 1, depth write on) with
 * renderOrder RIDER_RENDER_ORDER, i.e. after the board's underwater pass (BoardMesh, 20): that
 * pass must not see the rider in the depth buffer, otherwise it would paint the submerged board
 * over the rider's body.
 */
import * as THREE from 'three';
import type { RiderPose } from '../physics/Rider';

export interface RiderMeshOptions {
  suitColor?: number;
  accentColor?: number;
  skinColor?: number;
  hairColor?: number;
  castShadow?: boolean;
}

/** Bone lengths and radii for the 75 kg reference rider (m). */
const B = {
  torsoR: 0.15,
  torsoLen: 0.52,
  pelvisR: 0.105,
  hipHalf: 0.09,
  neckR: 0.05,
  headR: 0.105,
  thigh: 0.44,
  thighR: 0.07,
  shin: 0.43,
  shinR: 0.052,
  footR: 0.042,
  upperArm: 0.29,
  upperArmR: 0.047,
  forearm: 0.27,
  forearmR: 0.04,
  handR: 0.045,
  shoulderHalf: 0.19,
  ankleHeight: 0.075,
};

const Y = new THREE.Vector3(0, 1, 0);
export const RIDER_RENDER_ORDER = 30;

/** A capsule placed between two points (hemisphere centres at the ends). */
class Bone {
  readonly mesh: THREE.Mesh;
  constructor(
    material: THREE.Material,
    private readonly radius: number,
    private readonly length: number,
    castShadow: boolean,
  ) {
    this.mesh = new THREE.Mesh(new THREE.CapsuleGeometry(radius, length, 6, 12), material);
    this.mesh.castShadow = castShadow;
    this.mesh.receiveShadow = true;
    this.mesh.renderOrder = RIDER_RENDER_ORDER;
  }

  /** Place between a and b; `s` = rider scale (radius), `flat` = depth scale along `facing`. */
  place(a: THREE.Vector3, b: THREE.Vector3, s: number, facing?: THREE.Vector3, flat = 1): void {
    const m = this.mesh;
    m.position.addVectors(a, b).multiplyScalar(0.5);
    _d.subVectors(b, a);
    const len = _d.length();
    if (len < 1e-6) _d.copy(Y);
    else _d.multiplyScalar(1 / len);
    if (facing) {
      // full basis: y along the bone, z toward `facing` (orthogonalised)
      _z.copy(facing).addScaledVector(_d, -facing.dot(_d));
      if (_z.lengthSq() < 1e-8) _z.set(0, 0, 1).addScaledVector(_d, -_d.z);
      _z.normalize();
      _x.crossVectors(_d, _z);
      _m.makeBasis(_x, _d, _z);
      m.quaternion.setFromRotationMatrix(_m);
    } else {
      m.quaternion.setFromUnitVectors(Y, _d);
    }
    const sy = this.length > 0 ? Math.max(len, 1e-3) / this.length : s;
    m.scale.set(s, sy, s * flat);
  }

  get r(): number {
    return this.radius;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
  }
}

const _d = new THREE.Vector3();
const _x = new THREE.Vector3();
const _z = new THREE.Vector3();
const _m = new THREE.Matrix4();

/** Joint set (world space). */
interface Joints {
  pelvis: THREE.Vector3;
  neck: THREE.Vector3;
  head: THREE.Vector3;
  shoulderL: THREE.Vector3;
  shoulderR: THREE.Vector3;
  hipL: THREE.Vector3;
  hipR: THREE.Vector3;
  ankleL: THREE.Vector3;
  ankleR: THREE.Vector3;
  toeL: THREE.Vector3;
  toeR: THREE.Vector3;
  heelL: THREE.Vector3;
  heelR: THREE.Vector3;
  handL: THREE.Vector3;
  handR: THREE.Vector3;
  /** Chest facing direction. */
  chest: THREE.Vector3;
  /** Gaze direction. */
  gaze: THREE.Vector3;
  /** Knee / elbow bend hints. */
  kneePoleL: THREE.Vector3;
  kneePoleR: THREE.Vector3;
  elbowPoleL: THREE.Vector3;
  elbowPoleR: THREE.Vector3;
}

function makeJoints(): Joints {
  const v = (): THREE.Vector3 => new THREE.Vector3();
  return {
    pelvis: v(), neck: v(), head: v(), shoulderL: v(), shoulderR: v(), hipL: v(), hipR: v(),
    ankleL: v(), ankleR: v(), toeL: v(), toeR: v(), heelL: v(), heelR: v(), handL: v(), handR: v(),
    chest: v(), gaze: v(), kneePoleL: v(), kneePoleR: v(), elbowPoleL: v(), elbowPoleR: v(),
  };
}

const JOINT_KEYS = Object.keys(makeJoints()) as Array<keyof Joints>;
/** Joint positions (not directions / bend hints). */
const JOINT_POINTS: Array<keyof Joints> = [
  'pelvis', 'neck', 'head', 'shoulderL', 'shoulderR', 'hipL', 'hipR', 'ankleL', 'ankleR',
  'toeL', 'toeR', 'heelL', 'heelR', 'handL', 'handR',
];

function lerpJoints(a: Joints, b: Joints, t: number, out: Joints): void {
  for (const k of JOINT_KEYS) out[k].lerpVectors(a[k], b[k], t);
  out.chest.normalize();
  out.gaze.normalize();
}

/**
 * Two-bone IK: root a, target c, bone lengths l1 (a→b) and l2 (b→c), bend toward `pole`.
 * Writes the middle joint to `outB` and the reachable end point to `outC`.
 */
export function solveTwoBone(
  a: THREE.Vector3,
  c: THREE.Vector3,
  l1: number,
  l2: number,
  pole: THREE.Vector3,
  outB: THREE.Vector3,
  outC: THREE.Vector3,
): void {
  _ik.subVectors(c, a);
  let dist = _ik.length();
  if (dist < 1e-6) _ik.copy(pole).normalize();
  else _ik.multiplyScalar(1 / dist);
  dist = Math.min(Math.max(dist, Math.abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4);
  const x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(l1 * l1 - x * x, 0));
  _ib.copy(pole).addScaledVector(_ik, -pole.dot(_ik));
  if (_ib.lengthSq() < 1e-10) _ib.set(_ik.y, -_ik.x, 0);
  _ib.normalize();
  outB.copy(a).addScaledVector(_ik, x).addScaledVector(_ib, h);
  outC.copy(a).addScaledVector(_ik, dist);
}
const _ik = new THREE.Vector3();
const _ib = new THREE.Vector3();

export class RiderMesh {
  readonly object3d = new THREE.Group();
  private readonly materials: THREE.Material[] = [];
  private readonly bones: Bone[] = [];
  private readonly torso: Bone;
  private readonly pelvis: Bone;
  private readonly neck: Bone;
  private readonly thighL: Bone;
  private readonly thighR: Bone;
  private readonly shinL: Bone;
  private readonly shinR: Bone;
  private readonly footL: Bone;
  private readonly footR: Bone;
  private readonly upperL: Bone;
  private readonly upperR: Bone;
  private readonly foreL: Bone;
  private readonly foreR: Bone;
  private readonly head: THREE.Mesh;
  private readonly hair: THREE.Mesh;
  private readonly handLm: THREE.Mesh;
  private readonly handRm: THREE.Mesh;
  private readonly sphere: THREE.SphereGeometry;
  private readonly hairGeo: THREE.SphereGeometry;

  // scratch
  private readonly prone = makeJoints();
  private readonly stand = makeJoints();
  private readonly j = makeJoints();
  private readonly N = new THREE.Vector3();
  private readonly U = new THREE.Vector3();
  private readonly R = new THREE.Vector3();
  private readonly BU = new THREE.Vector3();
  private readonly F = new THREE.Vector3();
  private readonly L = new THREE.Vector3();
  /** Stance nose direction (⟂ stance up), torso axis, hip axis, toe direction. */
  private readonly SN = new THREE.Vector3();
  private readonly TD = new THREE.Vector3();
  private readonly HA = new THREE.Vector3();
  private readonly T = new THREE.Vector3();
  private readonly t0 = new THREE.Vector3();
  private readonly t1 = new THREE.Vector3();
  private readonly t2 = new THREE.Vector3();
  private readonly knee = new THREE.Vector3();
  private readonly end = new THREE.Vector3();
  private readonly q = new THREE.Quaternion();

  constructor(opts: RiderMeshOptions = {}) {
    this.object3d.name = 'rider';
    const cast = opts.castShadow ?? true;
    const suit = new THREE.MeshStandardMaterial({ name: 'wetsuit', color: opts.suitColor ?? 0x15181c, roughness: 0.62, metalness: 0 });
    const accent = new THREE.MeshStandardMaterial({ name: 'wetsuit-panel', color: opts.accentColor ?? 0x1f4f6e, roughness: 0.55 });
    const skin = new THREE.MeshStandardMaterial({ name: 'skin', color: opts.skinColor ?? 0xb57a58, roughness: 0.58 });
    const hair = new THREE.MeshStandardMaterial({ name: 'hair', color: opts.hairColor ?? 0x2a1d14, roughness: 0.8 });
    this.materials.push(suit, accent, skin, hair);
    for (const m of this.materials) {
      m.transparent = true; // drawn after the board's underwater pass (see header)
      m.opacity = 1;
    }
    const bone = (mat: THREE.Material, r: number, len: number): Bone => {
      const b = new Bone(mat, r, len, cast);
      this.bones.push(b);
      this.object3d.add(b.mesh);
      return b;
    };
    this.torso = bone(accent, B.torsoR, B.torsoLen - 0.18);
    this.pelvis = bone(suit, B.pelvisR, 2 * B.hipHalf);
    this.neck = bone(skin, B.neckR, 0.08);
    this.thighL = bone(suit, B.thighR, B.thigh);
    this.thighR = bone(suit, B.thighR, B.thigh);
    this.shinL = bone(suit, B.shinR, B.shin);
    this.shinR = bone(suit, B.shinR, B.shin);
    this.footL = bone(skin, B.footR, 0.17);
    this.footR = bone(skin, B.footR, 0.17);
    this.upperL = bone(suit, B.upperArmR, B.upperArm);
    this.upperR = bone(suit, B.upperArmR, B.upperArm);
    this.foreL = bone(suit, B.forearmR, B.forearm);
    this.foreR = bone(suit, B.forearmR, B.forearm);
    this.sphere = new THREE.SphereGeometry(1, 20, 14);
    this.hairGeo = new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, 0, Math.PI * 0.55);
    const ball = (mat: THREE.Material, geo: THREE.BufferGeometry): THREE.Mesh => {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = cast;
      m.receiveShadow = true;
      m.renderOrder = RIDER_RENDER_ORDER;
      this.object3d.add(m);
      return m;
    };
    this.head = ball(skin, this.sphere);
    this.hair = ball(hair, this.hairGeo);
    this.handLm = ball(skin, this.sphere);
    this.handRm = ball(skin, this.sphere);
  }

  /**
   * Lay out the figure.
   * @param pose        physics pose (world; may be an interpolated copy)
   * @param boardQuat   board orientation (local → world), for the deck frame
   * @param scale       rider.scale (body size relative to 75 kg)
   * @param paddling    true while the arms are stroking (else the prone hands rest on the rails)
   * @param time        seconds, for the floating idle motion
   * @param waterHeight water surface height at the rider (fallen: keeps the head out)
   */
  update(pose: RiderPose, boardQuat: THREE.Quaternion, scale: number, paddling: boolean, time: number, waterHeight?: number): void {
    const s = scale;
    // board frame: nose N, deck normal BU, right rail R
    this.N.set(1, 0, 0).applyQuaternion(boardQuat);
    this.BU.set(0, 1, 0).applyQuaternion(boardQuat);
    this.R.set(0, 0, 1).applyQuaternion(boardQuat);
    if (pose.stance === 'fallen') {
      this.layoutFloating(pose, s, time, this.j, waterHeight);
    } else {
      const w = pose.stance === 'standing' ? 1 : pose.stance === 'popping' ? pose.popProgress : 0;
      if (w < 1) this.layoutProne(pose, s, paddling, this.prone);
      if (w > 0) this.layoutStanding(pose, s, this.stand);
      if (w <= 0) copyJoints(this.prone, this.j);
      else if (w >= 1) copyJoints(this.stand, this.j);
      else lerpJoints(this.prone, this.stand, w, this.j);
    }
    this.apply(this.j, s);
  }

  set visible(v: boolean) {
    this.object3d.visible = v;
  }

  dispose(): void {
    this.object3d.removeFromParent();
    for (const b of this.bones) b.dispose();
    this.sphere.dispose();
    this.hairGeo.dispose();
    for (const m of this.materials) m.dispose();
  }

  // -------------------------------------------------------------------------------------------

  /** Lying on the deck, chest raised; COM ≈ 12 cm above the deck. */
  private layoutProne(pose: RiderPose, s: number, paddling: boolean, j: Joints): void {
    const N = this.N;
    const BU = this.BU;
    const R = this.R;
    const c = pose.com;
    j.pelvis.copy(c).addScaledVector(N, -0.06 * s).addScaledVector(BU, -0.01 * s);
    j.neck.copy(c).addScaledVector(N, 0.6 * s).addScaledVector(BU, 0.07 * s);
    j.head.copy(j.neck).addScaledVector(N, 0.12 * s).addScaledVector(BU, 0.1 * s);
    j.chest.copy(BU).negate();
    j.gaze.copy(N).addScaledVector(BU, 0.25).normalize();
    const sh = this.t0.copy(j.neck).addScaledVector(N, -0.07 * s);
    j.shoulderL.copy(sh).addScaledVector(R, -B.shoulderHalf * s);
    j.shoulderR.copy(sh).addScaledVector(R, B.shoulderHalf * s);
    j.hipL.copy(j.pelvis).addScaledVector(R, -B.hipHalf * s);
    j.hipR.copy(j.pelvis).addScaledVector(R, B.hipHalf * s);
    // legs together along the board, lower legs lifted a little
    for (const side of [-1, 1]) {
      const ankle = side < 0 ? j.ankleL : j.ankleR;
      const toe = side < 0 ? j.toeL : j.toeR;
      const heel = side < 0 ? j.heelL : j.heelR;
      const pole = side < 0 ? j.kneePoleL : j.kneePoleR;
      ankle.copy(j.pelvis).addScaledVector(N, -0.86 * s).addScaledVector(R, side * 0.08 * s).addScaledVector(BU, 0.1 * s);
      toe.copy(ankle).addScaledVector(N, -0.15 * s).addScaledVector(BU, -0.02 * s);
      heel.copy(ankle).addScaledVector(BU, 0.06 * s).addScaledVector(N, -0.02 * s);
      pole.copy(BU).negate();
    }
    // hands: physics paddling hands, or resting on the rails beside the chest
    if (paddling) {
      j.handL.copy(pose.handLeft);
      j.handR.copy(pose.handRight);
      j.elbowPoleL.copy(BU).addScaledVector(R, -0.6).addScaledVector(N, -0.4);
      j.elbowPoleR.copy(BU).addScaledVector(R, 0.6).addScaledVector(N, -0.4);
    } else {
      j.handL.copy(c).addScaledVector(N, 0.3 * s).addScaledVector(R, -0.24 * s).addScaledVector(BU, -0.07 * s);
      j.handR.copy(c).addScaledVector(N, 0.3 * s).addScaledVector(R, 0.24 * s).addScaledVector(BU, -0.07 * s);
      j.elbowPoleL.copy(R).negate().addScaledVector(BU, 0.6);
      j.elbowPoleR.copy(R).addScaledVector(BU, 0.6);
    }
  }

  /** Sideways regular stance on the physics feet, spine along the stance up axis. */
  private layoutStanding(pose: RiderPose, s: number, j: Joints): void {
    const U = this.U.copy(pose.up).normalize();
    // facing = right rail ⟂ U; stance nose direction = U × F (the rider's left)
    const F = this.F.copy(this.R).addScaledVector(U, -this.R.dot(U)).normalize();
    const SN = this.SN.crossVectors(U, F).normalize();
    const crouch = pose.crouch;
    // hips flex with the crouch: chest goes over the toes, pelvis back
    const beta = 0.22 + 0.5 * crouch;
    const sb = Math.sin(beta);
    const cb = Math.cos(beta);
    j.pelvis.copy(pose.com).addScaledVector(U, -0.04 * s).addScaledVector(F, -0.12 * sb * s);
    const TD = this.TD.copy(U).multiplyScalar(cb).addScaledVector(F, sb).normalize();
    j.neck.copy(j.pelvis).addScaledVector(TD, B.torsoLen * s);
    // chest opens toward the nose a little
    const gamma = 0.35;
    j.chest.copy(F).multiplyScalar(Math.cos(gamma)).addScaledVector(SN, Math.sin(gamma));
    j.chest.addScaledVector(TD, -j.chest.dot(TD)).normalize();
    // shoulder axis (rider's left) = torso axis × chest
    const Ls = this.L.crossVectors(TD, j.chest).normalize();
    j.head.copy(j.neck).addScaledVector(TD, 0.14 * s).addScaledVector(j.chest, 0.02 * s);
    j.gaze.copy(SN).multiplyScalar(0.85).addScaledVector(F, 0.35).addScaledVector(U, -0.15).normalize();
    const sh = this.t0.copy(j.neck).addScaledVector(TD, -0.07 * s);
    j.shoulderL.copy(sh).addScaledVector(Ls, B.shoulderHalf * s);
    j.shoulderR.copy(sh).addScaledVector(Ls, -B.shoulderHalf * s);
    // hip axis between the feet line and the shoulder line
    const HA = this.HA.copy(SN).multiplyScalar(Math.cos(gamma * 0.5)).addScaledVector(F, -Math.sin(gamma * 0.5)).normalize();
    j.hipL.copy(j.pelvis).addScaledVector(HA, B.hipHalf * s);
    j.hipR.copy(j.pelvis).addScaledVector(HA, -B.hipHalf * s);
    // feet: left = front foot, right = back foot; toes toward the right rail, the front foot
    // angled toward the nose
    const BU = this.BU;
    for (const side of [-1, 1]) {
      const front = side < 0;
      const footPt = front ? pose.frontFoot : pose.backFoot;
      const ankle = front ? j.ankleL : j.ankleR;
      const toe = front ? j.toeL : j.toeR;
      const heel = front ? j.heelL : j.heelR;
      const pole = front ? j.kneePoleL : j.kneePoleR;
      const ang = front ? 0.35 : 0.1;
      const T = this.T.copy(this.R).multiplyScalar(Math.cos(ang)).addScaledVector(this.N, Math.sin(ang)).normalize();
      ankle.copy(footPt).addScaledVector(BU, B.ankleHeight * s).addScaledVector(T, -0.03 * s);
      toe.copy(footPt).addScaledVector(BU, B.footR * s).addScaledVector(T, 0.15 * s);
      heel.copy(footPt).addScaledVector(BU, B.footR * s).addScaledVector(T, -0.07 * s);
      // knees over the toes, the back knee turned in toward the front foot
      pole.copy(F).addScaledVector(SN, front ? 0.1 : 0.45).normalize();
    }
    // arms out for balance: front arm toward the nose, back arm toward the tail; both follow
    // the lean toward the rail
    const lean = pose.lean;
    j.handL.copy(j.shoulderL).addScaledVector(SN, 0.4 * s).addScaledVector(U, -0.3 * s).addScaledVector(F, (0.16 + 0.25 * lean) * s);
    j.handR.copy(j.shoulderR).addScaledVector(SN, -0.3 * s).addScaledVector(U, -0.36 * s).addScaledVector(F, (0.2 + 0.25 * lean) * s);
    j.elbowPoleL.copy(U).negate().addScaledVector(F, -0.4).addScaledVector(SN, -0.3);
    j.elbowPoleR.copy(U).negate().addScaledVector(F, -0.4).addScaledVector(SN, 0.3);
  }

  /** Treading water after a wipeout: body upright-ish, head out, legs kicking, arms sculling. */
  private layoutFloating(pose: RiderPose, s: number, time: number, j: Joints, waterHeight?: number): void {
    const N = this.N.copy(pose.forward);
    N.y = 0;
    if (N.lengthSq() < 1e-8) N.set(1, 0, 0);
    N.normalize();
    const R = this.R.crossVectors(N, Y).normalize();
    // body axis leans back from vertical a little (floating on the back with the head up)
    const U = this.U.copy(Y).multiplyScalar(Math.cos(0.5)).addScaledVector(N, -Math.sin(0.5)).normalize();
    const c = pose.com;
    j.pelvis.copy(c).addScaledVector(U, -0.22 * s);
    j.neck.copy(j.pelvis).addScaledVector(U, B.torsoLen * s);
    j.head.copy(j.neck).addScaledVector(U, 0.14 * s).addScaledVector(N, 0.03 * s);
    j.chest.copy(N).addScaledVector(Y, 0.3).normalize();
    j.gaze.copy(N);
    const sh = this.t0.copy(j.neck).addScaledVector(U, -0.07 * s);
    j.shoulderL.copy(sh).addScaledVector(R, -B.shoulderHalf * s);
    j.shoulderR.copy(sh).addScaledVector(R, B.shoulderHalf * s);
    j.hipL.copy(j.pelvis).addScaledVector(R, -B.hipHalf * s);
    j.hipR.copy(j.pelvis).addScaledVector(R, B.hipHalf * s);
    // eggbeater kick
    for (const side of [-1, 1]) {
      const ph = time * 2.6 + (side < 0 ? 0 : Math.PI);
      const ankle = side < 0 ? j.ankleL : j.ankleR;
      const toe = side < 0 ? j.toeL : j.toeR;
      const heel = side < 0 ? j.heelL : j.heelR;
      const pole = side < 0 ? j.kneePoleL : j.kneePoleR;
      ankle.copy(j.pelvis).addScaledVector(U, -0.72 * s).addScaledVector(R, side * (0.2 + 0.08 * Math.cos(ph)) * s).addScaledVector(N, 0.12 * Math.sin(ph) * s);
      toe.copy(ankle).addScaledVector(N, 0.12 * s).addScaledVector(U, -0.08 * s);
      heel.copy(ankle).addScaledVector(N, -0.04 * s);
      pole.copy(N).addScaledVector(R, side * 0.5);
    }
    // sculling just under the surface
    const scull = Math.sin(time * 2.3);
    j.handL.copy(j.shoulderL).addScaledVector(R, -0.38 * s).addScaledVector(N, (0.12 + 0.12 * scull) * s).addScaledVector(Y, -0.22 * s);
    j.handR.copy(j.shoulderR).addScaledVector(R, 0.38 * s).addScaledVector(N, (0.12 - 0.12 * scull) * s).addScaledVector(Y, -0.22 * s);
    j.elbowPoleL.copy(R).negate().addScaledVector(N, -0.5);
    j.elbowPoleR.copy(R).addScaledVector(N, -0.5);
    // keep the head out when near the surface (visual only)
    if (waterHeight !== undefined && Number.isFinite(waterHeight)) {
      const lift = Math.min(Math.max(waterHeight + 0.07 * s - j.head.y, 0), 0.6);
      if (lift > 0) for (const k of JOINT_POINTS) j[k].y += lift;
    }
  }

  /** Place all meshes from a joint set (limbs via two-bone IK). */
  private apply(j: Joints, s: number): void {
    // torso (flattened toward the chest), pelvis, neck, head
    _d.subVectors(j.neck, j.pelvis).normalize();
    const a = this.t0.copy(j.pelvis).addScaledVector(_d, 0.06 * s);
    const b = this.t1.copy(j.neck).addScaledVector(_d, -0.12 * s);
    this.torso.place(a, b, s, j.chest, 0.7);
    this.pelvis.place(j.hipL, j.hipR, s, j.chest, 0.85);
    const n0 = this.t0.copy(j.neck).addScaledVector(_d, -0.02 * s);
    this.neck.place(n0, j.head, s);
    this.head.position.copy(j.head);
    this.head.scale.setScalar(B.headR * s);
    this.hair.position.copy(j.head).addScaledVector(j.gaze, -0.012 * s);
    this.q.setFromUnitVectors(Y, this.t2.subVectors(j.head, j.neck).normalize().addScaledVector(j.gaze, -0.6).normalize());
    this.hair.quaternion.copy(this.q);
    this.hair.scale.setScalar(B.headR * 1.06 * s);

    // legs
    this.leg(j.hipL, j.ankleL, j.kneePoleL, j.heelL, j.toeL, this.thighL, this.shinL, this.footL, s);
    this.leg(j.hipR, j.ankleR, j.kneePoleR, j.heelR, j.toeR, this.thighR, this.shinR, this.footR, s);
    // arms
    this.arm(j.shoulderL, j.handL, j.elbowPoleL, this.upperL, this.foreL, this.handLm, s);
    this.arm(j.shoulderR, j.handR, j.elbowPoleR, this.upperR, this.foreR, this.handRm, s);
  }

  private leg(hip: THREE.Vector3, ankle: THREE.Vector3, pole: THREE.Vector3, heel: THREE.Vector3, toe: THREE.Vector3, thigh: Bone, shin: Bone, foot: Bone, s: number): void {
    solveTwoBone(hip, ankle, B.thigh * s, B.shin * s, pole, this.knee, this.end);
    thigh.place(hip, this.knee, s);
    shin.place(this.knee, this.end, s);
    // the foot follows the reachable ankle
    _x.subVectors(this.end, ankle);
    this.t1.copy(heel).add(_x);
    this.t2.copy(toe).add(_x);
    foot.place(this.t1, this.t2, s);
  }

  private arm(shoulder: THREE.Vector3, hand: THREE.Vector3, pole: THREE.Vector3, upper: Bone, fore: Bone, handMesh: THREE.Mesh, s: number): void {
    solveTwoBone(shoulder, hand, B.upperArm * s, B.forearm * s, pole, this.knee, this.end);
    upper.place(shoulder, this.knee, s);
    fore.place(this.knee, this.end, s);
    handMesh.position.copy(this.end);
    handMesh.scale.setScalar(B.handR * s);
  }
}

function copyJoints(a: Joints, out: Joints): void {
  for (const k of JOINT_KEYS) out[k].copy(a[k]);
}
