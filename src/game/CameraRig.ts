/**
 * Camera modes (C cycles):
 *   follow  behind/above the board along its (smoothed) nose heading; mouse drag orbits around
 *           the board, wheel zooms, double-click recentres.
 *   orbit   OrbitControls around the moving board (the target follows the board).
 *   beach   a filmer on the beach (x ≈ 122 m) with a telephoto lens tracking the board.
 *   side    a filmer in the water off to the side, perpendicular to the swell, lagging behind.
 * Every mode keeps the camera above the water surface (ocean.heightAt at the render time — the
 * caller sets the ocean time before `update`).
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { OceanModel } from '../ocean/waveModel';
import type { Stance } from '../physics/Rider';

export type CameraMode = 'follow' | 'orbit' | 'beach' | 'side';
export const CAMERA_MODES: CameraMode[] = ['follow', 'orbit', 'beach', 'side'];

/** What the rig tracks (interpolated render state). */
export interface CameraTarget {
  /** Board centre of mass. */
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  velocity: THREE.Vector3;
  /** Rider centre of mass. */
  riderCom: THREE.Vector3;
  stance: Stance;
  /** Horizontal unit direction the wave under the board travels in (x, 0, z). */
  waveDir: THREE.Vector3;
  /** True while the board is riding a wave (follow cam moves toward the shore side). */
  riding: boolean;
}

const FOV = { follow: 60, orbit: 55, side: 48 };
/** Minimum camera height above the water surface, m. */
const MIN_CLEARANCE = 0.45;

export class CameraRig {
  mode: CameraMode = 'follow';
  readonly controls: OrbitControls;
  /** Follow-cam zoom multiplier (wheel). */
  zoom = 1;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly ocean: OceanModel;
  private readonly dom: HTMLElement;
  private snapNext = true;
  private yaw = 0;
  private yawOffset = 0;
  private pitchOffset = 0;
  private sideSign = 1;
  /** Follow cam: smoothed lateral offset toward the shore while riding, m. */
  private shoreShift = 0;
  private readonly focus = new THREE.Vector3();
  private readonly look = new THREE.Vector3();
  private readonly desired = new THREE.Vector3();
  private readonly tmp = new THREE.Vector3();
  private readonly fwd = new THREE.Vector3();
  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  constructor(camera: THREE.PerspectiveCamera, dom: HTMLElement, ocean: OceanModel) {
    this.camera = camera;
    this.ocean = ocean;
    this.dom = dom;
    this.controls = new OrbitControls(camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxDistance = 400;
    this.controls.minDistance = 1.5;
    this.controls.enabled = false;
    dom.addEventListener('pointerdown', this.onDown);
    window.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    dom.addEventListener('wheel', this.onWheel, { passive: true });
    dom.addEventListener('dblclick', this.onDblClick);
  }

  setMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.controls.enabled = mode === 'orbit';
    if (mode === 'orbit') {
      this.controls.target.copy(this.focus);
      this.controls.update();
    }
    this.snapNext = true;
  }

  /** Next mode in CAMERA_MODES; returns it. */
  cycle(): CameraMode {
    const i = CAMERA_MODES.indexOf(this.mode);
    this.setMode(CAMERA_MODES[(i + 1) % CAMERA_MODES.length]);
    return this.mode;
  }

  /** Jump straight to the target pose on the next update (after a teleport). */
  snap(): void {
    this.snapNext = true;
  }

  /**
   * Per frame, after the ocean is set to the render time.
   * @param dt real seconds since the last frame (smoothing)
   */
  update(dt: number, t: CameraTarget): void {
    // explicit snaps (teleports, mode changes) reposition every mode; a non-finite/large dt
    // (paused rendering) only disables the smoothing
    const forced = this.snapNext;
    const snap = forced || !Number.isFinite(dt) || dt > 0.5;
    this.snapNext = false;
    const cam = this.camera;
    const standing = t.stance === 'standing' || t.stance === 'popping';
    // focus: between the board and the rider's COM
    this.desired.copy(t.position).lerp(t.riderCom, t.stance === 'fallen' ? 0.3 : 0.5);
    this.desired.y = Math.max(this.desired.y, t.position.y);
    const kf = snap ? 1 : 1 - Math.exp(-dt / 0.06);
    this.focus.lerp(this.desired, kf);
    if (snap) this.focus.copy(this.desired);

    // smoothed nose heading (horizontal)
    this.fwd.set(1, 0, 0).applyQuaternion(t.quaternion);
    const h = Math.hypot(this.fwd.x, this.fwd.z);
    if (h > 0.2) {
      const target = Math.atan2(this.fwd.z, this.fwd.x);
      if (snap) this.yaw = target;
      else {
        const d = wrapPi(target - this.yaw);
        this.yaw += d * (1 - Math.exp(-dt / 0.45));
      }
    }

    switch (this.mode) {
      case 'follow': {
        this.setFov(FOV.follow, snap, dt);
        const dist = (standing ? 6.2 : t.stance === 'fallen' ? 6 : 4.6) * this.zoom;
        const height = (standing ? 2.3 : 1.7) * Math.sqrt(this.zoom);
        const yaw = this.yaw + this.yawOffset;
        const elev = Math.atan2(height, dist) + this.pitchOffset;
        const cy = Math.cos(yaw);
        const sy = Math.sin(yaw);
        const r = Math.hypot(dist, height);
        this.desired.set(
          this.focus.x - cy * r * Math.cos(elev),
          this.focus.y + r * Math.sin(elev),
          this.focus.z - sy * r * Math.cos(elev),
        );
        // riding down the line: slide out in front of the face (toward the shore, ⟂ heading) so
        // the camera looks back up the wave instead of straight along the crest
        const wx = t.waveDir.x;
        const wz = t.waveDir.z;
        const along = wx * cy + wz * sy;
        let px = wx - along * cy;
        let pz = wz - along * sy;
        const pl = Math.hypot(px, pz);
        const shiftTarget = t.riding && pl > 0.3 ? 2.6 * this.zoom : 0;
        this.shoreShift = snap ? shiftTarget : this.shoreShift + (shiftTarget - this.shoreShift) * (1 - Math.exp(-dt / 0.8));
        if (pl > 1e-3) {
          px /= pl;
          pz /= pl;
          this.desired.x += px * this.shoreShift;
          this.desired.z += pz * this.shoreShift;
        }
        this.moveTo(this.desired, snap ? 1 : 1 - Math.exp(-dt / 0.1));
        this.look.copy(this.focus).addScaledVector(this.tmp.set(cy, 0, sy), standing ? 2.2 : 1.4);
        this.look.y += 0.35;
        break;
      }
      case 'orbit': {
        const c = this.controls;
        // carry the orbit with the board
        this.tmp.subVectors(this.focus, c.target);
        c.target.add(this.tmp);
        cam.position.add(this.tmp);
        if (forced) {
          cam.position.copy(this.focus).add(this.tmp.set(-6, 2.5, 3));
        }
        c.update();
        this.clampAboveWater();
        this.setFov(FOV.orbit, snap, dt);
        return;
      }
      case 'beach': {
        // up the beach and off to −Z of the board, so the line of sight points toward +Z, away
        // from the default sun (over the sea toward −X/−Z); telephoto framing ≈ 22 m tall
        this.desired.set(124, 4.2, THREE.MathUtils.clamp(this.focus.z * 0.85 - 55, -300, 300));
        this.moveTo(this.desired, snap ? 1 : 1 - Math.exp(-dt / 2));
        this.look.copy(this.focus);
        this.look.y += 0.6;
        const d = cam.position.distanceTo(this.look);
        const fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(11 / Math.max(d, 1))), 3, 50);
        this.setFov(fov, snap, dt);
        break;
      }
      case 'side': {
        // keep to the side the board is heading toward (along the shore), with hysteresis
        const vz = t.velocity.z;
        if (Math.abs(vz) > 1.5) this.sideSign = Math.sign(vz);
        this.desired.set(this.focus.x + 7, this.focus.y + 1.6, this.focus.z + this.sideSign * 17);
        this.desired.y = Math.max(this.desired.y, 1.4);
        this.moveTo(this.desired, snap ? 1 : 1 - Math.exp(-dt / 0.7));
        this.look.copy(this.focus);
        this.look.y += 0.3;
        this.setFov(FOV.side, snap, dt);
        break;
      }
    }
    this.clampAboveWater();
    cam.lookAt(this.look);
  }

  dispose(): void {
    this.controls.dispose();
    this.dom.removeEventListener('pointerdown', this.onDown);
    window.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    this.dom.removeEventListener('wheel', this.onWheel);
    this.dom.removeEventListener('dblclick', this.onDblClick);
  }

  // -------------------------------------------------------------------------------------------

  private moveTo(p: THREE.Vector3, k: number): void {
    this.camera.position.lerp(p, k);
  }

  private clampAboveWater(): void {
    const p = this.camera.position;
    const h = this.ocean.heightAt(p.x, p.z) + MIN_CLEARANCE;
    if (p.y < h) p.y = h;
  }

  private setFov(fov: number, snap: boolean, dt: number): void {
    const cam = this.camera;
    const f = snap ? fov : cam.fov + (fov - cam.fov) * (1 - Math.exp(-dt / 0.5));
    if (Math.abs(f - cam.fov) > 1e-4) {
      cam.fov = f;
      cam.updateProjectionMatrix();
    }
  }

  private readonly onDown = (e: PointerEvent): void => {
    if (this.mode !== 'follow' || e.button !== 0) return;
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  };
  private readonly onMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.yawOffset += dx * 0.006;
    this.pitchOffset = THREE.MathUtils.clamp(this.pitchOffset + dy * 0.004, -0.25, 1.1);
  };
  private readonly onUp = (): void => {
    this.dragging = false;
  };
  private readonly onWheel = (e: WheelEvent): void => {
    if (this.mode !== 'follow') return;
    this.zoom = THREE.MathUtils.clamp(this.zoom * Math.exp(e.deltaY * 0.001), 0.4, 4);
  };
  private readonly onDblClick = (): void => {
    this.yawOffset = 0;
    this.pitchOffset = 0;
    this.zoom = 1;
  };
}

function wrapPi(a: number): number {
  return a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
}
