/**
 * Camera modes (C cycles):
 *   follow  behind/above the board along its (smoothed) nose heading; mouse drag orbits around
 *           the board, wheel zooms, double-click recentres. Two automatic framings blend in:
 *           - look-back (prone, a set wave's crest approaching from behind): the view swings
 *             round to face the sea, so the wave you are about to catch is in frame;
 *           - riding: the camera drops low (≈ 0.7 m above the water: below the crest) into the
 *             flats in front of the face, behind the rider, looking along the line and up the
 *             face (55° from the crest line toward the face), so the slope, the lip and the line
 *             ahead read.
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
  /** True while the rider is riding a wave (follow cam: low, in front of the face). */
  riding: boolean;
  /** A set wave's crest is approaching from behind (follow cam: look back at it). */
  lookBack: boolean;
}

const FOV = { follow: 60, followRide: 66, orbit: 55, side: 48 };
/** Follow cam while riding: angle of the view from the crest line toward the face (rad), distance
 * behind the focus (m), height above the water under the camera (m). */
const RIDE_VIEW = { angle: (55 * Math.PI) / 180, dist: 5, height: 0.7 };
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
  /** Follow cam: blend toward the riding framing / the look-back (0..1). */
  private rideBlend = 0;
  private backBlend = 0;
  /** Follow cam: which way the look-back swings round (±1), the riding direction along the
   * crest (±1, hysteresis) and the smoothed riding view azimuth (rad). */
  private backSign = 1;
  private rideSide = 1;
  private rideYaw = 0;
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
        const kb = (tau: number): number => (snap ? 1 : 1 - Math.exp(-dt / tau));
        // automatic framings (see the header)
        const wx = t.waveDir.x;
        const wz = t.waveDir.z;
        const wl = Math.hypot(wx, wz);
        const riding = t.riding && wl > 0.5;
        if (riding) {
          // riding direction along the crest (tangent (−wz, wx)), with hysteresis
          const vt = (-wz * t.velocity.x + wx * t.velocity.z) / wl;
          if (Math.abs(vt) > 1.5 || this.rideBlend < 0.01) this.rideSide = vt >= 0 ? 1 : -1;
          // view: along the line, turned toward the face (seaward = −waveDir)
          const a = RIDE_VIEW.angle;
          const ex = (-wz * this.rideSide * Math.cos(a) - wx * Math.sin(a)) / wl;
          const ez = (wx * this.rideSide * Math.cos(a) - wz * Math.sin(a)) / wl;
          const target = Math.atan2(ez, ex);
          this.rideYaw = this.rideBlend < 0.01 ? target : this.rideYaw + wrapPi(target - this.rideYaw) * kb(0.5);
        }
        this.rideBlend += ((riding ? 1 : 0) - this.rideBlend) * kb(riding ? 0.7 : 0.4);
        const back = t.lookBack && t.stance === 'prone' && wl > 0.5;
        if (back && this.backBlend < 0.01) {
          // swing round on the side that is shorter from the current heading
          const sea = Math.atan2(-wz, -wx);
          this.backSign = wrapPi(sea - this.yaw) >= 0 ? 1 : -1;
        }
        this.backBlend += ((back ? 1 : 0) - this.backBlend) * kb(back ? 0.8 : 0.5);
        let yaw = this.yaw;
        if (this.backBlend > 1e-3 && wl > 0.5) {
          // face the sea, a little off-axis so the rider doesn't hide the wave
          let d = wrapPi(Math.atan2(-wz, -wx) + this.backSign * 0.25 - yaw);
          if (Math.abs(d) > 2.6 && Math.sign(d) !== this.backSign) d += this.backSign * 2 * Math.PI;
          yaw += d * this.backBlend;
        }
        if (this.rideBlend > 1e-3) yaw += wrapPi(this.rideYaw - yaw) * this.rideBlend;
        yaw += this.yawOffset;
        const rb = this.rideBlend;
        this.setFov(FOV.follow + (FOV.followRide - FOV.follow) * rb, snap, dt);
        const dist = ((standing ? 6.2 : t.stance === 'fallen' ? 6 : 4.6) * (1 - rb) + RIDE_VIEW.dist * rb) * this.zoom;
        const height = (standing ? 2.3 : 1.7) * Math.sqrt(this.zoom);
        const elev = Math.atan2(height, dist) + this.pitchOffset;
        const cy = Math.cos(yaw);
        const sy = Math.sin(yaw);
        const r = Math.hypot(dist, height);
        this.desired.set(
          this.focus.x - cy * r * Math.cos(elev),
          this.focus.y + r * Math.sin(elev),
          this.focus.z - sy * r * Math.cos(elev),
        );
        if (rb > 1e-3) {
          // riding: low over the water in front of the face (looking up at the rider and the lip),
          // never far below the rider
          const water = this.ocean.heightAt(this.desired.x, this.desired.z);
          const low = Math.max(water + RIDE_VIEW.height * Math.sqrt(this.zoom), this.focus.y - 1.6) + this.pitchOffset * dist;
          this.desired.y += (low - this.desired.y) * rb;
        }
        this.moveTo(this.desired, snap ? 1 : 1 - Math.exp(-dt / 0.1));
        this.look.copy(this.focus).addScaledVector(this.tmp.set(cy, 0, sy), (standing ? 2.2 : 1.4) + 0.6 * rb);
        this.look.y += 0.35 + 0.25 * rb;
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
