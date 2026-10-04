import { Quaternion, Vector3 } from 'three';

/**
 * Rigid body with a diagonal inertia tensor in its local (principal) frame.
 *
 * State: centre-of-mass position, orientation (local -> world), linear velocity and angular
 * velocity (both world frame). Forces/torques accumulate until `clearForces()`.
 *
 * Integration is semi-implicit (symplectic) Euler: velocities first, then positions with the new
 * velocities. The gyroscopic term ω × Iω is integrated with the implicit midpoint rule in the
 * body frame (conserves |L| and kinetic energy; stable for bodies with very different principal
 * moments such as a surfboard). The quaternion is advanced with the exact
 * rotation for a constant ω over the step and renormalised.
 *
 * Nothing here allocates after construction.
 */
export class RigidBody {
  /** Centre of mass, world. */
  readonly position = new Vector3();
  /** Orientation: rotates local vectors into the world frame. */
  readonly quaternion = new Quaternion();
  /** Linear velocity of the centre of mass, world. */
  readonly velocity = new Vector3();
  /** Angular velocity, world. */
  readonly angularVelocity = new Vector3();
  /** Accumulated force, world. */
  readonly force = new Vector3();
  /** Accumulated torque about the centre of mass, world. */
  readonly torque = new Vector3();

  mass = 1;
  invMass = 1;
  /** Principal moments of inertia about local x, y, z (kg m²). */
  readonly inertia = new Vector3(1, 1, 1);
  readonly invInertia = new Vector3(1, 1, 1);

  /** Row-major rotation matrix local -> world. Refreshed by `updateDerived()`. */
  readonly R = new Float64Array(9);
  /** Row-major world inertia tensor R I Rᵀ. */
  readonly inertiaWorld = new Float64Array(9);
  /** Row-major world inverse inertia tensor R I⁻¹ Rᵀ. */
  readonly invInertiaWorld = new Float64Array(9);

  private readonly tmp = new Vector3();

  constructor(mass = 1, inertia?: Vector3) {
    this.setMass(mass);
    if (inertia) this.setInertia(inertia.x, inertia.y, inertia.z);
    this.updateDerived();
  }

  setMass(m: number): void {
    this.mass = m;
    this.invMass = m > 0 ? 1 / m : 0;
  }

  setInertia(ix: number, iy: number, iz: number): void {
    this.inertia.set(ix, iy, iz);
    this.invInertia.set(ix > 0 ? 1 / ix : 0, iy > 0 ? 1 / iy : 0, iz > 0 ? 1 / iz : 0);
    this.updateDerived();
  }

  /** Recompute the rotation matrix and world inertia tensors from the quaternion. */
  updateDerived(): void {
    const q = this.quaternion;
    const x = q.x, y = q.y, z = q.z, w = q.w;
    const R = this.R;
    R[0] = 1 - 2 * (y * y + z * z);
    R[1] = 2 * (x * y - z * w);
    R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w);
    R[4] = 1 - 2 * (x * x + z * z);
    R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);
    R[7] = 2 * (y * z + x * w);
    R[8] = 1 - 2 * (x * x + y * y);
    const I = this.inertia;
    const Ii = this.invInertia;
    for (let i = 0; i < 3; i++) {
      for (let j = i; j < 3; j++) {
        const a = R[i * 3] * R[j * 3];
        const b = R[i * 3 + 1] * R[j * 3 + 1];
        const c = R[i * 3 + 2] * R[j * 3 + 2];
        const v = a * I.x + b * I.y + c * I.z;
        const vi = a * Ii.x + b * Ii.y + c * Ii.z;
        this.inertiaWorld[i * 3 + j] = v;
        this.inertiaWorld[j * 3 + i] = v;
        this.invInertiaWorld[i * 3 + j] = vi;
        this.invInertiaWorld[j * 3 + i] = vi;
      }
    }
  }

  clearForces(): void {
    this.force.set(0, 0, 0);
    this.torque.set(0, 0, 0);
  }

  addForce(f: Vector3): void {
    this.force.add(f);
  }

  addTorque(t: Vector3): void {
    this.torque.add(t);
  }

  /** Add a world force acting at a world point (adds the torque about the centre of mass). */
  addForceAtPoint(f: Vector3, p: Vector3): void {
    this.force.add(f);
    const rx = p.x - this.position.x;
    const ry = p.y - this.position.y;
    const rz = p.z - this.position.z;
    this.torque.x += ry * f.z - rz * f.y;
    this.torque.y += rz * f.x - rx * f.z;
    this.torque.z += rx * f.y - ry * f.x;
  }

  /** Velocity of the material point currently at world position p. */
  pointVelocity(p: Vector3, out: Vector3): Vector3 {
    const w = this.angularVelocity;
    const rx = p.x - this.position.x;
    const ry = p.y - this.position.y;
    const rz = p.z - this.position.z;
    return out.set(
      this.velocity.x + w.y * rz - w.z * ry,
      this.velocity.y + w.z * rx - w.x * rz,
      this.velocity.z + w.x * ry - w.y * rx,
    );
  }

  /** Local direction -> world direction (uses the cached matrix). */
  localDirToWorld(d: Vector3, out: Vector3): Vector3 {
    const R = this.R;
    const x = d.x, y = d.y, z = d.z;
    return out.set(R[0] * x + R[1] * y + R[2] * z, R[3] * x + R[4] * y + R[5] * z, R[6] * x + R[7] * y + R[8] * z);
  }

  /** World direction -> local direction. */
  worldDirToLocal(d: Vector3, out: Vector3): Vector3 {
    const R = this.R;
    const x = d.x, y = d.y, z = d.z;
    return out.set(R[0] * x + R[3] * y + R[6] * z, R[1] * x + R[4] * y + R[7] * z, R[2] * x + R[5] * y + R[8] * z);
  }

  /** Local point (relative to the centre of mass) -> world point. */
  localToWorld(p: Vector3, out: Vector3): Vector3 {
    this.localDirToWorld(p, out);
    return out.add(this.position);
  }

  /** World point -> local point. */
  worldToLocal(p: Vector3, out: Vector3): Vector3 {
    this.tmp.subVectors(p, this.position);
    return this.worldDirToLocal(this.tmp, out);
  }

  /**
   * Gyroscopic update of the angular velocity (no external torque): Euler's equations
   * I ω̇ = −ω × Iω in the body frame, integrated with the implicit midpoint rule, which conserves
   * both |L|² and the kinetic energy exactly and is stable for very unequal principal moments.
   * Solved with three Newton iterations on the midpoint m = (ω + ω')/2:
   *   G(m) = 2 I (m − ω) + h m × I m = 0,   J = 2 I + h (skew(m) I − skew(I m)).
   */
  applyGyroscopic(dt: number): void {
    const R = this.R;
    const w = this.angularVelocity;
    // body-frame angular velocity
    const wx = R[0] * w.x + R[3] * w.y + R[6] * w.z;
    const wy = R[1] * w.x + R[4] * w.y + R[7] * w.z;
    const wz = R[2] * w.x + R[5] * w.y + R[8] * w.z;
    const Ix = this.inertia.x, Iy = this.inertia.y, Iz = this.inertia.z;
    let mx = wx, my = wy, mz = wz;
    for (let it = 0; it < 3; it++) {
      const Lx = Ix * mx, Ly = Iy * my, Lz = Iz * mz;
      const gx = 2 * Ix * (mx - wx) + dt * (my * Lz - mz * Ly);
      const gy = 2 * Iy * (my - wy) + dt * (mz * Lx - mx * Lz);
      const gz = 2 * Iz * (mz - wz) + dt * (mx * Ly - my * Lx);
      const j00 = 2 * Ix, j01 = dt * (-mz * Iy + Lz), j02 = dt * (my * Iz - Ly);
      const j10 = dt * (mz * Ix - Lz), j11 = 2 * Iy, j12 = dt * (-mx * Iz + Lx);
      const j20 = dt * (-my * Ix + Ly), j21 = dt * (mx * Iy - Lx), j22 = 2 * Iz;
      const c00 = j11 * j22 - j12 * j21;
      const c01 = j02 * j21 - j01 * j22;
      const c02 = j01 * j12 - j02 * j11;
      const det = j00 * c00 + j10 * c01 + j20 * c02;
      if (Math.abs(det) < 1e-300) break;
      const inv = 1 / det;
      const c10 = j12 * j20 - j10 * j22;
      const c11 = j00 * j22 - j02 * j20;
      const c12 = j02 * j10 - j00 * j12;
      const c20 = j10 * j21 - j11 * j20;
      const c21 = j01 * j20 - j00 * j21;
      const c22 = j00 * j11 - j01 * j10;
      mx -= inv * (c00 * gx + c01 * gy + c02 * gz);
      my -= inv * (c10 * gx + c11 * gy + c12 * gz);
      mz -= inv * (c20 * gx + c21 * gy + c22 * gz);
    }
    const nx = 2 * mx - wx;
    const ny = 2 * my - wy;
    const nz = 2 * mz - wz;
    w.set(R[0] * nx + R[1] * ny + R[2] * nz, R[3] * nx + R[4] * ny + R[5] * nz, R[6] * nx + R[7] * ny + R[8] * nz);
  }

  /** v += F/m dt; ω: gyroscopic step, then ω += I⁻¹ τ dt. */
  integrateVelocity(dt: number): void {
    this.velocity.addScaledVector(this.force, this.invMass * dt);
    this.applyGyroscopic(dt);
    const Ii = this.invInertiaWorld;
    const t = this.torque;
    const w = this.angularVelocity;
    w.x += dt * (Ii[0] * t.x + Ii[1] * t.y + Ii[2] * t.z);
    w.y += dt * (Ii[3] * t.x + Ii[4] * t.y + Ii[5] * t.z);
    w.z += dt * (Ii[6] * t.x + Ii[7] * t.y + Ii[8] * t.z);
  }

  /** x += v dt; q advanced by the exact rotation of ω over dt, renormalised; derived data refreshed. */
  integratePosition(dt: number): void {
    this.position.addScaledVector(this.velocity, dt);
    const w = this.angularVelocity;
    const wl = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    if (wl > 1e-12) {
      const half = 0.5 * wl * dt;
      const s = Math.sin(half) / wl;
      const dx = w.x * s, dy = w.y * s, dz = w.z * s, dw = Math.cos(half);
      const q = this.quaternion;
      const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
      // q = dq * q
      q.set(
        dw * qx + dx * qw + dy * qz - dz * qy,
        dw * qy + dy * qw + dz * qx - dx * qz,
        dw * qz + dz * qw + dx * qy - dy * qx,
        dw * qw - dx * qx - dy * qy - dz * qz,
      );
      q.normalize();
    }
    this.updateDerived();
  }

  /** Semi-implicit Euler step using the accumulated force and torque. */
  integrate(dt: number): void {
    this.integrateVelocity(dt);
    this.integratePosition(dt);
  }

  /** Angular momentum about the centre of mass, world. */
  angularMomentum(out: Vector3): Vector3 {
    const I = this.inertiaWorld;
    const w = this.angularVelocity;
    return out.set(
      I[0] * w.x + I[1] * w.y + I[2] * w.z,
      I[3] * w.x + I[4] * w.y + I[5] * w.z,
      I[6] * w.x + I[7] * w.y + I[8] * w.z,
    );
  }

  /** Total kinetic energy, J. */
  kineticEnergy(): number {
    const w = this.angularVelocity;
    const I = this.inertiaWorld;
    const Lx = I[0] * w.x + I[1] * w.y + I[2] * w.z;
    const Ly = I[3] * w.x + I[4] * w.y + I[5] * w.z;
    const Lz = I[6] * w.x + I[7] * w.y + I[8] * w.z;
    return 0.5 * this.mass * this.velocity.lengthSq() + 0.5 * (w.x * Lx + w.y * Ly + w.z * Lz);
  }
}
