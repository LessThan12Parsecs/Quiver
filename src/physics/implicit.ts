/**
 * Linearly-implicit Euler step for the coupled board + rider system.
 *
 * Generalised velocity u = [v_board (3), ω_board (3, world), v_rider (3)].
 * Forces are evaluated explicitly (Q) and their stiff parts are linearised as a damping matrix
 * C = −∂Q/∂u and a stiffness matrix K = −∂Q/∂x (built from rank-1 terms). One step solves
 *
 *     (M + h C + h² K) Δu = h (Q − h K u)
 *
 * which is unconditionally stable for the linear parts. This matters because a 3–6 kg board
 * carries a 75 kg rider on stiff "legs" (relative mode ≈ 80 Hz, effective mass at the rider's
 * hips only a few hundred grams in roll) and planing pressure damping on the light board
 * (c/m ≈ 700 s⁻¹ at 6 m/s): both are unstable with explicit integration at 240 Hz.
 */
export const NDOF = 9;

export class ImplicitSystem {
  /** Generalised forces: board force, board torque about its COM, rider force (world). */
  readonly Q = new Float64Array(NDOF);
  /** Damping matrix (upper triangle used), row-major 9×9. */
  readonly C = new Float64Array(NDOF * NDOF);
  /** Stiffness matrix (upper triangle used). */
  readonly K = new Float64Array(NDOF * NDOF);
  /** Mass matrix (upper triangle used). */
  readonly M = new Float64Array(NDOF * NDOF);
  readonly du = new Float64Array(NDOF);
  private readonly A = new Float64Array(NDOF * NDOF);
  private readonly rhs = new Float64Array(NDOF);
  private readonly y = new Float64Array(NDOF);
  private readonly J = new Float64Array(NDOF);

  /** Clear forces and the damping/stiffness matrices (M is set by the caller before forces). */
  clear(): void {
    this.Q.fill(0);
    this.C.fill(0);
    this.K.fill(0);
  }

  /** Force (fx, fy, fz) on the board at offset r (world, relative to the board COM). */
  addBoardForce(fx: number, fy: number, fz: number, rx: number, ry: number, rz: number): void {
    const Q = this.Q;
    Q[0] += fx;
    Q[1] += fy;
    Q[2] += fz;
    Q[3] += ry * fz - rz * fy;
    Q[4] += rz * fx - rx * fz;
    Q[5] += rx * fy - ry * fx;
  }

  addBoardTorque(tx: number, ty: number, tz: number): void {
    this.Q[3] += tx;
    this.Q[4] += ty;
    this.Q[5] += tz;
  }

  addRiderForce(fx: number, fy: number, fz: number): void {
    this.Q[6] += fx;
    this.Q[7] += fy;
    this.Q[8] += fz;
  }

  /**
   * Damping c along unit direction n at board offset r: adds c·J Jᵀ with J = [n, r × n, 0]
   * (the velocity of the board point along n is J·u).
   */
  addBoardDamping(nx: number, ny: number, nz: number, rx: number, ry: number, rz: number, c: number): void {
    if (c > 0) addBoardRank1(this.C, nx, ny, nz, rx, ry, rz, c);
  }

  /** Stiffness k along unit direction n at board offset r (a spring on that board point): K += k·J Jᵀ. */
  addBoardStiffness(nx: number, ny: number, nz: number, rx: number, ry: number, rz: number, k: number): void {
    if (k > 0) addBoardRank1(this.K, nx, ny, nz, rx, ry, rz, k);
  }

  /** Rank-1 mass m along unit direction n at board offset r (added mass): M += m·J Jᵀ. */
  addBoardMass(nx: number, ny: number, nz: number, rx: number, ry: number, rz: number, m: number): void {
    if (m > 0) addBoardRank1(this.M, nx, ny, nz, rx, ry, rz, m);
  }

  /** Isotropic damping c on the board's angular velocity. */
  addAngularDamping(c: number): void {
    this.C[30] += c;
    this.C[40] += c;
    this.C[50] += c;
  }

  /** Isotropic damping c on the rider's velocity. */
  addRiderDamping(c: number): void {
    this.C[60] += c;
    this.C[70] += c;
    this.C[80] += c;
  }

  /** Damping kd and stiffness ks on the rider along unit axis a (rider-only: ground contact). */
  addRiderSpring(ax: number, ay: number, az: number, kd: number, ks: number): void {
    const C = this.C, K = this.K;
    C[60] += kd * ax * ax; C[61] += kd * ax * ay; C[62] += kd * ax * az;
    C[70] += kd * ay * ay; C[71] += kd * ay * az;
    C[80] += kd * az * az;
    K[60] += ks * ax * ax; K[61] += ks * ax * ay; K[62] += ks * ax * az;
    K[70] += ks * ay * ay; K[71] += ks * ay * az;
    K[80] += ks * az * az;
  }

  /**
   * Rank-1 term between the rider and a board point: the relative velocity along unit axis a,
   *   a·(v_rider − v_board − ω × r) = J·u,  J = [−a, −(r × a), a].
   * Adds kd·J Jᵀ to C and ks·J Jᵀ to K.
   */
  addLink(ax: number, ay: number, az: number, rx: number, ry: number, rz: number, kd: number, ks: number): void {
    const J = this.J;
    J[0] = -ax;
    J[1] = -ay;
    J[2] = -az;
    J[3] = -(ry * az - rz * ay);
    J[4] = -(rz * ax - rx * az);
    J[5] = -(rx * ay - ry * ax);
    J[6] = ax;
    J[7] = ay;
    J[8] = az;
    for (let i = 0; i < NDOF; i++) {
      const ji = J[i];
      if (ji === 0) continue;
      for (let j = i; j < NDOF; j++) {
        const p = ji * J[j];
        this.C[i * NDOF + j] += kd * p;
        this.K[i * NDOF + j] += ks * p;
      }
    }
  }

  /** J·u for the link Jacobian of `addLink` (relative velocity along a). */
  static linkVelocity(u: Float64Array, ax: number, ay: number, az: number, rx: number, ry: number, rz: number): number {
    const j3 = -(ry * az - rz * ay);
    const j4 = -(rz * ax - rx * az);
    const j5 = -(rx * ay - ry * ax);
    return -ax * u[0] - ay * u[1] - az * u[2] + j3 * u[3] + j4 * u[4] + j5 * u[5] + ax * u[6] + ay * u[7] + az * u[8];
  }

  /**
   * Solve (M + hC + h²K) Δu = h (Q − h K u) by Cholesky. Result in `du`.
   * M, C, K use their upper triangles.
   */
  solve(h: number, u: Float64Array): Float64Array {
    const n = NDOF;
    const A = this.A;
    const M = this.M;
    const C = this.C;
    const K = this.K;
    const rhs = this.rhs;
    const h2 = h * h;
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const k = i * n + j;
        A[k] = M[k] + h * C[k] + h2 * K[k];
      }
    }
    for (let i = 0; i < n; i++) {
      let ku = 0;
      for (let j = 0; j < n; j++) ku += (i <= j ? K[i * n + j] : K[j * n + i]) * u[j];
      rhs[i] = h * (this.Q[i] - h * ku);
    }
    // Cholesky A = L Lᵀ, L stored in the lower triangle of A (reading the upper triangle of A).
    for (let j = 0; j < n; j++) {
      let d = A[j * n + j];
      for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
      if (!(d > 0)) d = 1e-12;
      const ljj = Math.sqrt(d);
      A[j * n + j] = ljj;
      for (let i = j + 1; i < n; i++) {
        let s = A[j * n + i]; // upper (j, i) == lower (i, j) of the symmetric matrix
        for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
        A[i * n + j] = s / ljj;
      }
    }
    const y = this.y;
    for (let i = 0; i < n; i++) {
      let s = rhs[i];
      for (let k = 0; k < i; k++) s -= A[i * n + k] * y[k];
      y[i] = s / A[i * n + i];
    }
    const du = this.du;
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= A[k * n + i] * du[k];
      du[i] = s / A[i * n + i];
    }
    return du;
  }
}

/** A += s·J Jᵀ (upper triangle of the board block) with J = [n, r × n]. */
function addBoardRank1(A: Float64Array, nx: number, ny: number, nz: number, rx: number, ry: number, rz: number, s: number): void {
  const j0 = nx, j1 = ny, j2 = nz;
  const j3 = ry * nz - rz * ny;
  const j4 = rz * nx - rx * nz;
  const j5 = rx * ny - ry * nx;
  A[0] += s * j0 * j0; A[1] += s * j0 * j1; A[2] += s * j0 * j2; A[3] += s * j0 * j3; A[4] += s * j0 * j4; A[5] += s * j0 * j5;
  A[10] += s * j1 * j1; A[11] += s * j1 * j2; A[12] += s * j1 * j3; A[13] += s * j1 * j4; A[14] += s * j1 * j5;
  A[20] += s * j2 * j2; A[21] += s * j2 * j3; A[22] += s * j2 * j4; A[23] += s * j2 * j5;
  A[30] += s * j3 * j3; A[31] += s * j3 * j4; A[32] += s * j3 * j5;
  A[40] += s * j4 * j4; A[41] += s * j4 * j5;
  A[50] += s * j5 * j5;
}
