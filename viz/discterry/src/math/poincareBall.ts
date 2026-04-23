import { R_SAFE } from "./constants";
import { poincareGeodesicXY } from "./poincareGeodesic";

const EPS = 1e-14;

function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  return ax * bx + ay * by + az * bz;
}

function norm3(x: number, y: number, z: number): number {
  return Math.hypot(x, y, z);
}

/** Gyrovector Möbius addition in the unit Poincaré ball (curvature −1), x ⊕ y. */
export function mobiusBallAdd(
  x1: number,
  x2: number,
  x3: number,
  y1: number,
  y2: number,
  y3: number,
): [number, number, number] {
  const x2n = x1 * x1 + x2 * x2 + x3 * x3;
  const y2n = y1 * y1 + y2 * y2 + y3 * y3;
  const xy = x1 * y1 + x2 * y2 + x3 * y3;
  const den = 1 + 2 * xy + x2n * y2n;
  if (den < 1e-30) return [0, 0, 0];
  const inv = 1 / den;
  const c1 = 1 + 2 * xy + y2n;
  const c2 = 1 - x2n;
  return [
    (c1 * x1 + c2 * y1) * inv,
    (c1 * x2 + c2 * y2) * inv,
    (c1 * x3 + c2 * y3) * inv,
  ];
}

/** Isometry of 𝔹³ sending ``a`` to ``0``: ``x ↦ (-a) ⊕ x``. */
export function mobiusBallToOrigin(
  ax: number,
  ay: number,
  az: number,
  x1: number,
  x2: number,
  x3: number,
): [number, number, number] {
  return mobiusBallAdd(-ax, -ay, -az, x1, x2, x3);
}

export function mobiusBallArrays(
  px: Float32Array,
  py: Float32Array,
  pz: Float32Array,
  ax: number,
  ay: number,
  az: number,
  outx: Float32Array,
  outy: Float32Array,
  outz: Float32Array,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    const [ox, oy, oz] = mobiusBallToOrigin(ax, ay, az, px[i], py[i], pz[i]);
    outx[i] = ox;
    outy[i] = oy;
    outz[i] = oz;
  }
}

/** Clamp ‖p‖ below ``R_SAFE`` (Blaschke-style denominator safety for ball maps). */
export function clampP0(x: number, y: number, z: number): [number, number, number] {
  const r = norm3(x, y, z);
  if (r < 1e-12) throw new Error("focus ball coordinate is ~0; pick another protein");
  if (r >= R_SAFE) {
    const s = R_SAFE / r;
    return [x * s, y * s, z * s];
  }
  return [x, y, z];
}

/**
 * Sample the Poincaré-ball geodesic between ``a`` and ``b`` in ℝ³ (|·| < 1),
 * using the totally geodesic disk through ``0``, ``a``, ``b`` reduced to 2D.
 * Writes ``n`` points as ``out[k*3], out[k*3+1], out[k*3+2]``.
 */
export function poincareBallGeodesicSamples(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  n: number,
  out: Float32Array,
): void {
  if (n < 2) return;
  const gx = new Float32Array(n);
  const gy = new Float32Array(n);

  const na = norm3(ax, ay, az);
  const nb = norm3(bx, by, bz);

  if (na < EPS && nb < EPS) {
    for (let k = 0; k < n; k++) {
      out[k * 3] = 0;
      out[k * 3 + 1] = 0;
      out[k * 3 + 2] = 0;
    }
    return;
  }

  let e1x: number;
  let e1y: number;
  let e1z: number;
  if (na < EPS) {
    e1x = bx / nb;
    e1y = by / nb;
    e1z = bz / nb;
  } else {
    e1x = ax / na;
    e1y = ay / na;
    e1z = az / na;
  }

  const t0 = dot3(bx, by, bz, e1x, e1y, e1z);
  let e2rx = bx - t0 * e1x;
  let e2ry = by - t0 * e1y;
  let e2rz = bz - t0 * e1z;
  let ne2 = norm3(e2rx, e2ry, e2rz);

  if (ne2 < EPS) {
    /* Collinear with origin: chord = geodesic */
    for (let k = 0; k < n; k++) {
      const u = k / (n - 1 || 1);
      out[k * 3] = ax + u * (bx - ax);
      out[k * 3 + 1] = ay + u * (by - ay);
      out[k * 3 + 2] = az + u * (bz - az);
    }
    return;
  }

  e2rx /= ne2;
  e2ry /= ne2;
  e2rz /= ne2;

  const z1r = dot3(ax, ay, az, e1x, e1y, e1z);
  const z1i = dot3(ax, ay, az, e2rx, e2ry, e2rz);
  const z2r = dot3(bx, by, bz, e1x, e1y, e1z);
  const z2i = dot3(bx, by, bz, e2rx, e2ry, e2rz);

  poincareGeodesicXY(z1r, z1i, z2r, z2i, n, gx, gy);

  for (let k = 0; k < n; k++) {
    const gr = gx[k];
    const gi = gy[k];
    out[k * 3] = gr * e1x + gi * e2rx;
    out[k * 3 + 1] = gr * e1y + gi * e2ry;
    out[k * 3 + 2] = gr * e1z + gi * e2rz;
  }
}
