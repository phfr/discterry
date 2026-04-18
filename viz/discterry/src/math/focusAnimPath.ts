import type { Complex } from "./mobius";
import { clampZ0 } from "./mobius";
import { clampP0, poincareBallGeodesicSamples } from "./poincareBall";
import { poincareGeodesicXY } from "./poincareGeodesic";

/** Samples along the disk geodesic between two z0 endpoints (reuse for all frames of one transition). */
export const Z0_GEODESIC_SAMPLES = 64;

/** Same sample count as disk z0 path; ball geodesic stores xyz triples in ``out`` (length ≥ 3n). */
export const P0_GEODESIC_SAMPLES = Z0_GEODESIC_SAMPLES;

export function easeInOutCubic(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2;
}

/** Fill gx, gy with disk geodesic from z0a to z0b (length n). */
export function fillZ0Geodesic(z0a: Complex, z0b: Complex, n: number, gx: Float32Array, gy: Float32Array): void {
  poincareGeodesicXY(z0a.re, z0a.im, z0b.re, z0b.im, n, gx, gy);
}

/** u in [0,1] along the precomputed geodesic polyline; result clamped for Möbius stability. */
export function z0AtGeodesicParameter(gx: Float32Array, gy: Float32Array, u: number): Complex {
  const n = gx.length;
  if (n < 2) return clampZ0({ re: gx[0] ?? 0, im: gy[0] ?? 0 });
  const t = Math.max(0, Math.min(1, u));
  const f = t * (n - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(n - 1, i0 + 1);
  const w = f - i0;
  const re = gx[i0] * (1 - w) + gx[i1] * w;
  const im = gy[i0] * (1 - w) + gy[i1] * w;
  return clampZ0({ re, im });
}

/** Fill ``out`` with Poincaré-ball geodesic samples from ``a`` to ``b`` (``n`` points, 3 floats each). */
export function fillP0BallGeodesic(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  n: number,
  out: Float32Array,
): void {
  poincareBallGeodesicSamples(ax, ay, az, bx, by, bz, n, out);
}

/** ``u`` in [0,1] along the precomputed ball geodesic polyline; result clamped for map stability. */
export function p0AtBallGeodesicParameter(
  out: Float32Array,
  n: number,
  u: number,
): { x: number; y: number; z: number } {
  if (n < 2) {
    const [x, y, z] = clampP0(out[0] ?? 0, out[1] ?? 0, out[2] ?? 0);
    return { x, y, z };
  }
  const t = Math.max(0, Math.min(1, u));
  const f = t * (n - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(n - 1, i0 + 1);
  const w = f - i0;
  const x = (out[i0 * 3] ?? 0) * (1 - w) + (out[i1 * 3] ?? 0) * w;
  const y = (out[i0 * 3 + 1] ?? 0) * (1 - w) + (out[i1 * 3 + 1] ?? 0) * w;
  const z = (out[i0 * 3 + 2] ?? 0) * (1 - w) + (out[i1 * 3 + 2] ?? 0) * w;
  const [cx, cy, cz] = clampP0(x, y, z);
  return { x: cx, y: cy, z: cz };
}
