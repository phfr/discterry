import type { Complex } from "./mobius";
import { clampZ0 } from "./mobius";
import { poincareGeodesicXY } from "./poincareGeodesic";

/** Samples along the disk geodesic between two z0 endpoints (reuse for all frames of one transition). */
export const Z0_GEODESIC_SAMPLES = 64;

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
