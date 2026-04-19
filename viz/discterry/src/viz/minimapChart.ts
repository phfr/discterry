import type { Complex } from "../math/mobius";

/** North-pole (0,0,1) stereographic chart: ball (or ℝ³) → (u,v) plane (can be unbounded near the pole). */
export function stereoNorthFromBall(x: number, y: number, z: number): [number, number] {
  const r = Math.hypot(x, y, z);
  if (r < 1e-15) return [0, 0];
  const X = x / r;
  const Y = y / r;
  const Z = z / r;
  const d = 1 - Z;
  if (Math.abs(d) < 1e-12) return [Math.sign(X) * 1e6, Math.sign(Y) * 1e6];
  return [X / d, Y / d];
}

/** Same chart but clamped so minimap bounds and clicks stay usable (raw stereo explodes near Z→+1 on S²). */
export const STEREO_MINIMAP_CAP = 9;

export function stereoNorthFromBallChart(x: number, y: number, z: number): [number, number] {
  const [u0, v0] = stereoNorthFromBall(x, y, z);
  if (!Number.isFinite(u0) || !Number.isFinite(v0)) return [0, 0];
  const u = Math.max(-STEREO_MINIMAP_CAP, Math.min(STEREO_MINIMAP_CAP, u0));
  const v = Math.max(-STEREO_MINIMAP_CAP, Math.min(STEREO_MINIMAP_CAP, v0));
  return [u, v];
}

/** Inverse north stereographic: (u,v) → unit sphere (X,Y,Z). */
export function stereoNorthToUnitSphere(u: number, v: number): [number, number, number] {
  const s = u * u + v * v;
  const d = 1 + s;
  if (d < 1e-20) return [0, 0, -1];
  return [(2 * u) / d, (2 * v) / d, (s - 1) / d];
}

export type MinimapBounds2d = { cx: number; cy: number; half: number };

/** Fixed axis-aligned bounds for native Z-disk (zx, zy), padded to ≥ square containing unit disk if needed. */
export function boundsNativeZDisk(zx: Float32Array, zy: Float32Array, n: number): MinimapBounds2d {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = zx[i]!;
    const y = zy[i]!;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) {
    return { cx: 0, cy: 0, half: 1.06 };
  }
  const pad = 0.06;
  let half = Math.max(maxX - minX, maxY - minY) * 0.5 + pad;
  half = Math.max(half, 1.06);
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  return { cx, cy, half };
}

export type MinimapBoundsStereo = { cx: number; cy: number; half: number };

/** Bounds in stereographic (u,v) over all raw ball vertices. */
export function boundsStereoRaw(px: Float32Array, py: Float32Array, pz: Float32Array, n: number): MinimapBoundsStereo {
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    const [u, v] = stereoNorthFromBallChart(px[i]!, py[i]!, pz[i]!);
    if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
    minU = Math.min(minU, u);
    maxU = Math.max(maxU, u);
    minV = Math.min(minV, v);
    maxV = Math.max(maxV, v);
  }
  if (!Number.isFinite(minU)) {
    return { cx: 0, cy: 0, half: 1.2 };
  }
  const pad = 0.08;
  let half = Math.max(maxU - minU, maxV - minV) * 0.5 + pad;
  half = Math.max(half, 1.2);
  const cx = (minU + maxU) * 0.5;
  const cy = (minV + maxV) * 0.5;
  return { cx, cy, half };
}

/** Map world (wx, wy) to minimap canvas pixel coords (inner square). */
export function worldToMinimapPixel(
  wx: number,
  wy: number,
  cx: number,
  cy: number,
  half: number,
  inner: number,
  pad: number,
): [number, number] {
  const nx = (wx - cx) / half;
  const ny = (wy - cy) / half;
  const px = pad + ((nx + 1) * 0.5) * inner;
  const py = pad + ((1 - ny) * 0.5) * inner;
  return [px, py];
}

/** Inverse of worldToMinimapPixel: canvas (px,py) → world (wx, wy). */
export function minimapPixelToWorld(
  px: number,
  py: number,
  cx: number,
  cy: number,
  half: number,
  inner: number,
  pad: number,
): [number, number] {
  const nx = ((px - pad) / inner) * 2 - 1;
  const ny = -(((py - pad) / inner) * 2 - 1);
  return [cx + nx * half, cy + ny * half];
}

/** Click on 2D native Z minimap → complex z0 (caller clamps). */
export function nativeZClickToComplex(
  px: number,
  py: number,
  cx: number,
  cy: number,
  half: number,
  inner: number,
  pad: number,
): Complex {
  const [wx, wy] = minimapPixelToWorld(px, py, cx, cy, half, inner, pad);
  return { re: wx, im: wy };
}

/** Click on stereographic minimap → (u,v) chart coords (then invert to ball). */
export function stereoMinimapClickToUV(
  px: number,
  py: number,
  cx: number,
  cy: number,
  half: number,
  inner: number,
  pad: number,
): [number, number] {
  return minimapPixelToWorld(px, py, cx, cy, half, inner, pad);
}
