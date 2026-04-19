import type { Complex } from "../math/mobius";

/** 3D minimap chart mode (planar) or WebGL globe (handled outside this module). */
export type Minimap3dMode =
  | "stereo_north"
  | "stereo_south"
  | "ortho_xy"
  | "ortho_xz"
  | "equirect"
  | "lambert_north"
  | "gnomonic_north"
  | "globe_webgl";

export function isPlanarMinimap3dMode(m: Minimap3dMode): m is Exclude<Minimap3dMode, "globe_webgl"> {
  return m !== "globe_webgl";
}

const EPS_Z = 1e-4;
const GNOMIC_CAP = 9;

export type MinimapBoundsStereo = { cx: number; cy: number; half: number };

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

function normUnit(x: number, y: number, z: number): [number, number, number] {
  const r = Math.hypot(x, y, z);
  if (r < 1e-15) return [0, 0, 1];
  return [x / r, y / r, z / r];
}

/** Raw vertex position → unit direction on S². */
export function normalizeBallDir(x: number, y: number, z: number): [number, number, number] {
  return normUnit(x, y, z);
}

/** Forward: unit ball direction (from raw x,y,z) → 2D chart (wx, wy) for minimap bounds and drawing. */
export function ballDirToChart(
  mode: Exclude<Minimap3dMode, "globe_webgl">,
  x: number,
  y: number,
  z: number,
): [number, number] {
  const [X, Y, Z] = normalizeBallDir(x, y, z);
  switch (mode) {
    case "stereo_north":
      return stereoNorthFromBallChart(x, y, z);
    case "stereo_south":
      return stereoNorthFromBallChart(x, y, -z);
    case "ortho_xy":
      return [X, Y];
    case "ortho_xz":
      return [X, Z];
    case "equirect":
      return [Math.atan2(Y, X), Math.asin(Math.max(-1, Math.min(1, Z)))];
    case "lambert_north": {
      if (Z <= -1 + 1e-5) {
        const ang = Math.atan2(Y, X);
        const R = 1.99;
        return [R * Math.cos(ang), R * Math.sin(ang)];
      }
      const k = Math.sqrt(2 / (1 + Z));
      let wx = k * X;
      let wy = k * Y;
      const rho = Math.hypot(wx, wy);
      const cap = 1.99;
      if (rho > cap) {
        const s = cap / rho;
        wx *= s;
        wy *= s;
      }
      return [wx, wy];
    }
    case "gnomonic_north": {
      const d = Math.max(Z, EPS_Z);
      let wx = X / d;
      let wy = Y / d;
      wx = Math.max(-GNOMIC_CAP, Math.min(GNOMIC_CAP, wx));
      wy = Math.max(-GNOMIC_CAP, Math.min(GNOMIC_CAP, wy));
      return [wx, wy];
    }
    default:
      return [0, 0];
  }
}

/** Inverse chart (wx, wy) → unit sphere direction. */
export function chartToUnitSphere(
  mode: Exclude<Minimap3dMode, "globe_webgl">,
  wx: number,
  wy: number,
): [number, number, number] {
  switch (mode) {
    case "stereo_north":
      return stereoNorthToUnitSphere(wx, wy);
    case "stereo_south": {
      const [sx, sy, sz] = stereoNorthToUnitSphere(wx, wy);
      return normUnit(sx, sy, -sz);
    }
    case "ortho_xy": {
      const s2 = wx * wx + wy * wy;
      if (s2 >= 1 - EPS_Z * EPS_Z) {
        const t = 1 / Math.sqrt(s2);
        return normUnit(wx * t, wy * t, 0);
      }
      const zz = Math.sqrt(Math.max(0, 1 - wx * wx - wy * wy));
      return normUnit(wx, wy, zz);
    }
    case "ortho_xz": {
      const s2 = wx * wx + wy * wy;
      if (s2 >= 1 - EPS_Z * EPS_Z) {
        const t = 1 / Math.sqrt(s2);
        return normUnit(wx * t, 0, wy * t);
      }
      const yy = Math.sqrt(Math.max(0, 1 - wx * wx - wy * wy));
      return normUnit(wx, yy, wy);
    }
    case "equirect": {
      const cosLat = Math.cos(wy);
      return normUnit(cosLat * Math.cos(wx), cosLat * Math.sin(wx), Math.sin(wy));
    }
    case "lambert_north": {
      const rho = Math.hypot(wx, wy);
      if (rho < 1e-10) return [0, 0, 1];
      let z = 1 - rho * rho * 0.5;
      z = Math.max(-1, Math.min(1, z));
      const rxy = Math.sqrt(Math.max(0, 1 - z * z));
      const f = rho > 1e-12 ? rxy / rho : 0;
      return normUnit(f * wx, f * wy, z);
    }
    case "gnomonic_north":
      return normUnit(wx, wy, 1);
    default:
      return [0, 0, 1];
  }
}

/** Axis-aligned bounds in chart (wx, wy) over all vertices. */
export function boundsMinimap3dChart(
  px: Float32Array,
  py: Float32Array,
  pz: Float32Array,
  n: number,
  mode: Exclude<Minimap3dMode, "globe_webgl">,
): MinimapBoundsStereo {
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    const [u, v] = ballDirToChart(mode, px[i]!, py[i]!, pz[i]!);
    if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
    minU = Math.min(minU, u);
    maxU = Math.max(maxU, u);
    minV = Math.min(minV, v);
    maxV = Math.max(maxV, v);
  }
  if (!Number.isFinite(minU)) {
    return { cx: 0, cy: 0, half: 1.2 };
  }
  const pad = mode === "equirect" ? 0.06 : 0.08;
  let half = Math.max(maxU - minU, maxV - minV) * 0.5 + pad;
  half = Math.max(half, mode === "equirect" ? 0.55 : 1.0);
  const cx = (minU + maxU) * 0.5;
  const cy = (minV + maxV) * 0.5;
  return { cx, cy, half };
}

/** Canvas pixel → unit sphere point for 3D minimap chart pick (caller scales toward interior). */
export function minimapChartClickToUnitSphere(
  mode: Exclude<Minimap3dMode, "globe_webgl">,
  canvasPx: number,
  canvasPy: number,
  cx: number,
  cy: number,
  half: number,
  inner: number,
  pad: number,
): [number, number, number] {
  const [wx, wy] = minimapPixelToWorld(canvasPx, canvasPy, cx, cy, half, inner, pad);
  return chartToUnitSphere(mode, wx, wy);
}

export type MinimapBounds2d = { cx: number; cy: number; half: number };

/** 2D minimap chart: native disk vs alternate (wx, wy) views of the same disk coordinates. */
export type Minimap2dMode =
  | "native_disk"
  | "polar_euclidean"
  | "hyperbolic_polar"
  | "upper_half_plane"
  | "sqrt_branch";

const DISK_EPS = 1e-7;

function principalSqrt(zx: number, zy: number): [number, number] {
  const r = Math.hypot(zx, zy);
  if (r < 1e-15) return [0, 0];
  const ang = Math.atan2(zy, zx);
  const sr = Math.sqrt(r);
  const h = ang * 0.5;
  return [sr * Math.cos(h), sr * Math.sin(h)];
}

/** Disk vertex (zx, zy) as complex z → chart (wx, wy) for minimap layout and drawing. */
export function diskPointToChart(mode: Minimap2dMode, zx: number, zy: number): [number, number] {
  switch (mode) {
    case "native_disk":
      return [zx, zy];
    case "polar_euclidean": {
      const r = Math.hypot(zx, zy);
      return [r, Math.atan2(zy, zx)];
    }
    case "hyperbolic_polar": {
      const r = Math.hypot(zx, zy);
      const rc = Math.min(r, 1 - DISK_EPS);
      return [Math.atanh(rc), Math.atan2(zy, zx)];
    }
    case "upper_half_plane": {
      /* w = i * (1+z) / (1-z), z = zx + i zy */
      const ar = -zy;
      const ai = 1 + zx;
      const cr = 1 - zx;
      const ci = -zy;
      const d2 = cr * cr + ci * ci;
      if (d2 < 1e-20) return [0, 10];
      const inv = 1 / d2;
      return [inv * (ar * cr + ai * ci), inv * (ai * cr - ar * ci)];
    }
    case "sqrt_branch":
      return principalSqrt(zx, zy);
    default:
      return [0, 0];
  }
}

/** Chart (wx, wy) → disk complex z (caller applies clampZ0). */
export function chartWorldToDisk(mode: Minimap2dMode, wx: number, wy: number): Complex {
  switch (mode) {
    case "native_disk":
      return { re: wx, im: wy };
    case "polar_euclidean": {
      const rho = Math.min(Math.max(wx, 0), 1 - DISK_EPS);
      const c = Math.cos(wy);
      const s = Math.sin(wy);
      return { re: rho * c, im: rho * s };
    }
    case "hyperbolic_polar": {
      const rho = Math.min(Math.max(Math.tanh(wx), 0), 1 - DISK_EPS);
      const c = Math.cos(wy);
      const s = Math.sin(wy);
      return { re: rho * c, im: rho * s };
    }
    case "upper_half_plane": {
      /* z = (w - i) / (w + i), w = wx + i wy */
      const ar = wx;
      const ai = wy - 1;
      const cr = wx;
      const ci = wy + 1;
      const d2 = cr * cr + ci * ci;
      if (d2 < 1e-20) return { re: 0, im: 0 };
      const inv = 1 / d2;
      return { re: inv * (ar * cr + ai * ci), im: inv * (ai * cr - ar * ci) };
    }
    case "sqrt_branch": {
      const zre = wx * wx - wy * wy;
      const zim = 2 * wx * wy;
      return { re: zre, im: zim };
    }
    default:
      return { re: 0, im: 0 };
  }
}

/** Axis-aligned bounds in 2D chart (wx, wy) over all disk vertices. */
export function boundsMinimap2dChart(zx: Float32Array, zy: Float32Array, n: number, mode: Minimap2dMode): MinimapBounds2d {
  if (mode === "native_disk") {
    return boundsNativeZDisk(zx, zy, n);
  }
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    const [u, v] = diskPointToChart(mode, zx[i]!, zy[i]!);
    if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
    minU = Math.min(minU, u);
    maxU = Math.max(maxU, u);
    minV = Math.min(minV, v);
    maxV = Math.max(maxV, v);
  }
  if (!Number.isFinite(minU)) {
    return { cx: 0, cy: 0, half: 1.06 };
  }
  const pad = 0.08;
  let half = Math.max(maxU - minU, maxV - minV) * 0.5 + pad;
  const minHalf =
    mode === "polar_euclidean" || mode === "hyperbolic_polar"
      ? 0.55
      : mode === "upper_half_plane"
        ? 0.85
        : 0.45;
  half = Math.max(half, minHalf);
  const cx = (minU + maxU) * 0.5;
  const cy = (minV + maxV) * 0.5;
  return { cx, cy, half };
}

/** Canvas pixel → disk z for 2D minimap chart pick (caller clamps with clampZ0). */
export function minimap2dChartClickToComplex(
  mode: Minimap2dMode,
  canvasPx: number,
  canvasPy: number,
  cx: number,
  cy: number,
  half: number,
  inner: number,
  pad: number,
): Complex {
  const [wx, wy] = minimapPixelToWorld(canvasPx, canvasPy, cx, cy, half, inner, pad);
  return chartWorldToDisk(mode, wx, wy);
}

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
