import { R_SAFE } from "./constants";

export type Complex = { re: number; im: number };

export function clampZ0(z: Complex): Complex {
  const r = Math.hypot(z.re, z.im);
  if (r < 1e-12) throw new Error("focus disk coordinate is ~0; pick another protein");
  if (r >= R_SAFE) {
    const s = R_SAFE / r;
    return { re: z.re * s, im: z.im * s };
  }
  return { re: z.re, im: z.im };
}

/** w = (z - z0) / (1 - conj(z0) * z) for complex z on unit disk. */
export function mobiusZ(zre: number, zim: number, z0: Complex): Complex {
  const { re: a, im: b } = z0;
  const numr = zre - a;
  const numi = zim - b;
  const dr = 1 - (a * zre + b * zim);
  const di = a * zim - b * zre;
  const d2 = dr * dr + di * di;
  if (d2 < 1e-20) return { re: 0, im: 0 };
  const invr = dr / d2;
  const invi = -di / d2;
  return {
    re: numr * invr - numi * invi,
    im: numr * invi + numi * invr,
  };
}

export function mobiusDiskArrays(
  zx: Float32Array,
  zy: Float32Array,
  z0: Complex,
  outWx: Float32Array,
  outWy: Float32Array,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    const w = mobiusZ(zx[i], zy[i], z0);
    outWx[i] = w.re;
    outWy[i] = w.im;
  }
}
