import { EDGE_Z_BOUND, GEODESIC_N } from "../math/constants";
import type { Complex } from "../math/mobius";
import { mobiusZ } from "../math/mobius";
import { poincareGeodesicXY } from "../math/poincareGeodesic";

/** Append line segment pairs (in W after focus `z0`) for the Poincaré geodesic from vertex `ia` to `ib` in Z. */
export function appendGeodesicLineSegments(
  zx: Float32Array,
  zy: Float32Array,
  z0: Complex,
  ia: number,
  ib: number,
  out: Float32Array,
  outOffset: { i: number },
): void {
  const z1r = zx[ia];
  const z1i = zy[ia];
  const z2r = zx[ib];
  const z2i = zy[ib];
  if (Math.hypot(z1r, z1i) >= EDGE_Z_BOUND || Math.hypot(z2r, z2i) >= EDGE_Z_BOUND) return;

  const w1 = mobiusZ(z1r, z1i, z0);
  const w2 = mobiusZ(z2r, z2i, z0);
  if (Math.hypot(w1.re, w1.im) >= EDGE_Z_BOUND || Math.hypot(w2.re, w2.im) >= EDGE_Z_BOUND) return;

  const n = GEODESIC_N;
  const gx = new Float32Array(n);
  const gy = new Float32Array(n);
  poincareGeodesicXY(w1.re, w1.im, w2.re, w2.im, n, gx, gy);

  let o = outOffset.i;
  for (let k = 0; k < n - 1; k++) {
    out[o++] = gx[k];
    out[o++] = gy[k];
    out[o++] = 0;
    out[o++] = gx[k + 1];
    out[o++] = gy[k + 1];
    out[o++] = 0;
  }
  outOffset.i = o;
}
