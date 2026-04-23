import { EDGE_Z_BOUND, GEODESIC_N } from "../math/constants";
import { mobiusBallToOrigin } from "../math/poincareBall";
import { poincareBallGeodesicSamples } from "../math/poincareBall";

function normBound(x: number, y: number, z: number): number {
  return Math.hypot(x, y, z);
}

/** Append line segment pairs (in W after focus ``p0``) for the Poincaré-ball geodesic from ``ia`` to ``ib``. */
export function appendGeodesicLineSegments3d(
  px: Float32Array,
  py: Float32Array,
  pz: Float32Array,
  p0x: number,
  p0y: number,
  p0z: number,
  ia: number,
  ib: number,
  out: Float32Array,
  outOffset: { i: number },
): void {
  const x1 = px[ia];
  const y1 = py[ia];
  const z1 = pz[ia];
  const x2 = px[ib];
  const y2 = py[ib];
  const z2 = pz[ib];
  if (normBound(x1, y1, z1) >= EDGE_Z_BOUND || normBound(x2, y2, z2) >= EDGE_Z_BOUND) return;

  const [w1x, w1y, w1z] = mobiusBallToOrigin(p0x, p0y, p0z, x1, y1, z1);
  const [w2x, w2y, w2z] = mobiusBallToOrigin(p0x, p0y, p0z, x2, y2, z2);
  if (normBound(w1x, w1y, w1z) >= EDGE_Z_BOUND || normBound(w2x, w2y, w2z) >= EDGE_Z_BOUND) return;

  const n = GEODESIC_N;
  const g = new Float32Array(n * 3);
  poincareBallGeodesicSamples(w1x, w1y, w1z, w2x, w2y, w2z, n, g);

  let o = outOffset.i;
  for (let k = 0; k < n - 1; k++) {
    out[o++] = g[k * 3];
    out[o++] = g[k * 3 + 1];
    out[o++] = g[k * 3 + 2];
    out[o++] = g[(k + 1) * 3];
    out[o++] = g[(k + 1) * 3 + 1];
    out[o++] = g[(k + 1) * 3 + 2];
  }
  outOffset.i = o;
}
