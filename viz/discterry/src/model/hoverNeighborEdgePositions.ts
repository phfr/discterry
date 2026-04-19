import type { Complex } from "../math/mobius";
import { GEODESIC_N } from "../math/constants";
import type { GraphCSR } from "./graphSearch";

export type HoverNeighborGraph2d = {
  csr: GraphCSR;
  zx: Float32Array;
  zy: Float32Array;
  z0: Complex;
};

export type HoverNeighborGraph3d = {
  csr: GraphCSR;
  px: Float32Array;
  py: Float32Array;
  pz: Float32Array;
  p0x: number;
  p0y: number;
  p0z: number;
};
import { appendGeodesicLineSegments } from "./geodesicLineBuffer";
import { appendGeodesicLineSegments3d } from "./geodesicLineBuffer3d";

const vertsPerEdge = (GEODESIC_N - 1) * 2 * 3;

/** Poincaré-disk geodesics from `center` to each CSR neighbor (raw W after focus `z0`). */
export function buildHoverNeighborLinePositions2d(
  csr: GraphCSR,
  zx: Float32Array,
  zy: Float32Array,
  z0: Complex,
  center: number,
): Float32Array {
  if (center < 0 || center >= csr.n) return new Float32Array(0);
  const r0 = csr.rowOff[center]!;
  const r1 = csr.rowOff[center + 1]!;
  const deg = r1 - r0;
  if (deg <= 0) return new Float32Array(0);
  const out = new Float32Array(deg * vertsPerEdge);
  const off = { i: 0 };
  for (let e = r0; e < r1; e++) {
    const v = csr.colInd[e]!;
    appendGeodesicLineSegments(zx, zy, z0, center, v, out, off);
  }
  return out.subarray(0, off.i);
}

/** Poincaré-ball geodesics from `center` to each CSR neighbor (raw W after focus `p0`). */
export function buildHoverNeighborLinePositions3d(
  csr: GraphCSR,
  px: Float32Array,
  py: Float32Array,
  pz: Float32Array,
  p0x: number,
  p0y: number,
  p0z: number,
  center: number,
): Float32Array {
  if (center < 0 || center >= csr.n) return new Float32Array(0);
  const r0 = csr.rowOff[center]!;
  const r1 = csr.rowOff[center + 1]!;
  const deg = r1 - r0;
  if (deg <= 0) return new Float32Array(0);
  const out = new Float32Array(deg * vertsPerEdge);
  const off = { i: 0 };
  for (let e = r0; e < r1; e++) {
    const v = csr.colInd[e]!;
    appendGeodesicLineSegments3d(px, py, pz, p0x, p0y, p0z, center, v, out, off);
  }
  return out.subarray(0, off.i);
}
