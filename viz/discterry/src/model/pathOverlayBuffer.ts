import type { Complex } from "../math/mobius";
import { GEODESIC_N } from "../math/constants";
import type { GraphBundle } from "../data/loadBundle";
import { appendGeodesicLineSegments } from "./geodesicLineBuffer";

export type PathOverlayBuffer = {
  positions: Float32Array;
  nFloats: number;
};

const vertsPerEdge = (GEODESIC_N - 1) * 2 * 3;

export function buildPathOverlayFromEdges(
  bundle: GraphBundle,
  z0: Complex,
  edges: ReadonlyArray<readonly [number, number]>,
): PathOverlayBuffer | null {
  if (edges.length === 0) return null;
  const out = new Float32Array(edges.length * vertsPerEdge);
  const off = { i: 0 };
  const { x: zx, y: zy } = bundle;
  for (const [a, b] of edges) {
    appendGeodesicLineSegments(zx, zy, z0, a, b, out, off);
  }
  if (off.i === 0) return null;
  return { positions: out.subarray(0, off.i), nFloats: off.i };
}

export function buildPathOverlayFromVertexPath(
  bundle: GraphBundle,
  z0: Complex,
  path: ReadonlyArray<number>,
): PathOverlayBuffer | null {
  if (path.length < 2) return null;
  const edges: [number, number][] = [];
  for (let i = 0; i < path.length - 1; i++) edges.push([path[i]!, path[i + 1]!]);
  return buildPathOverlayFromEdges(bundle, z0, edges);
}
