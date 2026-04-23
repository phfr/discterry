import type { Complex } from "../math/mobius";
import { GEODESIC_N } from "../math/constants";
import type { GraphBundle } from "../data/loadBundle";
import type { GraphBundle3d } from "../data/loadBundle3d";
import { appendGeodesicLineSegments } from "./geodesicLineBuffer";
import { appendGeodesicLineSegments3d } from "./geodesicLineBuffer3d";

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

export function buildPathOverlayFromEdges3d(
  bundle: GraphBundle3d,
  p0x: number,
  p0y: number,
  p0z: number,
  edges: ReadonlyArray<readonly [number, number]>,
): PathOverlayBuffer | null {
  if (edges.length === 0) return null;
  const out = new Float32Array(edges.length * vertsPerEdge);
  const off = { i: 0 };
  const { x: zx, y: zy, z: zz } = bundle;
  for (const [a, b] of edges) {
    appendGeodesicLineSegments3d(zx, zy, zz, p0x, p0y, p0z, a, b, out, off);
  }
  if (off.i === 0) return null;
  return { positions: out.subarray(0, off.i), nFloats: off.i };
}

export function buildPathOverlayFromVertexPath3d(
  bundle: GraphBundle3d,
  p0x: number,
  p0y: number,
  p0z: number,
  path: ReadonlyArray<number>,
): PathOverlayBuffer | null {
  if (path.length < 2) return null;
  const edges: [number, number][] = [];
  for (let i = 0; i < path.length - 1; i++) edges.push([path[i]!, path[i + 1]!]);
  return buildPathOverlayFromEdges3d(bundle, p0x, p0y, p0z, edges);
}
