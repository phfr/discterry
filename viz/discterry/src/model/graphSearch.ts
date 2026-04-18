import type { GraphBundle } from "../data/loadBundle";
import type { GraphBundle3d } from "../data/loadBundle3d";

export type GraphEdgeBundle = Pick<GraphBundle, "src" | "dst" | "vertex"> | Pick<GraphBundle3d, "src" | "dst" | "vertex">;

export type GraphCSR = {
  n: number;
  rowOff: Int32Array;
  colInd: Int32Array;
};

/** Undirected CSR: each edge (u,v) appears in both adjacency lists. */
export function buildUndirectedCSR(src: Int32Array, dst: Int32Array, n: number): GraphCSR {
  const deg = new Int32Array(n);
  for (let e = 0; e < src.length; e++) {
    const u = src[e]!;
    const v = dst[e]!;
    if (u < 0 || u >= n || v < 0 || v >= n) continue;
    deg[u]++;
    deg[v]++;
  }
  const rowOff = new Int32Array(n + 1);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    rowOff[i] = sum;
    sum += deg[i]!;
  }
  rowOff[n] = sum;
  const colInd = new Int32Array(sum);
  const wr = new Int32Array(n);
  for (let i = 0; i < n; i++) wr[i] = rowOff[i]!;
  for (let e = 0; e < src.length; e++) {
    const u = src[e]!;
    const v = dst[e]!;
    if (u < 0 || u >= n || v < 0 || v >= n) continue;
    colInd[wr[u]!] = v;
    wr[u]++;
    colInd[wr[v]!] = u;
    wr[v]++;
  }
  return { n, rowOff, colInd };
}

export function bfsShortestPath(csr: GraphCSR, start: number, goal: number): number[] | null {
  if (start === goal) return [start];
  if (start < 0 || start >= csr.n || goal < 0 || goal >= csr.n) return null;
  const vis = new Uint8Array(csr.n);
  const parent = new Int32Array(csr.n);
  parent.fill(-1);
  const q = new Int32Array(csr.n);
  let qh = 0;
  let qt = 0;
  q[qt++] = start;
  vis[start] = 1;
  parent[start] = start;
  while (qh < qt) {
    const u = q[qh++]!;
    const r0 = csr.rowOff[u]!;
    const r1 = csr.rowOff[u + 1]!;
    for (let e = r0; e < r1; e++) {
      const v = csr.colInd[e]!;
      if (vis[v]) continue;
      vis[v] = 1;
      parent[v] = u;
      if (v === goal) {
        const path: number[] = [];
        let x = goal;
        while (true) {
          path.push(x);
          if (x === start) break;
          x = parent[x]!;
          if (x < 0) return null;
        }
        path.reverse();
        return path;
      }
      q[qt++] = v;
    }
  }
  return null;
}

/** `parent[v]` is predecessor on BFS tree, or `-1` if unreachable; `parent[root] === root`. */
export function bfsParentTree(csr: GraphCSR, root: number): Int32Array {
  const parent = new Int32Array(csr.n);
  parent.fill(-1);
  if (root < 0 || root >= csr.n) return parent;
  const q = new Int32Array(csr.n);
  let qh = 0;
  let qt = 0;
  parent[root] = root;
  q[qt++] = root;
  while (qh < qt) {
    const u = q[qh++]!;
    const r0 = csr.rowOff[u]!;
    const r1 = csr.rowOff[u + 1]!;
    for (let e = r0; e < r1; e++) {
      const v = csr.colInd[e]!;
      if (parent[v] !== -1) continue;
      parent[v] = u;
      q[qt++] = v;
    }
  }
  return parent;
}

export function collectFilteredBfsTreeEdges(
  parent: Int32Array,
  root: number,
  allowed: ReadonlySet<number> | null,
  maxEdges: number,
): [number, number][] {
  const out: [number, number][] = [];
  for (let c = 0; c < parent.length && out.length < maxEdges; c++) {
    if (c === root) continue;
    const p = parent[c]!;
    if (p < 0 || p === c) continue;
    if (allowed !== null && !allowed.has(c)) continue;
    out.push([p, c]);
  }
  return out;
}

/** Hyperbolic distance in the Poincaré disk between two W-plane points (after focus map). */
export function poincareDiskDistance(wx0: number, wy0: number, wx1: number, wy1: number): number {
  const r0 = wx0 * wx0 + wy0 * wy0;
  const r1 = wx1 * wx1 + wy1 * wy1;
  const dx = wx1 - wx0;
  const dy = wy1 - wy0;
  const chord2 = dx * dx + dy * dy;
  const den = (1 - r0) * (1 - r1);
  if (den < 1e-20) return Number.POSITIVE_INFINITY;
  return Math.acosh(1 + (2 * chord2) / den);
}

/** Upper bound on seed count for Prim MST overlay (pairwise O(k²) work). */
export const PRIM_MAX_SEEDS = 48;

/** Prim MST on seed indices using pairwise hyperbolic distance in W at current embedding. */
export function primHyperbolicSeedEdges(
  wx: Float32Array,
  wy: Float32Array,
  seedIndices: number[],
): { edges: [number, number][]; skipped: boolean } {
  const k = seedIndices.length;
  if (k < 2) return { edges: [], skipped: false };
  if (k > PRIM_MAX_SEEDS) return { edges: [], skipped: true };
  const distW = (ia: number, ib: number) =>
    poincareDiskDistance(wx[ia]!, wy[ia]!, wx[ib]!, wy[ib]!);

  const inTree = new Uint8Array(k);
  const nearestD = new Float64Array(k);
  const nearestFrom = new Int32Array(k);
  nearestD.fill(Number.POSITIVE_INFINITY);
  nearestFrom.fill(-1);
  inTree[0] = 1;
  for (let j = 1; j < k; j++) {
    const ia = seedIndices[0]!;
    const ib = seedIndices[j]!;
    nearestD[j] = distW(ia, ib);
    nearestFrom[j] = 0;
  }
  const edges: [number, number][] = [];
  for (let it = 1; it < k; it++) {
    let bestJ = -1;
    let bestD = Number.POSITIVE_INFINITY;
    for (let j = 0; j < k; j++) {
      if (inTree[j]) continue;
      if (nearestD[j]! < bestD) {
        bestD = nearestD[j]!;
        bestJ = j;
      }
    }
    if (bestJ < 0) break;
    const t = nearestFrom[bestJ]!;
    if (t < 0) break;
    edges.push([seedIndices[t]!, seedIndices[bestJ]!]);
    inTree[bestJ] = 1;
    for (let j = 0; j < k; j++) {
      if (inTree[j]) continue;
      const ia = seedIndices[bestJ]!;
      const ib = seedIndices[j]!;
      const d = distW(ia, ib);
      if (d < nearestD[j]!) {
        nearestD[j] = d;
        nearestFrom[j] = bestJ;
      }
    }
  }
  return { edges, skipped: false };
}

export function bundleToCSR(bundle: GraphEdgeBundle): GraphCSR {
  return buildUndirectedCSR(bundle.src, bundle.dst, bundle.vertex.length);
}
