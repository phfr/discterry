import type { GraphBundle } from "../data/loadBundle";
import type { Complex } from "../math/mobius";
import { mobiusDiskArrays } from "../math/mobius";
import type { GraphCSR } from "./graphSearch";
import { poincareDiskDistance } from "./graphSearch";

/** Max seeds for hop matrix / scatter (each extra seed adds a full-graph BFS). */
export const SEED_PAIR_METRICS_MAX_SEEDS = 40;

export type OrderedSeeds = { idx: number[]; names: string[] };

/** Unique bundle indices for applied seed names, preserving first-seen order. */
export function buildOrderedSeedIndices(bundle: GraphBundle, seedNamesOrdered: string[]): OrderedSeeds {
  const idx: number[] = [];
  const names: string[] = [];
  const seen = new Set<number>();
  for (const raw of seedNamesOrdered) {
    const j = bundle.nameToIndex.get(raw.trim());
    if (j === undefined || seen.has(j)) continue;
    seen.add(j);
    idx.push(j);
    names.push(bundle.vertex[j]!);
  }
  return { idx, names };
}

export function fillMobiusW(
  bundle: GraphBundle,
  z0: Complex,
  wx: Float32Array,
  wy: Float32Array,
): void {
  mobiusDiskArrays(bundle.x, bundle.y, z0, wx, wy, bundle.vertex.length);
}

/** Symmetric k×k hyperbolic distances in W (upper triangle filled; diagonal 0). */
export function fillHyperbolicPairMatrix(
  wx: Float32Array,
  wy: Float32Array,
  idx: readonly number[],
  out: Float32Array,
): void {
  const k = idx.length;
  for (let i = 0; i < k; i++) {
    const ia = idx[i]!;
    for (let j = 0; j < k; j++) {
      const ib = idx[j]!;
      out[i * k + j] =
        i === j ? 0 : poincareDiskDistance(wx[ia]!, wy[ia]!, wx[ib]!, wy[ib]!);
    }
  }
}

/** Symmetric k×k Euclidean chord distance in (wx, wy). */
export function fillEuclideanWPairMatrix(
  wx: Float32Array,
  wy: Float32Array,
  idx: readonly number[],
  out: Float32Array,
): void {
  const k = idx.length;
  for (let i = 0; i < k; i++) {
    const ia = idx[i]!;
    for (let j = 0; j < k; j++) {
      if (i === j) {
        out[i * k + j] = 0;
        continue;
      }
      const ib = idx[j]!;
      out[i * k + j] = Math.hypot(wx[ib]! - wx[ia]!, wy[ib]! - wy[ia]!);
    }
  }
}

/** BFS hop count from `start` into `dist` (-1 = unreachable). Reuses `dist` buffer. */
export function bfsHopDistancesInto(csr: GraphCSR, start: number, dist: Int32Array): void {
  dist.fill(-1);
  if (start < 0 || start >= csr.n) return;
  const q = new Int32Array(csr.n);
  let qh = 0;
  let qt = 0;
  dist[start] = 0;
  q[qt++] = start;
  while (qh < qt) {
    const u = q[qh++]!;
    const du = dist[u]!;
    const r0 = csr.rowOff[u]!;
    const r1 = csr.rowOff[u + 1]!;
    for (let e = r0; e < r1; e++) {
      const v = csr.colInd[e]!;
      if (dist[v] !== -1) continue;
      dist[v] = du + 1;
      q[qt++] = v;
    }
  }
}

/** k×k shortest-path hop counts between seed indices (-1 if disconnected). */
export function fillGraphHopPairMatrix(
  csr: GraphCSR,
  idx: readonly number[],
  out: Int32Array,
  scratch: Int32Array,
): void {
  const k = idx.length;
  for (let si = 0; si < k; si++) {
    const src = idx[si]!;
    bfsHopDistancesInto(csr, src, scratch);
    for (let sj = 0; sj < k; sj++) {
      const t = idx[sj]!;
      out[si * k + sj] = scratch[t]!;
    }
  }
}

export type PairScatterPoint = { hops: number; hyp: number; i: number; j: number };

export function collectScatterPoints(
  hyp: Float32Array,
  hops: Int32Array,
  k: number,
): PairScatterPoint[] {
  const pts: PairScatterPoint[] = [];
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const h = hops[i * k + j]!;
      const d = hyp[i * k + j]!;
      if (h < 0 || !Number.isFinite(d) || d > 1e10) continue;
      pts.push({ hops: h, hyp: d, i, j });
    }
  }
  return pts;
}

/** Pearson r on paired arrays (empty if n<2). */
export function pearsonCorrelation(xs: number[], ys: number[]): number | null {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i]!;
    my += ys[i]!;
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - mx;
    const dy = ys[i]! - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const den = Math.sqrt(sxx * syy);
  if (den < 1e-20) return null;
  return sxy / den;
}
