import { GEODESIC_N } from "../math/constants";
import type { Complex } from "../math/mobius";
import { mobiusDiskArrays } from "../math/mobius";
import type { GraphBundle } from "../data/loadBundle";
import { appendGeodesicLineSegments } from "./geodesicLineBuffer";
import { buildSeedMask, classifySeedEdges } from "./graphFilter";

/** Which non-seed vertices get green disk markers (edges are always seed-touching only). */
export type NonSeedShowMode = "all" | "seed_only" | "seed_and_neighbors";

function includeGreenNonSeed(mode: NonSeedShowMode, seedEdgeVertex: Uint8Array, i: number): boolean {
  if (mode === "all") return true;
  if (mode === "seed_only") return false;
  return seedEdgeVertex[i] === 1;
}

export type SceneStats = {
  nodesTotal: number;
  nodesRendered: number;
  /** Nodes not drawn: |W| past rim band, not a seed, not endpoint of a seed-touching edge. */
  nodesHiddenRim: number;
  /** Seeds that lie in the rim band but are still drawn. */
  nodesSparedSeedRim: number;
  /** Non-seeds on at least one seed-touching edge, in rim band, still drawn. */
  nodesSparedSeedEdgeRim: number;
  edgesSeedTouching: number;
  edgesDrawn: number;
  /**
   * Seed-touching edges not drawn because an endpoint has |Z| or |W| ≥ EDGE_Z_BOUND
   * (notebook clip — not the same as rim culling).
   */
  edgesSkippedBoundary: number;
};

export type SceneBuffers = {
  lineOnePositions: Float32Array;
  lineBothPositions: Float32Array;
  pointsOther: Float32Array;
  pointsSeed: Float32Array;
  /** Graph vertex index for each `pointsOther` instance (green), same order as triples. */
  otherGraphIndex: Int32Array;
  /** Graph vertex index for each `pointsSeed` instance (red). */
  seedGraphIndex: Int32Array;
  /** Same length as `nPointsSeed`, order matches `pointsSeed` triples. */
  seedLabels: string[];
  nLineOneVerts: number;
  nLineBothVerts: number;
  nPointsOther: number;
  nPointsSeed: number;
  stats: SceneStats;
};

export function computeScene(
  bundle: GraphBundle,
  z0: Complex,
  seedNames: Set<string>,
  rimCullEps: number,
  nonSeedShowMode: NonSeedShowMode,
): SceneBuffers {
  const { x: zx, y: zy, src, dst, nameToIndex } = bundle;
  const n = zx.length;

  const isSeed = buildSeedMask(n, seedNames, nameToIndex);
  const { both, one } = classifySeedEdges(src, dst, isSeed);

  const seedEdgeVertex = new Uint8Array(n);
  for (const ei of both) {
    seedEdgeVertex[src[ei]] = 1;
    seedEdgeVertex[dst[ei]] = 1;
  }
  for (const ei of one) {
    seedEdgeVertex[src[ei]] = 1;
    seedEdgeVertex[dst[ei]] = 1;
  }

  const wx = new Float32Array(n);
  const wy = new Float32Array(n);
  mobiusDiskArrays(zx, zy, z0, wx, wy, n);

  const rim = 1 - Math.max(0, rimCullEps);

  const otherPts: number[] = [];
  const seedPts: number[] = [];
  const otherIdx: number[] = [];
  const seedIdx: number[] = [];
  const seedLabels: string[] = [];
  let hiddenNonSeed = 0;
  let sparedSeed = 0;
  let sparedEdge = 0;

  for (let i = 0; i < n; i++) {
    const rimWouldCull = Math.hypot(wx[i], wy[i]) > rim;
    const show = !rimWouldCull || isSeed[i] === 1 || seedEdgeVertex[i] === 1;

    if (rimWouldCull) {
      if (isSeed[i]) sparedSeed++;
      else if (seedEdgeVertex[i]) {
        if (includeGreenNonSeed(nonSeedShowMode, seedEdgeVertex, i)) sparedEdge++;
      } else hiddenNonSeed++;
    }

    if (!show) continue;

    if (isSeed[i]) {
      /* Slight +Z so seeds sit in front of coplanar green sprites at the same (x,y). */
      seedPts.push(wx[i], wy[i], 0.004);
      seedIdx.push(i);
      seedLabels.push(bundle.vertex[i]);
    } else if (includeGreenNonSeed(nonSeedShowMode, seedEdgeVertex, i)) {
      otherPts.push(wx[i], wy[i], 0);
      otherIdx.push(i);
    }
  }

  const vertsPerEdge = (GEODESIC_N - 1) * 2 * 3;
  const lineBothPositions = new Float32Array(both.length * vertsPerEdge);
  const lineOnePositions = new Float32Array(one.length * vertsPerEdge);
  const ob = { i: 0 };
  const oo = { i: 0 };

  const edgesTouching = both.length + one.length;
  let drawnBoth = 0;
  let drawnOne = 0;
  for (const ei of both) {
    const before = ob.i;
    appendGeodesicLineSegments(zx, zy, z0, src[ei], dst[ei], lineBothPositions, ob);
    if (ob.i > before) drawnBoth++;
  }
  for (const ei of one) {
    const before = oo.i;
    appendGeodesicLineSegments(zx, zy, z0, src[ei], dst[ei], lineOnePositions, oo);
    if (oo.i > before) drawnOne++;
  }

  const nodesRendered = otherPts.length / 3 + seedPts.length / 3;
  const edgesDrawn = drawnBoth + drawnOne;

  return {
    lineBothPositions: lineBothPositions.subarray(0, ob.i),
    lineOnePositions: lineOnePositions.subarray(0, oo.i),
    pointsOther: Float32Array.from(otherPts),
    pointsSeed: Float32Array.from(seedPts),
    otherGraphIndex: Int32Array.from(otherIdx),
    seedGraphIndex: Int32Array.from(seedIdx),
    seedLabels,
    nLineBothVerts: ob.i,
    nLineOneVerts: oo.i,
    nPointsOther: otherPts.length / 3,
    nPointsSeed: seedPts.length / 3,
    stats: {
      nodesTotal: n,
      nodesRendered,
      nodesHiddenRim: hiddenNonSeed,
      nodesSparedSeedRim: sparedSeed,
      nodesSparedSeedEdgeRim: sparedEdge,
      edgesSeedTouching: edgesTouching,
      edgesDrawn,
      edgesSkippedBoundary: edgesTouching - edgesDrawn,
    },
  };
}
