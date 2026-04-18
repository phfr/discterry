import { BACKGROUND_NONSEED_EDGE_MAX, GEODESIC_N } from "../math/constants";
import { mobiusBallArrays } from "../math/poincareBall";
import type { GraphBundle3d } from "../data/loadBundle3d";
import { appendGeodesicLineSegments3d } from "./geodesicLineBuffer3d";
import { buildSeedMask, classifySeedEdges } from "./graphFilter";
import type { NonSeedShowMode, SceneStats } from "./computeScene";

function includeGreenNonSeed(mode: NonSeedShowMode, seedEdgeVertex: Uint8Array, i: number): boolean {
  if (mode === "all") return true;
  if (mode === "seed_only") return false;
  return seedEdgeVertex[i] === 1;
}

export type SceneBuffers3d = {
  lineOnePositions: Float32Array;
  lineBothPositions: Float32Array;
  lineBgPositions: Float32Array;
  pointsOther: Float32Array;
  pointsSeed: Float32Array;
  otherGraphIndex: Int32Array;
  seedGraphIndex: Int32Array;
  seedLabels: string[];
  nLineOneVerts: number;
  nLineBothVerts: number;
  nLineBgVerts: number;
  nPointsOther: number;
  nPointsSeed: number;
  stats: SceneStats;
};

export function computeScene3d(
  bundle: GraphBundle3d,
  p0x: number,
  p0y: number,
  p0z: number,
  seedNames: Set<string>,
  rimCullEps: number,
  nonSeedShowMode: NonSeedShowMode,
  drawAllNonSeedEdges: boolean,
): SceneBuffers3d {
  const { x: px, y: py, z: pz, src, dst, nameToIndex } = bundle;
  const n = px.length;

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
  const wz = new Float32Array(n);
  mobiusBallArrays(px, py, pz, p0x, p0y, p0z, wx, wy, wz, n);

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
    const rw = Math.hypot(wx[i], wy[i], wz[i]);
    const rimWouldCull = rw > rim;
    const show = !rimWouldCull || isSeed[i] === 1 || seedEdgeVertex[i] === 1;

    if (rimWouldCull) {
      if (isSeed[i]) sparedSeed++;
      else if (seedEdgeVertex[i]) {
        if (includeGreenNonSeed(nonSeedShowMode, seedEdgeVertex, i)) sparedEdge++;
      } else hiddenNonSeed++;
    }

    if (!show) continue;

    if (isSeed[i]) {
      seedPts.push(wx[i], wy[i], wz[i]);
      seedIdx.push(i);
      seedLabels.push(bundle.vertex[i]);
    } else if (includeGreenNonSeed(nonSeedShowMode, seedEdgeVertex, i)) {
      otherPts.push(wx[i], wy[i], wz[i]);
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
    appendGeodesicLineSegments3d(px, py, pz, p0x, p0y, p0z, src[ei], dst[ei], lineBothPositions, ob);
    if (ob.i > before) drawnBoth++;
  }
  for (const ei of one) {
    const before = oo.i;
    appendGeodesicLineSegments3d(px, py, pz, p0x, p0y, p0z, src[ei], dst[ei], lineOnePositions, oo);
    if (oo.i > before) drawnOne++;
  }

  const none: number[] = [];
  let bgPool = 0;
  let bgSubmitted = 0;
  if (drawAllNonSeedEdges) {
    for (let ei = 0; ei < src.length; ei++) {
      const a = src[ei]!;
      const b = dst[ei]!;
      if (isSeed[a] === 0 && isSeed[b] === 0) bgPool++;
    }
    if (bgPool > 0) {
      const toTake = Math.min(bgPool, BACKGROUND_NONSEED_EDGE_MAX);
      let k = -1;
      for (let ei = 0; ei < src.length; ei++) {
        const a = src[ei]!;
        const b = dst[ei]!;
        if (isSeed[a] !== 0 || isSeed[b] !== 0) continue;
        k++;
        if (Math.floor(((k + 1) * toTake) / bgPool) > Math.floor((k * toTake) / bgPool)) {
          none.push(ei);
        }
      }
    }
    bgSubmitted = none.length;
  }
  const lineBgPositions = new Float32Array(none.length * vertsPerEdge);
  const ogb = { i: 0 };
  let drawnBg = 0;
  for (const ei of none) {
    const before = ogb.i;
    appendGeodesicLineSegments3d(px, py, pz, p0x, p0y, p0z, src[ei]!, dst[ei]!, lineBgPositions, ogb);
    if (ogb.i > before) drawnBg++;
  }

  const nodesRendered = otherPts.length / 3 + seedPts.length / 3;
  const edgesDrawn = drawnBoth + drawnOne;

  return {
    lineBothPositions: lineBothPositions.subarray(0, ob.i),
    lineOnePositions: lineOnePositions.subarray(0, oo.i),
    lineBgPositions: lineBgPositions.subarray(0, ogb.i),
    pointsOther: Float32Array.from(otherPts),
    pointsSeed: Float32Array.from(seedPts),
    otherGraphIndex: Int32Array.from(otherIdx),
    seedGraphIndex: Int32Array.from(seedIdx),
    seedLabels,
    nLineBothVerts: ob.i,
    nLineOneVerts: oo.i,
    nLineBgVerts: ogb.i,
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
      edgesBackgroundDrawn: drawnBg,
      edgesBackgroundPool: drawAllNonSeedEdges ? bgPool : 0,
      edgesBackgroundSubmitted: drawAllNonSeedEdges ? bgSubmitted : 0,
    },
  };
}
