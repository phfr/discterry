import type { GraphBundle } from "../data/loadBundle";
import type { GraphBundle3d } from "../data/loadBundle3d";
import type { Complex } from "../math/mobius";
import { mobiusDiskArrays } from "../math/mobius";
import { mobiusBallArrays } from "../math/poincareBall";
import { primHyperbolicSeedEdges, primHyperbolicSeedEdgesBall } from "./graphSearch";
import { buildPathOverlayFromEdges, buildPathOverlayFromEdges3d } from "./pathOverlayBuffer";
import type { PathOverlayBuffer } from "./pathOverlayBuffer";

function parseSeeds(text: string): Set<string> {
  const s = new Set<string>();
  for (const part of text.split(/\s+/)) {
    const t = part.trim();
    if (t) s.add(t);
  }
  return s;
}

export type PrimMstOverlayFailureReason =
  | "need_two_seeds"
  | "too_many_seeds"
  | "clip";

export type PrimMstOverlayResult =
  | { ok: true; buf: PathOverlayBuffer; seedCount: number }
  | { ok: false; reason: PrimMstOverlayFailureReason };

/** Same geometry as PathTrees “Show Prim MST (seeds)”: Prim on applied seeds in hyperbolic W at `z0`. */
export function tryBuildPrimMstOverlay(
  bundle: GraphBundle,
  z0: Complex,
  appliedSeedsText: string,
): PrimMstOverlayResult {
  const n = bundle.vertex.length;
  const wx = new Float32Array(n);
  const wy = new Float32Array(n);
  mobiusDiskArrays(bundle.x, bundle.y, z0, wx, wy, n);

  const seeds = parseSeeds(appliedSeedsText);
  const idx: number[] = [];
  const seen = new Set<number>();
  for (const name of seeds) {
    const j = bundle.nameToIndex.get(name);
    if (j !== undefined && !seen.has(j)) {
      seen.add(j);
      idx.push(j);
    }
  }
  if (idx.length < 2) return { ok: false, reason: "need_two_seeds" };
  const { edges, skipped } = primHyperbolicSeedEdges(wx, wy, idx);
  if (skipped) return { ok: false, reason: "too_many_seeds" };
  const buf = buildPathOverlayFromEdges(bundle, z0, edges);
  if (!buf) return { ok: false, reason: "clip" };
  return { ok: true, buf, seedCount: idx.length };
}

/** Prim MST on seeds in the Poincaré ball at focus ``p₀`` (same combinatorics as disk overlay). */
export function tryBuildPrimMstOverlay3d(
  bundle: GraphBundle3d,
  p0x: number,
  p0y: number,
  p0z: number,
  appliedSeedsText: string,
): PrimMstOverlayResult {
  const n = bundle.vertex.length;
  const wx = new Float32Array(n);
  const wy = new Float32Array(n);
  const wz = new Float32Array(n);
  mobiusBallArrays(bundle.x, bundle.y, bundle.z, p0x, p0y, p0z, wx, wy, wz, n);

  const seeds = parseSeeds(appliedSeedsText);
  const idx: number[] = [];
  const seen = new Set<number>();
  for (const name of seeds) {
    const j = bundle.nameToIndex.get(name);
    if (j !== undefined && !seen.has(j)) {
      seen.add(j);
      idx.push(j);
    }
  }
  if (idx.length < 2) return { ok: false, reason: "need_two_seeds" };
  const { edges, skipped } = primHyperbolicSeedEdgesBall(wx, wy, wz, idx);
  if (skipped) return { ok: false, reason: "too_many_seeds" };
  const buf = buildPathOverlayFromEdges3d(bundle, p0x, p0y, p0z, edges);
  if (!buf) return { ok: false, reason: "clip" };
  return { ok: true, buf, seedCount: idx.length };
}
