import type { GraphBundle, MetaJson } from "./data/loadBundle";
import type { GraphBundle3d } from "./data/loadBundle3d";

function fmtDisk(n: number): string {
  return Number.isFinite(n) ? n.toFixed(6) : String(n);
}

function appendRunMetaLines(lines: string[], runMeta: MetaJson | null | undefined): void {
  if (!runMeta) return;
  const bits: string[] = [];
  if (runMeta.beta !== undefined) bits.push(`β=${fmtDisk(runMeta.beta)}`);
  if (runMeta.mu !== undefined) bits.push(`μ=${fmtDisk(runMeta.mu)}`);
  if (runMeta.dimension !== undefined) bits.push(`D=${runMeta.dimension}`);
  if (runMeta.radius_s_d !== undefined) bits.push(`radius_S^D=${fmtDisk(runMeta.radius_s_d)}`);
  if (runMeta.radius_h_d1 !== undefined) bits.push(`radius_H_D+1=${fmtDisk(runMeta.radius_h_d1)}`);
  if (runMeta.kappa_min !== undefined) bits.push(`κ_min=${fmtDisk(runMeta.kappa_min)}`);
  if (bits.length) lines.push(`run: ${bits.join("  ")}`);
  if (runMeta.run_subdir) lines.push(`run_subdir: ${runMeta.run_subdir}`);
  if (bits.length || runMeta.run_subdir) lines.push("—");
}

/** Disk canvas hover: headline + monospace details (no duplicate name/degree in body). */
export type NodeDiskHoverTooltip = {
  name: string;
  degree: string;
  details: string;
};

function appendGeometryLines(lines: string[], bundle: GraphBundle, j: number): void {
  const x = bundle.x[j];
  const y = bundle.y[j];
  const rz = Math.hypot(x, y);
  const ang = (Math.atan2(y, x) * 180) / Math.PI;
  lines.push(
    `index: ${j}`,
    `disk x: ${fmtDisk(x)}`,
    `disk y: ${fmtDisk(y)}`,
    `|z|: ${fmtDisk(rz)}`,
    `arg z (deg): ${fmtDisk(ang)}`,
  );
  if (bundle.infKappa) lines.push(`Inf.Kappa: ${fmtDisk(bundle.infKappa[j]!)}`);
  if (bundle.infHypRad) lines.push(`Inf.Hyp.Rad: ${fmtDisk(bundle.infHypRad[j]!)}`);
  if (bundle.infPos1 && bundle.infPos2 && bundle.infPos3) {
    lines.push(
      `Inf.Pos.1: ${fmtDisk(bundle.infPos1[j]!)}`,
      `Inf.Pos.2: ${fmtDisk(bundle.infPos2[j]!)}`,
      `Inf.Pos.3: ${fmtDisk(bundle.infPos3[j]!)}`,
    );
  }
}

/** Rich hover for WebGL disk node cursor (structured; rendered in DiskView). */
export function nodeDiskHoverTooltipForIndex(
  bundle: GraphBundle,
  graphIndex: number,
  degrees: Int32Array | null,
  runMeta?: MetaJson | null,
): NodeDiskHoverTooltip | null {
  if (graphIndex < 0 || graphIndex >= bundle.vertex.length) return null;
  const name = bundle.vertex[graphIndex]!;
  const d = degrees && graphIndex < degrees.length ? degrees[graphIndex]! : null;
  const degreeStr = d !== null ? String(d) : "—";
  const lines: string[] = [];
  appendRunMetaLines(lines, runMeta);
  appendGeometryLines(lines, bundle, graphIndex);
  return { name, degree: degreeStr, details: lines.join("\n") };
}

function appendGeometryLines3d(lines: string[], bundle: GraphBundle3d, j: number): void {
  lines.push(
    `index: ${j}`,
    `ball x: ${fmtDisk(bundle.x[j])}`,
    `ball y: ${fmtDisk(bundle.y[j])}`,
    `ball z: ${fmtDisk(bundle.z[j])}`,
    `|p|: ${fmtDisk(Math.hypot(bundle.x[j], bundle.y[j], bundle.z[j]))}`,
  );
  if (bundle.infKappa) lines.push(`Inf.Kappa: ${fmtDisk(bundle.infKappa[j]!)}`);
  if (bundle.infHypRad) lines.push(`Inf.Hyp.Rad: ${fmtDisk(bundle.infHypRad[j]!)}`);
  if (bundle.infPos1 && bundle.infPos2 && bundle.infPos3 && bundle.infPos4) {
    lines.push(
      `Inf.Pos.1: ${fmtDisk(bundle.infPos1[j]!)}`,
      `Inf.Pos.2: ${fmtDisk(bundle.infPos2[j]!)}`,
      `Inf.Pos.3: ${fmtDisk(bundle.infPos3[j]!)}`,
      `Inf.Pos.4: ${fmtDisk(bundle.infPos4[j]!)}`,
    );
  }
}

/** Hover payload for 3D Poincaré-ball view (`#3d`). */
export function nodeDiskHoverTooltipForGraph3d(
  bundle: GraphBundle3d,
  graphIndex: number,
  degrees: Int32Array | null,
  runMeta?: MetaJson | null,
): NodeDiskHoverTooltip | null {
  if (graphIndex < 0 || graphIndex >= bundle.vertex.length) return null;
  const name = bundle.vertex[graphIndex]!;
  const d = degrees && graphIndex < degrees.length ? degrees[graphIndex]! : null;
  const degreeStr = d !== null ? String(d) : "—";
  const lines: string[] = [];
  appendRunMetaLines(lines, runMeta);
  appendGeometryLines3d(lines, bundle, graphIndex);
  return { name, degree: degreeStr, details: lines.join("\n") };
}

/** Same text as the focus list `title` — Parquet disk coords, D-Mercator columns, degree, optional run header. */
export function nodeListTooltip(
  bundle: GraphBundle,
  name: string,
  degree: number | null,
  runMeta?: MetaJson | null,
): string {
  const j = bundle.nameToIndex.get(name);
  if (j === undefined) return name;
  const lines: string[] = [];
  appendRunMetaLines(lines, runMeta);
  lines.push(`vertex: ${bundle.vertex[j]}`);
  appendGeometryLines(lines, bundle, j);
  lines.push(`degree: ${degree !== null ? String(degree) : "—"}`);
  return lines.join("\n");
}

export function nodeListTooltipForIndex(
  bundle: GraphBundle,
  graphIndex: number,
  degrees: Int32Array | null,
  runMeta?: MetaJson | null,
): string {
  if (graphIndex < 0 || graphIndex >= bundle.vertex.length) return "";
  const d = degrees && graphIndex < degrees.length ? degrees[graphIndex]! : null;
  return nodeListTooltip(bundle, bundle.vertex[graphIndex]!, d, runMeta);
}

/** Native-style list `title` for 3D bundle (ball coords + Inf.* + degree). */
export function nodeListTooltip3d(
  bundle: GraphBundle3d,
  name: string,
  degree: number | null,
  runMeta?: MetaJson | null,
): string {
  const j = bundle.nameToIndex.get(name);
  if (j === undefined) return name;
  const lines: string[] = [];
  appendRunMetaLines(lines, runMeta);
  lines.push(`vertex: ${bundle.vertex[j]}`);
  appendGeometryLines3d(lines, bundle, j);
  lines.push(`degree: ${degree !== null ? String(degree) : "—"}`);
  return lines.join("\n");
}
