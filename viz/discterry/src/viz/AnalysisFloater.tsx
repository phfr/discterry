import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphBundle } from "../data/loadBundle";
import type { Complex } from "../math/mobius";
import type { GraphCSR } from "../model/graphSearch";
import {
  buildOrderedSeedIndices,
  collectScatterPoints,
  fillEuclideanWPairMatrix,
  fillGraphHopPairMatrix,
  fillHyperbolicPairMatrix,
  fillMobiusW,
  pearsonCorrelation,
  SEED_PAIR_METRICS_MAX_SEEDS,
} from "../model/seedPairMetrics";

type TabId = "deg" | "angular" | "seedPairs";
type SeedPairView = "hyp" | "eucl" | "hops" | "scatter";
type DegYMetric = "kappa" | "hypRad";

const SCATTER_MAX_POINTS = 12000;
const MARGIN = 40;
/** Space reserved under heatmap matrix for color bar + labels + title. */
const HEATMAP_LEGEND_RESERVE = 50;

function rgbCosineHeat(t: number): [number, number, number] {
  return [Math.floor(30 + t * 180), Math.floor(40 + (1 - t) * 120), Math.floor(80 + t * 100)];
}

function rgbSeedPairHeat(t: number): [number, number, number] {
  return [Math.floor(40 + t * 140), Math.floor(50 + (1 - t) * 100), Math.floor(120 + t * 90)];
}

/** Horizontal gradient bar: low t = min value, high t = max value. */
function drawHeatmapColorLegend(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  minV: number,
  maxV: number,
  format: (v: number) => string,
  rgbAtT: (t: number) => [number, number, number],
  title: string,
  legendExtras?: { disconnected?: { rgb: [number, number, number]; label: string } },
) {
  const hasDisc = !!legendExtras?.disconnected;
  const barH = 10;
  const barY = h - (hasDisc ? 38 : 26);
  const x0 = MARGIN;
  const barW = Math.max(60, w - 2 * MARGIN);
  const steps = Math.min(120, Math.max(24, Math.floor(barW)));
  for (let s = 0; s < steps; s++) {
    const t = s / (steps - 1);
    const [r, g, b] = rgbAtT(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    const px = x0 + (s / (steps - 1)) * barW;
    ctx.fillRect(px, barY, Math.ceil(barW / steps) + 0.5, barH);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.strokeRect(x0, barY, barW, barH);
  ctx.fillStyle = "#b0b0bc";
  ctx.font = "9px system-ui,sans-serif";
  ctx.textBaseline = "bottom";
  ctx.textAlign = "left";
  ctx.fillText(format(minV), x0, barY - 2);
  ctx.textAlign = "right";
  ctx.fillText(format(maxV), x0 + barW, barY - 2);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  if (legendExtras?.disconnected) {
    const { rgb, label } = legendExtras.disconnected;
    const sw = 14;
    const sy = h - 18;
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(x0, sy, sw, 8);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.strokeRect(x0, sy, sw, 8);
    ctx.fillStyle = "#9595a0";
    ctx.font = "9px system-ui,sans-serif";
    ctx.fillText(label, x0 + sw + 5, sy + 7);
  }
  ctx.textAlign = "center";
  ctx.fillStyle = "#8e8e96";
  ctx.font = "10px system-ui,sans-serif";
  const ttext = title.length > 78 ? `${title.slice(0, 77)}…` : title;
  ctx.fillText(ttext, w / 2, h - 6);
  ctx.textAlign = "left";
}

function norm3(a: number, b: number, c: number): [number, number, number] {
  const n = Math.hypot(a, b, c);
  if (n < 1e-20) return [0, 0, 0];
  return [a / n, b / n, c / n];
}

function drawDegreeMetricScatter(
  canvas: HTMLCanvasElement,
  bundle: GraphBundle,
  degrees: Int32Array,
  stride: number,
  yMetric: DegYMetric,
) {
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || w < 10 || h < 10) return;
  ctx.fillStyle = "#121218";
  ctx.fillRect(0, 0, w, h);
  const yCol = yMetric === "kappa" ? bundle.infKappa : bundle.infHypRad;
  const yNeed = yMetric === "kappa" ? "inf_kappa" : "inf_hyp_rad";
  if (!yCol) {
    ctx.fillStyle = "#9a9aa8";
    ctx.font = "12px system-ui,sans-serif";
    ctx.fillText(`Re-export nodes.parquet (needs ${yNeed}).`, MARGIN, MARGIN + 8);
    return;
  }
  const n = bundle.vertex.length;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i += stride) {
    xs.push(degrees[i]!);
    ys.push(yCol[i]!);
  }
  let minD = Infinity;
  let maxD = -Infinity;
  let minK = Infinity;
  let maxK = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    minD = Math.min(minD, xs[i]!);
    maxD = Math.max(maxD, xs[i]!);
    minK = Math.min(minK, ys[i]!);
    maxK = Math.max(maxK, ys[i]!);
  }
  if (minD === maxD) {
    minD -= 1;
    maxD += 1;
  }
  if (minK === maxK) {
    minK -= 1;
    maxK += 1;
  }
  const plotW = w - 2 * MARGIN;
  const plotH = h - 2 * MARGIN;
  const sx = (d: number) => MARGIN + ((d - minD) / (maxD - minD)) * plotW;
  const sy = (k: number) => h - MARGIN - ((k - minK) / (maxK - minK)) * plotH;
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  for (let t = 0; t <= 4; t++) {
    const x = MARGIN + (t / 4) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, MARGIN);
    ctx.lineTo(x, h - MARGIN);
    ctx.stroke();
    const y = MARGIN + (t / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(MARGIN, y);
    ctx.lineTo(w - MARGIN, y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(120, 200, 255, 0.55)";
  for (let i = 0; i < xs.length; i++) {
    ctx.beginPath();
    ctx.arc(sx(xs[i]!), sy(ys[i]!), 1.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#b8b8c8";
  ctx.font = "11px system-ui,sans-serif";
  ctx.fillText("degree", w / 2 - 24, h - 10);
  ctx.save();
  ctx.translate(12, h / 2 + 24);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yMetric === "kappa" ? "Inf.Kappa" : "Inf.Hyp.Rad", 0, 0);
  ctx.restore();
  if (stride > 1) {
    ctx.fillStyle = "#6e6e78";
    ctx.font = "10px system-ui,sans-serif";
    ctx.fillText(`subsampled (stride ${stride})`, MARGIN, 14);
  }
}

function drawSeedCosineHeatmap(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
  bundle: GraphBundle,
  seedNames: string[],
) {
  const w = cssW;
  const h = cssH;
  const ctx = canvas.getContext("2d");
  if (!ctx || w < 10 || h < 10) return;
  ctx.fillStyle = "#121218";
  ctx.fillRect(0, 0, w, h);
  if (!bundle.infPos1 || !bundle.infPos2 || !bundle.infPos3) {
    ctx.fillStyle = "#9a9aa8";
    ctx.font = "12px system-ui,sans-serif";
    ctx.fillText("Re-export nodes.parquet (needs inf_pos_*).", MARGIN, MARGIN + 8);
    return;
  }
  const idx: number[] = [];
  const names: string[] = [];
  for (const name of seedNames) {
    const j = bundle.nameToIndex.get(name.trim());
    if (j !== undefined) {
      idx.push(j);
      names.push(name.trim());
    }
  }
  const k = idx.length;
  if (k === 0) {
    ctx.fillStyle = "#9a9aa8";
    ctx.fillText("No applied seeds in bundle.", MARGIN, MARGIN + 8);
    return;
  }
  const norms: [number, number, number][] = [];
  for (const j of idx) {
    norms.push(
      norm3(bundle.infPos1![j]!, bundle.infPos2![j]!, bundle.infPos3![j]!),
    );
  }
  const cosMat: number[][] = [];
  for (let i = 0; i < k; i++) {
    cosMat[i] = [];
    for (let j = 0; j < k; j++) {
      const a = norms[i]!;
      const b = norms[j]!;
      cosMat[i]![j] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
  }
  const labelW = Math.min(100, Math.floor(w * 0.22));
  const cell = Math.min(
    28,
    Math.floor((w - labelW - MARGIN) / k),
    Math.floor((h - MARGIN * 2 - HEATMAP_LEGEND_RESERVE) / k),
  );
  const ox = labelW + 8;
  const oy = MARGIN;
  let minC = 1;
  let maxC = -1;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = cosMat[i]![j]!;
      minC = Math.min(minC, v);
      maxC = Math.max(maxC, v);
    }
  }
  if (minC === maxC) {
    minC -= 0.01;
    maxC += 0.01;
  }
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = cosMat[i]![j]!;
      const t = (v - minC) / (maxC - minC);
      const [r, g, b] = rgbCosineHeat(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.fillStyle = "#c4c4d0";
  ctx.font = "10px system-ui,sans-serif";
  for (let j = 0; j < k; j++) {
    const name = names[j] ?? "";
    const short = name.length > 12 ? `${name.slice(0, 11)}…` : name;
    ctx.save();
    ctx.translate(ox + j * cell + cell / 2, oy - 4);
    ctx.rotate(-0.35);
    ctx.fillText(short, 0, 0);
    ctx.restore();
  }
  for (let i = 0; i < k; i++) {
    const name = names[i] ?? "";
    const short = name.length > 10 ? `${name.slice(0, 9)}…` : name;
    ctx.fillText(short, 4, oy + i * cell + cell / 2 + 3);
  }
  drawHeatmapColorLegend(
    ctx,
    w,
    h,
    minC,
    maxC,
    (v) => v.toFixed(2),
    rgbCosineHeat,
    "cos(angle) seed×seed on Inf.Pos",
  );
}

function drawSeedPairHeatmap(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
  values: Float32Array | Int32Array,
  k: number,
  names: string[],
  opts: { title: string; isHops: boolean },
) {
  const w = cssW;
  const h = cssH;
  const ctx = canvas.getContext("2d");
  if (!ctx || w < 10 || h < 10) return;
  ctx.fillStyle = "#121218";
  ctx.fillRect(0, 0, w, h);
  if (k < 2) {
    ctx.fillStyle = "#9a9aa8";
    ctx.font = "12px system-ui,sans-serif";
    ctx.fillText("Need ≥2 applied seeds.", MARGIN, MARGIN + 8);
    return;
  }
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      const v = values[i * k + j]!;
      if (opts.isHops && v < 0) continue;
      if (!opts.isHops && !Number.isFinite(v)) continue;
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  }
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) {
    minV = 0;
    maxV = 1;
  } else if (minV === maxV) {
    minV -= 1e-6;
    maxV += 1e-6;
  }
  const labelW = Math.min(100, Math.floor(w * 0.22));
  const cell = Math.min(
    28,
    Math.floor((w - labelW - MARGIN) / k),
    Math.floor((h - MARGIN * 2 - HEATMAP_LEGEND_RESERVE) / k),
  );
  const ox = labelW + 8;
  const oy = MARGIN;
  let anyDisconnected = false;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = values[i * k + j]!;
      if (opts.isHops && v < 0) {
        anyDisconnected = true;
        ctx.fillStyle = "rgb(45,42,58)";
        ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
        continue;
      }
      if (!opts.isHops && (!Number.isFinite(v) || v > 1e9)) {
        ctx.fillStyle = "rgb(90,35,35)";
        ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
        continue;
      }
      const t = i === j ? 0 : (v - minV) / (maxV - minV);
      const [r, g, b] = rgbSeedPairHeat(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.fillStyle = "#c4c4d0";
  ctx.font = "10px system-ui,sans-serif";
  for (let j = 0; j < k; j++) {
    const name = names[j] ?? "";
    const short = name.length > 12 ? `${name.slice(0, 11)}…` : name;
    ctx.save();
    ctx.translate(ox + j * cell + cell / 2, oy - 4);
    ctx.rotate(-0.35);
    ctx.fillText(short, 0, 0);
    ctx.restore();
  }
  for (let i = 0; i < k; i++) {
    const name = names[i] ?? "";
    const short = name.length > 10 ? `${name.slice(0, 9)}…` : name;
    ctx.fillText(short, 4, oy + i * cell + cell / 2 + 3);
  }
  const fmt = opts.isHops
    ? (v: number) => String(Math.round(v))
    : (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));
  drawHeatmapColorLegend(ctx, w, h, minV, maxV, fmt, rgbSeedPairHeat, opts.title, {
    disconnected:
      opts.isHops && anyDisconnected
        ? { rgb: [45, 42, 58], label: "disconnected" }
        : undefined,
  });
}

function drawSeedPairScatter(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
  pts: { hops: number; hyp: number }[],
  r: number | null,
) {
  const w = cssW;
  const h = cssH;
  const ctx = canvas.getContext("2d");
  if (!ctx || w < 10 || h < 10) return;
  ctx.fillStyle = "#121218";
  ctx.fillRect(0, 0, w, h);
  if (pts.length === 0) {
    ctx.fillStyle = "#9a9aa8";
    ctx.font = "12px system-ui,sans-serif";
    ctx.fillText("No finite pairs (check hops / hyperbolic distances).", MARGIN, MARGIN + 8);
    return;
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.hops);
    maxX = Math.max(maxX, p.hops);
    minY = Math.min(minY, p.hyp);
    maxY = Math.max(maxY, p.hyp);
  }
  if (minX === maxX) {
    minX -= 0.5;
    maxX += 0.5;
  }
  if (minY === maxY) {
    minY -= 1e-6;
    maxY += 1e-6;
  }
  const plotW = w - 2 * MARGIN;
  const plotH = h - 2 * MARGIN - 22;
  const sx = (x: number) => MARGIN + ((x - minX) / (maxX - minX)) * plotW;
  const sy = (y: number) => h - MARGIN - 18 - ((y - minY) / (maxY - minY)) * plotH;
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  for (let t = 0; t <= 4; t++) {
    const x = MARGIN + (t / 4) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, MARGIN);
    ctx.lineTo(x, h - MARGIN - 18);
    ctx.stroke();
    const y = MARGIN + (t / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(MARGIN, y);
    ctx.lineTo(w - MARGIN, y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(140, 200, 255, 0.75)";
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(sx(p.hops), sy(p.hyp), 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#b8b8c8";
  ctx.font = "11px system-ui,sans-serif";
  ctx.fillText("graph hops (shortest path)", w / 2 - 70, h - 8);
  ctx.save();
  ctx.translate(12, h / 2 - 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("hyperbolic dist (W)", 0, 0);
  ctx.restore();
  if (r !== null) {
    ctx.fillStyle = "#8e8e96";
    ctx.font = "10px system-ui,sans-serif";
    ctx.fillText(`Pearson r ≈ ${r.toFixed(3)}`, MARGIN, 14);
  }
}

type Props = {
  open: boolean;
  onClose: () => void;
  bundle: GraphBundle | null;
  degrees: Int32Array | null;
  /** Applied seed names (order preserved for matrix labels). */
  seedNamesOrdered: string[];
  z0: Complex | null;
  csr: GraphCSR | null;
};

export function AnalysisFloater({
  open,
  onClose,
  bundle,
  degrees,
  seedNamesOrdered,
  z0,
  csr,
}: Props) {
  const [tab, setTab] = useState<TabId>("deg");
  const [seedPairView, setSeedPairView] = useState<SeedPairView>("hyp");
  const [degYMetric, setDegYMetric] = useState<DegYMetric>("kappa");
  const [pos, setPos] = useState({ left: 16, bottom: 16 });
  const dragRef = useRef<{ sx: number; sy: number; l0: number; b0: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);

  const stride = useMemo(() => {
    if (!bundle || !degrees) return 1;
    const n = bundle.vertex.length;
    return n > SCATTER_MAX_POINTS ? Math.ceil(n / SCATTER_MAX_POINTS) : 1;
  }, [bundle, degrees]);

  type SeedPairMatrices =
    | { error: string }
    | {
        k: number;
        idx: number[];
        names: string[];
        hyp: Float32Array;
        eucl: Float32Array;
        hops: Int32Array | null;
        hopError: string | null;
      };

  const seedPairMatrices = useMemo((): SeedPairMatrices | null => {
    if (tab !== "seedPairs" || !bundle) return null;
    if (!z0) return { error: "No focus embedding (z₀). Set a focus protein." };
    const { idx, names } = buildOrderedSeedIndices(bundle, seedNamesOrdered);
    if (idx.length < 2) return { error: "Need ≥2 applied seeds." };
    const n = bundle.vertex.length;
    const wx = new Float32Array(n);
    const wy = new Float32Array(n);
    fillMobiusW(bundle, z0, wx, wy);
    const k = idx.length;
    const hyp = new Float32Array(k * k);
    const eucl = new Float32Array(k * k);
    fillHyperbolicPairMatrix(wx, wy, idx, hyp);
    fillEuclideanWPairMatrix(wx, wy, idx, eucl);
    let hops: Int32Array | null = null;
    let hopError: string | null = null;
    if (csr && (seedPairView === "hops" || seedPairView === "scatter")) {
      if (k > SEED_PAIR_METRICS_MAX_SEEDS) {
        hopError = `Hop metrics need ≤${SEED_PAIR_METRICS_MAX_SEEDS} seeds (have ${k}).`;
      } else {
        hops = new Int32Array(k * k);
        const scratch = new Int32Array(csr.n);
        fillGraphHopPairMatrix(csr, idx, hops, scratch);
      }
    }
    return { k, idx, names, hyp, eucl, hops, hopError };
  }, [tab, bundle, z0, seedNamesOrdered, csr, seedPairView]);

  const seedPairSummary = useMemo(() => {
    if (tab !== "seedPairs" || !seedPairMatrices) return null;
    if ("error" in seedPairMatrices) return seedPairMatrices.error;
    const { k, hyp, eucl, hops, hopError } = seedPairMatrices;
    let sumH = 0;
    let cntH = 0;
    let sumE = 0;
    let cntE = 0;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        const d = hyp[i * k + j]!;
        if (Number.isFinite(d) && d < 1e9) {
          sumH += d;
          cntH++;
        }
        sumE += eucl[i * k + j]!;
        cntE++;
      }
    }
    const meanHyp = cntH ? sumH / cntH : null;
    const meanEucl = cntE ? sumE / cntE : null;
    let hopPart = "";
    if (hopError) hopPart = ` ${hopError}`;
    else if (hops) {
      let sumG = 0;
      let cntG = 0;
      let disc = 0;
      for (let i = 0; i < k; i++) {
        for (let j = i + 1; j < k; j++) {
          const g = hops[i * k + j]!;
          if (g < 0) disc++;
          else {
            sumG += g;
            cntG++;
          }
        }
      }
      hopPart = ` Mean graph hops (connected pairs): ${cntG ? (sumG / cntG).toFixed(2) : "—"}. Disconnected pairs: ${disc}.`;
    }
    return `${k} seeds. Mean hyperbolic W (pairs): ${meanHyp !== null ? meanHyp.toFixed(4) : "—"}. Mean |ΔW|₂ (pairs): ${meanEucl !== null ? meanEucl.toFixed(4) : "—"}.${hopPart}`;
  }, [tab, seedPairMatrices]);

  const redraw = useCallback(() => {
    const el = canvasRef.current;
    if (!el || !open) return;
    const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
    const rect = el.getBoundingClientRect();
    const cwCss = Math.max(1, rect.width);
    const chCss = Math.max(1, rect.height);
    const cw = Math.max(1, Math.floor(rect.width * dpr));
    const ch = Math.max(1, Math.floor(rect.height * dpr));
    if (el.width !== cw || el.height !== ch) {
      el.width = cw;
      el.height = ch;
    }
    const ctx = el.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!bundle) return;
    if (tab === "deg") {
      if (!degrees) return;
      drawDegreeMetricScatter(el, bundle, degrees, stride, degYMetric);
    } else if (tab === "angular") {
      drawSeedCosineHeatmap(el, cwCss, chCss, bundle, seedNamesOrdered);
    } else if (tab === "seedPairs") {
      if (!seedPairMatrices) return;
      if ("error" in seedPairMatrices) {
        const c2 = el.getContext("2d");
        if (!c2) return;
        c2.fillStyle = "#121218";
        c2.fillRect(0, 0, cwCss, chCss);
        c2.fillStyle = "#9a9aa8";
        c2.font = "12px system-ui,sans-serif";
        const msg = seedPairMatrices.error;
        for (let i = 0, y = MARGIN + 8; i < 4; i++, y += 16) {
          const line = msg.slice(i * 52, (i + 1) * 52);
          if (!line) break;
          c2.fillText(line, MARGIN, y);
        }
        return;
      }
      const { k, names, hyp, eucl, hops, hopError } = seedPairMatrices;
      if (seedPairView === "hyp") {
        drawSeedPairHeatmap(el, cwCss, chCss, hyp, k, names, {
          title: "Hyperbolic distance (W) seed×seed at current focus",
          isHops: false,
        });
      } else if (seedPairView === "eucl") {
        drawSeedPairHeatmap(el, cwCss, chCss, eucl, k, names, {
          title: "Euclidean chord in W plane seed×seed",
          isHops: false,
        });
      } else if (seedPairView === "hops") {
        if (hopError || !hops) {
          const c2 = el.getContext("2d");
          if (!c2) return;
          c2.fillStyle = "#121218";
          c2.fillRect(0, 0, cwCss, chCss);
          c2.fillStyle = "#9a9aa8";
          c2.font = "12px system-ui,sans-serif";
          c2.fillText(hopError ?? "Graph not ready.", MARGIN, MARGIN + 8);
        } else {
          drawSeedPairHeatmap(el, cwCss, chCss, hops, k, names, {
            title: "Shortest-path hop count seed×seed (dark = disconnected)",
            isHops: true,
          });
        }
      } else {
        if (hopError || !hops) {
          const c2 = el.getContext("2d");
          if (!c2) return;
          c2.fillStyle = "#121218";
          c2.fillRect(0, 0, cwCss, chCss);
          c2.fillStyle = "#9a9aa8";
          c2.font = "12px system-ui,sans-serif";
          c2.fillText(hopError ?? "Need CSR for scatter.", MARGIN, MARGIN + 8);
        } else {
          const pts = collectScatterPoints(hyp, hops, k);
          const xs = pts.map((p) => p.hops);
          const ys = pts.map((p) => p.hyp);
          const r = pearsonCorrelation(xs, ys);
          drawSeedPairScatter(el, cwCss, chCss, pts, r);
        }
      }
    }
  }, [
    open,
    bundle,
    degrees,
    tab,
    stride,
    seedNamesOrdered,
    degYMetric,
    seedPairMatrices,
    seedPairView,
  ]);

  useEffect(() => {
    if (!open) return;
    redraw();
    const ro = new ResizeObserver(() => redraw());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [open, redraw]);

  const onHeaderPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    dragRef.current = {
      sx: e.clientX,
      sy: e.clientY,
      l0: pos.left,
      b0: pos.bottom,
    };
    headerRef.current?.setPointerCapture(e.pointerId);
  };
  const onHeaderPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    setPos({
      left: d.l0 + (e.clientX - d.sx),
      bottom: d.b0 + (d.sy - e.clientY),
    });
  };
  const onHeaderPointerUp = (e: React.PointerEvent) => {
    dragRef.current = null;
    try {
      headerRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      /* noop */
    }
  };

  if (!open) return null;

  return (
    <div
      className="analysisFloater"
      style={{ left: pos.left, bottom: pos.bottom }}
      ref={wrapRef}
      role="dialog"
      aria-label="Analysis"
    >
      <div
        ref={headerRef}
        className="analysisFloaterHeader"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        onPointerCancel={onHeaderPointerUp}
      >
        <span className="analysisFloaterTitle">Analysis</span>
        <button
          type="button"
          className="analysisFloaterClose"
          onClick={onClose}
          onPointerDown={(e) => e.stopPropagation()}
          aria-label="Close analysis"
        >
          ×
        </button>
      </div>
      <div className="analysisFloaterTabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "deg"}
          className={tab === "deg" ? "analysisTab analysisTabOn" : "analysisTab"}
          onClick={() => setTab("deg")}
        >
          Degree · κ · r
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "angular"}
          className={tab === "angular" ? "analysisTab analysisTabOn" : "analysisTab"}
          onClick={() => setTab("angular")}
        >
          Angular
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "seedPairs"}
          className={tab === "seedPairs" ? "analysisTab analysisTabOn" : "analysisTab"}
          onClick={() => setTab("seedPairs")}
        >
          Seed pairs
        </button>
      </div>
      {tab === "seedPairs" ? (
        <div className="analysisDegYRow" role="group" aria-label="Seed pair metrics">
          <label className="analysisDegYLabel">
            <input
              type="radio"
              name="seedPairV"
              checked={seedPairView === "hyp"}
              onChange={() => setSeedPairView("hyp")}
            />
            W hyp
          </label>
          <label className="analysisDegYLabel">
            <input
              type="radio"
              name="seedPairV"
              checked={seedPairView === "eucl"}
              onChange={() => setSeedPairView("eucl")}
            />
            W Euclid
          </label>
          <label className="analysisDegYLabel">
            <input
              type="radio"
              name="seedPairV"
              checked={seedPairView === "hops"}
              onChange={() => setSeedPairView("hops")}
            />
            Graph hops
          </label>
          <label className="analysisDegYLabel">
            <input
              type="radio"
              name="seedPairV"
              checked={seedPairView === "scatter"}
              onChange={() => setSeedPairView("scatter")}
            />
            Hops vs W
          </label>
        </div>
      ) : null}
      {tab === "deg" && bundle?.infHypRad ? (
        <div className="analysisDegYRow" role="group" aria-label="Y axis metric">
          <label className="analysisDegYLabel">
            <input
              type="radio"
              name="degY"
              checked={degYMetric === "kappa"}
              onChange={() => setDegYMetric("kappa")}
            />
            Y: κ
          </label>
          <label className="analysisDegYLabel">
            <input
              type="radio"
              name="degY"
              checked={degYMetric === "hypRad"}
              onChange={() => setDegYMetric("hypRad")}
            />
            Y: Hyp.Rad
          </label>
        </div>
      ) : null}
      {tab === "seedPairs" && seedPairSummary ? (
        <div className="analysisSeedPairSummary">{seedPairSummary}</div>
      ) : null}
      <div className="analysisFloaterBody">
        <canvas ref={canvasRef} className="analysisFloaterCanvas" />
      </div>
    </div>
  );
}

