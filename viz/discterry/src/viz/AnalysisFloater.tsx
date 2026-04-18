import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphBundle } from "../data/loadBundle";

type TabId = "deg" | "angular";
type DegYMetric = "kappa" | "hypRad";

const SCATTER_MAX_POINTS = 12000;
const MARGIN = 40;

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
  bundle: GraphBundle,
  seedNames: string[],
) {
  const w = canvas.width;
  const h = canvas.height;
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
  const cell = Math.min(28, Math.floor((w - labelW - MARGIN) / k), Math.floor((h - MARGIN * 2) / k));
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
      const r = Math.floor(30 + t * 180);
      const g = Math.floor(40 + (1 - t) * 120);
      const b = Math.floor(80 + t * 100);
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
  ctx.fillStyle = "#8e8e96";
  ctx.font = "10px system-ui,sans-serif";
  ctx.fillText("cos(angle) seed×seed on Inf.Pos", ox, h - 8);
}

type Props = {
  open: boolean;
  onClose: () => void;
  bundle: GraphBundle | null;
  degrees: Int32Array | null;
  /** Applied seed names (order preserved for matrix labels). */
  seedNamesOrdered: string[];
};

export function AnalysisFloater({ open, onClose, bundle, degrees, seedNamesOrdered }: Props) {
  const [tab, setTab] = useState<TabId>("deg");
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

  const redraw = useCallback(() => {
    const el = canvasRef.current;
    if (!el || !open) return;
    const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
    const rect = el.getBoundingClientRect();
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
    } else {
      drawSeedCosineHeatmap(el, bundle, seedNamesOrdered);
    }
  }, [open, bundle, degrees, tab, stride, seedNamesOrdered, degYMetric]);

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
      </div>
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
      <div className="analysisFloaterBody">
        <canvas ref={canvasRef} className="analysisFloaterCanvas" />
      </div>
    </div>
  );
}

