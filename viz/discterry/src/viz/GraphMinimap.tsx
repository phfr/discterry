import { useCallback, useLayoutEffect, useMemo, useRef } from "react";
import type { GraphBundle } from "../data/loadBundle";
import type { GraphBundle3d } from "../data/loadBundle3d";
import type { Complex } from "../math/mobius";
import { clampP0 } from "../math/poincareBall";
import type { Vec3 } from "../p0FromProtein";
import {
  boundsNativeZDisk,
  boundsStereoRaw,
  stereoNorthFromBallChart,
  stereoNorthToUnitSphere,
  nativeZClickToComplex,
  stereoMinimapClickToUV,
  worldToMinimapPixel,
} from "./minimapChart";

/** Place Mobius center well inside the ball so the main view responds clearly to minimap picks. */
const P0_FROM_MINIMAP_INTERIOR = 0.52;

/** Extra margin so rim dots and focus ring are not clipped at the canvas edge. */
const PAD = 20;
/** Skip drawing every k-th vertex when graph is huge (still show structure). */
const MINIMAP_DRAW_STRIDE_THRESHOLD = 25_000;

export type GraphMinimapProps = {
  mode: "2d" | "3d";
  graph2d: GraphBundle | null;
  graph3d: GraphBundle3d | null;
  /** Shown immediately when starting a focus animation (same as main UI key), not only after `appliedFocus` commits. */
  focusIndicatorName: string;
  seeds: Set<string>;
  onChartPick2d: (z: Complex) => void;
  onChartPick3d: (p: Vec3) => void;
  onPickFocus: (name: string, opts?: { skipAnimation?: boolean; source?: "minimap" }) => void;
};

type Layout2d = {
  kind: "2d";
  n: number;
  stride: number;
  cx: number;
  cy: number;
  half: number;
  zx: Float32Array;
  zy: Float32Array;
  vertex: string[];
  nameToIndex: Map<string, number>;
};

type Layout3d = {
  kind: "3d";
  n: number;
  stride: number;
  cx: number;
  cy: number;
  half: number;
  px: Float32Array;
  py: Float32Array;
  pz: Float32Array;
  vertex: string[];
  nameToIndex: Map<string, number>;
};

type Layout = Layout2d | Layout3d | null;
type LayoutNonNull = Layout2d | Layout3d;

function buildLayout(mode: "2d" | "3d", graph2d: GraphBundle | null, graph3d: GraphBundle3d | null): Layout {
  if (mode === "2d" && graph2d) {
    const n = graph2d.vertex.length;
    const b = boundsNativeZDisk(graph2d.x, graph2d.y, n);
    const stride = n > MINIMAP_DRAW_STRIDE_THRESHOLD ? Math.ceil(n / MINIMAP_DRAW_STRIDE_THRESHOLD) : 1;
    return {
      kind: "2d",
      n,
      stride,
      cx: b.cx,
      cy: b.cy,
      half: b.half,
      zx: graph2d.x,
      zy: graph2d.y,
      vertex: graph2d.vertex,
      nameToIndex: graph2d.nameToIndex,
    };
  }
  if (mode === "3d" && graph3d) {
    const n = graph3d.vertex.length;
    const b = boundsStereoRaw(graph3d.x, graph3d.y, graph3d.z, n);
    const stride = n > MINIMAP_DRAW_STRIDE_THRESHOLD ? Math.ceil(n / MINIMAP_DRAW_STRIDE_THRESHOLD) : 1;
    return {
      kind: "3d",
      n,
      stride,
      cx: b.cx,
      cy: b.cy,
      half: b.half,
      px: graph3d.x,
      py: graph3d.y,
      pz: graph3d.z,
      vertex: graph3d.vertex,
      nameToIndex: graph3d.nameToIndex,
    };
  }
  return null;
}

/** Unit disk boundary in minimap CSS pixels (native Z rim). */
function pathUnitDisk2d(
  ctx: CanvasRenderingContext2D,
  layout: Layout2d,
  inner: number,
  pad: number,
) {
  const steps = 96;
  ctx.beginPath();
  for (let k = 0; k <= steps; k++) {
    const t = (k / steps) * Math.PI * 2;
    const wx = Math.cos(t);
    const wy = Math.sin(t);
    const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function drawRimUnitCircle2d(
  ctx: CanvasRenderingContext2D,
  layout: Layout2d,
  inner: number,
  pad: number,
  alpha: number,
) {
  ctx.strokeStyle = `rgba(160,170,200,${alpha})`;
  ctx.lineWidth = 1;
  pathUnitDisk2d(ctx, layout, inner, pad);
  ctx.stroke();
}

const R_GREEN = 0.9;
const R_SEED = 1.75;
const R_FOCUS = 2.25;
const R_FOCUS_RING = 4.25;

function layoutBaseCacheSig(layout: LayoutNonNull, cssW: number, cssH: number): string {
  const a = layout.vertex[0] ?? "";
  const b = layout.vertex[layout.n - 1] ?? "";
  return `${layout.kind}|${layout.n}|${layout.stride}|${layout.cx}|${layout.cy}|${layout.half}|${cssW}|${cssH}|${a}|${b}`;
}

/** Draw static background + rim (2d) + all vertices as small green dots (CSS pixel space). */
function drawMinimapBaseLayer(
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  layout: LayoutNonNull,
  inner: number,
  pad: number,
) {
  ctx.clearRect(0, 0, cssW, cssH);
  if (layout.kind === "2d") {
    const L = layout;
    ctx.save();
    pathUnitDisk2d(ctx, L, inner, pad);
    ctx.clip();
    ctx.fillStyle = "rgba(14,14,16,0.92)";
    ctx.fill();
    drawRimUnitCircle2d(ctx, L, inner, pad, 0.35);
    for (let i = 0; i < L.n; i += L.stride) {
      const wx = L.zx[i]!;
      const wy = L.zy[i]!;
      const [px, py] = worldToMinimapPixel(wx, wy, L.cx, L.cy, L.half, inner, pad);
      ctx.fillStyle = "rgba(70, 200, 120, 0.28)";
      ctx.beginPath();
      ctx.arc(px, py, R_GREEN, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  } else {
    /* 3D stereo: transparent background (no full-rect fill); dots only. */
    for (let i = 0; i < layout.n; i += layout.stride) {
      const [wx, wy] = stereoNorthFromBallChart(layout.px[i]!, layout.py[i]!, layout.pz[i]!);
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      ctx.fillStyle = "rgba(70, 200, 120, 0.28)";
      ctx.beginPath();
      ctx.arc(px, py, R_GREEN, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function nearestVertexAtMinimapPixel(
  layout: LayoutNonNull,
  clickPx: number,
  clickPy: number,
  inner: number,
  pad: number,
  maxDistCssPx: number,
): number {
  const maxSq = maxDistCssPx * maxDistCssPx;
  let bestI = -1;
  let bestD = maxSq;
  if (layout.kind === "2d") {
    for (let i = 0; i < layout.n; i += layout.stride) {
      const wx = layout.zx[i]!;
      const wy = layout.zy[i]!;
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      const dx = clickPx - px;
      const dy = clickPy - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }
  } else {
    for (let i = 0; i < layout.n; i += layout.stride) {
      const [wx, wy] = stereoNorthFromBallChart(layout.px[i]!, layout.py[i]!, layout.pz[i]!);
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      const dx = clickPx - px;
      const dy = clickPy - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }
  }
  return bestI;
}

function drawCanvas(
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  layout: LayoutNonNull,
  seeds: Set<string>,
  focusName: string,
  baseCache: HTMLCanvasElement,
  baseCacheSigRef: { current: string },
) {
  const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
  const w = Math.floor(cssW * dpr);
  const h = Math.floor(cssH * dpr);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.scale(dpr, dpr);

  const inner = Math.min(cssW, cssH) - PAD * 2;
  const pad = PAD;
  if (inner < 20) return;

  const sig = layoutBaseCacheSig(layout, cssW, cssH);
  if (baseCacheSigRef.current !== sig) {
    baseCacheSigRef.current = sig;
    baseCache.width = w;
    baseCache.height = h;
    const bctx = baseCache.getContext("2d");
    if (!bctx) return;
    bctx.setTransform(1, 0, 0, 1, 0, 0);
    bctx.clearRect(0, 0, w, h);
    bctx.scale(dpr, dpr);
    drawMinimapBaseLayer(bctx, cssW, cssH, layout, inner, pad);
  }

  ctx.drawImage(baseCache, 0, 0, cssW, cssH);

  const focusIdx =
    focusName.trim().length > 0 ? layout.nameToIndex.get(focusName.trim()) ?? -1 : -1;

  if (layout.kind === "2d") {
    ctx.save();
    pathUnitDisk2d(ctx, layout, inner, pad);
    ctx.clip();
    for (let i = 0; i < layout.n; i += layout.stride) {
      if (i === focusIdx) continue;
      const name = layout.vertex[i]!;
      if (!seeds.has(name)) continue;
      const wx = layout.zx[i]!;
      const wy = layout.zy[i]!;
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      ctx.fillStyle = "rgba(255,55,55,1)";
      ctx.beginPath();
      ctx.arc(px, py, R_SEED, 0, Math.PI * 2);
      ctx.fill();
    }
    if (focusIdx >= 0) {
      const wx = layout.zx[focusIdx]!;
      const wy = layout.zy[focusIdx]!;
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath();
      ctx.arc(px, py, R_FOCUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, R_FOCUS_RING, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  } else {
    for (let i = 0; i < layout.n; i += layout.stride) {
      if (i === focusIdx) continue;
      const name = layout.vertex[i]!;
      if (!seeds.has(name)) continue;
      const [wx, wy] = stereoNorthFromBallChart(layout.px[i]!, layout.py[i]!, layout.pz[i]!);
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      ctx.fillStyle = "rgba(255,55,55,1)";
      ctx.beginPath();
      ctx.arc(px, py, R_SEED, 0, Math.PI * 2);
      ctx.fill();
    }
    if (focusIdx >= 0) {
      const [wx, wy] = stereoNorthFromBallChart(layout.px[focusIdx]!, layout.py[focusIdx]!, layout.pz[focusIdx]!);
      const [px, py] = worldToMinimapPixel(wx, wy, layout.cx, layout.cy, layout.half, inner, pad);
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath();
      ctx.arc(px, py, R_FOCUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, R_FOCUS_RING, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

/** Max distance (CSS px) from a vertex dot to register a node pick vs chart pan. */
const MINIMAP_VERTEX_HIT_PX = 13;
/** After this many ms with primary button down, minimap focus picks apply without geodesic animation. */
const MINIMAP_LONG_PRESS_SKIP_ANIM_MS = 1000;

export function GraphMinimap({
  mode,
  graph2d,
  graph3d,
  focusIndicatorName,
  seeds,
  onChartPick2d,
  onChartPick3d,
  onPickFocus,
}: GraphMinimapProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseCacheRef = useRef<HTMLCanvasElement | null>(null);
  const baseCacheSigRef = useRef("");
  /** `performance.now()` when primary pointer went down on canvas; 0 if not tracking. */
  const pointerDownAtRef = useRef(0);
  const layout = useMemo(() => buildLayout(mode, graph2d, graph3d), [mode, graph2d, graph3d]);

  const redraw = useCallback(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas || !layout) return;
    if (!baseCacheRef.current) baseCacheRef.current = document.createElement("canvas");
    const baseCache = baseCacheRef.current;
    const cssW = Math.max(1, wrap.clientWidth);
    const cssH = Math.max(1, wrap.clientHeight);
    const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawCanvas(ctx, cssW, cssH, layout, seeds, focusIndicatorName, baseCache, baseCacheSigRef);
  }, [layout, seeds, focusIndicatorName]);

  useLayoutEffect(() => {
    redraw();
    const ro = new ResizeObserver(() => redraw());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [redraw]);

  const applyAtClient = useCallback(
    (clientX: number, clientY: number, rect: DOMRect) => {
      if (!layout) return;
      const cssW = rect.width;
      const cssH = rect.height;
      const inner = Math.min(cssW, cssH) - PAD * 2;
      const pad = PAD;
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const longHold =
        pointerDownAtRef.current > 0 &&
        performance.now() - pointerDownAtRef.current >= MINIMAP_LONG_PRESS_SKIP_ANIM_MS;
      const hitIdx = nearestVertexAtMinimapPixel(layout, px, py, inner, pad, MINIMAP_VERTEX_HIT_PX);
      if (hitIdx >= 0) {
        onPickFocus(layout.vertex[hitIdx]!, { skipAnimation: longHold, source: "minimap" });
        return;
      }
      if (layout.kind === "2d") {
        const z = nativeZClickToComplex(px, py, layout.cx, layout.cy, layout.half, inner, pad);
        onChartPick2d(z);
      } else {
        const [u, v] = stereoMinimapClickToUV(px, py, layout.cx, layout.cy, layout.half, inner, pad);
        const [sx, sy, sz] = stereoNorthToUnitSphere(u, v);
        try {
          const s = P0_FROM_MINIMAP_INTERIOR;
          const [x, y, z] = clampP0(s * sx, s * sy, s * sz);
          onChartPick3d({ x, y, z });
        } catch {
          /* clampP0 can throw for degenerate direction */
        }
      }
    },
    [layout, onChartPick2d, onChartPick3d, onPickFocus],
  );

  const endPointerTracking = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    pointerDownAtRef.current = 0;
    try {
      (e.currentTarget as HTMLCanvasElement).releasePointerCapture(e.pointerId);
    } catch {
      /* not captured */
    }
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!layout || e.button !== 0) return;
      pointerDownAtRef.current = performance.now();
      (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
      const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect();
      applyAtClient(e.clientX, e.clientY, rect);
    },
    [layout, applyAtClient],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!layout || !(e.buttons & 1) || pointerDownAtRef.current <= 0) return;
      const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect();
      applyAtClient(e.clientX, e.clientY, rect);
    },
    [layout, applyAtClient],
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (e.button !== 0) return;
      endPointerTracking(e);
    },
    [endPointerTracking],
  );

  const onPointerCancel = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      endPointerTracking(e);
    },
    [endPointerTracking],
  );

  if (!layout) return null;

  return (
    <div className="graphMinimap">
      <div ref={wrapRef} className="graphMinimapCanvasWrap">
        <canvas
          ref={canvasRef}
          className="graphMinimapCanvas"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
        />
      </div>
    </div>
  );
}
