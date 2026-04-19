import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import "./App.css";
import { loadGraphBundle, loadMeta, type GraphBundle, type MetaJson } from "./data/loadBundle";
import { loadGraphBundle3d, loadMeta3d, type GraphBundle3d } from "./data/loadBundle3d";
import { RIM_CULL_EPS, RIM_CULL_EPS_SLIDER_MAX } from "./math/constants";
import {
  easeInOutCubic,
  fillP0BallGeodesic,
  fillZ0Geodesic,
  p0AtBallGeodesicParameter,
  P0_GEODESIC_SAMPLES,
  z0AtGeodesicParameter,
  Z0_GEODESIC_SAMPLES,
} from "./math/focusAnimPath";
import { clampZ0, type Complex } from "./math/mobius";
import {
  computeScene,
  type NonSeedShowMode,
  type SceneBuffers,
  type SceneStats,
} from "./model/computeScene";
import { computeScene3d, type SceneBuffers3d } from "./model/computeScene3d";
import { bundleToCSR } from "./model/graphSearch";
import type { HoverNeighborGraph2d, HoverNeighborGraph3d } from "./model/hoverNeighborEdgePositions";
import { tryBuildPrimMstOverlay, tryBuildPrimMstOverlay3d } from "./model/primMstOverlay";
import type { PathOverlayBuffer } from "./model/pathOverlayBuffer";
import { z0FromProtein } from "./z0FromProtein";
import { p0FromProtein, type Vec3 } from "./p0FromProtein";
import {
  nodeDiskHoverTooltipForGraph3d,
  nodeDiskHoverTooltipForIndex,
  nodeListTooltip,
  nodeListTooltip3d,
} from "./nodeListTooltip";
import { analysisKeyboardGuard } from "./analysisKeyboardGuard";
import { AnalysisFloater } from "./viz/AnalysisFloater";
import { BallView3d, type BallView3dHandle, type BallView3dNodeInteraction } from "./viz/BallView3d";
import { DiskView, type DiskViewHandle, type DiskViewNodeInteraction } from "./viz/DiskView";
import { PathTreesFloater } from "./viz/PathTreesFloater";
import type { Minimap2dMode, Minimap3dMode } from "./viz/minimapChart";
import { GraphMinimap } from "./viz/GraphMinimap";

const DEFAULT_RADIAL_SCALE_MIN = 0.2;
const DEFAULT_RADIAL_SCALE_MAX = 1.3;
const DEFAULT_NODE_SIZE_MUL = 0.5;
/** Floor for radial instance scale and for zoom-size multiplier (see DiskView). */
const DEFAULT_NODE_MIN_MUL = 0.1;

const PRIM_MST_FLASH_HOLD_MS = 2000;
const PRIM_MST_FLASH_FADE_MS = 300;

function parseSeeds(text: string): Set<string> {
  const s = new Set<string>();
  for (const part of text.split(/\s+/)) {
    const t = part.trim();
    if (t) s.add(t);
  }
  return s;
}

function webGpuSupported(): boolean {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

function vertexDegrees(bundle: { src: Int32Array; dst: Int32Array; vertex: string[] }): Int32Array {
  const n = bundle.vertex.length;
  const d = new Int32Array(n);
  const { src, dst } = bundle;
  for (let ei = 0; ei < src.length; ei++) {
    d[src[ei]]++;
    d[dst[ei]]++;
  }
  return d;
}

/** Sorted clickable names: applied seeds valid for the bundle. */
function focusPickerNames(
  bundle: { nameToIndex: Map<string, number> },
  appliedSeedsText: string,
): string[] {
  const seeds = [...parseSeeds(appliedSeedsText)].filter((n) => bundle.nameToIndex.has(n));
  seeds.sort((a, b) => a.localeCompare(b));
  return seeds;
}

const FOCUS_SEARCH_MAX = 28;

/** Case-insensitive prefix then substring; capped for UI responsiveness. */
function filterVertexNames(vertices: readonly string[], query: string, max: number): string[] {
  const t = query.trim().toLowerCase();
  if (!t) return [];
  const pref: string[] = [];
  const sub: string[] = [];
  for (const name of vertices) {
    const l = name.toLowerCase();
    if (l.startsWith(t)) pref.push(name);
    else if (l.includes(t)) sub.push(name);
  }
  pref.sort((a, b) => a.localeCompare(b));
  sub.sort((a, b) => a.localeCompare(b));
  return [...pref, ...sub].slice(0, max);
}

/** Applied seed names in first-seen order (from applied text), bundle-valid only. */
function appliedSeedNamesOrdered(
  bundle: { vertex: string[]; nameToIndex: Map<string, number> },
  appliedSeedsText: string,
): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const part of appliedSeedsText.split(/\s+/)) {
    const t = part.trim();
    if (!t || !bundle.nameToIndex.has(t) || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
}

export default function App() {
  const [dataMode, setDataMode] = useState<"2d" | "3d">(() =>
    typeof window !== "undefined" && window.location.hash === "#3d" ? "3d" : "2d",
  );
  const [bundle, setBundle] = useState<GraphBundle | null>(null);
  const [bundle3d, setBundle3d] = useState<GraphBundle3d | null>(null);
  const [runMeta, setRunMeta] = useState<MetaJson | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [seedsDraft, setSeedsDraft] = useState("");
  const [appliedFocus, setAppliedFocus] = useState("");
  const [appliedSeedsText, setAppliedSeedsText] = useState("");
  const [formErr, setFormErr] = useState<string | null>(null);
  const [rimCullEps, setRimCullEps] = useState(RIM_CULL_EPS);
  const [showSeedLabels, setShowSeedLabels] = useState(true);
  const [showCrosshair, setShowCrosshair] = useState(false);
  const [centerWeightedSizes, setCenterWeightedSizes] = useState(true);
  const [radialScaleMin, setRadialScaleMin] = useState(DEFAULT_RADIAL_SCALE_MIN);
  const [radialScaleMax, setRadialScaleMax] = useState(DEFAULT_RADIAL_SCALE_MAX);
  const [nodeSizeMul, setNodeSizeMul] = useState(DEFAULT_NODE_SIZE_MUL);
  const [compensateZoomNodes, setCompensateZoomNodes] = useState(true);
  const [nodeMinMul, setNodeMinMul] = useState(DEFAULT_NODE_MIN_MUL);
  const [edgeOpacity, setEdgeOpacity] = useState(0.45);
  const [nonSeedShowMode, setNonSeedShowMode] = useState<NonSeedShowMode>("all");
  const [focusAnimTarget, setFocusAnimTarget] = useState<string | null>(null);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [pathTreesOpen, setPathTreesOpen] = useState(false);
  /** Shift+U: draw all graph edges; non-seed–non-seed segments are faint light orange (see DiskView). */
  const [showAllGraphEdges, setShowAllGraphEdges] = useState(false);
  /** Advanced: white incident edges on node hover (2D disk + 3D ball). */
  const [showHoverNeighborEdges, setShowHoverNeighborEdges] = useState(true);
  const [minimapVisible, setMinimapVisible] = useState(true);
  const [minimap2dMode, setMinimap2dMode] = useState<Minimap2dMode>("native_disk");
  const [minimap3dMode, setMinimap3dMode] = useState<Minimap3dMode>("stereo_north");
  const [seedPanelOpen, setSeedPanelOpen] = useState(true);
  /** Row highlight in seed list: list clicks, or main-viz pick only when that vertex is already a seed. */
  const [seedListHighlightName, setSeedListHighlightName] = useState("");
  /** When set, main chart origin follows minimap click (native Z / stereo), not the focused protein’s disk/ball point. */
  const [chartOrigin2dOverride, setChartOrigin2dOverride] = useState<Complex | null>(null);
  const [chartOrigin3dOverride, setChartOrigin3dOverride] = useState<Vec3 | null>(null);
  const [focusSearch, setFocusSearch] = useState("");
  const [focusSearchHl, setFocusSearchHl] = useState(0);
  const diskViewRef = useRef<DiskViewHandle>(null);
  const ballViewRef = useRef<BallView3dHandle>(null);
  const nodeInteractionRef = useRef<DiskViewNodeInteraction | BallView3dNodeInteraction | null>(null);
  const appliedFocusRef = useRef(appliedFocus);
  const appliedSeedsTextRef = useRef(appliedSeedsText);
  const seedsDraftRef = useRef(seedsDraft);
  const focusAnimTargetRef = useRef<string | null>(null);
  const focusAnimGenRef = useRef(0);
  const focusAnimRafRef = useRef(0);
  const isFocusAnimatingRef = useRef(false);
  const z0AnimRef = useRef<Complex | null>(null);
  const p0AnimRef = useRef<Vec3 | null>(null);

  useLayoutEffect(() => {
    appliedFocusRef.current = appliedFocus;
  }, [appliedFocus]);
  useLayoutEffect(() => {
    appliedSeedsTextRef.current = appliedSeedsText;
  }, [appliedSeedsText]);
  useLayoutEffect(() => {
    seedsDraftRef.current = seedsDraft;
  }, [seedsDraft]);
  useLayoutEffect(() => {
    focusAnimTargetRef.current = focusAnimTarget;
  }, [focusAnimTarget]);

  /** When applied focus name changes, align camera with data origin (w=0); keep zoom and Shift+Möbius from sticking across unrelated focus changes. */
  useLayoutEffect(() => {
    if (!appliedFocus.trim()) return;
    diskViewRef.current?.recenterPreservingZoom();
  }, [appliedFocus]);

  const webGpuError = useMemo(
    () => (webGpuSupported() ? null : "WebGPU required"),
    [],
  );

  useEffect(() => {
    const onHash = () => {
      const next = window.location.hash === "#3d" ? "3d" : "2d";
      setDataMode((m) => (m === next ? m : next));
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    setChartOrigin2dOverride(null);
    setChartOrigin3dOverride(null);
  }, [dataMode]);

  /** Shift+M: toggle 2D / 3D (`#3d` hash). Plain M on 2D disk: Prim MST overlay (see keydown handler below). */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (analysisKeyboardGuard(e)) return;
      if (e.code !== "KeyM" || !e.shiftKey) return;
      e.preventDefault();
      window.location.hash = window.location.hash === "#3d" ? "" : "#3d";
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const base = import.meta.env.BASE_URL;
        const seedsSnapshot = appliedSeedsTextRef.current;
        const draftSnapshot = seedsDraftRef.current;
        const focusSnapshot = appliedFocusRef.current;

        type Loaded = GraphBundle | GraphBundle3d;
        let b: Loaded;
        let meta: MetaJson | null;
        if (dataMode === "3d") {
          const [g, m] = await Promise.all([loadGraphBundle3d(base), loadMeta3d(base)]);
          if (cancelled) return;
          setBundle(null);
          setBundle3d(g);
          b = g;
          meta = m ?? null;
        } else {
          const [g, m] = await Promise.all([loadGraphBundle(base), loadMeta(base)]);
          if (cancelled) return;
          setBundle3d(null);
          setBundle(g);
          b = g;
          meta = m ?? null;
        }

        setRunMeta(meta);
        const defF = meta?.default_focus?.trim() || b.vertex[0] || "";
        const defSeeds = (meta?.default_seeds?.join(" ") || defF).replace(/\s+/g, " ").trim();

        const filterNamesToBundle = (text: string): string => {
          const parts = text.split(/\s+/).map((p) => p.trim()).filter(Boolean);
          const ok = parts.filter((p) => b.nameToIndex.has(p));
          return ok.join(" ");
        };

        const mergedSubset = draftSnapshot.trim() || seedsSnapshot.trim();
        const kept = filterNamesToBundle(mergedSubset);
        const seedsToApply = kept.length > 0 ? kept : defSeeds;

        let focusTo = focusSnapshot.trim();
        if (!focusTo || !b.nameToIndex.has(focusTo)) {
          const sp = parseSeeds(seedsToApply);
          focusTo = [...sp].find((n) => b.nameToIndex.has(n)) ?? defF;
        }

        setSeedsDraft(seedsToApply);
        setAppliedSeedsText(seedsToApply);
        setAppliedFocus(focusTo);
        setLoadErr(null);
      } catch (e) {
        if (!cancelled) setLoadErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dataMode]);

  useEffect(() => {
    if (!isFocusAnimatingRef.current) return;
    focusAnimGenRef.current += 1;
    cancelAnimationFrame(focusAnimRafRef.current);
    isFocusAnimatingRef.current = false;
    setFocusAnimTarget(null);
    z0AnimRef.current = null;
    p0AnimRef.current = null;
  }, [appliedSeedsText, rimCullEps, bundle, bundle3d, nonSeedShowMode]);

  useEffect(() => {
    const h = seedListHighlightName.trim();
    if (!h) return;
    if (!parseSeeds(appliedSeedsText).has(h)) setSeedListHighlightName("");
  }, [appliedSeedsText, seedListHighlightName]);

  useEffect(() => {
    return () => {
      focusAnimGenRef.current += 1;
      cancelAnimationFrame(focusAnimRafRef.current);
    };
  }, []);

  const z0Current = useMemo(() => {
    if (!bundle) return null;
    if (chartOrigin2dOverride !== null) return clampZ0(chartOrigin2dOverride);
    if (!appliedFocus.trim()) return null;
    try {
      return z0FromProtein(bundle, appliedFocus);
    } catch {
      return null;
    }
  }, [bundle, appliedFocus, chartOrigin2dOverride]);

  const p0Ball = useMemo(() => {
    if (!bundle3d) return null;
    if (chartOrigin3dOverride !== null) return chartOrigin3dOverride;
    if (!appliedFocus.trim()) return null;
    try {
      return p0FromProtein(bundle3d, appliedFocus);
    } catch {
      return null;
    }
  }, [bundle3d, appliedFocus, chartOrigin3dOverride]);

  const graphCSR = useMemo(() => {
    if (bundle) return bundleToCSR(bundle);
    if (bundle3d) return bundleToCSR(bundle3d);
    return null;
  }, [bundle, bundle3d]);

  const hoverNeighborGraph2d = useMemo((): HoverNeighborGraph2d | null => {
    if (!bundle || !graphCSR || !z0Current) return null;
    return { csr: graphCSR, zx: bundle.x, zy: bundle.y, z0: z0Current };
  }, [bundle, graphCSR, z0Current]);

  const hoverNeighborGraph3d = useMemo((): HoverNeighborGraph3d | null => {
    if (!bundle3d || !graphCSR || !p0Ball) return null;
    return {
      csr: graphCSR,
      px: bundle3d.x,
      py: bundle3d.y,
      pz: bundle3d.z,
      p0x: p0Ball.x,
      p0y: p0Ball.y,
      p0z: p0Ball.z,
    };
  }, [bundle3d, graphCSR, p0Ball]);

  const graphInteractionKey = useMemo(() => {
    const o2 =
      chartOrigin2dOverride !== null
        ? `|z0o:${chartOrigin2dOverride.re.toFixed(4)},${chartOrigin2dOverride.im.toFixed(4)}`
        : "";
    const o3 =
      chartOrigin3dOverride !== null
        ? `|p0o:${chartOrigin3dOverride.x.toFixed(4)},${chartOrigin3dOverride.y.toFixed(4)},${chartOrigin3dOverride.z.toFixed(4)}`
        : "";
    if (dataMode === "3d" && bundle3d) {
      return `3d|${bundle3d.vertex.length}|${appliedFocus}|${appliedSeedsText}|${rimCullEps}|${nonSeedShowMode}${o3}`;
    }
    if (bundle) {
      return `2d|${bundle.vertex.length}|${appliedFocus}|${appliedSeedsText}|${rimCullEps}|${nonSeedShowMode}${o2}`;
    }
    return "";
  }, [
    dataMode,
    bundle,
    bundle3d,
    appliedFocus,
    appliedSeedsText,
    rimCullEps,
    nonSeedShowMode,
    chartOrigin2dOverride,
    chartOrigin3dOverride,
  ]);

  const [pathOverlayPinned, setPathOverlayPinned] = useState<{
    buf: PathOverlayBuffer;
    key: string;
  } | null>(null);
  const pathOverlay =
    pathOverlayPinned && pathOverlayPinned.key === graphInteractionKey ? pathOverlayPinned.buf : null;

  const primMstFlashActiveRef = useRef(false);
  const primMstFadeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const primMstFadeRafRef = useRef(0);

  const cancelPrimMstFadeTimers = useCallback(() => {
    if (primMstFadeTimeoutRef.current !== null) {
      clearTimeout(primMstFadeTimeoutRef.current);
      primMstFadeTimeoutRef.current = null;
    }
    if (primMstFadeRafRef.current) {
      cancelAnimationFrame(primMstFadeRafRef.current);
      primMstFadeRafRef.current = 0;
    }
  }, []);

  useLayoutEffect(() => {
    cancelPrimMstFadeTimers();
    primMstFlashActiveRef.current = false;
    diskViewRef.current?.setPathOverlayOpacityMultiplier(1);
    ballViewRef.current?.setPathOverlayOpacityMultiplier(1);
    setPathOverlayPinned(null);
  }, [graphInteractionKey, cancelPrimMstFadeTimers]);

  const handlePathOverlayChange = useCallback(
    (b: PathOverlayBuffer | null) => {
      cancelPrimMstFadeTimers();
      primMstFlashActiveRef.current = false;
      diskViewRef.current?.setPathOverlayOpacityMultiplier(1);
      ballViewRef.current?.setPathOverlayOpacityMultiplier(1);
      setPathOverlayPinned(b && graphInteractionKey ? { buf: b, key: graphInteractionKey } : null);
    },
    [graphInteractionKey, cancelPrimMstFadeTimers],
  );

  const scene: SceneBuffers | SceneBuffers3d | null = useMemo(() => {
    const seeds = parseSeeds(appliedSeedsText);
    if (seeds.size === 0) return null;
    if (bundle3d && p0Ball) {
      try {
        return computeScene3d(
          bundle3d,
          p0Ball.x,
          p0Ball.y,
          p0Ball.z,
          seeds,
          rimCullEps,
          nonSeedShowMode,
          showAllGraphEdges,
        );
      } catch {
        return null;
      }
    }
    if (bundle && z0Current) {
      try {
        return computeScene(bundle, z0Current, seeds, rimCullEps, nonSeedShowMode, showAllGraphEdges);
      } catch {
        return null;
      }
    }
    return null;
  }, [
    bundle,
    bundle3d,
    appliedSeedsText,
    rimCullEps,
    nonSeedShowMode,
    showAllGraphEdges,
    z0Current,
    p0Ball,
  ]);

  const pickerGraph = bundle ?? bundle3d;
  const minimapSeeds = useMemo(() => parseSeeds(appliedSeedsText), [appliedSeedsText]);

  useEffect(() => {
    if (!pickerGraph) {
      setChartOrigin2dOverride(null);
      setChartOrigin3dOverride(null);
    }
  }, [pickerGraph]);

  const pickerNames = useMemo(
    () => (pickerGraph ? focusPickerNames(pickerGraph, appliedSeedsText) : []),
    [pickerGraph, appliedSeedsText],
  );

  const degrees = useMemo(
    () => (pickerGraph ? vertexDegrees(pickerGraph) : null),
    [pickerGraph],
  );

  const analysisSeedOrder = useMemo(
    () => (pickerGraph ? appliedSeedNamesOrdered(pickerGraph, appliedSeedsText) : []),
    [pickerGraph, appliedSeedsText],
  );

  const focusSearchMatches = useMemo(() => {
    if (!pickerGraph) return [];
    return filterVertexNames(pickerGraph.vertex, focusSearch, FOCUS_SEARCH_MAX);
  }, [pickerGraph, focusSearch]);

  useLayoutEffect(() => {
    setFocusSearchHl((i) => {
      if (focusSearchMatches.length === 0) return 0;
      return Math.min(i, focusSearchMatches.length - 1);
    });
  }, [focusSearchMatches]);

  useEffect(() => {
    if (webGpuError) return;
    const onKey = (e: KeyboardEvent) => {
      if (analysisKeyboardGuard(e)) return;
      if (e.code === "KeyA") {
        e.preventDefault();
        setAnalysisOpen((o) => !o);
        return;
      }
      if (e.code === "KeyS") {
        e.preventDefault();
        setPathTreesOpen((o) => !o);
        return;
      }
      if (bundle3d) {
        if (e.code === "KeyR") {
          e.preventDefault();
          ballViewRef.current?.resetView();
        } else if (e.code === "KeyF") {
          e.preventDefault();
          ballViewRef.current?.fitSubgraphToView();
        } else if (e.code === "KeyU" && e.shiftKey) {
          e.preventDefault();
          setShowAllGraphEdges((o) => !o);
        } else if (e.code === "KeyM" && !e.shiftKey) {
          e.preventDefault();
          if (!bundle3d || !p0Ball) return;
          cancelPrimMstFadeTimers();
          primMstFlashActiveRef.current = false;
          const r = tryBuildPrimMstOverlay3d(
            bundle3d,
            p0Ball.x,
            p0Ball.y,
            p0Ball.z,
            appliedSeedsText,
          );
          if (!r.ok) return;
          primMstFlashActiveRef.current = true;
          ballViewRef.current?.setPathOverlayOpacityMultiplier(1);
          setPathOverlayPinned({ buf: r.buf, key: graphInteractionKey });
          primMstFadeTimeoutRef.current = setTimeout(() => {
            primMstFadeTimeoutRef.current = null;
            if (!primMstFlashActiveRef.current) return;
            const t0 = performance.now();
            const tick = (now: number) => {
              if (!primMstFlashActiveRef.current) return;
              const u = Math.min(1, (now - t0) / PRIM_MST_FLASH_FADE_MS);
              ballViewRef.current?.setPathOverlayOpacityMultiplier(1 - u);
              if (u < 1) {
                primMstFadeRafRef.current = requestAnimationFrame(tick);
              } else {
                primMstFadeRafRef.current = 0;
                if (primMstFlashActiveRef.current) {
                  setPathOverlayPinned(null);
                  primMstFlashActiveRef.current = false;
                }
                ballViewRef.current?.setPathOverlayOpacityMultiplier(1);
              }
            };
            primMstFadeRafRef.current = requestAnimationFrame(tick);
          }, PRIM_MST_FLASH_HOLD_MS);
        }
        return;
      }
      if (!bundle) return;
      if (e.code === "KeyR") {
        e.preventDefault();
        diskViewRef.current?.resetView();
      } else if (e.code === "KeyF") {
        e.preventDefault();
        diskViewRef.current?.fitSubgraphToView();
      } else if (e.code === "KeyU" && e.shiftKey) {
        e.preventDefault();
        setShowAllGraphEdges((o) => !o);
      } else if (e.code === "KeyM" && !e.shiftKey) {
        e.preventDefault();
        if (!z0Current) return;
        cancelPrimMstFadeTimers();
        primMstFlashActiveRef.current = false;
        const r = tryBuildPrimMstOverlay(bundle, z0Current, appliedSeedsText);
        if (!r.ok) return;
        primMstFlashActiveRef.current = true;
        diskViewRef.current?.setPathOverlayOpacityMultiplier(1);
        setPathOverlayPinned({ buf: r.buf, key: graphInteractionKey });
        primMstFadeTimeoutRef.current = setTimeout(() => {
          primMstFadeTimeoutRef.current = null;
          if (!primMstFlashActiveRef.current) return;
          const t0 = performance.now();
          const tick = (now: number) => {
            if (!primMstFlashActiveRef.current) return;
            const u = Math.min(1, (now - t0) / PRIM_MST_FLASH_FADE_MS);
            diskViewRef.current?.setPathOverlayOpacityMultiplier(1 - u);
            if (u < 1) {
              primMstFadeRafRef.current = requestAnimationFrame(tick);
            } else {
              primMstFadeRafRef.current = 0;
              if (primMstFlashActiveRef.current) {
                setPathOverlayPinned(null);
                primMstFlashActiveRef.current = false;
              }
              diskViewRef.current?.setPathOverlayOpacityMultiplier(1);
            }
          };
          primMstFadeRafRef.current = requestAnimationFrame(tick);
        }, PRIM_MST_FLASH_HOLD_MS);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      cancelPrimMstFadeTimers();
      primMstFlashActiveRef.current = false;
    };
  }, [
    webGpuError,
    bundle,
    bundle3d,
    z0Current,
    p0Ball,
    appliedSeedsText,
    graphInteractionKey,
    cancelPrimMstFadeTimers,
  ]);

  const commitSeedsText = useCallback(
    (text: string): boolean => {
      setFormErr(null);
      const bg = bundle ?? bundle3d;
      if (!bg) return false;
      const seeds = parseSeeds(text);
      if (seeds.size === 0) {
        setFormErr("Need ≥1 seed");
        return false;
      }
      const unknown = [...seeds].filter((n) => !bg.nameToIndex.has(n));
      if (unknown.length) {
        setFormErr(`Unknown: ${unknown.slice(0, 5).join(", ")}`);
        return false;
      }
      setSeedsDraft(text);
      setAppliedSeedsText(text);
      const f = appliedFocus.trim();
      if (!f || !seeds.has(f) || !bg.nameToIndex.has(f)) {
        const first = [...seeds].find((n) => bg.nameToIndex.has(n));
        if (first) setAppliedFocus(first);
      }
      return true;
    },
    [bundle, bundle3d, appliedFocus],
  );

  const onApplySeeds = useCallback(() => {
    commitSeedsText(seedsDraft);
  }, [commitSeedsText, seedsDraft]);

  const shiftPickGraphIndex = useCallback(
    (graphIndex: number) => {
      const bg = bundle ?? bundle3d;
      if (!bg || graphIndex < 0 || graphIndex >= bg.vertex.length) return;
      const name = bg.vertex[graphIndex]!;
      const seeds = parseSeeds(seedsDraft);
      if (seeds.has(name)) {
        const tokens = seedsDraft
          .split(/\s+/)
          .map((p) => p.trim())
          .filter(Boolean)
          .filter((t) => t !== name);
        if (tokens.length === 0) {
          setFormErr("Need ≥1 seed");
          return;
        }
        commitSeedsText(tokens.join(" "));
        return;
      }
      const trimmed = seedsDraft.trim();
      const next = trimmed ? `${trimmed} ${name}` : name;
      commitSeedsText(next);
    },
    [bundle, bundle3d, seedsDraft, commitSeedsText],
  );

  const onMinimapChart2d = useCallback((z: Complex) => {
    setChartOrigin2dOverride(clampZ0(z));
    requestAnimationFrame(() => {
      diskViewRef.current?.recenterPreservingZoom();
    });
  }, []);

  const onMinimapChart3d = useCallback((p: Vec3) => {
    setChartOrigin3dOverride(p);
  }, []);

  const onPickFocus = useCallback(
    (
      name: string,
      opts?: { skipAnimation?: boolean; source?: "viz" | "list" | "search" | "minimap" },
    ) => {
      setFormErr(null);
      setChartOrigin2dOverride(null);
      setChartOrigin3dOverride(null);
      const t = name.trim();
      const skipAnimation = !!opts?.skipAnimation;
      const src = opts?.source ?? "viz";
      if (src === "list" && t) setSeedListHighlightName(t);

      const syncHighlightForVizOrSearch = () => {
        if (src !== "viz" && src !== "search") return;
        const seedsSet = parseSeeds(appliedSeedsTextRef.current);
        setSeedListHighlightName(t && seedsSet.has(t) ? t : "");
      };

      if (webGpuError) {
        syncHighlightForVizOrSearch();
        setAppliedFocus(t);
        return;
      }
      if (bundle3d) {
        if (!isFocusAnimatingRef.current && t === appliedFocusRef.current.trim()) return;
        if (isFocusAnimatingRef.current && t === (focusAnimTargetRef.current ?? "").trim()) return;

        let p0End: Vec3;
        try {
          p0End = p0FromProtein(bundle3d, t);
        } catch {
          return;
        }

        const seeds = parseSeeds(appliedSeedsText);
        if (seeds.size === 0) {
          syncHighlightForVizOrSearch();
          setAppliedFocus(t);
          return;
        }

        if (showAllGraphEdges) {
          isFocusAnimatingRef.current = false;
          p0AnimRef.current = null;
          setFocusAnimTarget(null);
          syncHighlightForVizOrSearch();
          setAppliedFocus(t);
          return;
        }

        if (skipAnimation) {
          cancelAnimationFrame(focusAnimRafRef.current);
          focusAnimGenRef.current += 1;
          isFocusAnimatingRef.current = false;
          p0AnimRef.current = null;
          setFocusAnimTarget(null);
          try {
            const buf = computeScene3d(
              bundle3d,
              p0End.x,
              p0End.y,
              p0End.z,
              seeds,
              rimCullEps,
              nonSeedShowMode,
              showAllGraphEdges,
            );
            ballViewRef.current?.applySceneBuffers(buf);
          } catch {
            syncHighlightForVizOrSearch();
            setAppliedFocus(t);
            return;
          }
          syncHighlightForVizOrSearch();
          setAppliedFocus(t);
          return;
        }

        cancelAnimationFrame(focusAnimRafRef.current);
        const gen = ++focusAnimGenRef.current;

        let p0Start: Vec3;
        try {
          const fromName = appliedFocusRef.current.trim();
          p0Start = p0AnimRef.current ?? p0FromProtein(bundle3d, fromName || t);
        } catch {
          syncHighlightForVizOrSearch();
          setAppliedFocus(t);
          return;
        }

        const p0Geo = new Float32Array(P0_GEODESIC_SAMPLES * 3);
        fillP0BallGeodesic(
          p0Start.x,
          p0Start.y,
          p0Start.z,
          p0End.x,
          p0End.y,
          p0End.z,
          P0_GEODESIC_SAMPLES,
          p0Geo,
        );

        isFocusAnimatingRef.current = true;
        syncHighlightForVizOrSearch();
        setFocusAnimTarget(t);

        const FOCUS_MS = 1000;
        const t0 = performance.now();

        const tick = (now: number) => {
          if (gen !== focusAnimGenRef.current) return;
          const tLin = Math.min(1, (now - t0) / FOCUS_MS);
          const u = easeInOutCubic(tLin);
          const p0 = p0AtBallGeodesicParameter(p0Geo, P0_GEODESIC_SAMPLES, u);
          p0AnimRef.current = p0;
          let buf: SceneBuffers3d | null = null;
          try {
            buf = computeScene3d(
              bundle3d,
              p0.x,
              p0.y,
              p0.z,
              seeds,
              rimCullEps,
              nonSeedShowMode,
              showAllGraphEdges,
            );
          } catch {
            focusAnimGenRef.current += 1;
            isFocusAnimatingRef.current = false;
            p0AnimRef.current = null;
            setFocusAnimTarget(null);
            syncHighlightForVizOrSearch();
            setAppliedFocus(t);
            return;
          }
          ballViewRef.current?.applySceneBuffers(buf);
          if (tLin < 1) {
            focusAnimRafRef.current = requestAnimationFrame(tick);
          } else {
            isFocusAnimatingRef.current = false;
            p0AnimRef.current = null;
            setFocusAnimTarget(null);
            setAppliedFocus(t);
          }
        };
        focusAnimRafRef.current = requestAnimationFrame(tick);
        return;
      }
      if (!bundle) return;
      if (!isFocusAnimatingRef.current && t === appliedFocusRef.current.trim()) return;
      if (isFocusAnimatingRef.current && t === (focusAnimTargetRef.current ?? "").trim()) return;

      cancelAnimationFrame(focusAnimRafRef.current);
      const gen = ++focusAnimGenRef.current;

      let z0End: Complex;
      try {
        z0End = z0FromProtein(bundle, t);
      } catch {
        return;
      }

      const seeds = parseSeeds(appliedSeedsText);
      if (seeds.size === 0) {
        syncHighlightForVizOrSearch();
        setAppliedFocus(t);
        return;
      }

      if (showAllGraphEdges) {
        isFocusAnimatingRef.current = false;
        z0AnimRef.current = null;
        setFocusAnimTarget(null);
        diskViewRef.current?.recenterPreservingZoom();
        syncHighlightForVizOrSearch();
        setAppliedFocus(t);
        return;
      }

      if (skipAnimation) {
        isFocusAnimatingRef.current = false;
        z0AnimRef.current = null;
        setFocusAnimTarget(null);
        diskViewRef.current?.recenterPreservingZoom();
        try {
          const buf = computeScene(bundle, z0End, seeds, rimCullEps, nonSeedShowMode, showAllGraphEdges);
          diskViewRef.current?.applySceneBuffers(buf);
        } catch {
          syncHighlightForVizOrSearch();
          setAppliedFocus(t);
          return;
        }
        syncHighlightForVizOrSearch();
        setAppliedFocus(t);
        return;
      }

      let z0Start: Complex;
      try {
        const fromName = appliedFocusRef.current.trim();
        z0Start = z0AnimRef.current ?? z0FromProtein(bundle, fromName || t);
      } catch {
        syncHighlightForVizOrSearch();
        setAppliedFocus(t);
        return;
      }

      const gx = new Float32Array(Z0_GEODESIC_SAMPLES);
      const gy = new Float32Array(Z0_GEODESIC_SAMPLES);
      fillZ0Geodesic(z0Start, z0End, Z0_GEODESIC_SAMPLES, gx, gy);

      /* Pan/Möbius are camera-only; data focus puts the gene at w=0 — clear offsets so that stays centered. */
      diskViewRef.current?.recenterPreservingZoom();

      isFocusAnimatingRef.current = true;
      syncHighlightForVizOrSearch();
      setFocusAnimTarget(t);

      const FOCUS_MS = 1000;
      const t0 = performance.now();

      const tick = (now: number) => {
        if (gen !== focusAnimGenRef.current) return;
        const tLin = Math.min(1, (now - t0) / FOCUS_MS);
        const u = easeInOutCubic(tLin);
        const z0 = z0AtGeodesicParameter(gx, gy, u);
        z0AnimRef.current = z0;
        let buf: SceneBuffers | null = null;
        try {
          buf = computeScene(bundle, z0, seeds, rimCullEps, nonSeedShowMode, showAllGraphEdges);
        } catch {
          focusAnimGenRef.current += 1;
          isFocusAnimatingRef.current = false;
          z0AnimRef.current = null;
          setFocusAnimTarget(null);
          syncHighlightForVizOrSearch();
          setAppliedFocus(t);
          return;
        }
        diskViewRef.current?.applySceneBuffers(buf);
        if (tLin < 1) {
          focusAnimRafRef.current = requestAnimationFrame(tick);
        } else {
          isFocusAnimatingRef.current = false;
          z0AnimRef.current = null;
          setFocusAnimTarget(null);
          setAppliedFocus(t);
        }
      };
      focusAnimRafRef.current = requestAnimationFrame(tick);
    },
    [bundle, bundle3d, appliedSeedsText, rimCullEps, nonSeedShowMode, showAllGraphEdges, webGpuError],
  );

  useLayoutEffect(() => {
    if (bundle3d && degrees) {
      nodeInteractionRef.current = {
        tooltipForGraphIndex: (i: number) => nodeDiskHoverTooltipForGraph3d(bundle3d, i, degrees, runMeta),
        pickGraphIndex: (i: number) => {
          if (i < 0 || i >= bundle3d.vertex.length) return;
          onPickFocus(bundle3d.vertex[i]!);
        },
        shiftPickGraphIndex,
      };
      return;
    }
    if (bundle && degrees) {
      nodeInteractionRef.current = {
        tooltipForGraphIndex: (i: number) => nodeDiskHoverTooltipForIndex(bundle, i, degrees, runMeta),
        pickGraphIndex: (i: number) => {
          if (i < 0 || i >= bundle.vertex.length) return;
          onPickFocus(bundle.vertex[i]!);
        },
        shiftPickGraphIndex,
      };
      return;
    }
    nodeInteractionRef.current = null;
  }, [bundle, bundle3d, degrees, onPickFocus, runMeta, shiftPickGraphIndex]);

  const stats: SceneStats | null = scene?.stats ?? null;
  const focusUiKey = focusAnimTarget?.trim() || appliedFocus.trim();

  return (
    <div
      className="shell"
    >
      {bundle3d ? (
        <BallView3d
          ref={ballViewRef}
          scene={scene as SceneBuffers3d | null}
          pathOverlay={pathOverlay}
          webGpuError={webGpuError}
          showSeedLabels={showSeedLabels}
          nodeInteractionRef={nodeInteractionRef as RefObject<BallView3dNodeInteraction | null>}
          centerWeightedSizes={centerWeightedSizes}
          radialScaleMin={radialScaleMin}
          radialScaleMax={radialScaleMax}
          nodeSizeMul={nodeSizeMul}
          compensateZoomNodes={compensateZoomNodes}
          nodeMinMul={nodeMinMul}
          edgeOpacity={edgeOpacity}
          showHoverNeighborEdges={showHoverNeighborEdges}
          hoverNeighborGraph={hoverNeighborGraph3d}
        />
      ) : (
        <DiskView
          ref={diskViewRef}
          scene={scene as SceneBuffers | null}
          pathOverlay={pathOverlay}
          webGpuError={webGpuError}
          showSeedLabels={showSeedLabels}
          showCrosshair={showCrosshair}
          centerWeightedSizes={centerWeightedSizes}
          radialScaleMin={radialScaleMin}
          radialScaleMax={radialScaleMax}
          nodeSizeMul={nodeSizeMul}
          compensateZoomNodes={compensateZoomNodes}
          nodeMinMul={nodeMinMul}
          edgeOpacity={edgeOpacity}
          showHoverNeighborEdges={showHoverNeighborEdges}
          hoverNeighborGraph={hoverNeighborGraph2d}
          nodeInteractionRef={nodeInteractionRef as RefObject<DiskViewNodeInteraction | null>}
        />
      )}

      {bundle || bundle3d ? (
        <AnalysisFloater
          open={analysisOpen}
          onClose={() => setAnalysisOpen(false)}
          bundle={bundle ?? bundle3d}
          chartMode={bundle3d ? "3d" : "2d"}
          degrees={degrees}
          seedNamesOrdered={analysisSeedOrder}
          z0={z0Current}
          p0Ball={p0Ball}
          csr={graphCSR}
        />
      ) : null}

      {bundle || bundle3d ? (
        <PathTreesFloater
          open={pathTreesOpen}
          onClose={() => setPathTreesOpen(false)}
          bundle={bundle ?? bundle3d}
          scene={scene}
          chartMode={bundle3d ? "3d" : "2d"}
          z0={z0Current}
          p0Ball={p0Ball}
          csr={graphCSR}
          pickerNames={pickerNames}
          appliedFocus={appliedFocus}
          appliedSeedsText={appliedSeedsText}
          onPathOverlayChange={handlePathOverlayChange}
        />
      ) : null}

      {pickerGraph && minimapVisible ? (
        <GraphMinimap
          mode={bundle3d ? "3d" : "2d"}
          graph2d={bundle}
          graph3d={bundle3d}
          minimap2dMode={minimap2dMode}
          minimap3dMode={minimap3dMode}
          focusIndicatorName={focusUiKey}
          seeds={minimapSeeds}
          onChartPick2d={onMinimapChart2d}
          onChartPick3d={onMinimapChart3d}
          onPickFocus={onPickFocus}
        />
      ) : null}

      <details className="advancedPanel">
        <summary>Advanced</summary>
        <div className="advancedInner">
          <label
            className="seedLabelsCb"
            title="When on, green/red node markers shrink partially with Euclidean zoom (50% of full inverse scale) so they grow less on screen. Off = world size unchanged when zooming."
          >
            <input
              type="checkbox"
              checked={compensateZoomNodes}
              onChange={(e) => setCompensateZoomNodes(e.target.checked)}
            />
            <span>Shrink nodes when zooming (50%)</span>
          </label>
          <button
            type="button"
            className="resetViewBtn"
            onClick={() =>
              bundle3d ? ballViewRef.current?.resetView() : diskViewRef.current?.resetView()
            }
            disabled={!!webGpuError}
          >
            Reset view
          </button>
          <label
            className="advancedSlider"
            title="Hide nodes with |W| past the rim band (seeds and seed-touching edges can still be shown)."
          >
            <span className="advancedSliderLabel">rimcull</span>
            <input
              type="range"
              min={0}
              max={RIM_CULL_EPS_SLIDER_MAX}
              step={0.0005}
              value={Math.min(rimCullEps, RIM_CULL_EPS_SLIDER_MAX)}
              onChange={(e) => setRimCullEps(Number(e.target.value))}
              aria-valuetext={`rimcull ${rimCullEps.toFixed(4)}`}
            />
            <span className="advancedSliderVal">{rimCullEps.toFixed(4)}</span>
          </label>
          <label className="seedLabelsCb">
            <input
              type="checkbox"
              checked={showSeedLabels}
              onChange={(e) => setShowSeedLabels(e.target.checked)}
            />
            <span>Show seed labels</span>
          </label>
          <label className="seedLabelsCb">
            <input
              type="checkbox"
              checked={showCrosshair}
              onChange={(e) => setShowCrosshair(e.target.checked)}
            />
            <span>Show crosshair</span>
          </label>
          <label className="seedLabelsCb" title="Overview map (native Z / stereo) bottom-left.">
            <input
              type="checkbox"
              checked={minimapVisible}
              onChange={(e) => setMinimapVisible(e.target.checked)}
            />
            <span>Show overview minimap</span>
          </label>
          {bundle ? (
            <label
              className="advancedSelect"
              title="Chart for the bottom-left 2D overview (native disk vs polar / half-plane / sqrt views of the same data)."
            >
              <span className="advancedSelectLabel">2D overview</span>
              <select
                value={minimap2dMode}
                onChange={(e) => setMinimap2dMode(e.target.value as Minimap2dMode)}
                aria-label="2D overview chart"
              >
                <option value="native_disk">Native disk (re, im)</option>
                <option value="polar_euclidean">Polar (|z|, arg z)</option>
                <option value="hyperbolic_polar">Hyperbolic radius + angle</option>
                <option value="upper_half_plane">Upper half-plane (Cayley)</option>
                <option value="sqrt_branch">Principal √z</option>
              </select>
            </label>
          ) : null}
          {bundle3d ? (
            <label
              className="advancedSelect"
              title="Projection for the bottom-left 3D overview: planar charts on canvas, or a small WebGL globe with orbit and pick."
            >
              <span className="advancedSelectLabel">3D overview</span>
              <select
                value={minimap3dMode}
                onChange={(e) => setMinimap3dMode(e.target.value as Minimap3dMode)}
                aria-label="3D overview projection"
              >
                <option value="stereo_north">North stereographic</option>
                <option value="stereo_south">South stereographic</option>
                <option value="ortho_xy">Orthographic (XY)</option>
                <option value="ortho_xz">Orthographic (XZ)</option>
                <option value="equirect">Equirectangular (lon/lat)</option>
                <option value="lambert_north">Lambert azimuthal (north)</option>
                <option value="gnomonic_north">Gnomonic (north, capped)</option>
                <option value="globe_webgl">Globe (WebGL)</option>
              </select>
            </label>
          ) : null}
          <label
            className="seedLabelsCb"
            title="While the node tooltip is active, draw all graph edges incident on that vertex in white (Poincaré geodesics)."
          >
            <input
              type="checkbox"
              checked={showHoverNeighborEdges}
              onChange={(e) => setShowHoverNeighborEdges(e.target.checked)}
            />
            <span>Hover: highlight neighbor edges</span>
          </label>
          <label
            className="advancedSlider"
            title="Opacity of additive blue one-seed edges on the disk and in the 3D ball."
          >
            <span className="advancedSliderLabel">Edge opacity</span>
            <input
              type="range"
              min={0.02}
              max={1}
              step={0.02}
              value={edgeOpacity}
              onChange={(e) => setEdgeOpacity(Number(e.target.value))}
              aria-valuetext={`edge opacity ${edgeOpacity.toFixed(2)}`}
            />
            <span className="advancedSliderVal">{edgeOpacity.toFixed(2)}</span>
          </label>
          <label
            className="advancedSelect"
            title="Green disks: all non-seeds in view, none, or only vertices on a seed-touching edge. Red seeds and geodesics unchanged."
          >
            <span className="advancedSelectLabel">Show nodes</span>
            <select
              value={nonSeedShowMode}
              onChange={(e) => {
                const v = e.target.value;
                if (v === "all" || v === "seed_only" || v === "seed_and_neighbors") {
                  setNonSeedShowMode(v);
                }
              }}
              aria-label="Show nodes"
            >
              <option value="all">All</option>
              <option value="seed_only">Only seed</option>
              <option value="seed_and_neighbors">Only seed and neighbors</option>
            </select>
          </label>
          <label className="seedLabelsCb" title="Larger near disk center (after focus map); same rule for green and red nodes.">
            <input
              type="checkbox"
              checked={centerWeightedSizes}
              onChange={(e) => setCenterWeightedSizes(e.target.checked)}
            />
            <span>Size nodes by distance to center</span>
          </label>
          <label className="advancedSlider" title="Scales green point sprites and both disk radii (world units).">
            <span className="advancedSliderLabel">Node size</span>
            <input
              type="range"
              min={0.25}
              max={2.5}
              step={0.025}
              value={nodeSizeMul}
              onChange={(e) => setNodeSizeMul(Number(e.target.value))}
              aria-valuetext={`node size ${nodeSizeMul.toFixed(2)}×`}
            />
            <span className="advancedSliderVal">{nodeSizeMul.toFixed(2)}×</span>
          </label>
          <label
            className="advancedSlider"
            title="Raises the smallest nodes: floor on radial scale (when distance sizing is on) and on zoom-size multiplier for disks and green points."
          >
            <span className="advancedSliderLabel">Node minimum</span>
            <input
              type="range"
              min={0.02}
              max={0.55}
              step={0.01}
              value={nodeMinMul}
              onChange={(e) => setNodeMinMul(Number(e.target.value))}
              aria-valuetext={`node minimum ${nodeMinMul.toFixed(2)}`}
            />
            <span className="advancedSliderVal">{nodeMinMul.toFixed(2)}</span>
          </label>
          <label
            className={`advancedSlider${centerWeightedSizes ? "" : " advancedSliderDisabled"}`}
            title="Scale factor at the rim (|W|≈1 in the plane); only used when distance sizing is on."
          >
            <span className="advancedSliderLabel">Rim scale</span>
            <input
              type="range"
              min={0.05}
              max={1.5}
              step={0.01}
              value={radialScaleMin}
              disabled={!centerWeightedSizes}
              onChange={(e) => {
                const v = Number(e.target.value);
                setRadialScaleMin(v);
                setRadialScaleMax((M) => Math.max(M, v));
              }}
              aria-valuetext={`rim scale ${radialScaleMin.toFixed(2)}`}
            />
            <span className="advancedSliderVal">{radialScaleMin.toFixed(2)}</span>
          </label>
          <label
            className={`advancedSlider${centerWeightedSizes ? "" : " advancedSliderDisabled"}`}
            title="Scale factor at disk center; only used when distance sizing is on."
          >
            <span className="advancedSliderLabel">Center scale</span>
            <input
              type="range"
              min={0.3}
              max={2.5}
              step={0.01}
              value={radialScaleMax}
              disabled={!centerWeightedSizes}
              onChange={(e) => {
                const v = Number(e.target.value);
                setRadialScaleMax(v);
                setRadialScaleMin((m) => Math.min(m, v));
              }}
              aria-valuetext={`center scale ${radialScaleMax.toFixed(2)}`}
            />
            <span className="advancedSliderVal">{radialScaleMax.toFixed(2)}</span>
          </label>
        </div>
      </details>

      <details
        className="floatPanel"
        open={seedPanelOpen}
        onToggle={(e) => {
          /* Mirror native `open` — do not preventDefault or invert; that fights the DOM and can loop open/closed. */
          setSeedPanelOpen(e.currentTarget.open);
        }}
      >
        <summary>Seeds &amp; focus</summary>
        <div className="floatPanelBody">
        {loadErr ? <div className="errLine">{loadErr}</div> : null}
        <textarea
          className="seedsTa"
          value={seedsDraft}
          onChange={(e) => setSeedsDraft(e.target.value)}
          spellCheck={false}
          aria-label="Seeds"
        />
        <button
          type="button"
          className="applyMini"
          onClick={onApplySeeds}
          disabled={!pickerGraph}
          title="Apply seeds"
        >
          ↵
        </button>
        {formErr ? <div className="errLine">{formErr}</div> : null}
        <ul className="nameList" aria-label="Focus">
          {pickerNames.map((name) => {
            const idx = pickerGraph?.nameToIndex.get(name);
            const deg = idx !== undefined && degrees ? degrees[idx] : null;
            const tip = pickerGraph
              ? bundle3d
                ? nodeListTooltip3d(bundle3d, name, deg, runMeta)
                : nodeListTooltip(bundle!, name, deg, runMeta)
              : name;
            return (
              <li key={name}>
                <button
                  type="button"
                  className={name === seedListHighlightName.trim() ? "nameBtn nameBtnOn" : "nameBtn"}
                  onClick={() => onPickFocus(name, { source: "list" })}
                  title={tip}
                >
                  <span className="nameBtnLabel">{name}</span>
                  {deg !== null ? <span className="nameDeg">{deg}</span> : null}
                </button>
              </li>
            );
          })}
        </ul>
        {pickerGraph ? (
          <div className="focusSearchWrap">
            <div className="focusSearchRow">
              <label className="focusSearchLabel" htmlFor="focusSearchInput">
                Search
              </label>
              <div className="focusSearchInputRow">
                <input
                  id="focusSearchInput"
                  className="focusSearchInput"
                  type="text"
                  value={focusSearch}
                  onChange={(e) => setFocusSearch(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowDown") {
                      e.preventDefault();
                      if (focusSearchMatches.length === 0) return;
                      setFocusSearchHl((h) => Math.min(h + 1, focusSearchMatches.length - 1));
                    } else if (e.key === "ArrowUp") {
                      e.preventDefault();
                      setFocusSearchHl((h) => Math.max(h - 1, 0));
                    } else if (e.key === "Enter") {
                      e.preventDefault();
                      const exact = focusSearch.trim();
                      if (focusSearchMatches.length > 0) {
                        onPickFocus(focusSearchMatches[focusSearchHl]!, { source: "search" });
                        setFocusSearch("");
                      } else if (exact && pickerGraph.nameToIndex.has(exact)) {
                        onPickFocus(exact, { source: "search" });
                        setFocusSearch("");
                      }
                    } else if (e.key === "Escape") {
                      setFocusSearch("");
                    }
                  }}
                  placeholder="Type gene… Enter to focus"
                  autoComplete="off"
                  spellCheck={false}
                  aria-autocomplete="list"
                  aria-controls="focusSearchList"
                  aria-expanded={focusSearch.trim().length > 0 && focusSearchMatches.length > 0}
                  disabled={!!webGpuError}
                />
                {focusSearch.trim() && focusSearchMatches.length > 0 ? (
                  <ul id="focusSearchList" className="focusSearchList" role="listbox">
                    {focusSearchMatches.map((name, i) => (
                      <li key={name} role="presentation">
                        <button
                          type="button"
                          role="option"
                          aria-selected={i === focusSearchHl}
                          className={i === focusSearchHl ? "focusSearchItem focusSearchItemOn" : "focusSearchItem"}
                          onMouseEnter={() => setFocusSearchHl(i)}
                          onMouseDown={(ev) => ev.preventDefault()}
                          onClick={() => {
                            onPickFocus(name, { source: "search" });
                            setFocusSearch("");
                          }}
                        >
                          {name}
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </div>
          </div>
        ) : null}
        </div>
      </details>

      <div className="statusBox" aria-live="polite">
        {stats ? (
          <>
            <div className="statusRow">
              <span className="statusK">nodes</span>
              <span className="statusV">
                {stats.nodesRendered} / {stats.nodesTotal}
              </span>
            </div>
            <div className="statusRow">
              <span className="statusK">hidden rim</span>
              <span className="statusV">{stats.nodesHiddenRim}</span>
            </div>
            <div className="statusRow sub">
              <span className="statusK">spared seed</span>
              <span className="statusV">{stats.nodesSparedSeedRim}</span>
            </div>
            <div className="statusRow sub">
              <span className="statusK">spared edge</span>
              <span className="statusV">{stats.nodesSparedSeedEdgeRim}</span>
            </div>
            <div className="statusRow">
              <span className="statusK">edges drawn</span>
              <span className="statusV">{stats.edgesDrawn}</span>
            </div>
            {stats.edgesBackgroundPool > 0 ? (
              <div
                className="statusRow sub"
                title={
                  stats.edgesBackgroundPool > stats.edgesBackgroundSubmitted
                    ? `Capped at ${stats.edgesBackgroundSubmitted} of ${stats.edgesBackgroundPool} non-seed edges so the UI stays responsive. Shift+U to toggle.`
                    : "Non-seed–non-seed edges (Shift+U to toggle)."
                }
              >
                <span className="statusK">edges non-seed</span>
                <span className="statusV">
                  {stats.edgesBackgroundDrawn}
                  {stats.edgesBackgroundPool > stats.edgesBackgroundSubmitted
                    ? ` / ${stats.edgesBackgroundPool}`
                    : ""}
                </span>
              </div>
            ) : null}
            <div className="statusRow">
              <span className="statusK">edges seed</span>
              <span className="statusV">{stats.edgesSeedTouching}</span>
            </div>
            <div
              className="statusRow"
              title="Endpoints with |Z| or |W| ≥ 0.999 in disk coords (notebook clip). Not rim hide."
            >
              <span className="statusK">edges skip clip</span>
              <span className="statusV">{stats.edgesSkippedBoundary}</span>
            </div>
          </>
        ) : (
          <div className="statusRow muted">—</div>
        )}
      </div>
    </div>
  );
}
