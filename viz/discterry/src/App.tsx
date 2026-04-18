import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { loadGraphBundle, loadMeta, type GraphBundle, type MetaJson } from "./data/loadBundle";
import { RIM_CULL_EPS, RIM_CULL_EPS_SLIDER_MAX } from "./math/constants";
import {
  easeInOutCubic,
  fillZ0Geodesic,
  z0AtGeodesicParameter,
  Z0_GEODESIC_SAMPLES,
} from "./math/focusAnimPath";
import type { Complex } from "./math/mobius";
import {
  computeScene,
  type NonSeedShowMode,
  type SceneBuffers,
  type SceneStats,
} from "./model/computeScene";
import { bundleToCSR } from "./model/graphSearch";
import type { PathOverlayBuffer } from "./model/pathOverlayBuffer";
import { z0FromProtein } from "./z0FromProtein";
import { nodeListTooltip, nodeListTooltipForIndex } from "./nodeListTooltip";
import { analysisKeyboardGuard } from "./analysisKeyboardGuard";
import { AnalysisFloater } from "./viz/AnalysisFloater";
import { DiskView, type DiskViewHandle, type DiskViewNodeInteraction } from "./viz/DiskView";
import { PathTreesFloater } from "./viz/PathTreesFloater";

const DEFAULT_RADIAL_SCALE_MIN = 0.2;
const DEFAULT_RADIAL_SCALE_MAX = 1.3;
const DEFAULT_NODE_SIZE_MUL = 0.5;
/** Floor for radial instance scale and for zoom-size multiplier (see DiskView). */
const DEFAULT_NODE_MIN_MUL = 0.1;

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

function vertexDegrees(bundle: GraphBundle): Int32Array {
  const n = bundle.vertex.length;
  const d = new Int32Array(n);
  const { src, dst } = bundle;
  for (let ei = 0; ei < src.length; ei++) {
    d[src[ei]]++;
    d[dst[ei]]++;
  }
  return d;
}

/** Sorted clickable names: valid applied seeds, plus current focus if in bundle but not listed. */
function focusPickerNames(bundle: GraphBundle, appliedSeedsText: string, appliedFocus: string): string[] {
  const seeds = [...parseSeeds(appliedSeedsText)].filter((n) => bundle.nameToIndex.has(n));
  const set = new Set(seeds);
  const f = appliedFocus.trim();
  if (f && bundle.nameToIndex.has(f) && !set.has(f)) seeds.push(f);
  seeds.sort((a, b) => a.localeCompare(b));
  return seeds;
}

/** Applied seed names in first-seen order (from applied text), bundle-valid only. */
function appliedSeedNamesOrdered(bundle: GraphBundle, appliedSeedsText: string): string[] {
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
  const [bundle, setBundle] = useState<GraphBundle | null>(null);
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
  const [nonSeedShowMode, setNonSeedShowMode] = useState<NonSeedShowMode>("all");
  const [focusAnimTarget, setFocusAnimTarget] = useState<string | null>(null);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [pathTreesOpen, setPathTreesOpen] = useState(false);
  const diskViewRef = useRef<DiskViewHandle>(null);
  const nodeInteractionRef = useRef<DiskViewNodeInteraction | null>(null);
  const appliedFocusRef = useRef(appliedFocus);
  const focusAnimTargetRef = useRef<string | null>(null);
  const focusAnimGenRef = useRef(0);
  const focusAnimRafRef = useRef(0);
  const isFocusAnimatingRef = useRef(false);
  const z0AnimRef = useRef<Complex | null>(null);

  useLayoutEffect(() => {
    appliedFocusRef.current = appliedFocus;
  }, [appliedFocus]);
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
    let cancelled = false;
    void (async () => {
      try {
        const [b, meta] = await Promise.all([
          loadGraphBundle(import.meta.env.BASE_URL),
          loadMeta(import.meta.env.BASE_URL),
        ]);
        if (cancelled) return;
        setBundle(b);
        setRunMeta(meta ?? null);
        const defF = meta?.default_focus?.trim() || b.vertex[0] || "";
        const defSeeds = (meta?.default_seeds?.join(" ") || defF).replace(/\s+/g, " ").trim();
        setSeedsDraft(defSeeds);
        setAppliedFocus(defF);
        setAppliedSeedsText(defSeeds);
        setLoadErr(null);
      } catch (e) {
        if (!cancelled) setLoadErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isFocusAnimatingRef.current) return;
    focusAnimGenRef.current += 1;
    cancelAnimationFrame(focusAnimRafRef.current);
    isFocusAnimatingRef.current = false;
    setFocusAnimTarget(null);
    z0AnimRef.current = null;
  }, [appliedSeedsText, rimCullEps, bundle, nonSeedShowMode]);

  useEffect(() => {
    return () => {
      focusAnimGenRef.current += 1;
      cancelAnimationFrame(focusAnimRafRef.current);
    };
  }, []);

  const z0Current = useMemo(() => {
    if (!bundle || !appliedFocus.trim()) return null;
    try {
      return z0FromProtein(bundle, appliedFocus);
    } catch {
      return null;
    }
  }, [bundle, appliedFocus]);

  const graphCSR = useMemo(() => (bundle ? bundleToCSR(bundle) : null), [bundle]);

  const graphInteractionKey = useMemo(
    () =>
      bundle
        ? `${bundle.vertex.length}|${appliedFocus}|${appliedSeedsText}|${rimCullEps}|${nonSeedShowMode}`
        : "",
    [bundle, appliedFocus, appliedSeedsText, rimCullEps, nonSeedShowMode],
  );

  const [pathOverlayPinned, setPathOverlayPinned] = useState<{
    buf: PathOverlayBuffer;
    key: string;
  } | null>(null);
  const pathOverlay =
    pathOverlayPinned && pathOverlayPinned.key === graphInteractionKey ? pathOverlayPinned.buf : null;

  const scene: SceneBuffers | null = useMemo(() => {
    if (!bundle || !appliedFocus.trim() || !z0Current) return null;
    try {
      const seeds = parseSeeds(appliedSeedsText);
      if (seeds.size === 0) return null;
      return computeScene(bundle, z0Current, seeds, rimCullEps, nonSeedShowMode);
    } catch {
      return null;
    }
  }, [bundle, appliedFocus, appliedSeedsText, rimCullEps, nonSeedShowMode, z0Current]);

  const pickerNames = useMemo(
    () => (bundle ? focusPickerNames(bundle, appliedSeedsText, appliedFocus) : []),
    [bundle, appliedSeedsText, appliedFocus],
  );

  const degrees = useMemo(() => (bundle ? vertexDegrees(bundle) : null), [bundle]);

  const analysisSeedOrder = useMemo(
    () => (bundle ? appliedSeedNamesOrdered(bundle, appliedSeedsText) : []),
    [bundle, appliedSeedsText],
  );

  useEffect(() => {
    if (webGpuError || !bundle) return;
    const onKey = (e: KeyboardEvent) => {
      if (analysisKeyboardGuard(e)) return;
      if (e.code === "KeyR") {
        e.preventDefault();
        diskViewRef.current?.resetView();
      } else if (e.code === "KeyF") {
        e.preventDefault();
        diskViewRef.current?.fitSubgraphToView();
      } else if (e.code === "KeyA") {
        e.preventDefault();
        setAnalysisOpen((o) => !o);
      } else if (e.code === "KeyS") {
        e.preventDefault();
        setPathTreesOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [webGpuError, bundle]);

  const onApplySeeds = useCallback(() => {
    setFormErr(null);
    if (!bundle) return;
    const seeds = parseSeeds(seedsDraft);
    if (seeds.size === 0) {
      setFormErr("Need ≥1 seed");
      return;
    }
    const unknown = [...seeds].filter((n) => !bundle.nameToIndex.has(n));
    if (unknown.length) {
      setFormErr(`Unknown: ${unknown.slice(0, 5).join(", ")}`);
      return;
    }
    setAppliedSeedsText(seedsDraft);
    const f = appliedFocus.trim();
    if (!f || !seeds.has(f) || !bundle.nameToIndex.has(f)) {
      const first = [...seeds].find((n) => bundle.nameToIndex.has(n));
      if (first) setAppliedFocus(first);
    }
  }, [bundle, seedsDraft, appliedFocus]);

  const onPickFocus = useCallback(
    (name: string) => {
      setFormErr(null);
      const t = name.trim();
      if (webGpuError) {
        setAppliedFocus(t);
        return;
      }
      if (!bundle) return;
      if (!isFocusAnimatingRef.current && t === appliedFocusRef.current.trim()) return;
      if (isFocusAnimatingRef.current && t === (focusAnimTargetRef.current ?? "").trim()) return;

      cancelAnimationFrame(focusAnimRafRef.current);
      const gen = ++focusAnimGenRef.current;

      let z0Start: Complex;
      try {
        const fromName = appliedFocusRef.current.trim();
        z0Start = z0AnimRef.current ?? z0FromProtein(bundle, fromName || t);
      } catch {
        setAppliedFocus(t);
        return;
      }
      let z0End: Complex;
      try {
        z0End = z0FromProtein(bundle, t);
      } catch {
        return;
      }

      const seeds = parseSeeds(appliedSeedsText);
      if (seeds.size === 0) {
        setAppliedFocus(t);
        return;
      }

      const gx = new Float32Array(Z0_GEODESIC_SAMPLES);
      const gy = new Float32Array(Z0_GEODESIC_SAMPLES);
      fillZ0Geodesic(z0Start, z0End, Z0_GEODESIC_SAMPLES, gx, gy);

      /* Pan/Möbius are camera-only; data focus puts the gene at w=0 — clear offsets so that stays centered. */
      diskViewRef.current?.recenterPreservingZoom();

      isFocusAnimatingRef.current = true;
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
          buf = computeScene(bundle, z0, seeds, rimCullEps, nonSeedShowMode);
        } catch {
          focusAnimGenRef.current += 1;
          isFocusAnimatingRef.current = false;
          z0AnimRef.current = null;
          setFocusAnimTarget(null);
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
    [bundle, appliedSeedsText, rimCullEps, nonSeedShowMode, webGpuError],
  );

  useLayoutEffect(() => {
    if (!bundle || !degrees) {
      nodeInteractionRef.current = null;
      return;
    }
    nodeInteractionRef.current = {
      tooltipForGraphIndex: (i: number) => nodeListTooltipForIndex(bundle, i, degrees, runMeta),
      pickGraphIndex: (i: number) => {
        if (i < 0 || i >= bundle.vertex.length) return;
        onPickFocus(bundle.vertex[i]!);
      },
    };
  }, [bundle, degrees, onPickFocus, runMeta]);

  const stats: SceneStats | null = scene?.stats ?? null;
  const focusUiKey = focusAnimTarget?.trim() || appliedFocus.trim();

  return (
    <div
      className="shell"
      title="Shortcuts: R reset view, F fit subgraph, A analysis, S path/trees."
    >
      <DiskView
        ref={diskViewRef}
        scene={scene}
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
        nodeInteractionRef={nodeInteractionRef}
      />

      <AnalysisFloater
        open={analysisOpen}
        onClose={() => setAnalysisOpen(false)}
        bundle={bundle}
        degrees={degrees}
        seedNamesOrdered={analysisSeedOrder}
      />

      <PathTreesFloater
        open={pathTreesOpen}
        onClose={() => setPathTreesOpen(false)}
        bundle={bundle}
        scene={scene}
        z0={z0Current}
        csr={graphCSR}
        pickerNames={pickerNames}
        appliedFocus={appliedFocus}
        appliedSeedsText={appliedSeedsText}
        onPathOverlayChange={(b) =>
          setPathOverlayPinned(b && graphInteractionKey ? { buf: b, key: graphInteractionKey } : null)
        }
      />

      <details className="advancedPanel">
        <summary>Advanced</summary>
        <div className="advancedInner">
          <p className="advancedShortcutsHint">
            Shortcuts: <kbd>R</kbd> reset view, <kbd>F</kbd> fit subgraph, <kbd>A</kbd> analysis,{" "}
            <kbd>S</kbd> path/trees.
          </p>
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
            onClick={() => diskViewRef.current?.resetView()}
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

      <div className="floatPanel">
        {loadErr ? <div className="errLine">{loadErr}</div> : null}
        <textarea
          className="seedsTa"
          value={seedsDraft}
          onChange={(e) => setSeedsDraft(e.target.value)}
          spellCheck={false}
          aria-label="Seeds"
        />
        <button type="button" className="applyMini" onClick={onApplySeeds} disabled={!bundle} title="Apply seeds">
          ↵
        </button>
        {formErr ? <div className="errLine">{formErr}</div> : null}
        <ul className="nameList" aria-label="Focus">
          {pickerNames.map((name) => {
            const idx = bundle?.nameToIndex.get(name);
            const deg = idx !== undefined && degrees ? degrees[idx] : null;
            const tip = bundle ? nodeListTooltip(bundle, name, deg, runMeta) : name;
            return (
              <li key={name}>
                <button
                  type="button"
                  className={name === focusUiKey ? "nameBtn nameBtnOn" : "nameBtn"}
                  onClick={() => onPickFocus(name)}
                  title={tip}
                >
                  <span className="nameBtnLabel">{name}</span>
                  {deg !== null ? <span className="nameDeg">{deg}</span> : null}
                </button>
              </li>
            );
          })}
        </ul>
      </div>

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
