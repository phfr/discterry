import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { loadGraphBundle, loadMeta, type GraphBundle } from "./data/loadBundle";
import { RIM_CULL_EPS, RIM_CULL_EPS_SLIDER_MAX } from "./math/constants";
import { computeScene, type SceneBuffers, type SceneStats } from "./model/computeScene";
import { z0FromProtein } from "./z0FromProtein";
import { DiskView } from "./viz/DiskView";

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

export default function App() {
  const [bundle, setBundle] = useState<GraphBundle | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [seedsDraft, setSeedsDraft] = useState("");
  const [appliedFocus, setAppliedFocus] = useState("");
  const [appliedSeedsText, setAppliedSeedsText] = useState("");
  const [formErr, setFormErr] = useState<string | null>(null);
  const [rimCullEps, setRimCullEps] = useState(RIM_CULL_EPS);
  const [showSeedLabels, setShowSeedLabels] = useState(true);
  const [showCrosshair, setShowCrosshair] = useState(true);

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

  const scene: SceneBuffers | null = useMemo(() => {
    if (!bundle || !appliedFocus.trim()) return null;
    try {
      const z0 = z0FromProtein(bundle, appliedFocus);
      const seeds = parseSeeds(appliedSeedsText);
      if (seeds.size === 0) return null;
      return computeScene(bundle, z0, seeds, rimCullEps);
    } catch {
      return null;
    }
  }, [bundle, appliedFocus, appliedSeedsText, rimCullEps]);

  const pickerNames = useMemo(
    () => (bundle ? focusPickerNames(bundle, appliedSeedsText, appliedFocus) : []),
    [bundle, appliedSeedsText, appliedFocus],
  );

  const degrees = useMemo(() => (bundle ? vertexDegrees(bundle) : null), [bundle]);

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

  const onPickFocus = useCallback((name: string) => {
    setAppliedFocus(name);
    setFormErr(null);
  }, []);

  const stats: SceneStats | null = scene?.stats ?? null;

  return (
    <div className="shell">
      <DiskView
        scene={scene}
        webGpuError={webGpuError}
        showSeedLabels={showSeedLabels}
        showCrosshair={showCrosshair}
      />

      <div className="floatPanel">
        {loadErr ? <div className="errLine">{loadErr}</div> : null}
        <label className="rimSlider">
          <span className="rimSliderLabel">rimcull</span>
          <input
            type="range"
            min={0}
            max={RIM_CULL_EPS_SLIDER_MAX}
            step={0.0005}
            value={Math.min(rimCullEps, RIM_CULL_EPS_SLIDER_MAX)}
            onChange={(e) => setRimCullEps(Number(e.target.value))}
            aria-valuetext={`rimcull ${rimCullEps.toFixed(4)}`}
          />
          <span className="rimSliderVal">{rimCullEps.toFixed(4)}</span>
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
            return (
              <li key={name}>
                <button
                  type="button"
                  className={name === appliedFocus.trim() ? "nameBtn nameBtnOn" : "nameBtn"}
                  onClick={() => onPickFocus(name)}
                >
                  <span className="nameBtnLabel">{name}</span>
                  {deg !== null ? (
                    <span className="nameDeg" title="Graph degree (undirected)">
                      {deg}
                    </span>
                  ) : null}
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
