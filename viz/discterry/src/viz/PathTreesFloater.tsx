import { useRef, useState } from "react";
import type { GraphBundle } from "../data/loadBundle";
import type { GraphBundle3d } from "../data/loadBundle3d";
import type { Complex } from "../math/mobius";
import { PRIM_MAX_SEEDS, bfsShortestPath, type GraphCSR } from "../model/graphSearch";
import { tryBuildPrimMstOverlay, tryBuildPrimMstOverlay3d } from "../model/primMstOverlay";
import type { PathOverlayBuffer } from "../model/pathOverlayBuffer";
import { buildPathOverlayFromVertexPath, buildPathOverlayFromVertexPath3d } from "../model/pathOverlayBuffer";
type TabId = "graph" | "geometry";

type Props = {
  open: boolean;
  onClose: () => void;
  bundle: GraphBundle | GraphBundle3d | null;
  chartMode: "2d" | "3d";
  z0: Complex | null;
  p0Ball: { x: number; y: number; z: number } | null;
  csr: GraphCSR | null;
  pickerNames: string[];
  appliedSeedsText: string;
  onPathOverlayChange: (overlay: PathOverlayBuffer | null) => void;
};

export function PathTreesFloater({
  open,
  onClose,
  bundle,
  chartMode,
  z0,
  p0Ball,
  csr,
  pickerNames,
  appliedSeedsText,
  onPathOverlayChange,
}: Props) {
  const [tab, setTab] = useState<TabId>("graph");
  const [pos, setPos] = useState({ left: 360, bottom: 24 });
  const dragRef = useRef<{ sx: number; sy: number; l0: number; b0: number } | null>(null);
  const headerRef = useRef<HTMLDivElement>(null);

  const [pathFrom, setPathFrom] = useState("");
  const [pathTo, setPathTo] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const fromSel =
    pathFrom && pickerNames.includes(pathFrom) ? pathFrom : (pickerNames[0] ?? "");
  const toSel =
    pathTo && pickerNames.includes(pathTo)
      ? pathTo
      : pickerNames.length > 1
        ? pickerNames[1]!
        : (pickerNames[0] ?? "");

  const clearOverlay = () => {
    onPathOverlayChange(null);
    setMsg(null);
  };

  const showShortestPath = () => {
    setMsg(null);
    if (!bundle || !csr) {
      setMsg("Graph not ready.");
      return;
    }
    if (chartMode === "3d") {
      if (!p0Ball) {
        setMsg("Ball focus not ready.");
        return;
      }
      const b = bundle as GraphBundle3d;
      const ia = b.nameToIndex.get(fromSel.trim());
      const ib = b.nameToIndex.get(toSel.trim());
      if (ia === undefined || ib === undefined) {
        setMsg("Pick valid names from the list.");
        return;
      }
      const path = bfsShortestPath(csr, ia, ib);
      if (!path) {
        setMsg("No path (disconnected in this graph).");
        onPathOverlayChange(null);
        return;
      }
      const buf = buildPathOverlayFromVertexPath3d(b, p0Ball.x, p0Ball.y, p0Ball.z, path);
      if (!buf) {
        setMsg("Path lies past draw clip (boundary).");
        onPathOverlayChange(null);
        return;
      }
      onPathOverlayChange(buf);
      setMsg(`Shortest path: ${path.length} vertices.`);
      return;
    }
    if (!z0) {
      setMsg("Graph not ready.");
      return;
    }
    const b2 = bundle as GraphBundle;
    const ia = b2.nameToIndex.get(fromSel.trim());
    const ib = b2.nameToIndex.get(toSel.trim());
    if (ia === undefined || ib === undefined) {
      setMsg("Pick valid names from the list.");
      return;
    }
    const path = bfsShortestPath(csr, ia, ib);
    if (!path) {
      setMsg("No path (disconnected in this graph).");
      onPathOverlayChange(null);
      return;
    }
    const buf = buildPathOverlayFromVertexPath(b2, z0, path);
    if (!buf) {
      setMsg("Path lies past draw clip (boundary).");
      onPathOverlayChange(null);
      return;
    }
    onPathOverlayChange(buf);
    setMsg(`Shortest path: ${path.length} vertices.`);
  };

  const showPrimMst = () => {
    setMsg(null);
    if (!bundle) {
      setMsg("Embedding not ready.");
      return;
    }
    if (chartMode === "3d") {
      if (!p0Ball) {
        setMsg("Ball focus not ready.");
        return;
      }
      const r = tryBuildPrimMstOverlay3d(
        bundle as GraphBundle3d,
        p0Ball.x,
        p0Ball.y,
        p0Ball.z,
        appliedSeedsText,
      );
      if (!r.ok) {
        if (r.reason === "need_two_seeds") {
          setMsg("Need ≥2 applied seeds for Prim MST.");
        } else if (r.reason === "too_many_seeds") {
          setMsg(`Too many seeds (>${PRIM_MAX_SEEDS}); reduce seed list for Prim.`);
        } else {
          setMsg("MST edges skipped by boundary clip.");
        }
        onPathOverlayChange(null);
        return;
      }
      onPathOverlayChange(r.buf);
      setMsg(`Prim MST on ${r.seedCount} seeds (hyperbolic distance in 𝔹³ W at current focus).`);
      return;
    }
    if (!z0) {
      setMsg("Embedding not ready.");
      return;
    }
    const r = tryBuildPrimMstOverlay(bundle as GraphBundle, z0, appliedSeedsText);
    if (!r.ok) {
      if (r.reason === "need_two_seeds") {
        setMsg("Need ≥2 applied seeds for Prim MST.");
      } else if (r.reason === "too_many_seeds") {
        setMsg(`Too many seeds (>${PRIM_MAX_SEEDS}); reduce seed list for Prim.`);
      } else {
        setMsg("MST edges skipped by boundary clip.");
      }
      onPathOverlayChange(null);
      return;
    }
    onPathOverlayChange(r.buf);
    setMsg(`Prim MST on ${r.seedCount} seeds (hyperbolic distance in W at current focus).`);
  };

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
      className="pathTreesFloater"
      style={{ left: pos.left, bottom: pos.bottom }}
      role="dialog"
      aria-label="Path and trees"
    >
      <div
        ref={headerRef}
        className="analysisFloaterHeader"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        onPointerCancel={onHeaderPointerUp}
      >
        <span className="analysisFloaterTitle">Path / trees</span>
        <button
          type="button"
          className="analysisFloaterClose"
          onClick={onClose}
          onPointerDown={(e) => e.stopPropagation()}
          aria-label="Close path trees panel"
        >
          ×
        </button>
      </div>
      <div className="analysisFloaterTabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "graph"}
          className={tab === "graph" ? "analysisTab analysisTabOn" : "analysisTab"}
          onClick={() => setTab("graph")}
        >
          Graph distance
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "geometry"}
          className={tab === "geometry" ? "analysisTab analysisTabOn" : "analysisTab"}
          onClick={() => setTab("geometry")}
        >
          Geometry
        </button>
      </div>
      <div className="pathTreesFloaterBody">
        {tab === "graph" ? (
          <div className="pathTreesControls">
            <label className="pathTreesLabel">
              Path from
              <select
                className="pathTreesSelect"
                value={fromSel}
                onChange={(e) => setPathFrom(e.target.value)}
                aria-label="Path from"
              >
                {pickerNames.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="pathTreesLabel">
              Path to
              <select
                className="pathTreesSelect"
                value={toSel}
                onChange={(e) => setPathTo(e.target.value)}
                aria-label="Path to"
              >
                {pickerNames.map((n) => (
                  <option key={`t-${n}`} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <div className="pathTreesBtnRow">
              <button type="button" className="pathTreesBtn" onClick={showShortestPath}>
                Show shortest path
              </button>
              <button type="button" className="pathTreesBtn pathTreesBtnGhost" onClick={clearOverlay}>
                Clear overlay
              </button>
            </div>
            <p className="pathTreesHint">
              Shortest path is the BFS shortest route in the abstract graph, drawn as hyperbolic geodesics in the
              current {chartMode === "3d" ? "ball" : "disk"} view.
            </p>
          </div>
        ) : null}
        {tab === "geometry" ? (
          <div className="pathTreesControls">
            <p className="pathTreesHint">
              <strong>Prim MST</strong> on <strong>applied seeds</strong> only, using hyperbolic distance in{" "}
              {chartMode === "3d" ? "the Poincaré ball W" : "the W-plane"} at the current focus. Capped at 48
              seeds.
            </p>
            <div className="pathTreesBtnRow">
              <button type="button" className="pathTreesBtn" onClick={showPrimMst}>
                Show Prim MST (seeds)
              </button>
              <button type="button" className="pathTreesBtn pathTreesBtnGhost" onClick={clearOverlay}>
                Clear overlay
              </button>
            </div>
          </div>
        ) : null}
        {msg ? <p className="pathTreesMsg">{msg}</p> : null}
      </div>
    </div>
  );
}
