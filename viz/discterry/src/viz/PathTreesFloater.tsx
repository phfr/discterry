import { useMemo, useRef, useState } from "react";
import type { GraphBundle } from "../data/loadBundle";
import type { Complex } from "../math/mobius";
import { mobiusDiskArrays } from "../math/mobius";
import type { GraphCSR } from "../model/graphSearch";
import {
  bfsParentTree,
  bfsShortestPath,
  collectFilteredBfsTreeEdges,
  primHyperbolicSeedEdges,
} from "../model/graphSearch";
import type { PathOverlayBuffer } from "../model/pathOverlayBuffer";
import { buildPathOverlayFromEdges, buildPathOverlayFromVertexPath } from "../model/pathOverlayBuffer";
import type { SceneBuffers } from "../model/computeScene";

type TabId = "concepts" | "graph" | "geometry";

const BFS_TREE_MAX_DRAW_EDGES = 8000;

function parseSeeds(text: string): Set<string> {
  const s = new Set<string>();
  for (const part of text.split(/\s+/)) {
    const t = part.trim();
    if (t) s.add(t);
  }
  return s;
}

type Props = {
  open: boolean;
  onClose: () => void;
  bundle: GraphBundle | null;
  scene: SceneBuffers | null;
  z0: Complex | null;
  csr: GraphCSR | null;
  pickerNames: string[];
  appliedFocus: string;
  appliedSeedsText: string;
  onPathOverlayChange: (overlay: PathOverlayBuffer | null) => void;
};

export function PathTreesFloater({
  open,
  onClose,
  bundle,
  scene,
  z0,
  csr,
  pickerNames,
  appliedFocus,
  appliedSeedsText,
  onPathOverlayChange,
}: Props) {
  const [tab, setTab] = useState<TabId>("concepts");
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

  const renderedVertexSet = useMemo(() => {
    if (!scene) return null;
    const s = new Set<number>();
    for (let i = 0; i < scene.seedGraphIndex.length; i++) s.add(scene.seedGraphIndex[i]!);
    for (let i = 0; i < scene.otherGraphIndex.length; i++) s.add(scene.otherGraphIndex[i]!);
    return s;
  }, [scene]);

  const wxWy = useMemo(() => {
    if (!bundle || !z0) return null;
    const n = bundle.vertex.length;
    const wx = new Float32Array(n);
    const wy = new Float32Array(n);
    mobiusDiskArrays(bundle.x, bundle.y, z0, wx, wy, n);
    return { wx, wy };
  }, [bundle, z0]);

  const clearOverlay = () => {
    onPathOverlayChange(null);
    setMsg(null);
  };

  const showShortestPath = () => {
    setMsg(null);
    if (!bundle || !z0 || !csr) {
      setMsg("Graph not ready.");
      return;
    }
    const ia = bundle.nameToIndex.get(fromSel.trim());
    const ib = bundle.nameToIndex.get(toSel.trim());
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
    const buf = buildPathOverlayFromVertexPath(bundle, z0, path);
    if (!buf) {
      setMsg("Path lies past draw clip (boundary).");
      onPathOverlayChange(null);
      return;
    }
    onPathOverlayChange(buf);
    setMsg(`Shortest path: ${path.length} vertices.`);
  };

  const showBfsTree = () => {
    setMsg(null);
    if (!bundle || !z0 || !csr) {
      setMsg("Graph not ready.");
      return;
    }
    const root = bundle.nameToIndex.get(appliedFocus.trim());
    if (root === undefined) {
      setMsg("Focus protein not in graph.");
      return;
    }
    const parent = bfsParentTree(csr, root);
    const edges = collectFilteredBfsTreeEdges(parent, root, renderedVertexSet, BFS_TREE_MAX_DRAW_EDGES);
    if (edges.length === 0) {
      setMsg("No tree edges to draw (try widening seeds / show nodes).");
      onPathOverlayChange(null);
      return;
    }
    const buf = buildPathOverlayFromEdges(bundle, z0, edges);
    if (!buf) {
      setMsg("Tree edges skipped by boundary clip.");
      onPathOverlayChange(null);
      return;
    }
    onPathOverlayChange(buf);
    setMsg(`BFS tree from focus: ${edges.length} edges (capped / filtered to current view).`);
  };

  const showPrimMst = () => {
    setMsg(null);
    if (!bundle || !z0 || !wxWy) {
      setMsg("Embedding not ready.");
      return;
    }
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
    if (idx.length < 2) {
      setMsg("Need ≥2 applied seeds for Prim MST.");
      onPathOverlayChange(null);
      return;
    }
    const { edges, skipped } = primHyperbolicSeedEdges(wxWy.wx, wxWy.wy, idx);
    if (skipped) {
      setMsg(`Too many seeds (>${48}); reduce seed list for Prim.`);
      onPathOverlayChange(null);
      return;
    }
    const buf = buildPathOverlayFromEdges(bundle, z0, edges);
    if (!buf) {
      setMsg("MST edges skipped by boundary clip.");
      onPathOverlayChange(null);
      return;
    }
    onPathOverlayChange(buf);
    setMsg(`Prim MST on ${idx.length} seeds (hyperbolic distance in W at current focus).`);
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
          aria-selected={tab === "concepts"}
          className={tab === "concepts" ? "analysisTab analysisTabOn" : "analysisTab"}
          onClick={() => setTab("concepts")}
        >
          Concepts
        </button>
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
        {tab === "concepts" ? (
          <div className="pathTreesConcepts">
            <p>
              <strong>Graph shortest path</strong> uses <strong>unweighted</strong> PPI edges (BFS). The orange
              overlay draws each graph hop as a <strong>Poincaré geodesic</strong> in the current focus map—the same
              geometric connector as seed edges, not a different notion of “shortest” in hyperbolic space.
            </p>
            <p>
              A <strong>hyperbolic geodesic between two positions</strong> is the metric shortest path in the disk
              model; it need not follow PPI edges.
            </p>
            <p>
              A <strong>“geodesic tree”</strong> in navigation terms is often a spanning tree whose edges are drawn as
              geodesics (e.g. Prim MST among seeds by hyperbolic distance in <strong>W</strong> at the current focus)—
              a geometric object on a small set of vertices, not the same as the PPI BFS tree on the full graph.
            </p>
          </div>
        ) : null}
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
              <button type="button" className="pathTreesBtn" onClick={showBfsTree}>
                BFS tree from focus
              </button>
              <button type="button" className="pathTreesBtn pathTreesBtnGhost" onClick={clearOverlay}>
                Clear overlay
              </button>
            </div>
            <p className="pathTreesHint">
              BFS tree edges are limited to children that appear in the <strong>current disk view</strong> (green +
              red markers), up to {BFS_TREE_MAX_DRAW_EDGES} edges.
            </p>
          </div>
        ) : null}
        {tab === "geometry" ? (
          <div className="pathTreesControls">
            <p className="pathTreesHint">
              <strong>Prim MST</strong> on <strong>applied seeds</strong> only, using hyperbolic distance in the
              W-plane at the current focus. Capped at 48 seeds.
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
