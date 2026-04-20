import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  type RefObject,
} from "react";
import {
  WebGPURenderer,
  Scene,
  Group,
  Object3D,
  Mesh,
  PerspectiveCamera,
  LineSegments,
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  Line2NodeMaterial,
  Color,
  AdditiveBlending,
  InstancedMesh,
  MeshBasicMaterial,
  SphereGeometry,
  Matrix4,
  Vector3,
  Vector2,
  Quaternion,
  Raycaster,
  EdgesGeometry,
} from "three/webgpu";
import { LineSegments2 } from "three/addons/lines/webgpu/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { NodeDiskHoverTooltip } from "../nodeListTooltip";
import type { PathOverlayBuffer } from "../model/pathOverlayBuffer";
import type { SceneBuffers3d } from "../model/computeScene3d";
import {
  buildHoverNeighborLinePositions3d,
  type HoverNeighborGraph3d,
} from "../model/hoverNeighborEdgePositions";
import { VIEWER_RIM_GAMMA_MAX, VIEWER_RIM_GAMMA_MIN, VIEWER_RIM_WHEEL_SENS } from "../math/constants";

/**
 * Unit-sphere geometry is radius 1; instance matrix scale sets world radius.
 * Chosen so default `nodeSizeMul`≈0.5 matches disk-like footprint in the |w|≤1 ball.
 */
const OTHER_BALL_BASE = 0.0055;
/** Non-seed (green) spheres: visual scale relative to `OTHER_BALL_BASE` (seeds unchanged). */
const OTHER_BALL_VISUAL_SCALE = 0.5;
const SEED_BALL_BASE = 0.011;
const BALL_WIRE_SEGMENTS = 48;
/** Orbit distance at default framing — used like disk `zoom` for node size compensation. */
const CAM_REF_DIST = 2.85;
const ZOOM_NODE_COMP_STRENGTH = 0.5;
const PATH_OVERLAY_OPACITY_BASE = 0.55;
/** Prim MST / path overlay stroke width in 3D (DiskView uses 4; this is 3×). */
const PATH_OVERLAY_LINEWIDTH_3D = 8;
/** Seed–seed (“red”) edges in 3D: 1.5× DiskView `lineBoth` linewidth (4). */
const BOTH_SEED_LINEWIDTH_3D = 6;
/** Shift+drag: rotate graph content (trackball-style, camera-relative axes). */
const SHIFT_BALL_DRAG_ROT = 0.0055;

export type BallDisplaySizing = {
  centerWeighted: boolean;
  radialMin: number;
  radialMax: number;
  nodeSizeMul: number;
  compensateZoomNodes: boolean;
  nodeMinMul: number;
  /** Opacity of additive blue one-seed edges (`lineOne`). */
  edgeOpacity: number;
};

export type BallView3dHandle = {
  resetView: () => void;
  applySceneBuffers: (buf: SceneBuffers3d | null) => void;
  fitSubgraphToView: () => void;
  setPathOverlayOpacityMultiplier: (mult: number) => void;
};

export type BallView3dNodeInteraction = {
  tooltipForGraphIndex: (graphIndex: number) => NodeDiskHoverTooltip | null;
  pickGraphIndex: (graphIndex: number) => void;
  shiftPickGraphIndex?: (graphIndex: number) => void;
};

type OrbitState = {
  yaw: number;
  pitch: number;
  distance: number;
};

function defaultOrbit(): OrbitState {
  return { yaw: 0.55, pitch: 0.38, distance: 2.85 };
}

function orbitToPosition(o: OrbitState, out: InstanceType<typeof Vector3>): void {
  const cp = Math.cos(o.pitch);
  out.set(cp * Math.sin(o.yaw), Math.sin(o.pitch), cp * Math.cos(o.yaw)).multiplyScalar(o.distance);
}

type BallCtx = {
  renderer: InstanceType<typeof WebGPURenderer>;
  scene3: InstanceType<typeof Scene>;
  camera: InstanceType<typeof PerspectiveCamera>;
  host: HTMLDivElement;
  lineBg: InstanceType<typeof LineSegments> | null;
  lineBoth: InstanceType<typeof Mesh> | null;
  lineOne: InstanceType<typeof LineSegments> | null;
  meshOther: InstanceType<typeof InstancedMesh> | null;
  meshSeeds: InstanceType<typeof InstancedMesh> | null;
  linePath: InstanceType<typeof Mesh> | null;
  /** White incident edges for hovered vertex (LineSegments2, under spheres). */
  lineHoverNbr: InstanceType<typeof Mesh> | null;
  ballWire: InstanceType<typeof LineSegments> | null;
  /** Graph + overlay (not the unit wire); Shift+drag rotates this group. */
  contentRoot: InstanceType<typeof Group>;
  rimGammaRef: RefObject<number>;
  nodeTipEl: HTMLDivElement;
  raycaster: InstanceType<typeof Raycaster>;
  ndcPointer: InstanceType<typeof Vector2>;
  labelsLayer: HTMLDivElement;
  labelSpans: HTMLSpanElement[];
  raf: number;
};

function disposeLine(m: InstanceType<typeof LineSegments> | null, parent: InstanceType<typeof Object3D> | null) {
  if (!m || !parent) return;
  parent.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof LineBasicMaterial>).dispose();
}

function disposeWideLineSegments2(m: InstanceType<typeof Mesh> | null, parent: InstanceType<typeof Object3D> | null) {
  if (!m || !parent) return;
  parent.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof Line2NodeMaterial>).dispose();
}

function disposeInst(m: InstanceType<typeof InstancedMesh> | null, parent: InstanceType<typeof Object3D> | null) {
  if (!m || !parent) return;
  parent.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof MeshBasicMaterial>).dispose();
}

const _m4 = new Matrix4();
const _v3 = new Vector3();
const _vScale = new Vector3();
const _qId = new Quaternion();
const _seedLblView = new Vector3();
const _seedLblNdc = new Vector3();
const _seedLblWorld = new Vector3();
const _dragAxisRight = new Vector3();
const _dragAxisUp = new Vector3();
const _qDragH = new Quaternion();
const _qDragV = new Quaternion();
const _qDragDelta = new Quaternion();

/** Display map w ↦ |w|^(γ−1) w (unit sphere fixed); same idea as disk ``rimPreservingGamma``. */
function rimBallMapInPlace(v: InstanceType<typeof Vector3>, gamma: number): void {
  if (Math.abs(gamma - 1) < 1e-9) return;
  const r = v.length();
  if (r < 1e-15) {
    v.set(0, 0, 0);
    return;
  }
  const s = r ** (gamma - 1);
  v.multiplyScalar(s);
}

function setRimDisplayWorld(
  rawx: number,
  rawy: number,
  rawz: number,
  ctx: BallCtx,
  out: InstanceType<typeof Vector3>,
): void {
  out.set(rawx, rawy, rawz);
  rimBallMapInPlace(out, ctx.rimGammaRef.current ?? 1);
  out.applyQuaternion(ctx.contentRoot.quaternion);
}

type BallHoverEdgeState = {
  enabled: boolean;
  graph: HoverNeighborGraph3d | null;
  gid: number;
};

function fillRimGammaPositions(
  src: Float32Array,
  nFloats: number,
  gamma: number,
  dst: Float32Array,
): void {
  if (Math.abs(gamma - 1) < 1e-9) {
    dst.set(src.subarray(0, nFloats));
    return;
  }
  for (let i = 0; i < nFloats; i += 3) {
    const x = src[i]!;
    const y = src[i + 1]!;
    const z = src[i + 2]!;
    const r = Math.hypot(x, y, z);
    if (r < 1e-15) {
      dst[i] = 0;
      dst[i + 1] = 0;
      dst[i + 2] = 0;
    } else {
      const s = r ** (gamma - 1);
      dst[i] = x * s;
      dst[i + 1] = y * s;
      dst[i + 2] = z * s;
    }
  }
}

function refreshHoverNeighborEdges3d(
  ctx: BallCtx,
  buf: SceneBuffers3d | null,
  st: BallHoverEdgeState,
): void {
  const parent = ctx.contentRoot;
  disposeWideLineSegments2(ctx.lineHoverNbr, parent);
  ctx.lineHoverNbr = null;
  if (!buf || !st.enabled || !st.graph || st.gid < 0 || st.gid >= st.graph.csr.n) return;
  const g = st.graph;
  const raw = buildHoverNeighborLinePositions3d(
    g.csr,
    g.px,
    g.py,
    g.pz,
    g.p0x,
    g.p0y,
    g.p0z,
    st.gid,
  );
  if (raw.length < 6) return;
  const gamma = ctx.rimGammaRef.current ?? 1;
  const dst = new Float32Array(raw.length);
  fillRimGammaPositions(raw, raw.length, gamma, dst);
  const geom = new LineSegmentsGeometry();
  geom.setPositions(dst);
  const mat = new Line2NodeMaterial({
    color: 0xffffff,
    linewidth: 1,
    transparent: true,
    opacity: 0.9,
    depthTest: true,
    depthWrite: false,
    toneMapped: false,
  });
  const ln = new LineSegments2(geom, mat);
  ln.renderOrder = 18;
  parent.add(ln);
  ctx.lineHoverNbr = ln;
}

function radialScaleFromR(r: number, sMin: number, sMax: number): number {
  const t = Math.min(1, Math.max(0, r));
  const lo = Math.min(sMin, sMax);
  const hi = Math.max(sMin, sMax);
  return lo + (1 - t) * (hi - lo);
}

function nodeZoomMultiplier(camDist: number, compensate: boolean): number {
  if (!compensate) return 1;
  const inv = Math.min(2.2, Math.max(0.45, camDist / CAM_REF_DIST));
  return 1 + (inv - 1) * ZOOM_NODE_COMP_STRENGTH;
}

function clampedRadialScale(
  r: number,
  radialMin: number,
  radialMax: number,
  nodeMinMul: number,
): number {
  const lo = Math.min(radialMin, radialMax);
  const hi = Math.max(radialMin, radialMax);
  const s0 = radialScaleFromR(r, radialMin, radialMax);
  return Math.min(hi, Math.max(lo, Math.max(s0, nodeMinMul)));
}

function syncBallSeedLabels(ctx: BallCtx, buf: SceneBuffers3d | null, show: boolean) {
  const layer = ctx.labelsLayer;
  layer.innerHTML = "";
  ctx.labelSpans = [];
  if (!show || !buf || buf.nPointsSeed === 0) return;
  for (let i = 0; i < buf.nPointsSeed; i++) {
    const sp = document.createElement("span");
    sp.className = "seedLabel";
    sp.textContent = buf.seedLabels[i] ?? "";
    layer.appendChild(sp);
    ctx.labelSpans.push(sp);
  }
}

function updateBallSeedLabelPositions(ctx: BallCtx, buf: SceneBuffers3d | null, show: boolean) {
  const spans = ctx.labelSpans;
  if (!show || !buf || buf.nPointsSeed === 0 || spans.length !== buf.nPointsSeed) return;
  const { camera, renderer } = ctx;
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const p = buf.pointsSeed;
  const ox = 8;
  const oy = -7;
  for (let i = 0; i < spans.length; i++) {
    const ix = i * 3;
    const wx = p[ix]!;
    const wy = p[ix + 1]!;
    const wz = p[ix + 2]!;
    setRimDisplayWorld(wx, wy, wz, ctx, _seedLblWorld);
    _seedLblView.copy(_seedLblWorld).applyMatrix4(camera.matrixWorldInverse);
    if (_seedLblView.z >= 0) {
      spans[i].style.visibility = "hidden";
      continue;
    }
    _seedLblNdc.copy(_seedLblWorld).project(camera);
    spans[i].style.visibility = "visible";
    const sx = (_seedLblNdc.x * 0.5 + 0.5) * w + ox;
    const sy = (-_seedLblNdc.y * 0.5 + 0.5) * h + oy;
    spans[i].style.transform = `translate(${sx}px, ${sy}px)`;
  }
}

function applySceneToBall(
  ctx: BallCtx,
  buf: SceneBuffers3d | null,
  orbitRef: RefObject<OrbitState>,
  camera: InstanceType<typeof PerspectiveCamera>,
  sizing: BallDisplaySizing,
  pathOverlay: PathOverlayBuffer | null,
  pathOverlayOpacityMult: number,
  showSeedLabels: boolean,
  hoverEdgeRef: RefObject<BallHoverEdgeState>,
  altNeighborHoverRef: RefObject<boolean>,
) {
  const parent = ctx.contentRoot;
  const gamma = ctx.rimGammaRef.current ?? 1;
  disposeWideLineSegments2(ctx.lineHoverNbr, parent);
  ctx.lineHoverNbr = null;
  disposeLine(ctx.lineBg, parent);
  disposeWideLineSegments2(ctx.lineBoth, parent);
  disposeLine(ctx.lineOne, parent);
  disposeWideLineSegments2(ctx.linePath, parent);
  disposeInst(ctx.meshOther, parent);
  disposeInst(ctx.meshSeeds, parent);
  ctx.lineBg = ctx.lineBoth = ctx.lineOne = ctx.linePath = ctx.meshOther = ctx.meshSeeds = null;
  if (!buf) {
    syncBallSeedLabels(ctx, null, showSeedLabels);
    hoverEdgeRef.current.enabled = altNeighborHoverRef.current;
    refreshHoverNeighborEdges3d(ctx, null, hoverEdgeRef.current);
    return;
  }

  const mkLine = (positions: Float32Array, nFloats: number, color: number, opacity: number) => {
    if (nFloats < 6) return null;
    const dst = new Float32Array(nFloats);
    fillRimGammaPositions(positions, nFloats, gamma, dst);
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(dst, 3));
    const mat = new LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
    });
    const ln = new LineSegments(g, mat);
    parent.add(ln);
    return ln;
  };

  if (buf.nLineBgVerts > 0) {
    ctx.lineBg = mkLine(buf.lineBgPositions, buf.nLineBgVerts, 0xffc8a0, 0.12);
  }
  if (buf.nLineBothVerts > 0) {
    const n = buf.nLineBothVerts;
    if (n >= 6) {
      const dst = new Float32Array(n);
      fillRimGammaPositions(buf.lineBothPositions, n, gamma, dst);
      const geom = new LineSegmentsGeometry();
      geom.setPositions(dst);
      const mat = new Line2NodeMaterial({
        color: 0xff6644,
        linewidth: BOTH_SEED_LINEWIDTH_3D,
        transparent: true,
        opacity: 0.55,
        depthTest: true,
        depthWrite: true,
        toneMapped: false,
      });
      const ln = new LineSegments2(geom, mat);
      ln.renderOrder = 12;
      parent.add(ln);
      ctx.lineBoth = ln;
    }
  }
  if (buf.nLineOneVerts > 0) {
    const n = buf.nLineOneVerts;
    if (n >= 6) {
      const dst = new Float32Array(n);
      fillRimGammaPositions(buf.lineOnePositions, n, gamma, dst);
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(dst, 3));
      const blueOp = Math.max(0.02, Math.min(1, sizing.edgeOpacity));
      const mat = new LineBasicMaterial({
        color: 0x6699ff,
        transparent: true,
        opacity: blueOp,
        blending: AdditiveBlending,
        depthWrite: false,
      });
      const ln = new LineSegments(g, mat);
      parent.add(ln);
      ctx.lineOne = ln;
    }
  }

  const camDist = orbitRef.current?.distance ?? CAM_REF_DIST;
  const zm = nodeZoomMultiplier(camDist, sizing.compensateZoomNodes);

  const seedGeom = new SphereGeometry(1, 14, 12);
  const seedMat = new MeshBasicMaterial({ color: 0xff4466 });
  if (buf.nPointsSeed > 0) {
    const mesh = new InstancedMesh(seedGeom, seedMat, buf.nPointsSeed);
    for (let i = 0; i < buf.nPointsSeed; i++) {
      const ix = i * 3;
      const x = buf.pointsSeed[ix]!;
      const y = buf.pointsSeed[ix + 1]!;
      const z = buf.pointsSeed[ix + 2]!;
      const r = Math.hypot(x, y, z);
      const rad = sizing.centerWeighted
        ? clampedRadialScale(r, sizing.radialMin, sizing.radialMax, sizing.nodeMinMul)
        : 1;
      const s = SEED_BALL_BASE * sizing.nodeSizeMul * zm * rad;
      _v3.set(x, y, z);
      rimBallMapInPlace(_v3, gamma);
      _m4.compose(_v3, _qId, _vScale.set(s, s, s));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    parent.add(mesh);
    ctx.meshSeeds = mesh;
  } else seedGeom.dispose();

  const otherGeom = new SphereGeometry(1, 12, 10);
  const otherMat = new MeshBasicMaterial({
    color: 0x55cc77,
    transparent: true,
    opacity: 0.4,
    depthWrite: false,
  });
  if (buf.nPointsOther > 0) {
    const mesh = new InstancedMesh(otherGeom, otherMat, buf.nPointsOther);
    for (let i = 0; i < buf.nPointsOther; i++) {
      const ix = i * 3;
      const x = buf.pointsOther[ix]!;
      const y = buf.pointsOther[ix + 1]!;
      const z = buf.pointsOther[ix + 2]!;
      const r = Math.hypot(x, y, z);
      const rad = sizing.centerWeighted
        ? clampedRadialScale(r, sizing.radialMin, sizing.radialMax, sizing.nodeMinMul)
        : 1;
      const s = OTHER_BALL_BASE * OTHER_BALL_VISUAL_SCALE * sizing.nodeSizeMul * zm * rad;
      _v3.set(x, y, z);
      rimBallMapInPlace(_v3, gamma);
      _m4.compose(_v3, _qId, _vScale.set(s, s, s));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    parent.add(mesh);
    ctx.meshOther = mesh;
  } else otherGeom.dispose();

  const ov = pathOverlay;
  if (ov && ov.nFloats > 0 && ov.positions.length >= ov.nFloats) {
    const pathDst = new Float32Array(ov.nFloats);
    fillRimGammaPositions(ov.positions, ov.nFloats, gamma, pathDst);
    const geomPath = new LineSegmentsGeometry();
    geomPath.setPositions(pathDst);
    const om = Math.max(0, Math.min(1, pathOverlayOpacityMult));
    const mat = new Line2NodeMaterial({
      color: 0xff9922,
      linewidth: PATH_OVERLAY_LINEWIDTH_3D,
      transparent: true,
      opacity: PATH_OVERLAY_OPACITY_BASE * om,
      depthTest: true,
      depthWrite: false,
      toneMapped: false,
    });
    const ln = new LineSegments2(geomPath, mat);
    ln.renderOrder = 6;
    parent.add(ln);
    ctx.linePath = ln;
  }

  syncBallSeedLabels(ctx, buf, showSeedLabels);

  orbitToPosition(orbitRef.current, camera.position);
  camera.lookAt(0, 0, 0);
  hoverEdgeRef.current.enabled = altNeighborHoverRef.current;
  refreshHoverNeighborEdges3d(ctx, buf, hoverEdgeRef.current);
}

/** Recompute instance scales only (orbit distance / sizing); keeps line meshes. */
function refreshBallNodeScales(
  ctx: BallCtx,
  buf: SceneBuffers3d,
  orbitRef: RefObject<OrbitState>,
  sizing: BallDisplaySizing,
): void {
  const camDist = orbitRef.current?.distance ?? CAM_REF_DIST;
  const zm = nodeZoomMultiplier(camDist, sizing.compensateZoomNodes);

  const gamma = ctx.rimGammaRef.current ?? 1;
  if (ctx.meshSeeds && buf.nPointsSeed > 0) {
    const mesh = ctx.meshSeeds;
    for (let i = 0; i < buf.nPointsSeed; i++) {
      const ix = i * 3;
      const x = buf.pointsSeed[ix]!;
      const y = buf.pointsSeed[ix + 1]!;
      const z = buf.pointsSeed[ix + 2]!;
      const r = Math.hypot(x, y, z);
      const rad = sizing.centerWeighted
        ? clampedRadialScale(r, sizing.radialMin, sizing.radialMax, sizing.nodeMinMul)
        : 1;
      const s = SEED_BALL_BASE * sizing.nodeSizeMul * zm * rad;
      _v3.set(x, y, z);
      rimBallMapInPlace(_v3, gamma);
      _m4.compose(_v3, _qId, _vScale.set(s, s, s));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }
  if (ctx.meshOther && buf.nPointsOther > 0) {
    const mesh = ctx.meshOther;
    for (let i = 0; i < buf.nPointsOther; i++) {
      const ix = i * 3;
      const x = buf.pointsOther[ix]!;
      const y = buf.pointsOther[ix + 1]!;
      const z = buf.pointsOther[ix + 2]!;
      const r = Math.hypot(x, y, z);
      const rad = sizing.centerWeighted
        ? clampedRadialScale(r, sizing.radialMin, sizing.radialMax, sizing.nodeMinMul)
        : 1;
      const s = OTHER_BALL_BASE * OTHER_BALL_VISUAL_SCALE * sizing.nodeSizeMul * zm * rad;
      _v3.set(x, y, z);
      rimBallMapInPlace(_v3, gamma);
      _m4.compose(_v3, _qId, _vScale.set(s, s, s));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }
}

function pickGraphIndexAtEvent(
  e: PointerEvent,
  canvas: HTMLCanvasElement,
  camera: InstanceType<typeof PerspectiveCamera>,
  buf: SceneBuffers3d,
  ctx: BallCtx,
): number | null {
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  ctx.ndcPointer.x = ((e.clientX - rect.left) / w) * 2 - 1;
  ctx.ndcPointer.y = -((e.clientY - rect.top) / h) * 2 + 1;
  ctx.raycaster.setFromCamera(ctx.ndcPointer, camera);
  const objs: InstanceType<typeof InstancedMesh>[] = [];
  if (ctx.meshSeeds) objs.push(ctx.meshSeeds);
  if (ctx.meshOther) objs.push(ctx.meshOther);
  if (!objs.length) return null;
  const hits = ctx.raycaster.intersectObjects(objs, false);
  const hit = hits[0];
  if (hit?.instanceId !== undefined) {
    const mesh = hit.object as InstanceType<typeof InstancedMesh>;
    const iid = hit.instanceId;
    if (mesh === ctx.meshSeeds && iid >= 0 && iid < buf.seedGraphIndex.length) return buf.seedGraphIndex[iid]!;
    if (mesh === ctx.meshOther && iid >= 0 && iid < buf.otherGraphIndex.length) return buf.otherGraphIndex[iid]!;
  }
  /* Proximity pick in screen space */
  let bestD2 = Infinity;
  let bestGid: number | null = null;
  const pushCand = (gx: number, gy: number, gz: number, gid: number) => {
    setRimDisplayWorld(gx, gy, gz, ctx, _v3);
    _v3.project(camera);
    const sx = (0.5 * _v3.x + 0.5) * w + rect.left;
    const sy = (-0.5 * _v3.y + 0.5) * h + rect.top;
    const d2 = (sx - e.clientX) ** 2 + (sy - e.clientY) ** 2;
    const rPx = 22;
    if (d2 < rPx * rPx && d2 < bestD2) {
      bestD2 = d2;
      bestGid = gid;
    }
  };
  for (let i = 0; i < buf.nPointsSeed; i++) {
    const ix = i * 3;
    pushCand(buf.pointsSeed[ix]!, buf.pointsSeed[ix + 1]!, buf.pointsSeed[ix + 2]!, buf.seedGraphIndex[i]!);
  }
  for (let i = 0; i < buf.nPointsOther; i++) {
    const ix = i * 3;
    pushCand(buf.pointsOther[ix]!, buf.pointsOther[ix + 1]!, buf.pointsOther[ix + 2]!, buf.otherGraphIndex[i]!);
  }
  return bestGid;
}

export type BallView3dProps = {
  scene: SceneBuffers3d | null;
  pathOverlay: PathOverlayBuffer | null;
  webGpuError: string | null;
  showSeedLabels: boolean;
  nodeInteractionRef: RefObject<BallView3dNodeInteraction | null>;
  centerWeightedSizes: boolean;
  radialScaleMin: number;
  radialScaleMax: number;
  nodeSizeMul: number;
  compensateZoomNodes: boolean;
  nodeMinMul: number;
  edgeOpacity: number;
  /** CSR + ball positions + focus; white neighbor edges when hovering + Alt. */
  hoverNeighborGraph?: HoverNeighborGraph3d | null;
};

export const BallView3d = forwardRef<BallView3dHandle, BallView3dProps>(function BallView3d(
  {
    scene,
    pathOverlay,
    webGpuError,
    showSeedLabels,
    nodeInteractionRef,
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
    edgeOpacity,
    hoverNeighborGraph = null,
  }: BallView3dProps,
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<BallCtx | null>(null);
  const sceneRef = useRef(scene);
  const orbitRef = useRef<OrbitState>(defaultOrbit());
  const pathOverlayRef = useRef<PathOverlayBuffer | null>(null);
  const pathOverlayOpacityMultRef = useRef(1);
  const showLabelsRef = useRef(showSeedLabels);
  const dragRef = useRef({ active: false, downX: 0, downY: 0, lastX: 0, lastY: 0 });
  const ballSizingRef = useRef<BallDisplaySizing>({
    centerWeighted: centerWeightedSizes,
    radialMin: radialScaleMin,
    radialMax: radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
    edgeOpacity,
  });
  /** Rim power map γ for ball (same role as disk `rimPreservingGamma`). */
  const ballRimGammaRef = useRef(1);
  const altNeighborHoverRef = useRef(false);
  const ballHoverEdgeRef = useRef<BallHoverEdgeState>({
    enabled: false,
    graph: hoverNeighborGraph,
    gid: -1,
  });

  useLayoutEffect(() => {
    sceneRef.current = scene;
  }, [scene]);

  useLayoutEffect(() => {
    pathOverlayRef.current = pathOverlay ?? null;
  }, [pathOverlay]);

  useLayoutEffect(() => {
    showLabelsRef.current = showSeedLabels;
  }, [showSeedLabels]);

  useLayoutEffect(() => {
    ballHoverEdgeRef.current.graph = hoverNeighborGraph;
  }, [hoverNeighborGraph]);

  useLayoutEffect(() => {
    ballSizingRef.current = {
      centerWeighted: centerWeightedSizes,
      radialMin: radialScaleMin,
      radialMax: radialScaleMax,
      nodeSizeMul,
      compensateZoomNodes,
      nodeMinMul,
      edgeOpacity,
    };
  }, [
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
    edgeOpacity,
  ]);

  useImperativeHandle(ref, () => ({
    resetView() {
      orbitRef.current = defaultOrbit();
      ballRimGammaRef.current = 1;
      const c = ctxRef.current;
      const buf = sceneRef.current;
      if (c) {
        c.contentRoot.quaternion.identity();
        orbitToPosition(orbitRef.current, c.camera.position);
        c.camera.lookAt(0, 0, 0);
        if (buf) {
          applySceneToBall(
            c,
            buf,
            orbitRef,
            c.camera,
            ballSizingRef.current,
            pathOverlayRef.current,
            pathOverlayOpacityMultRef.current,
            showLabelsRef.current,
            ballHoverEdgeRef,
            altNeighborHoverRef,
          );
        }
      }
    },
    applySceneBuffers(buf: SceneBuffers3d | null) {
      sceneRef.current = buf;
      const c = ctxRef.current;
      if (!c) return;
      applySceneToBall(
        c,
        buf,
        orbitRef,
        c.camera,
        ballSizingRef.current,
        pathOverlayRef.current,
        pathOverlayOpacityMultRef.current,
        showLabelsRef.current,
        ballHoverEdgeRef,
        altNeighborHoverRef,
      );
    },
    setPathOverlayOpacityMultiplier(mult: number) {
      pathOverlayOpacityMultRef.current = Math.max(0, Math.min(1, mult));
      const c = ctxRef.current;
      const buf = sceneRef.current;
      if (!c) return;
      applySceneToBall(
        c,
        buf,
        orbitRef,
        c.camera,
        ballSizingRef.current,
        pathOverlayRef.current,
        pathOverlayOpacityMultRef.current,
        showLabelsRef.current,
        ballHoverEdgeRef,
        altNeighborHoverRef,
      );
    },
    fitSubgraphToView() {
      const buf = sceneRef.current;
      const c = ctxRef.current;
      if (!buf || !c || (buf.nPointsSeed === 0 && buf.nPointsOther === 0)) return;
      let maxR = 0.05;
      const acc = (p: Float32Array) => {
        for (let i = 0; i < p.length; i += 3) {
          const r = Math.hypot(p[i]!, p[i + 1]!, p[i + 2]!);
          maxR = Math.max(maxR, r);
        }
      };
      acc(buf.pointsSeed);
      acc(buf.pointsOther);
      const fov = (c.camera.fov * Math.PI) / 180;
      const dist = Math.max(1.2, (maxR * 1.35) / Math.tan(fov / 2));
      orbitRef.current.distance = dist;
      orbitToPosition(orbitRef.current, c.camera.position);
      c.camera.lookAt(0, 0, 0);
      if (buf) refreshBallNodeScales(c, buf, orbitRef, ballSizingRef.current);
    },
  }));

  useEffect(() => {
    const host = hostRef.current;
    if (!host || webGpuError) return;

    let disposed = false;
    const renderer = new WebGPURenderer({ antialias: true, alpha: true });
    const scene3 = new Scene();
    scene3.background = new Color(0x0e0e10);
    const camera = new PerspectiveCamera(50, 1, 0.08, 80);
    camera.position.set(2.2, 1.4, 2.4);
    camera.lookAt(0, 0, 0);

    const ballGeom = new SphereGeometry(1, BALL_WIRE_SEGMENTS, BALL_WIRE_SEGMENTS / 2);
    const edges = new EdgesGeometry(ballGeom);
    const ballMat = new LineBasicMaterial({
      color: 0x5a5a68,
      transparent: true,
      opacity: 0.175,
      depthWrite: false,
    });
    const ballWire = new LineSegments(edges, ballMat);
    scene3.add(ballWire);
    ballGeom.dispose();

    const contentRoot = new Group();
    scene3.add(contentRoot);

    const labelsLayer = document.createElement("div");
    labelsLayer.className = "diskSeedLabels";
    labelsLayer.setAttribute("aria-hidden", "true");

    const nodeTipEl = document.createElement("div");
    nodeTipEl.className = "diskNodeTooltip";
    nodeTipEl.style.display = "none";

    const raycaster = new Raycaster();
    const ndcPointer = new Vector2();

    const ctx: BallCtx = {
      renderer,
      scene3,
      camera,
      host,
      lineBg: null,
      lineBoth: null,
      lineOne: null,
      meshOther: null,
      meshSeeds: null,
      linePath: null,
      lineHoverNbr: null,
      ballWire,
      contentRoot,
      rimGammaRef: ballRimGammaRef,
      nodeTipEl,
      raycaster,
      ndcPointer,
      labelsLayer,
      labelSpans: [],
      raf: 0,
    };
    ctxRef.current = ctx;

    let shiftHeld = false;
    const syncShiftFromEvent = (e: PointerEvent | KeyboardEvent | WheelEvent) => {
      if (typeof e.getModifierState === "function") {
        shiftHeld = e.getModifierState("Shift");
      } else {
        shiftHeld = "shiftKey" in e && e.shiftKey === true;
      }
    };
    const syncAltFromEvent = (e: PointerEvent | KeyboardEvent | WheelEvent) => {
      if (typeof e.getModifierState === "function") {
        /* Alt + AltGraph: left/right Alt, AltGr layouts, and macOS Option (maps to Alt in browsers). */
        altNeighborHoverRef.current =
          e.getModifierState("Alt") || e.getModifierState("AltGraph");
      } else {
        const alt = "altKey" in e && e.altKey === true;
        const ag =
          "altGraphKey" in e &&
          (e as KeyboardEvent & { altGraphKey?: boolean }).altGraphKey === true;
        altNeighborHoverRef.current = alt || ag;
      }
    };
    const bumpBallHoverEdges = () => {
      ballHoverEdgeRef.current.enabled = altNeighborHoverRef.current;
      refreshHoverNeighborEdges3d(ctx, sceneRef.current, ballHoverEdgeRef.current);
    };
    const onWindowKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Shift") shiftHeld = true;
      syncAltFromEvent(e);
      bumpBallHoverEdges();
    };
    const onWindowKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") shiftHeld = false;
      syncAltFromEvent(e);
      bumpBallHoverEdges();
    };
    const onWindowBlur = () => {
      shiftHeld = false;
      altNeighborHoverRef.current = false;
      bumpBallHoverEdges();
    };
    window.addEventListener("keydown", onWindowKeyDown);
    window.addEventListener("keyup", onWindowKeyUp);
    window.addEventListener("blur", onWindowBlur);

    const hideNodeTip = () => {
      nodeTipEl.style.display = "none";
      nodeTipEl.replaceChildren();
    };
    const showNodeTip = (e: PointerEvent, tip: NodeDiskHoverTooltip | null) => {
      if (!tip?.name) {
        hideNodeTip();
        return;
      }
      nodeTipEl.replaceChildren();
      const head = document.createElement("div");
      head.className = "diskNodeTooltipHead";
      const nameEl = document.createElement("span");
      nameEl.className = "diskNodeTooltipName";
      nameEl.textContent = tip.name;
      const degEl = document.createElement("span");
      degEl.className = "diskNodeTooltipDeg";
      degEl.textContent = tip.degree === "—" ? "deg —" : `deg ${tip.degree}`;
      head.append(nameEl, degEl);
      const body = document.createElement("div");
      body.className = "diskNodeTooltipBody";
      body.textContent = tip.details;
      nodeTipEl.append(head, body);
      nodeTipEl.style.display = "block";
      const pad = 14;
      const rect = nodeTipEl.getBoundingClientRect();
      const tw = Math.ceil(rect.width);
      const th = Math.ceil(rect.height);
      nodeTipEl.style.left = `${Math.min(e.clientX + pad, Math.max(0, window.innerWidth - tw - 4))}px`;
      nodeTipEl.style.top = `${Math.min(e.clientY + pad, Math.max(0, window.innerHeight - th - 4))}px`;
    };

    const resize = () => {
      const w = Math.max(1, Math.floor(host.clientWidth));
      const h = Math.max(1, Math.floor(host.clientHeight || host.clientWidth));
      renderer.setSize(w, h, true);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      syncShiftFromEvent(e);
      syncAltFromEvent(e);
      const shiftWheel =
        shiftHeld ||
        e.shiftKey ||
        (typeof e.getModifierState === "function" && e.getModifierState("Shift"));
      const buf = sceneRef.current;
      if (shiftWheel) {
        const g = ballRimGammaRef.current;
        ballRimGammaRef.current = Math.min(
          VIEWER_RIM_GAMMA_MAX,
          Math.max(VIEWER_RIM_GAMMA_MIN, g * Math.exp(-e.deltaY * VIEWER_RIM_WHEEL_SENS)),
        );
        if (buf) {
          applySceneToBall(
            ctx,
            buf,
            orbitRef,
            camera,
            ballSizingRef.current,
            pathOverlayRef.current,
            pathOverlayOpacityMultRef.current,
            showLabelsRef.current,
            ballHoverEdgeRef,
            altNeighborHoverRef,
          );
        }
        return;
      }
      const o = orbitRef.current;
      o.distance = Math.min(24, Math.max(0.9, o.distance * Math.exp(e.deltaY * 0.0012)));
      orbitToPosition(o, camera.position);
      camera.lookAt(0, 0, 0);
      if (buf && ballSizingRef.current.compensateZoomNodes) {
        refreshBallNodeScales(ctx, buf, orbitRef, ballSizingRef.current);
      }
      bumpBallHoverEdges();
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      syncShiftFromEvent(e);
      syncAltFromEvent(e);
      dragRef.current = {
        active: true,
        downX: e.clientX,
        downY: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
      };
      hideNodeTip();
      ballHoverEdgeRef.current.gid = -1;
      bumpBallHoverEdges();
      (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      syncShiftFromEvent(e);
      syncAltFromEvent(e);
      const buf = sceneRef.current;
      const nip = nodeInteractionRef?.current;
      if (dragRef.current.active) {
        const dx = e.clientX - dragRef.current.lastX;
        const dy = e.clientY - dragRef.current.lastY;
        dragRef.current.lastX = e.clientX;
        dragRef.current.lastY = e.clientY;
        if (shiftHeld) {
          camera.updateMatrixWorld();
          _dragAxisRight.set(1, 0, 0).transformDirection(camera.matrixWorld);
          _dragAxisUp.set(0, 1, 0).transformDirection(camera.matrixWorld);
          _qDragH.setFromAxisAngle(_dragAxisUp, -dx * SHIFT_BALL_DRAG_ROT);
          _qDragV.setFromAxisAngle(_dragAxisRight, -dy * SHIFT_BALL_DRAG_ROT);
          _qDragDelta.multiplyQuaternions(_qDragH, _qDragV);
          contentRoot.quaternion.premultiply(_qDragDelta);
        } else {
          const o = orbitRef.current;
          o.yaw += dx * 0.0055;
          o.pitch = Math.max(-1.45, Math.min(1.45, o.pitch + dy * 0.0055));
          orbitToPosition(o, camera.position);
          camera.lookAt(0, 0, 0);
          if (buf && ballSizingRef.current.compensateZoomNodes) {
            refreshBallNodeScales(ctx, buf, orbitRef, ballSizingRef.current);
          }
        }
      } else if (nip && buf) {
        const gid = pickGraphIndexAtEvent(e, renderer.domElement, camera, buf, ctx);
        ballHoverEdgeRef.current.gid = gid ?? -1;
        if (gid === null) hideNodeTip();
        else showNodeTip(e, nip.tooltipForGraphIndex(gid));
        bumpBallHoverEdges();
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      syncShiftFromEvent(e);
      syncAltFromEvent(e);
      const wasDrag =
        Math.hypot(e.clientX - dragRef.current.downX, e.clientY - dragRef.current.downY) > 5;
      dragRef.current.active = false;
      try {
        (e.currentTarget as HTMLCanvasElement).releasePointerCapture(e.pointerId);
      } catch {
        /* */
      }
      const buf = sceneRef.current;
      const nip = nodeInteractionRef?.current;
      let skipHoverRefresh = false;
      if (!wasDrag && nip && buf) {
        const shiftPick =
          shiftHeld ||
          e.shiftKey ||
          (typeof e.getModifierState === "function" && e.getModifierState("Shift"));
        const gid = pickGraphIndexAtEvent(e, renderer.domElement, camera, buf, ctx);
        if (gid !== null) {
          if (shiftPick && nip.shiftPickGraphIndex) nip.shiftPickGraphIndex(gid);
          else if (!shiftPick) {
            nip.pickGraphIndex(gid);
            /* Focus pick: do not leave white hover-neighbor preview until next move. */
            ballHoverEdgeRef.current.gid = -1;
            bumpBallHoverEdges();
            hideNodeTip();
            skipHoverRefresh = true;
          }
        }
      }
      if (!skipHoverRefresh && nip && buf) {
        const gid = pickGraphIndexAtEvent(e, renderer.domElement, camera, buf, ctx);
        ballHoverEdgeRef.current.gid = gid ?? -1;
        if (gid === null) hideNodeTip();
        else showNodeTip(e, nip.tooltipForGraphIndex(gid));
        bumpBallHoverEdges();
      } else if (!skipHoverRefresh) hideNodeTip();
    };

    const onPointerLeave = () => {
      hideNodeTip();
      ballHoverEdgeRef.current.gid = -1;
      bumpBallHoverEdges();
    };

    const loop = () => {
      ctx.raf = requestAnimationFrame(loop);
      renderer.render(scene3, camera);
      updateBallSeedLabelPositions(ctx, sceneRef.current, showLabelsRef.current);
    };

    void (async () => {
      try {
        await renderer.init();
        if (disposed) return;
        host.appendChild(renderer.domElement);
        host.appendChild(labelsLayer);
        host.appendChild(nodeTipEl);
        renderer.domElement.style.display = "block";
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        resize();
        requestAnimationFrame(() => {
          if (!disposed) resize();
        });
        const el = renderer.domElement;
        el.addEventListener("wheel", onWheel, { passive: false });
        el.addEventListener("pointerdown", onPointerDown);
        el.addEventListener("pointermove", onPointerMove);
        el.addEventListener("pointerup", onPointerUp);
        el.addEventListener("pointercancel", onPointerUp);
        el.addEventListener("pointerleave", onPointerLeave);
        applySceneToBall(
          ctx,
          sceneRef.current,
          orbitRef,
          camera,
          ballSizingRef.current,
          pathOverlayRef.current,
          pathOverlayOpacityMultRef.current,
          showLabelsRef.current,
          ballHoverEdgeRef,
          altNeighborHoverRef,
        );
        loop();
      } catch (err) {
        console.error(err);
      }
    })();

    const ro = new ResizeObserver(resize);
    ro.observe(host);
    return () => {
      disposed = true;
      cancelAnimationFrame(ctx.raf);
      ro.disconnect();
      window.removeEventListener("keydown", onWindowKeyDown);
      window.removeEventListener("keyup", onWindowKeyUp);
      window.removeEventListener("blur", onWindowBlur);
      const el = ctx.renderer.domElement;
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
      el.removeEventListener("pointerleave", onPointerLeave);
      const cr = ctx.contentRoot;
      disposeLine(ctx.lineBg, cr);
      disposeWideLineSegments2(ctx.lineBoth, cr);
      disposeLine(ctx.lineOne, cr);
      disposeWideLineSegments2(ctx.linePath, cr);
      disposeWideLineSegments2(ctx.lineHoverNbr, cr);
      disposeInst(ctx.meshOther, cr);
      disposeInst(ctx.meshSeeds, cr);
      scene3.remove(cr);
      if (ctx.ballWire) {
        scene3.remove(ctx.ballWire);
        ctx.ballWire.geometry.dispose();
        (ctx.ballWire.material as InstanceType<typeof LineBasicMaterial>).dispose();
      }
      ctxRef.current = null;
      renderer.dispose();
      if (host.contains(el)) host.removeChild(el);
      if (labelsLayer.parentElement === host) host.removeChild(labelsLayer);
      if (nodeTipEl.parentElement === host) host.removeChild(nodeTipEl);
    };
  }, [webGpuError]);

  useEffect(() => {
    const c = ctxRef.current;
    if (!c || webGpuError) return;
    applySceneToBall(
      c,
      scene,
      orbitRef,
      c.camera,
      ballSizingRef.current,
      pathOverlayRef.current,
      pathOverlayOpacityMultRef.current,
      showSeedLabels,
      ballHoverEdgeRef,
      altNeighborHoverRef,
    );
  }, [
    scene,
    pathOverlay,
    webGpuError,
    showSeedLabels,
    hoverNeighborGraph,
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
    edgeOpacity,
  ]);

  const hostStyle = {
    position: "absolute" as const,
    inset: 0,
    background: "#0e0e10",
  };

  if (webGpuError) {
    return (
      <div ref={hostRef} style={hostStyle}>
        <div className="webgpuErr">{webGpuError}</div>
      </div>
    );
  }

  return <div ref={hostRef} style={hostStyle} />;
});
