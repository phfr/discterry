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
  OrthographicCamera,
  LineSegments,
  Mesh,
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  Line2NodeMaterial,
  Color,
  AdditiveBlending,
  InstancedMesh,
  MeshBasicMaterial,
  CircleGeometry,
  Matrix4,
  Vector3,
  Vector2,
  Quaternion,
  Raycaster,
} from "three/webgpu";
import { LineSegments2 } from "three/addons/lines/webgpu/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { SceneBuffers } from "../model/computeScene";
import {
  applyMobiusToW,
  identityMobius,
  incrementMobiusFromDrag,
  isIdentityMobius,
  type Mobius2,
} from "../math/viewerMobius";

const BASE_HALF_EXTENT = 1.06;
const ZOOM_MIN = 0.45;
/** Max zoom-in (ortho half-extent = 1.06 / zoom); 16 = 2× previous cap of 8. */
const ZOOM_MAX = 16;
const WHEEL_ZOOM_SENS = 0.0018;
/** Shift+wheel: radial display map w ↦ r^{γ-1} w (|w|=1 fixed; γ<1 spreads interior toward rim). */
const RIM_WHEEL_SENS = 0.0014;
const RIM_GAMMA_MIN = 0.38;
const RIM_GAMMA_MAX = 2.4;
/** When zoom node compensation is on: blend halfway from 1 to full 1/zoom (screen-stable radius). */
const ZOOM_NODE_COMP_STRENGTH = 0.5;

export type DiskViewTransform = {
  /** 1 = default framing; larger zooms in (smaller ortho half-extent). */
  zoom: number;
  panX: number;
  panY: number;
  mobius: Mobius2;
  /**
   * Radial-only viewer warp after Möbius: (x,y) ↦ r^{γ-1}(x,y), r = hypot(x,y). γ=1 off.
   * |w|=1 fixed; γ<1 pushes interior toward rim (wide “FOV”); γ>1 toward center.
   */
  rimPreservingGamma: number;
};

export type DiskViewHandle = {
  resetView: () => void;
  /** Clear Euclidean pan and viewer Möbius; keep zoom (call when data focus `z0` changes so w=0 stays centered). */
  recenterPreservingZoom: () => void;
  /** Zoom and pan so current scene node markers (seed + other) fit in view; uses viewer Möbius/rim as today. */
  fitSubgraphToView: () => void;
  /** Push GPU buffers without waiting for React `scene` prop (focus animation frames). */
  applySceneBuffers: (buf: SceneBuffers | null) => void;
};

/** Filled by App: map graph index → list tooltip text; pick sets focus by vertex index. */
export type DiskViewNodeInteraction = {
  tooltipForGraphIndex: (graphIndex: number) => string;
  pickGraphIndex: (graphIndex: number) => void;
};

function defaultDiskViewTransform(): DiskViewTransform {
  return { zoom: 1, panX: 0, panY: 0, mobius: identityMobius(), rimPreservingGamma: 1 };
}

function updateOrthographicCamera(
  camera: InstanceType<typeof OrthographicCamera>,
  v: DiskViewTransform,
): void {
  const half = BASE_HALF_EXTENT / v.zoom;
  camera.left = v.panX - half;
  camera.right = v.panX + half;
  camera.top = v.panY + half;
  camera.bottom = v.panY - half;
  camera.updateProjectionMatrix();
}

/** Map (wx, wy) through viewer Möbius when non-identity. */
function mapWxy(wre: number, wim: number, M: Mobius2): [number, number] {
  if (isIdentityMobius(M)) return [wre, wim];
  const o = applyMobiusToW(wre, wim, M);
  return [o.re, o.im];
}

/** After Möbius: radial power map fixing the rim circle (r=1) pointwise. */
function rimRadialExponent(wx: number, wy: number, gamma: number): [number, number] {
  if (Math.abs(gamma - 1) < 1e-9) return [wx, wy];
  const r = Math.hypot(wx, wy);
  if (r < 1e-15) return [0, 0];
  const s = Math.pow(r, gamma - 1);
  return [wx * s, wy * s];
}

function mapViewerWxy(wre: number, wim: number, v: DiskViewTransform): [number, number] {
  const [a, b] = mapWxy(wre, wim, v.mobius);
  return rimRadialExponent(a, b, v.rimPreservingGamma);
}

function transformLinePositions(src: Float32Array, v: DiskViewTransform): Float32Array {
  if (isIdentityMobius(v.mobius) && Math.abs(v.rimPreservingGamma - 1) < 1e-9) return src;
  const n = src.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 3) {
    const [x, y] = mapViewerWxy(src[i], src[i + 1], v);
    out[i] = x;
    out[i + 1] = y;
    out[i + 2] = src[i + 2];
  }
  return out;
}

/** Batched display params read inside `applyBuffers` (ref kept fresh for async WebGPU init). */
export type DiskDisplaySizing = {
  centerWeighted: boolean;
  radialMin: number;
  radialMax: number;
  nodeSizeMul: number;
  /** If true, shrink world-space node markers partially with zoom (see ZOOM_NODE_COMP_STRENGTH). */
  compensateZoomNodes: boolean;
  /** Floor on zoom scale and on radial instance scale (rim); keeps nodes from vanishing. */
  nodeMinMul: number;
};

type Props = {
  scene: SceneBuffers | null;
  webGpuError: string | null;
  showSeedLabels: boolean;
  showCrosshair: boolean;
  centerWeightedSizes: boolean;
  radialScaleMin: number;
  radialScaleMax: number;
  nodeSizeMul: number;
  compensateZoomNodes: boolean;
  nodeMinMul: number;
  nodeInteractionRef?: RefObject<DiskViewNodeInteraction | null>;
};

type ThreeCtx = {
  renderer: InstanceType<typeof WebGPURenderer>;
  scene3: InstanceType<typeof Scene>;
  camera: InstanceType<typeof OrthographicCamera>;
  host: HTMLDivElement;
  lineBoth: InstanceType<typeof Mesh> | null;
  lineOne: InstanceType<typeof LineSegments> | null;
  meshOther: InstanceType<typeof InstancedMesh> | null;
  meshSeeds: InstanceType<typeof InstancedMesh> | null;
  labelsLayer: HTMLDivElement;
  labelSpans: HTMLSpanElement[];
  labelProj: InstanceType<typeof Vector3>;
  crosshairH: InstanceType<typeof LineSegments>;
  crosshairV: InstanceType<typeof LineSegments>;
  raf: number;
  nodeTipEl: HTMLDivElement;
  raycaster: InstanceType<typeof Raycaster>;
  ndcPointer: InstanceType<typeof Vector2>;
};

const CLICK_DRAG_PX = 6;
/** Extra slack (screen px) for hover/click hit-testing when disk sprites are tiny. */
const NODE_HOVER_PAD_PX = 3;

const _pickProj = new Vector3();

function pickGraphIndexByScreenProximity(
  e: PointerEvent,
  camera: InstanceType<typeof OrthographicCamera>,
  buf: SceneBuffers,
  ctx: ThreeCtx,
  w: number,
  h: number,
  rect: DOMRect,
): number | null {
  const pad2 = NODE_HOVER_PAD_PX * NODE_HOVER_PAD_PX;
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  let bestD2 = Infinity;
  let bestG: number | null = null;

  const considerMesh = (mesh: InstanceType<typeof InstancedMesh> | null, gidAt: (i: number) => number | null) => {
    if (!mesh) return;
    const n = mesh.count;
    for (let i = 0; i < n; i++) {
      mesh.getMatrixAt(i, _m4);
      _m4.decompose(_vPos, _qId, _vScale);
      _pickProj.copy(_vPos).project(camera);
      const sx = (_pickProj.x * 0.5 + 0.5) * w;
      const sy = (-_pickProj.y * 0.5 + 0.5) * h;
      const dx = sx - px;
      const dy = sy - py;
      const d2 = dx * dx + dy * dy;
      if (d2 <= pad2 && d2 < bestD2) {
        const g = gidAt(i);
        if (g !== null) {
          bestD2 = d2;
          bestG = g;
        }
      }
    }
  };

  /* Seeds checked first so equal-distance ties favor the top layer. */
  considerMesh(ctx.meshSeeds, (i) =>
    i >= 0 && i < buf.seedGraphIndex.length ? buf.seedGraphIndex[i]! : null,
  );
  considerMesh(ctx.meshOther, (i) =>
    i >= 0 && i < buf.otherGraphIndex.length ? buf.otherGraphIndex[i]! : null,
  );
  return bestG;
}

function pickGraphIndexAtPointerEvent(
  e: PointerEvent,
  canvas: HTMLCanvasElement,
  camera: InstanceType<typeof OrthographicCamera>,
  buf: SceneBuffers | null,
  ctx: ThreeCtx,
  ndc: InstanceType<typeof Vector2>,
  raycaster: InstanceType<typeof Raycaster>,
): number | null {
  if (!buf || (!ctx.meshOther && !ctx.meshSeeds)) return null;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  ndc.x = ((e.clientX - rect.left) / w) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / h) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  const objs: InstanceType<typeof InstancedMesh>[] = [];
  if (ctx.meshSeeds) objs.push(ctx.meshSeeds);
  if (ctx.meshOther) objs.push(ctx.meshOther);
  if (!objs.length) return null;
  const hits = raycaster.intersectObjects(objs, false);
  const hit = hits[0];
  if (hit?.instanceId !== undefined) {
    const mesh = hit.object as InstanceType<typeof InstancedMesh>;
    const iid = hit.instanceId;
    if (mesh === ctx.meshSeeds && iid >= 0 && iid < buf.seedGraphIndex.length) return buf.seedGraphIndex[iid]!;
    if (mesh === ctx.meshOther && iid >= 0 && iid < buf.otherGraphIndex.length) return buf.otherGraphIndex[iid]!;
  }
  return pickGraphIndexByScreenProximity(e, camera, buf, ctx, w, h, rect);
}

/** World-space radius under ortho ~±1.06; opaque filled disks on top of lines/points. */
const SEED_DISK_RADIUS = 0.013;
const SEED_DISK_SEGMENTS = 40;
/** Smaller base disk for non-seed nodes when using instanced radial scaling. */
const OTHER_DISK_RADIUS = 0.0062;
const OTHER_DISK_SEGMENTS = 32;
/** Scale factor from W-plane radius r = hypot(x,y): larger near center (r→0). */
const _qId = new Quaternion();
const _vPos = new Vector3();
const _vScale = new Vector3();
const _m4 = new Matrix4();

function radialScaleFromR(r: number, sMin: number, sMax: number): number {
  const t = Math.min(1, Math.max(0, r));
  const lo = Math.min(sMin, sMax);
  const hi = Math.max(sMin, sMax);
  return lo + (1 - t) * (hi - lo);
}

function nodeZoomMultiplier(invZoom: number, compensate: boolean): number {
  if (!compensate) return 1;
  return 1 + (invZoom - 1) * ZOOM_NODE_COMP_STRENGTH;
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

function disposeLineMesh(m: InstanceType<typeof LineSegments> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof LineBasicMaterial>).dispose();
}

function disposeLineBothWide(m: InstanceType<typeof Mesh> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof Line2NodeMaterial>).dispose();
}

function disposeInstancedMesh(m: InstanceType<typeof InstancedMesh> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof MeshBasicMaterial>).dispose();
}

function syncSeedLabelDom(ctx: ThreeCtx, buf: SceneBuffers | null, show: boolean) {
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

function updateSeedLabelPositions(
  ctx: ThreeCtx,
  buf: SceneBuffers | null,
  show: boolean,
  viewRef: RefObject<DiskViewTransform>,
) {
  const spans = ctx.labelSpans;
  if (!show || !buf || buf.nPointsSeed === 0 || spans.length !== buf.nPointsSeed) return;
  const { camera, renderer, labelProj } = ctx;
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const p = buf.pointsSeed;
  const viewT = viewRef.current;
  const ox = 8;
  const oy = -7;
  for (let i = 0; i < spans.length; i++) {
    const [x, y] = mapViewerWxy(p[i * 3], p[i * 3 + 1], viewT);
    labelProj.set(x, y, p[i * 3 + 2]);
    labelProj.project(camera);
    const sx = (labelProj.x * 0.5 + 0.5) * w + ox;
    const sy = (-labelProj.y * 0.5 + 0.5) * h + oy;
    spans[i].style.transform = `translate(${sx}px, ${sy}px)`;
  }
}

function applyBuffers(
  ctx: ThreeCtx,
  buf: SceneBuffers | null,
  sizingRef: RefObject<DiskDisplaySizing>,
  viewRef: RefObject<DiskViewTransform>,
) {
  const { scene3 } = ctx;
  const {
    centerWeighted: weighted,
    radialMin,
    radialMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
  } = sizingRef.current;
  const v = viewRef.current;
  const zoom = v.zoom;
  const invZoom = 1 / Math.max(zoom, ZOOM_MIN);
  const nodeZoomMul = nodeZoomMultiplier(invZoom, compensateZoomNodes);
  const zm = Math.max(nodeZoomMul, nodeMinMul);
  const otherR = OTHER_DISK_RADIUS * nodeSizeMul * zm;
  const seedR = SEED_DISK_RADIUS * nodeSizeMul * zm;
  disposeLineBothWide(ctx.lineBoth, scene3);
  disposeLineMesh(ctx.lineOne, scene3);
  disposeInstancedMesh(ctx.meshOther, scene3);
  disposeInstancedMesh(ctx.meshSeeds, scene3);
  ctx.lineBoth = ctx.lineOne = ctx.meshOther = ctx.meshSeeds = null;
  if (!buf) return;

  /* Blue (additive) → green points → red both-seed lines → opaque seed disks on top. */
  if (buf.nLineOneVerts > 0) {
    const linePos = transformLinePositions(buf.lineOnePositions, v);
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(linePos, 3));
    const m = new LineBasicMaterial({
      color: 0x6699ff,
      transparent: true,
      opacity: 0.2,
      blending: AdditiveBlending,
      depthWrite: false,
    });
    const lineOne = new LineSegments(g, m);
    lineOne.renderOrder = 0;
    ctx.lineOne = lineOne;
    scene3.add(lineOne);
  }
  if (buf.nPointsOther > 0) {
    const po = buf.pointsOther;
    /* Always world-space disks (like red seeds) so Euclidean zoom scales them; old Points path capped ~56px. */
    const circleGeom = new CircleGeometry(otherR, OTHER_DISK_SEGMENTS);
    const mat = new MeshBasicMaterial({
      color: 0x55dd77,
      transparent: false,
      depthTest: true,
      depthWrite: false,
      toneMapped: false,
    });
    const mesh = new InstancedMesh(circleGeom, mat, buf.nPointsOther);
    for (let i = 0; i < buf.nPointsOther; i++) {
      const [x, y] = mapViewerWxy(po[i * 3], po[i * 3 + 1], v);
      const z = po[i * 3 + 2];
      const s = weighted
        ? clampedRadialScale(Math.hypot(x, y), radialMin, radialMax, nodeMinMul)
        : 1;
      _vPos.set(x, y, z);
      _m4.compose(_vPos, _qId, _vScale.set(s, s, 1));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.renderOrder = 5;
    mesh.frustumCulled = false;
    ctx.meshOther = mesh;
    scene3.add(mesh);
  }
  if (buf.nLineBothVerts > 0) {
    const bothPos = transformLinePositions(buf.lineBothPositions, v);
    const geom = new LineSegmentsGeometry();
    geom.setPositions(bothPos);
    const mat = new Line2NodeMaterial({
      color: 0xff2222,
      linewidth: 4,
      depthTest: true,
      depthWrite: true,
      toneMapped: false,
    });
    const lineBoth = new LineSegments2(geom, mat);
    lineBoth.renderOrder = 12;
    ctx.lineBoth = lineBoth;
    scene3.add(lineBoth);
  }
  if (buf.nPointsSeed > 0) {
    const circleGeom = new CircleGeometry(seedR, SEED_DISK_SEGMENTS);
    const mat = new MeshBasicMaterial({
      color: 0xff2a2a,
      transparent: false,
      opacity: 1,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    const mesh = new InstancedMesh(circleGeom, mat, buf.nPointsSeed);
    const p = buf.pointsSeed;
    for (let i = 0; i < buf.nPointsSeed; i++) {
      const [x, y] = mapViewerWxy(p[i * 3], p[i * 3 + 1], v);
      const z = p[i * 3 + 2];
      if (weighted) {
        const s = clampedRadialScale(Math.hypot(x, y), radialMin, radialMax, nodeMinMul);
        _vPos.set(x, y, z);
        _m4.compose(_vPos, _qId, _vScale.set(s, s, 1));
        mesh.setMatrixAt(i, _m4);
      } else {
        _m4.makeTranslation(x, y, z);
        mesh.setMatrixAt(i, _m4);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.renderOrder = 100;
    mesh.frustumCulled = false;
    ctx.meshSeeds = mesh;
    scene3.add(mesh);
  }
}

export const DiskView = forwardRef<DiskViewHandle, Props>(function DiskView(
  {
    scene,
    webGpuError,
    showSeedLabels,
    showCrosshair,
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
    nodeInteractionRef,
  }: Props,
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<ThreeCtx | null>(null);
  const diskViewTransformRef = useRef<DiskViewTransform>(defaultDiskViewTransform());
  const sceneRef = useRef(scene);
  const showLabelsRef = useRef(showSeedLabels);
  const showCrosshairRef = useRef(showCrosshair);
  const diskDisplayRef = useRef<DiskDisplaySizing>({
    centerWeighted: centerWeightedSizes,
    radialMin: radialScaleMin,
    radialMax: radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
  });
  useLayoutEffect(() => {
    sceneRef.current = scene;
  }, [scene]);
  useLayoutEffect(() => {
    showLabelsRef.current = showSeedLabels;
  }, [showSeedLabels]);
  useLayoutEffect(() => {
    showCrosshairRef.current = showCrosshair;
  }, [showCrosshair]);
  useLayoutEffect(() => {
    diskDisplayRef.current = {
      centerWeighted: centerWeightedSizes,
      radialMin: radialScaleMin,
      radialMax: radialScaleMax,
      nodeSizeMul,
      compensateZoomNodes,
      nodeMinMul,
    };
  }, [centerWeightedSizes, radialScaleMin, radialScaleMax, nodeSizeMul, compensateZoomNodes, nodeMinMul]);

  useImperativeHandle(ref, () => ({
    resetView() {
      diskViewTransformRef.current = defaultDiskViewTransform();
      const c = ctxRef.current;
      if (c) {
        updateOrthographicCamera(c.camera, diskViewTransformRef.current);
        applyBuffers(c, sceneRef.current, diskDisplayRef, diskViewTransformRef);
      }
    },
    recenterPreservingZoom() {
      const tr = diskViewTransformRef.current;
      tr.panX = 0;
      tr.panY = 0;
      tr.mobius = identityMobius();
      const c = ctxRef.current;
      if (c) {
        updateOrthographicCamera(c.camera, tr);
        applyBuffers(c, sceneRef.current, diskDisplayRef, diskViewTransformRef);
      }
    },
    fitSubgraphToView() {
      const buf = sceneRef.current;
      const c = ctxRef.current;
      if (!buf || !c) return;
      if (buf.nPointsSeed === 0 && buf.nPointsOther === 0) return;
      const tr = diskViewTransformRef.current;
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      const accumulate = (po: Float32Array) => {
        for (let i = 0; i < po.length; i += 3) {
          const [x, y] = mapViewerWxy(po[i], po[i + 1], tr);
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      };
      accumulate(buf.pointsSeed);
      accumulate(buf.pointsOther);
      if (!Number.isFinite(minX) || maxX <= minX || maxY <= minY) return;
      const cx = (minX + maxX) * 0.5;
      const cy = (minY + maxY) * 0.5;
      const PAD = 1.16;
      let w = (maxX - minX) * PAD;
      let h = (maxY - minY) * PAD;
      const eps = 1e-6;
      if (w < eps) w = 0.2;
      if (h < eps) h = 0.2;
      const zoomW = (2 * BASE_HALF_EXTENT) / w;
      const zoomH = (2 * BASE_HALF_EXTENT) / h;
      tr.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.min(zoomW, zoomH)));
      tr.panX = cx;
      tr.panY = cy;
      updateOrthographicCamera(c.camera, tr);
      applyBuffers(c, buf, diskDisplayRef, diskViewTransformRef);
    },
    applySceneBuffers(buf: SceneBuffers | null) {
      sceneRef.current = buf;
      const c = ctxRef.current;
      if (!c) return;
      applyBuffers(c, buf, diskDisplayRef, diskViewTransformRef);
      syncSeedLabelDom(c, buf, showLabelsRef.current);
    },
  }));

  useEffect(() => {
    const host = hostRef.current;
    if (!host || webGpuError) return;

    let disposed = false;
    const renderer = new WebGPURenderer({ antialias: true, alpha: true });
    const scene3 = new Scene();
    scene3.background = new Color(0x0e0e10);
    const camera = new OrthographicCamera(-1.06, 1.06, 1.06, -1.06, 0.1, 10);
    camera.position.set(0, 0, 2);
    camera.lookAt(0, 0, 0);
    updateOrthographicCamera(camera, diskViewTransformRef.current);

    const gridMat = new LineBasicMaterial({ color: 0x6e6e7a, opacity: 0.92, transparent: true });
    const g1 = new BufferGeometry().setFromPoints([
      { x: -1.05, y: 0, z: 0 },
      { x: 1.05, y: 0, z: 0 },
    ]);
    const g2 = new BufferGeometry().setFromPoints([
      { x: 0, y: -1.05, z: 0 },
      { x: 0, y: 1.05, z: 0 },
    ]);
    /*
     * Crosshair stays in scene W-plane (not composed with viewer Möbius): under Shift+drag
     * hyperbolic pan, diameters through 0 would become circular arcs if transformed with T.
     */
    const crosshairH = new LineSegments(g1, gridMat);
    const crosshairV = new LineSegments(g2, gridMat);
    crosshairH.visible = showCrosshairRef.current;
    crosshairV.visible = showCrosshairRef.current;
    scene3.add(crosshairH);
    scene3.add(crosshairV);

    const labelsLayer = document.createElement("div");
    labelsLayer.className = "diskSeedLabels";
    labelsLayer.setAttribute("aria-hidden", "true");

    const nodeTipEl = document.createElement("div");
    nodeTipEl.className = "diskNodeTooltip";
    nodeTipEl.style.display = "none";

    const raycaster = new Raycaster();
    const ndcPointer = new Vector2();

    const ctx: ThreeCtx = {
      renderer,
      scene3,
      camera,
      host,
      lineBoth: null,
      lineOne: null,
      meshOther: null,
      meshSeeds: null,
      labelsLayer,
      labelSpans: [],
      labelProj: new Vector3(),
      crosshairH,
      crosshairV,
      raf: 0,
      nodeTipEl,
      raycaster,
      ndcPointer,
    };
    ctxRef.current = ctx;

    const hideNodeTip = () => {
      nodeTipEl.style.display = "none";
      nodeTipEl.textContent = "";
    };
    const showNodeTip = (e: PointerEvent, text: string) => {
      if (!text) {
        hideNodeTip();
        return;
      }
      nodeTipEl.textContent = text;
      nodeTipEl.style.display = "block";
      const pad = 14;
      const tw = 240;
      const th = 100;
      nodeTipEl.style.left = `${Math.min(e.clientX + pad, window.innerWidth - tw)}px`;
      nodeTipEl.style.top = `${Math.min(e.clientY + pad, window.innerHeight - th)}px`;
    };

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight || w;
      renderer.setSize(w, h, false);
      updateOrthographicCamera(camera, diskViewTransformRef.current);
    };

    let dragActive = false;
    let lastPx = 0;
    let lastPy = 0;
    let downPx = 0;
    let downPy = 0;
    let pointerDidDrag = false;
    /** `e.shiftKey` is unreliable on some browsers during `setPointerCapture`; track Shift explicitly. */
    let shiftHeld = false;
    const syncShiftFromEvent = (e: PointerEvent | KeyboardEvent) => {
      if (typeof e.getModifierState === "function") {
        shiftHeld = e.getModifierState("Shift");
      } else {
        shiftHeld = "shiftKey" in e && e.shiftKey === true;
      }
    };
    const onWindowKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Shift") shiftHeld = true;
    };
    const onWindowKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") shiftHeld = false;
    };
    const onWindowBlur = () => {
      shiftHeld = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const tr = diskViewTransformRef.current;
      const shiftWheel =
        shiftHeld ||
        e.shiftKey ||
        (typeof e.getModifierState === "function" && e.getModifierState("Shift"));
      if (shiftWheel) {
        tr.rimPreservingGamma = Math.min(
          RIM_GAMMA_MAX,
          Math.max(RIM_GAMMA_MIN, tr.rimPreservingGamma * Math.exp(-e.deltaY * RIM_WHEEL_SENS)),
        );
        applyBuffers(ctx, sceneRef.current, diskDisplayRef, diskViewTransformRef);
      } else {
        tr.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, tr.zoom * Math.exp(-e.deltaY * WHEEL_ZOOM_SENS)));
        updateOrthographicCamera(camera, tr);
        applyBuffers(ctx, sceneRef.current, diskDisplayRef, diskViewTransformRef);
      }
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      dragActive = true;
      pointerDidDrag = false;
      downPx = e.clientX;
      downPy = e.clientY;
      syncShiftFromEvent(e);
      lastPx = e.clientX;
      lastPy = e.clientY;
      hideNodeTip();
      (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      syncShiftFromEvent(e);
      const canvas = renderer.domElement;
      const tr = diskViewTransformRef.current;
      const buf = sceneRef.current;
      if (dragActive) {
        if (Math.hypot(e.clientX - downPx, e.clientY - downPy) > CLICK_DRAG_PX) pointerDidDrag = true;
        const dx = e.clientX - lastPx;
        const dy = e.clientY - lastPy;
        lastPx = e.clientX;
        lastPy = e.clientY;
        const cw = Math.max(1, canvas.clientWidth);
        const ch = Math.max(1, canvas.clientHeight);
        const worldSpan = (2 * BASE_HALF_EXTENT) / tr.zoom;
        if (shiftHeld) {
          tr.mobius = incrementMobiusFromDrag(tr.mobius, dx, dy, cw, ch);
          applyBuffers(ctx, buf, diskDisplayRef, diskViewTransformRef);
        } else {
          tr.panX -= (dx / cw) * worldSpan;
          tr.panY += (dy / ch) * worldSpan;
          updateOrthographicCamera(camera, tr);
        }
      } else {
        const nip = nodeInteractionRef?.current;
        if (!nip || !buf) {
          hideNodeTip();
          return;
        }
        const gid = pickGraphIndexAtPointerEvent(e, canvas, camera, buf, ctx, ndcPointer, raycaster);
        if (gid === null) hideNodeTip();
        else showNodeTip(e, nip.tooltipForGraphIndex(gid));
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      const canvas = renderer.domElement;
      const buf = sceneRef.current;
      const wasDrag = pointerDidDrag;
      try {
        (e.currentTarget as HTMLCanvasElement).releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
      dragActive = false;
      if (!wasDrag) {
        const nip = nodeInteractionRef?.current;
        if (nip && buf) {
          const gid = pickGraphIndexAtPointerEvent(e, canvas, camera, buf, ctx, ndcPointer, raycaster);
          if (gid !== null) nip.pickGraphIndex(gid);
        }
      }
      const nip = nodeInteractionRef?.current;
      if (nip && buf) {
        const gid = pickGraphIndexAtPointerEvent(e, canvas, camera, buf, ctx, ndcPointer, raycaster);
        if (gid === null) hideNodeTip();
        else showNodeTip(e, nip.tooltipForGraphIndex(gid));
      } else hideNodeTip();
    };

    const onPointerLeave = () => {
      hideNodeTip();
    };

    const loop = () => {
      ctx.raf = requestAnimationFrame(loop);
      renderer.render(scene3, camera);
      updateSeedLabelPositions(ctx, sceneRef.current, showLabelsRef.current, diskViewTransformRef);
    };

    void (async () => {
      try {
        await renderer.init();
        if (disposed) return;
        host.appendChild(renderer.domElement);
        host.appendChild(labelsLayer);
        host.appendChild(nodeTipEl);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        resize();
        const el = renderer.domElement;
        window.addEventListener("keydown", onWindowKeyDown);
        window.addEventListener("keyup", onWindowKeyUp);
        window.addEventListener("blur", onWindowBlur);
        el.addEventListener("wheel", onWheel, { passive: false });
        el.addEventListener("pointerdown", onPointerDown);
        el.addEventListener("pointermove", onPointerMove);
        el.addEventListener("pointerup", onPointerUp);
        el.addEventListener("pointercancel", onPointerUp);
        el.addEventListener("pointerleave", onPointerLeave);
        applyBuffers(ctx, sceneRef.current, diskDisplayRef, diskViewTransformRef);
        syncSeedLabelDom(ctx, sceneRef.current, showLabelsRef.current);
        loop();
      } catch (e) {
        console.error(e);
      }
    })();

    const ro = new ResizeObserver(resize);
    ro.observe(host);

    return () => {
      disposed = true;
      cancelAnimationFrame(ctx.raf);
      ro.disconnect();
      if (renderer.domElement.parentElement === host) host.removeChild(renderer.domElement);
      if (labelsLayer.parentElement === host) host.removeChild(labelsLayer);
      if (nodeTipEl.parentElement === host) host.removeChild(nodeTipEl);
      window.removeEventListener("keydown", onWindowKeyDown);
      window.removeEventListener("keyup", onWindowKeyUp);
      window.removeEventListener("blur", onWindowBlur);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerUp);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      applyBuffers(ctx, null, diskDisplayRef, diskViewTransformRef);
      syncSeedLabelDom(ctx, null, false);
      renderer.dispose();
      ctxRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- read nodeInteractionRef.current in handlers; stable ref object
  }, [webGpuError]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || webGpuError) return;
    /* Keep Euclidean pan/zoom and viewer Möbius across focus/scene buffer updates; use Reset view to clear. */
    applyBuffers(ctx, scene, diskDisplayRef, diskViewTransformRef);
    syncSeedLabelDom(ctx, scene, showSeedLabels);
  }, [
    scene,
    showSeedLabels,
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
    webGpuError,
  ]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || webGpuError) return;
    ctx.crosshairH.visible = showCrosshair;
    ctx.crosshairV.visible = showCrosshair;
  }, [showCrosshair, webGpuError]);

  return (
    <div
      ref={hostRef}
      style={{
        position: "absolute",
        inset: 0,
        background: "#0e0e10",
      }}
    >
      {webGpuError ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
            color: "#f87171",
            font: "13px system-ui, sans-serif",
          }}
        >
          {webGpuError}
        </div>
      ) : null}
    </div>
  );
});

DiskView.displayName = "DiskView";
