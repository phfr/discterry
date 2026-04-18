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
  Points,
  PointsMaterial,
  Color,
  AdditiveBlending,
  InstancedMesh,
  MeshBasicMaterial,
  CircleGeometry,
  Matrix4,
  Vector3,
  Quaternion,
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
const ZOOM_MAX = 8;
const WHEEL_ZOOM_SENS = 0.0018;

export type DiskViewTransform = {
  /** 1 = default framing; larger zooms in (smaller ortho half-extent). */
  zoom: number;
  panX: number;
  panY: number;
  mobius: Mobius2;
};

export type DiskViewHandle = {
  resetView: () => void;
};

function defaultDiskViewTransform(): DiskViewTransform {
  return { zoom: 1, panX: 0, panY: 0, mobius: identityMobius() };
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

function transformLinePositions(src: Float32Array, M: Mobius2): Float32Array {
  if (isIdentityMobius(M)) return src;
  const n = src.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 3) {
    const [x, y] = mapWxy(src[i], src[i + 1], M);
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
};

type ThreeCtx = {
  renderer: InstanceType<typeof WebGPURenderer>;
  scene3: InstanceType<typeof Scene>;
  camera: InstanceType<typeof OrthographicCamera>;
  host: HTMLDivElement;
  lineBoth: InstanceType<typeof Mesh> | null;
  lineOne: InstanceType<typeof LineSegments> | null;
  ptsOther: InstanceType<typeof Points> | null;
  meshOther: InstanceType<typeof InstancedMesh> | null;
  meshSeeds: InstanceType<typeof InstancedMesh> | null;
  labelsLayer: HTMLDivElement;
  labelSpans: HTMLSpanElement[];
  labelProj: InstanceType<typeof Vector3>;
  crosshairH: InstanceType<typeof LineSegments>;
  crosshairV: InstanceType<typeof LineSegments>;
  raf: number;
};

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

function pointsSpriteSize(nodeMul: number): number {
  return Math.min(48, Math.max(2, 10 * nodeMul));
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

function disposePoints(m: InstanceType<typeof Points> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof PointsMaterial>).dispose();
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
  const { camera, renderer, labelProj: v } = ctx;
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const p = buf.pointsSeed;
  const M = viewRef.current.mobius;
  const ox = 8;
  const oy = -7;
  for (let i = 0; i < spans.length; i++) {
    const [x, y] = mapWxy(p[i * 3], p[i * 3 + 1], M);
    v.set(x, y, p[i * 3 + 2]);
    v.project(camera);
    const sx = (v.x * 0.5 + 0.5) * w + ox;
    const sy = (-v.y * 0.5 + 0.5) * h + oy;
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
  const { centerWeighted: weighted, radialMin, radialMax, nodeSizeMul } = sizingRef.current;
  const M = viewRef.current.mobius;
  const otherR = OTHER_DISK_RADIUS * nodeSizeMul;
  const seedR = SEED_DISK_RADIUS * nodeSizeMul;
  disposeLineBothWide(ctx.lineBoth, scene3);
  disposeLineMesh(ctx.lineOne, scene3);
  disposePoints(ctx.ptsOther, scene3);
  disposeInstancedMesh(ctx.meshOther, scene3);
  disposeInstancedMesh(ctx.meshSeeds, scene3);
  ctx.lineBoth = ctx.lineOne = ctx.ptsOther = ctx.meshOther = ctx.meshSeeds = null;
  if (!buf) return;

  /* Blue (additive) → green points → red both-seed lines → opaque seed disks on top. */
  if (buf.nLineOneVerts > 0) {
    const linePos = transformLinePositions(buf.lineOnePositions, M);
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
    if (weighted) {
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
        const [x, y] = mapWxy(po[i * 3], po[i * 3 + 1], M);
        const z = po[i * 3 + 2];
        const s = radialScaleFromR(Math.hypot(x, y), radialMin, radialMax);
        _vPos.set(x, y, z);
        _m4.compose(_vPos, _qId, _vScale.set(s, s, 1));
        mesh.setMatrixAt(i, _m4);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.renderOrder = 5;
      mesh.frustumCulled = false;
      ctx.meshOther = mesh;
      scene3.add(mesh);
    } else {
      const g = new BufferGeometry();
      if (isIdentityMobius(M)) {
        g.setAttribute("position", new Float32BufferAttribute(po, 3));
      } else {
        const tf = new Float32Array(po.length);
        for (let i = 0; i < po.length; i += 3) {
          const [x, y] = mapWxy(po[i], po[i + 1], M);
          tf[i] = x;
          tf[i + 1] = y;
          tf[i + 2] = po[i + 2];
        }
        g.setAttribute("position", new Float32BufferAttribute(tf, 3));
      }
      const m = new PointsMaterial({
        color: 0x55dd77,
        size: pointsSpriteSize(nodeSizeMul),
        sizeAttenuation: false,
        depthTest: true,
        /* Avoid large sprites writing depth into neighbors' pixels (hides nearby seed points). */
        depthWrite: false,
      });
      const pts = new Points(g, m);
      pts.renderOrder = 5;
      ctx.ptsOther = pts;
      scene3.add(pts);
    }
  }
  if (buf.nLineBothVerts > 0) {
    const bothPos = transformLinePositions(buf.lineBothPositions, M);
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
      const [x, y] = mapWxy(p[i * 3], p[i * 3 + 1], M);
      const z = p[i * 3 + 2];
      if (weighted) {
        const s = radialScaleFromR(Math.hypot(x, y), radialMin, radialMax);
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
  }: Props,
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<ThreeCtx | null>(null);
  const diskViewTransformRef = useRef<DiskViewTransform>(defaultDiskViewTransform());
  const prevSceneRef = useRef<SceneBuffers | null>(null);
  const sceneRef = useRef(scene);
  const showLabelsRef = useRef(showSeedLabels);
  const showCrosshairRef = useRef(showCrosshair);
  const diskDisplayRef = useRef<DiskDisplaySizing>({
    centerWeighted: centerWeightedSizes,
    radialMin: radialScaleMin,
    radialMax: radialScaleMax,
    nodeSizeMul,
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
    };
  }, [centerWeightedSizes, radialScaleMin, radialScaleMax, nodeSizeMul]);

  useImperativeHandle(ref, () => ({
    resetView() {
      diskViewTransformRef.current = defaultDiskViewTransform();
      const c = ctxRef.current;
      if (c) {
        updateOrthographicCamera(c.camera, diskViewTransformRef.current);
        applyBuffers(c, sceneRef.current, diskDisplayRef, diskViewTransformRef);
      }
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

    const ctx: ThreeCtx = {
      renderer,
      scene3,
      camera,
      host,
      lineBoth: null,
      lineOne: null,
      ptsOther: null,
      meshOther: null,
      meshSeeds: null,
      labelsLayer,
      labelSpans: [],
      labelProj: new Vector3(),
      crosshairH,
      crosshairV,
      raf: 0,
    };
    ctxRef.current = ctx;

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight || w;
      renderer.setSize(w, h, false);
      updateOrthographicCamera(camera, diskViewTransformRef.current);
    };

    let dragActive = false;
    let lastPx = 0;
    let lastPy = 0;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const tr = diskViewTransformRef.current;
      tr.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, tr.zoom * Math.exp(-e.deltaY * WHEEL_ZOOM_SENS)));
      updateOrthographicCamera(camera, tr);
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      dragActive = true;
      lastPx = e.clientX;
      lastPy = e.clientY;
      (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!dragActive) return;
      const dx = e.clientX - lastPx;
      const dy = e.clientY - lastPy;
      lastPx = e.clientX;
      lastPy = e.clientY;
      const tr = diskViewTransformRef.current;
      const canvas = renderer.domElement;
      const cw = Math.max(1, canvas.clientWidth);
      const ch = Math.max(1, canvas.clientHeight);
      const worldSpan = (2 * BASE_HALF_EXTENT) / tr.zoom;
      if (e.shiftKey) {
        tr.mobius = incrementMobiusFromDrag(tr.mobius, dx, dy, cw, ch);
        applyBuffers(ctx, sceneRef.current, diskDisplayRef, diskViewTransformRef);
      } else {
        tr.panX -= (dx / cw) * worldSpan;
        tr.panY += (dy / ch) * worldSpan;
        updateOrthographicCamera(camera, tr);
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      dragActive = false;
      try {
        (e.currentTarget as HTMLCanvasElement).releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
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
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        resize();
        const el = renderer.domElement;
        el.addEventListener("wheel", onWheel, { passive: false });
        el.addEventListener("pointerdown", onPointerDown);
        el.addEventListener("pointermove", onPointerMove);
        el.addEventListener("pointerup", onPointerUp);
        el.addEventListener("pointercancel", onPointerUp);
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
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerUp);
      applyBuffers(ctx, null, diskDisplayRef, diskViewTransformRef);
      syncSeedLabelDom(ctx, null, false);
      renderer.dispose();
      ctxRef.current = null;
    };
  }, [webGpuError]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || webGpuError) return;
    if (prevSceneRef.current !== scene) {
      prevSceneRef.current = scene;
      diskViewTransformRef.current = defaultDiskViewTransform();
      updateOrthographicCamera(ctx.camera, diskViewTransformRef.current);
    }
    applyBuffers(ctx, scene, diskDisplayRef, diskViewTransformRef);
    syncSeedLabelDom(ctx, scene, showSeedLabels);
  }, [scene, showSeedLabels, centerWeightedSizes, radialScaleMin, radialScaleMax, nodeSizeMul, webGpuError]);

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
