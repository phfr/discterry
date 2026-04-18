import { useEffect, useLayoutEffect, useRef } from "react";
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
} from "three/webgpu";
import { LineSegments2 } from "three/addons/lines/webgpu/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { SceneBuffers } from "../model/computeScene";

type Props = {
  scene: SceneBuffers | null;
  webGpuError: string | null;
  showSeedLabels: boolean;
  showCrosshair: boolean;
};

type ThreeCtx = {
  renderer: InstanceType<typeof WebGPURenderer>;
  scene3: InstanceType<typeof Scene>;
  camera: InstanceType<typeof OrthographicCamera>;
  host: HTMLDivElement;
  lineBoth: InstanceType<typeof Mesh> | null;
  lineOne: InstanceType<typeof LineSegments> | null;
  ptsOther: InstanceType<typeof Points> | null;
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

function updateSeedLabelPositions(ctx: ThreeCtx, buf: SceneBuffers | null, show: boolean) {
  const spans = ctx.labelSpans;
  if (!show || !buf || buf.nPointsSeed === 0 || spans.length !== buf.nPointsSeed) return;
  const { camera, renderer, labelProj: v } = ctx;
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  const p = buf.pointsSeed;
  const ox = 8;
  const oy = -7;
  for (let i = 0; i < spans.length; i++) {
    v.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    v.project(camera);
    const x = (v.x * 0.5 + 0.5) * w + ox;
    const y = (-v.y * 0.5 + 0.5) * h + oy;
    spans[i].style.transform = `translate(${x}px, ${y}px)`;
  }
}

function applyBuffers(ctx: ThreeCtx, buf: SceneBuffers | null) {
  const { scene3 } = ctx;
  disposeLineBothWide(ctx.lineBoth, scene3);
  disposeLineMesh(ctx.lineOne, scene3);
  disposePoints(ctx.ptsOther, scene3);
  disposeInstancedMesh(ctx.meshSeeds, scene3);
  ctx.lineBoth = ctx.lineOne = ctx.ptsOther = ctx.meshSeeds = null;
  if (!buf) return;

  /* Blue (additive) → green points → red both-seed lines → opaque seed disks on top. */
  if (buf.nLineOneVerts > 0) {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(buf.lineOnePositions, 3));
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
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(buf.pointsOther, 3));
    const m = new PointsMaterial({
      color: 0x55dd77,
      size: 10,
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
  if (buf.nLineBothVerts > 0) {
    const geom = new LineSegmentsGeometry();
    geom.setPositions(buf.lineBothPositions);
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
    const circleGeom = new CircleGeometry(SEED_DISK_RADIUS, SEED_DISK_SEGMENTS);
    const mat = new MeshBasicMaterial({
      color: 0xff2a2a,
      transparent: false,
      opacity: 1,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    const mesh = new InstancedMesh(circleGeom, mat, buf.nPointsSeed);
    const m4 = new Matrix4();
    const p = buf.pointsSeed;
    for (let i = 0; i < buf.nPointsSeed; i++) {
      m4.makeTranslation(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
      mesh.setMatrixAt(i, m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.renderOrder = 100;
    mesh.frustumCulled = false;
    ctx.meshSeeds = mesh;
    scene3.add(mesh);
  }
}

export function DiskView({ scene, webGpuError, showSeedLabels, showCrosshair }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<ThreeCtx | null>(null);
  const sceneRef = useRef(scene);
  const showLabelsRef = useRef(showSeedLabels);
  const showCrosshairRef = useRef(showCrosshair);
  useLayoutEffect(() => {
    sceneRef.current = scene;
  }, [scene]);
  useLayoutEffect(() => {
    showLabelsRef.current = showSeedLabels;
  }, [showSeedLabels]);
  useLayoutEffect(() => {
    showCrosshairRef.current = showCrosshair;
  }, [showCrosshair]);

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

    const gridMat = new LineBasicMaterial({ color: 0x6e6e7a, opacity: 0.92, transparent: true });
    const g1 = new BufferGeometry().setFromPoints([
      { x: -1.05, y: 0, z: 0 },
      { x: 1.05, y: 0, z: 0 },
    ]);
    const g2 = new BufferGeometry().setFromPoints([
      { x: 0, y: -1.05, z: 0 },
      { x: 0, y: 1.05, z: 0 },
    ]);
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
    };

    const loop = () => {
      ctx.raf = requestAnimationFrame(loop);
      renderer.render(scene3, camera);
      updateSeedLabelPositions(ctx, sceneRef.current, showLabelsRef.current);
    };

    void (async () => {
      try {
        await renderer.init();
        if (disposed) return;
        host.appendChild(renderer.domElement);
        host.appendChild(labelsLayer);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        resize();
        applyBuffers(ctx, sceneRef.current);
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
      applyBuffers(ctx, null);
      syncSeedLabelDom(ctx, null, false);
      renderer.dispose();
      ctxRef.current = null;
    };
  }, [webGpuError]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || webGpuError) return;
    applyBuffers(ctx, scene);
    syncSeedLabelDom(ctx, scene, showSeedLabels);
  }, [scene, showSeedLabels, webGpuError]);

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
}
