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
  PerspectiveCamera,
  LineSegments,
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  Color,
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
import type { NodeDiskHoverTooltip } from "../nodeListTooltip";
import type { SceneBuffers3d } from "../model/computeScene3d";

/**
 * Unit-sphere geometry is radius 1; instance matrix scale sets world radius.
 * Chosen so default `nodeSizeMul`≈0.5 matches disk-like footprint in the |w|≤1 ball.
 */
const OTHER_BALL_BASE = 0.0055;
const SEED_BALL_BASE = 0.011;
const BALL_WIRE_SEGMENTS = 48;
/** Orbit distance at default framing — used like disk `zoom` for node size compensation. */
const CAM_REF_DIST = 2.85;
const ZOOM_NODE_COMP_STRENGTH = 0.5;

export type BallDisplaySizing = {
  centerWeighted: boolean;
  radialMin: number;
  radialMax: number;
  nodeSizeMul: number;
  compensateZoomNodes: boolean;
  nodeMinMul: number;
};

export type BallView3dHandle = {
  resetView: () => void;
  applySceneBuffers: (buf: SceneBuffers3d | null) => void;
  fitSubgraphToView: () => void;
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
  lineBoth: InstanceType<typeof LineSegments> | null;
  lineOne: InstanceType<typeof LineSegments> | null;
  meshOther: InstanceType<typeof InstancedMesh> | null;
  meshSeeds: InstanceType<typeof InstancedMesh> | null;
  ballWire: InstanceType<typeof LineSegments> | null;
  nodeTipEl: HTMLDivElement;
  raycaster: InstanceType<typeof Raycaster>;
  ndcPointer: InstanceType<typeof Vector2>;
  raf: number;
};

function disposeLine(m: InstanceType<typeof LineSegments> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof LineBasicMaterial>).dispose();
}

function disposeInst(m: InstanceType<typeof InstancedMesh> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof MeshBasicMaterial>).dispose();
}

const _m4 = new Matrix4();
const _v3 = new Vector3();
const _vScale = new Vector3();
const _qId = new Quaternion();

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

function applySceneToBall(
  ctx: BallCtx,
  buf: SceneBuffers3d | null,
  orbitRef: RefObject<OrbitState>,
  camera: InstanceType<typeof PerspectiveCamera>,
  sizing: BallDisplaySizing,
) {
  const { scene3 } = ctx;
  disposeLine(ctx.lineBg, scene3);
  disposeLine(ctx.lineBoth, scene3);
  disposeLine(ctx.lineOne, scene3);
  disposeInst(ctx.meshOther, scene3);
  disposeInst(ctx.meshSeeds, scene3);
  ctx.lineBg = ctx.lineBoth = ctx.lineOne = ctx.meshOther = ctx.meshSeeds = null;
  if (!buf) return;

  const mkLine = (positions: Float32Array, color: number, opacity: number) => {
    if (positions.length < 6) return null;
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(positions, 3));
    const mat = new LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
    });
    const ln = new LineSegments(g, mat);
    scene3.add(ln);
    return ln;
  };

  if (buf.nLineBgVerts > 0) {
    ctx.lineBg = mkLine(buf.lineBgPositions, 0xffc8a0, 0.12);
  }
  if (buf.nLineBothVerts > 0) {
    ctx.lineBoth = mkLine(buf.lineBothPositions, 0xff6644, 0.55);
  }
  if (buf.nLineOneVerts > 0) {
    ctx.lineOne = mkLine(buf.lineOnePositions, 0x6699ff, 0.35);
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
      _m4.compose(_v3.set(x, y, z), _qId, _vScale.set(s, s, s));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    scene3.add(mesh);
    ctx.meshSeeds = mesh;
  } else seedGeom.dispose();

  const otherGeom = new SphereGeometry(1, 12, 10);
  const otherMat = new MeshBasicMaterial({ color: 0x55cc77 });
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
      const s = OTHER_BALL_BASE * sizing.nodeSizeMul * zm * rad;
      _m4.compose(_v3.set(x, y, z), _qId, _vScale.set(s, s, s));
      mesh.setMatrixAt(i, _m4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    scene3.add(mesh);
    ctx.meshOther = mesh;
  } else otherGeom.dispose();

  orbitToPosition(orbitRef.current, camera.position);
  camera.lookAt(0, 0, 0);
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
      _m4.compose(_v3.set(x, y, z), _qId, _vScale.set(s, s, s));
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
      const s = OTHER_BALL_BASE * sizing.nodeSizeMul * zm * rad;
      _m4.compose(_v3.set(x, y, z), _qId, _vScale.set(s, s, s));
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
    _v3.set(gx, gy, gz).project(camera);
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
  webGpuError: string | null;
  showSeedLabels: boolean;
  nodeInteractionRef: RefObject<BallView3dNodeInteraction | null>;
  centerWeightedSizes: boolean;
  radialScaleMin: number;
  radialScaleMax: number;
  nodeSizeMul: number;
  compensateZoomNodes: boolean;
  nodeMinMul: number;
};

export const BallView3d = forwardRef<BallView3dHandle, BallView3dProps>(function BallView3d(
  {
    scene,
    webGpuError,
    showSeedLabels: _showSeedLabels,
    nodeInteractionRef,
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
  }: BallView3dProps,
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<BallCtx | null>(null);
  const sceneRef = useRef(scene);
  const orbitRef = useRef<OrbitState>(defaultOrbit());
  const dragRef = useRef({ active: false, downX: 0, downY: 0, lastX: 0, lastY: 0 });
  const ballSizingRef = useRef<BallDisplaySizing>({
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
    ballSizingRef.current = {
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
      orbitRef.current = defaultOrbit();
      const c = ctxRef.current;
      if (c) {
        orbitToPosition(orbitRef.current, c.camera.position);
        c.camera.lookAt(0, 0, 0);
        const buf = sceneRef.current;
        if (buf) refreshBallNodeScales(c, buf, orbitRef, ballSizingRef.current);
      }
    },
    applySceneBuffers(buf: SceneBuffers3d | null) {
      sceneRef.current = buf;
      const c = ctxRef.current;
      if (!c) return;
      applySceneToBall(c, buf, orbitRef, c.camera, ballSizingRef.current);
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
      opacity: 0.35,
      depthWrite: false,
    });
    const ballWire = new LineSegments(edges, ballMat);
    scene3.add(ballWire);
    ballGeom.dispose();

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
      ballWire,
      nodeTipEl,
      raycaster,
      ndcPointer,
      raf: 0,
    };
    ctxRef.current = ctx;

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
      const o = orbitRef.current;
      o.distance = Math.min(24, Math.max(0.9, o.distance * Math.exp(e.deltaY * 0.0012)));
      orbitToPosition(o, camera.position);
      camera.lookAt(0, 0, 0);
      const buf = sceneRef.current;
      if (buf && ballSizingRef.current.compensateZoomNodes) {
        refreshBallNodeScales(ctx, buf, orbitRef, ballSizingRef.current);
      }
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      dragRef.current = {
        active: true,
        downX: e.clientX,
        downY: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
      };
      hideNodeTip();
      (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      const buf = sceneRef.current;
      const nip = nodeInteractionRef?.current;
      if (dragRef.current.active) {
        const dx = e.clientX - dragRef.current.lastX;
        const dy = e.clientY - dragRef.current.lastY;
        dragRef.current.lastX = e.clientX;
        dragRef.current.lastY = e.clientY;
        const o = orbitRef.current;
        o.yaw += dx * 0.0055;
        o.pitch = Math.max(-1.45, Math.min(1.45, o.pitch + dy * 0.0055));
        orbitToPosition(o, camera.position);
        camera.lookAt(0, 0, 0);
        if (buf && ballSizingRef.current.compensateZoomNodes) {
          refreshBallNodeScales(ctx, buf, orbitRef, ballSizingRef.current);
        }
      } else if (nip && buf) {
        const gid = pickGraphIndexAtEvent(e, renderer.domElement, camera, buf, ctx);
        if (gid === null) hideNodeTip();
        else showNodeTip(e, nip.tooltipForGraphIndex(gid));
      }
    };

    const onPointerUp = (e: PointerEvent) => {
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
      if (!wasDrag && nip && buf) {
        const shiftPick =
          e.shiftKey || (typeof e.getModifierState === "function" && e.getModifierState("Shift"));
        const gid = pickGraphIndexAtEvent(e, renderer.domElement, camera, buf, ctx);
        if (gid !== null) {
          if (shiftPick && nip.shiftPickGraphIndex) nip.shiftPickGraphIndex(gid);
          else if (!shiftPick) nip.pickGraphIndex(gid);
        }
      }
      if (nip && buf) {
        const gid = pickGraphIndexAtEvent(e, renderer.domElement, camera, buf, ctx);
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
    };

    void (async () => {
      try {
        await renderer.init();
        if (disposed) return;
        host.appendChild(renderer.domElement);
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
        applySceneToBall(ctx, sceneRef.current, orbitRef, camera, ballSizingRef.current);
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
      const el = ctx.renderer.domElement;
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
      el.removeEventListener("pointerleave", onPointerLeave);
      disposeLine(ctx.lineBg, scene3);
      disposeLine(ctx.lineBoth, scene3);
      disposeLine(ctx.lineOne, scene3);
      disposeInst(ctx.meshOther, scene3);
      disposeInst(ctx.meshSeeds, scene3);
      if (ctx.ballWire) {
        scene3.remove(ctx.ballWire);
        ctx.ballWire.geometry.dispose();
        (ctx.ballWire.material as InstanceType<typeof LineBasicMaterial>).dispose();
      }
      ctxRef.current = null;
      renderer.dispose();
      if (host.contains(el)) host.removeChild(el);
      if (nodeTipEl.parentElement === host) host.removeChild(nodeTipEl);
    };
  }, [webGpuError]);

  useEffect(() => {
    const c = ctxRef.current;
    if (!c || webGpuError) return;
    applySceneToBall(c, scene, orbitRef, c.camera, ballSizingRef.current);
  }, [
    scene,
    webGpuError,
    centerWeightedSizes,
    radialScaleMin,
    radialScaleMax,
    nodeSizeMul,
    compensateZoomNodes,
    nodeMinMul,
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
