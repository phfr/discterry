import { useEffect, useLayoutEffect, useRef } from "react";
import {
  WebGPURenderer,
  Scene,
  OrthographicCamera,
  LineSegments,
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  Points,
  PointsMaterial,
  Color,
  Line,
} from "three/webgpu";
import type { SceneBuffers } from "../model/computeScene";

type Props = {
  scene: SceneBuffers | null;
  webGpuError: string | null;
};

type ThreeCtx = {
  renderer: InstanceType<typeof WebGPURenderer>;
  scene3: InstanceType<typeof Scene>;
  camera: InstanceType<typeof OrthographicCamera>;
  host: HTMLDivElement;
  lineBoth: InstanceType<typeof LineSegments> | null;
  lineOne: InstanceType<typeof LineSegments> | null;
  ptsOther: InstanceType<typeof Points> | null;
  ptsSeed: InstanceType<typeof Points> | null;
  raf: number;
};

function disposeLineMesh(m: InstanceType<typeof LineSegments> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof LineBasicMaterial>).dispose();
}

function disposePoints(m: InstanceType<typeof Points> | null, scene3: InstanceType<typeof Scene>) {
  if (!m) return;
  scene3.remove(m);
  m.geometry.dispose();
  (m.material as InstanceType<typeof PointsMaterial>).dispose();
}

function applyBuffers(ctx: ThreeCtx, buf: SceneBuffers | null) {
  const { scene3 } = ctx;
  disposeLineMesh(ctx.lineBoth, scene3);
  disposeLineMesh(ctx.lineOne, scene3);
  disposePoints(ctx.ptsOther, scene3);
  disposePoints(ctx.ptsSeed, scene3);
  ctx.lineBoth = ctx.lineOne = ctx.ptsOther = ctx.ptsSeed = null;
  if (!buf) return;

  if (buf.nLineBothVerts > 0) {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(buf.lineBothPositions, 3));
    const m = new LineBasicMaterial({
      color: 0xff5555,
      linewidth: 2,
      transparent: true,
      opacity: 0.9,
    });
    ctx.lineBoth = new LineSegments(g, m);
    scene3.add(ctx.lineBoth);
  }
  if (buf.nLineOneVerts > 0) {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(buf.lineOnePositions, 3));
    const m = new LineBasicMaterial({
      color: 0x6699ff,
      transparent: true,
      opacity: 0.2,
    });
    ctx.lineOne = new LineSegments(g, m);
    scene3.add(ctx.lineOne);
  }
  if (buf.nPointsOther > 0) {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(buf.pointsOther, 3));
    const m = new PointsMaterial({
      color: 0x7a7a82,
      size: 4,
      sizeAttenuation: false,
    });
    ctx.ptsOther = new Points(g, m);
    scene3.add(ctx.ptsOther);
  }
  if (buf.nPointsSeed > 0) {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(buf.pointsSeed, 3));
    const m = new PointsMaterial({
      color: 0xff6b6b,
      size: 8,
      sizeAttenuation: false,
    });
    ctx.ptsSeed = new Points(g, m);
    scene3.add(ctx.ptsSeed);
  }
}

export function DiskView({ scene, webGpuError }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<ThreeCtx | null>(null);
  const sceneRef = useRef(scene);
  useLayoutEffect(() => {
    sceneRef.current = scene;
  }, [scene]);

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

    const ringN = 160;
    // WebGPURenderer does not support LineLoop; use Line strip with repeated first vertex to close the circle.
    const ringPos = new Float32Array((ringN + 1) * 3);
    for (let i = 0; i <= ringN; i++) {
      const t = (i / ringN) * Math.PI * 2;
      ringPos[i * 3] = Math.cos(t);
      ringPos[i * 3 + 1] = Math.sin(t);
      ringPos[i * 3 + 2] = 0;
    }
    const ringGeom = new BufferGeometry();
    ringGeom.setAttribute("position", new Float32BufferAttribute(ringPos, 3));
    scene3.add(new Line(ringGeom, new LineBasicMaterial({ color: 0x5a5a62 })));

    const gridMat = new LineBasicMaterial({ color: 0x3a3a42, opacity: 0.65, transparent: true });
    const g1 = new BufferGeometry().setFromPoints([
      { x: -1.05, y: 0, z: 0 },
      { x: 1.05, y: 0, z: 0 },
    ]);
    const g2 = new BufferGeometry().setFromPoints([
      { x: 0, y: -1.05, z: 0 },
      { x: 0, y: 1.05, z: 0 },
    ]);
    scene3.add(new LineSegments(g1, gridMat));
    scene3.add(new LineSegments(g2, gridMat));

    const ctx: ThreeCtx = {
      renderer,
      scene3,
      camera,
      host,
      lineBoth: null,
      lineOne: null,
      ptsOther: null,
      ptsSeed: null,
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
    };

    void (async () => {
      try {
        await renderer.init();
        if (disposed) return;
        host.appendChild(renderer.domElement);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        resize();
        applyBuffers(ctx, sceneRef.current);
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
      applyBuffers(ctx, null);
      renderer.dispose();
      ctxRef.current = null;
    };
  }, [webGpuError]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || webGpuError) return;
    applyBuffers(ctx, scene);
  }, [scene, webGpuError]);

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
