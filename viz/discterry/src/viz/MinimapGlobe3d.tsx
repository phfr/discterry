import { useCallback, useEffect, useRef } from "react";
import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Raycaster,
  Scene,
  SphereGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { GraphBundle3d } from "../data/loadBundle3d";
import { normalizeBallDir } from "./minimapChart";

const MINIMAP_VERTEX_STRIDE_THRESHOLD = 25_000;
/** Draw nodes slightly outside the wireframe unit sphere to avoid z-fighting with rim lines. */
const POINTS_RADIUS = 1.035;
const WIREFRAME_RADIUS = 0.992;
/** Target point diameter in CSS pixels (scaled by DPR in resize). */
const POINT_PIXEL_BASE = 1.75;
const POINT_OPACITY = 0.38;

export type MinimapGlobe3dProps = {
  graph3d: GraphBundle3d;
  seeds: Set<string>;
  focusIndicatorName: string;
  onPickFocus: (name: string, opts?: { skipAnimation?: boolean; source?: "minimap" }) => void;
};

function strideForN(n: number): number {
  return n > MINIMAP_VERTEX_STRIDE_THRESHOLD ? Math.ceil(n / MINIMAP_VERTEX_STRIDE_THRESHOLD) : 1;
}

function fillPointColors(
  vertexIndices: readonly number[],
  vertex: readonly string[],
  seeds: Set<string>,
  focusName: string,
  out: Float32Array,
): void {
  const focus = focusName.trim();
  const cGreen = new Color(0.27, 0.78, 0.47);
  const cSeed = new Color(1, 0.22, 0.22);
  const cFocus = new Color(0.95, 0.95, 0.95);
  for (let k = 0; k < vertexIndices.length; k++) {
    const vi = vertexIndices[k]!;
    const name = vertex[vi] ?? "";
    const c = name === focus ? cFocus : seeds.has(name) ? cSeed : cGreen;
    out[k * 3] = c.r;
    out[k * 3 + 1] = c.g;
    out[k * 3 + 2] = c.b;
  }
}

export function MinimapGlobe3d({ graph3d, seeds, focusIndicatorName, onPickFocus }: MinimapGlobe3dProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const pointsRef = useRef<Points | null>(null);
  const vertexIndicesRef = useRef<number[]>([]);

  const updatePointColors = useCallback(() => {
    const pts = pointsRef.current;
    const geom = pts?.geometry;
    if (!geom) return;
    const attr = geom.getAttribute("color") as Float32BufferAttribute | undefined;
    const idx = vertexIndicesRef.current;
    if (!attr || idx.length === 0) return;
    const arr = attr.array as Float32Array;
    fillPointColors(idx, graph3d.vertex, seeds, focusIndicatorName, arr);
    attr.needsUpdate = true;
  }, [graph3d.vertex, seeds, focusIndicatorName]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;

    const scene = new Scene();
    const camera = new PerspectiveCamera(42, 1, 0.05, 100);
    camera.position.set(0.35, 0.55, 2.15);
    camera.lookAt(0, 0, 0);

    const renderer = new WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x0e0e10, 1);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.touchAction = "none";
    wrap.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.9;
    controls.minDistance = 1.25;
    controls.maxDistance = 5;

    const sphereWire = new Mesh(
      new SphereGeometry(WIREFRAME_RADIUS, 32, 24),
      new MeshBasicMaterial({
        color: 0x8890aa,
        wireframe: true,
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
      }),
    );
    sphereWire.renderOrder = 0;
    scene.add(sphereWire);

    const n = graph3d.vertex.length;
    const stride = strideForN(n);
    const idx: number[] = [];
    for (let i = 0; i < n; i += stride) {
      idx.push(i);
    }
    vertexIndicesRef.current = idx;

    const positions = new Float32Array(idx.length * 3);
    const colors = new Float32Array(idx.length * 3);
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k]!;
      const [nx, ny, nz] = normalizeBallDir(graph3d.x[i]!, graph3d.y[i]!, graph3d.z[i]!);
      positions[k * 3] = nx * POINTS_RADIUS;
      positions[k * 3 + 1] = ny * POINTS_RADIUS;
      positions[k * 3 + 2] = nz * POINTS_RADIUS;
    }
    fillPointColors(idx, graph3d.vertex, seeds, focusIndicatorName, colors);

    const geom = new BufferGeometry();
    geom.setAttribute("position", new Float32BufferAttribute(positions, 3));
    geom.setAttribute("color", new Float32BufferAttribute(colors, 3));
    geom.computeBoundingSphere();

    /* Pixel-sized sprites: world-unit + attenuation often shrinks gl_PointSize below 1px here. */
    const mat = new PointsMaterial({
      vertexColors: true,
      size: POINT_PIXEL_BASE,
      sizeAttenuation: false,
      transparent: true,
      opacity: POINT_OPACITY,
      depthTest: true,
      depthWrite: false,
    });
    const points = new Points(geom, mat);
    points.frustumCulled = false;
    points.renderOrder = 2;
    pointsRef.current = points;
    scene.add(points);

    const raycaster = new Raycaster();
    raycaster.params.Points = { threshold: 0.045 };

    let raf = 0;
    let disposed = false;

    const resize = () => {
      const w = Math.max(1, wrap.clientWidth);
      const h = Math.max(1, wrap.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      const pr = Math.min(window.devicePixelRatio ?? 1, 2);
      renderer.setPixelRatio(pr);
      renderer.setSize(w, h, false);
      /* Point size is in framebuffer pixels; keep dots small (~1.5–2 CSS px). */
      mat.size = Math.max(1, Math.round(POINT_PIXEL_BASE * pr));
    };

    const tick = () => {
      if (disposed) return;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };

    resize();
    const ro = new ResizeObserver(() => resize());
    ro.observe(wrap);
    tick();

    const ndcScratch = new Vector2();
    const vScratch = new Vector3();

    const pickNearestScreen = (clientX: number, clientY: number): number => {
      const rect = renderer.domElement.getBoundingClientRect();
      const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;
      ndcScratch.set(ndcX, ndcY);

      raycaster.setFromCamera(ndcScratch, camera);
      const hits = raycaster.intersectObject(points, false);
      if (hits.length > 0 && hits[0]!.index != null) {
        const k = hits[0]!.index;
        const vi = vertexIndicesRef.current[k];
        if (vi != null) return vi;
      }

      let bestI = -1;
      /* Squared NDC distance; ~0.0018 ≈ 4% of half-width (~6–8px on minimap). */
      let bestD = 0.0018;
      const id = vertexIndicesRef.current;
      for (let k = 0; k < id.length; k++) {
        const i = id[k]!;
        vScratch.set(graph3d.x[i]!, graph3d.y[i]!, graph3d.z[i]!).normalize().multiplyScalar(POINTS_RADIUS);
        vScratch.project(camera);
        const dx = vScratch.x - ndcX;
        const dy = vScratch.y - ndcY;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      return bestI;
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const hit = pickNearestScreen(e.clientX, e.clientY);
      if (hit >= 0) {
        e.stopPropagation();
        onPickFocus(graph3d.vertex[hit]!, { skipAnimation: false, source: "minimap" });
      }
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown, true);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown, true);
      controls.dispose();
      scene.traverse((obj) => {
        if (obj instanceof Points || obj instanceof Mesh) {
          obj.geometry?.dispose();
          const m = obj.material;
          if (Array.isArray(m)) m.forEach((x) => x.dispose());
          else m?.dispose();
        }
      });
      renderer.dispose();
      if (renderer.domElement.parentElement === wrap) {
        wrap.removeChild(renderer.domElement);
      }
      pointsRef.current = null;
      vertexIndicesRef.current = [];
    };
    /* seeds / focusIndicatorName: initial colors in this effect; updates via updatePointColors (avoids WebGL remount each focus). */
  // eslint-disable-next-line react-hooks/exhaustive-deps -- only remount when graph or handler identity changes
  }, [graph3d, onPickFocus]);

  useEffect(() => {
    updatePointColors();
  }, [updatePointColors]);

  return <div ref={wrapRef} className="minimapGlobe3dWrap" />;
}
