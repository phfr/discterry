import type { GraphBundle3d } from "./data/loadBundle3d";
import { clampP0 } from "./math/poincareBall";

export type Vec3 = { x: number; y: number; z: number };

export function p0FromProtein(bundle: GraphBundle3d, name: string): Vec3 {
  const j = bundle.nameToIndex.get(name.trim());
  if (j === undefined) throw new Error(`Unknown protein: ${name.trim()}`);
  const [x, y, z] = clampP0(bundle.x[j], bundle.y[j], bundle.z[j]);
  return { x, y, z };
}
