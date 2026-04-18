import type { GraphBundle } from "./data/loadBundle";
import { clampZ0, type Complex } from "./math/mobius";

export function z0FromProtein(bundle: GraphBundle, name: string): Complex {
  const j = bundle.nameToIndex.get(name.trim());
  if (j === undefined) throw new Error(`Unknown protein: ${name.trim()}`);
  return clampZ0({ re: bundle.x[j], im: bundle.y[j] });
}
