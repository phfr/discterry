/** Edge indices touching the seed set (>=1 seed endpoint), split into both-seed vs one-seed. */

export function classifySeedEdges(
  src: Int32Array,
  dst: Int32Array,
  isSeed: Uint8Array,
): { both: number[]; one: number[] } {
  const both: number[] = [];
  const one: number[] = [];
  for (let ei = 0; ei < src.length; ei++) {
    const a = src[ei];
    const b = dst[ei];
    const sa = isSeed[a];
    const sb = isSeed[b];
    if (!sa && !sb) continue;
    if (sa && sb) both.push(ei);
    else one.push(ei);
  }
  return { both, one };
}

export function buildSeedMask(n: number, seedNames: Set<string>, nameToIndex: Map<string, number>): Uint8Array {
  const m = new Uint8Array(n);
  for (const raw of seedNames) {
    const j = nameToIndex.get(raw.trim());
    if (j !== undefined) m[j] = 1;
  }
  return m;
}
