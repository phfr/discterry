import { tableFromIPC, type Table } from "apache-arrow";
import initWasm, { readParquet } from "parquet-wasm";

export type GraphBundle = {
  vertex: string[];
  x: Float32Array;
  y: Float32Array;
  src: Int32Array;
  dst: Int32Array;
  nameToIndex: Map<string, number>;
};

let wasmReady = false;

async function ensureParquetWasm(): Promise<void> {
  if (wasmReady) return;
  await initWasm();
  wasmReady = true;
}

function colStrings(t: Table, name: string): string[] {
  const v = t.getChild(name);
  if (!v) throw new Error(`missing column ${name}`);
  const out: string[] = [];
  for (let i = 0; i < v.length; i++) out.push(String(v.get(i)));
  return out;
}

function colFloat32(t: Table, name: string): Float32Array {
  const v = t.getChild(name);
  if (!v) throw new Error(`missing column ${name}`);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = Number(v.get(i));
  return out;
}

function colInt32(t: Table, name: string): Int32Array {
  const v = t.getChild(name);
  if (!v) throw new Error(`missing column ${name}`);
  const out = new Int32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = Number(v.get(i));
  return out;
}

export async function loadGraphBundle(baseUrl = ""): Promise<GraphBundle> {
  await ensureParquetWasm();
  const root = baseUrl.replace(/\/$/, "");

  const [nodesBuf, edgesBuf] = await Promise.all([
    fetch(`${root}/data/nodes.parquet`).then((r) => {
      if (!r.ok) throw new Error(`nodes.parquet: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(`${root}/data/edges.parquet`).then((r) => {
      if (!r.ok) throw new Error(`edges.parquet: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);

  const nodesWasm = readParquet(new Uint8Array(nodesBuf));
  const edgesWasm = readParquet(new Uint8Array(edgesBuf));
  const nodesTable = tableFromIPC(nodesWasm.intoIPCStream());
  const edgesTable = tableFromIPC(edgesWasm.intoIPCStream());

  const vertex = colStrings(nodesTable, "vertex");
  const x = colFloat32(nodesTable, "x");
  const y = colFloat32(nodesTable, "y");
  const src = colInt32(edgesTable, "src");
  const dst = colInt32(edgesTable, "dst");

  const nameToIndex = new Map<string, number>();
  for (let i = 0; i < vertex.length; i++) {
    nameToIndex.set(String(vertex[i]).trim(), i);
  }

  return { vertex, x, y, src, dst, nameToIndex };
}

export type MetaJson = {
  run_subdir?: string;
  default_focus?: string;
  default_seeds?: string[];
};

export async function loadMeta(baseUrl = ""): Promise<MetaJson | null> {
  const root = baseUrl.replace(/\/$/, "");
  const r = await fetch(`${root}/data/meta.json`);
  if (!r.ok) return null;
  return (await r.json()) as MetaJson;
}
