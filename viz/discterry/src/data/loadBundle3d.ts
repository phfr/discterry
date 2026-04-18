import { tableFromIPC, type Table } from "apache-arrow";
import initWasm, { readParquet } from "parquet-wasm";

export type GraphBundle3d = {
  vertex: string[];
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  infKappa: Float32Array | null;
  infHypRad: Float32Array | null;
  infPos1: Float32Array | null;
  infPos2: Float32Array | null;
  infPos3: Float32Array | null;
  infPos4: Float32Array | null;
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

function colFloat32Optional(t: Table, name: string, expectedLen: number): Float32Array | null {
  const v = t.getChild(name);
  if (!v || v.length !== expectedLen) return null;
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

export async function loadGraphBundle3d(baseUrl = ""): Promise<GraphBundle3d> {
  await ensureParquetWasm();
  const root = baseUrl.replace(/\/$/, "");

  const [nodesBuf, edgesBuf] = await Promise.all([
    fetch(`${root}/data3d/nodes.parquet`).then((r) => {
      if (!r.ok) throw new Error(`data3d/nodes.parquet: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(`${root}/data3d/edges.parquet`).then((r) => {
      if (!r.ok) throw new Error(`data3d/edges.parquet: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);

  const nodesWasm = readParquet(new Uint8Array(nodesBuf));
  const edgesWasm = readParquet(new Uint8Array(edgesBuf));
  const nodesTable = tableFromIPC(nodesWasm.intoIPCStream());
  const edgesTable = tableFromIPC(edgesWasm.intoIPCStream());

  const vertex = colStrings(nodesTable, "vertex");
  const n = vertex.length;
  const x = colFloat32(nodesTable, "x");
  const y = colFloat32(nodesTable, "y");
  const z = colFloat32(nodesTable, "z");
  const infKappa = colFloat32Optional(nodesTable, "inf_kappa", n);
  const infHypRad = colFloat32Optional(nodesTable, "inf_hyp_rad", n);
  const infPos1 = colFloat32Optional(nodesTable, "inf_pos_1", n);
  const infPos2 = colFloat32Optional(nodesTable, "inf_pos_2", n);
  const infPos3 = colFloat32Optional(nodesTable, "inf_pos_3", n);
  const infPos4 = colFloat32Optional(nodesTable, "inf_pos_4", n);
  const src = colInt32(edgesTable, "src");
  const dst = colInt32(edgesTable, "dst");

  const nameToIndex = new Map<string, number>();
  for (let i = 0; i < vertex.length; i++) {
    nameToIndex.set(String(vertex[i]).trim(), i);
  }

  return {
    vertex,
    x,
    y,
    z,
    infKappa,
    infHypRad,
    infPos1,
    infPos2,
    infPos3,
    infPos4,
    src,
    dst,
    nameToIndex,
  };
}

export async function loadMeta3d(baseUrl = ""): Promise<import("./loadBundle").MetaJson | null> {
  const root = baseUrl.replace(/\/$/, "");
  const r = await fetch(`${root}/data3d/meta.json`);
  if (!r.ok) return null;
  return (await r.json()) as import("./loadBundle").MetaJson;
}
