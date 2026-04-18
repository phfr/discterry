declare module "three/addons/lines/webgpu/LineSegments2.js" {
  import type { Mesh } from "three/webgpu";

  /** Wide line segments for WebGPU (`Line2NodeMaterial`). */
  export const LineSegments2: new (geometry?: object, material?: object) => Mesh & { readonly isLineSegments2: true };
}

declare module "three/addons/lines/LineSegmentsGeometry.js" {
  import type { InstancedBufferGeometry } from "three/webgpu";

  export class LineSegmentsGeometry extends InstancedBufferGeometry {
    setPositions(array: Float32Array | number[]): this;
  }
}
