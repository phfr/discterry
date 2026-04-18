/** Match `02d_disk_focus_mobius.ipynb`: inward nudge when |z0| is near the boundary. */
export const R_SAFE = 0.998;

/** Samples per Poincaré geodesic polyline. */
export const GEODESIC_N = 96;

/** Skip drawing edges if an endpoint has |z| >= this in the original disk (notebook uses 0.999). */
export const EDGE_Z_BOUND = 0.999;

/** Default rim margin: hide nodes with |W| > 1 - RIM_CULL_EPS after Möbius (display-only). */
export const RIM_CULL_EPS = 0.008;

/** Upper bound for the rim-cull slider in the UI (larger = more aggressive hiding). */
export const RIM_CULL_EPS_SLIDER_MAX = 0.04;
