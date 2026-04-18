/**
 * Inward nudge for focus `z0` when |z0| is extremely close to 1 (Blaschke denominator).
 * The notebook uses 0.998, but many real vertices (e.g. CATSPER1 ~0.99966) lie above that;
 * nudging `z0` while leaving graph coords at the raw rim point breaks `T(z_focus)=0`, so the
 * focus no longer sits on the crosshair. Use a much tighter margin so typical rim genes still
 * use the true disk coordinate as the Möbius center.
 */
export const R_SAFE = 0.999999;

/** Samples per Poincaré geodesic polyline. */
export const GEODESIC_N = 96;

/**
 * Max non-seed–non-seed edges drawn when “show all edges” (Shift+U) is on.
 * Full graphs can have 500k+ such edges; without a cap the main thread allocates ~1GB+ and hangs.
 */
export const BACKGROUND_NONSEED_EDGE_MAX = 25_000;

/** Skip drawing edges if an endpoint has |z| >= this in the original disk (notebook uses 0.999). */
export const EDGE_Z_BOUND = 0.999;

/** Default rim margin: hide nodes with |W| > 1 - RIM_CULL_EPS after Möbius (display-only). */
export const RIM_CULL_EPS = 0;

/** Upper bound for the rim-cull slider in the UI (larger = more aggressive hiding). */
export const RIM_CULL_EPS_SLIDER_MAX = 0.04;
