import type { Complex } from "./mobius";

/** 2×2 complex Möbius matrix [[a,b],[c,d]] acting as w ↦ (a w + b) / (c w + d). */
export type Mobius2 = {
  a: Complex;
  b: Complex;
  c: Complex;
  d: Complex;
};

const C0: Complex = { re: 0, im: 0 };
const C1: Complex = { re: 1, im: 0 };

function cadd(p: Complex, q: Complex): Complex {
  return { re: p.re + q.re, im: p.im + q.im };
}

function csub(p: Complex, q: Complex): Complex {
  return { re: p.re - q.re, im: p.im - q.im };
}

function cmul(p: Complex, q: Complex): Complex {
  return { re: p.re * q.re - p.im * q.im, im: p.re * q.im + p.im * q.re };
}

function cdiv(p: Complex, q: Complex): Complex {
  const d2 = q.re * q.re + q.im * q.im;
  if (d2 < 1e-30) return C0;
  const invr = q.re / d2;
  const invi = -q.im / d2;
  return {
    re: p.re * invr - p.im * invi,
    im: p.re * invi + p.im * invr,
  };
}

function cconj(p: Complex): Complex {
  return { re: p.re, im: -p.im };
}

/** Determinant ad − bc. */
function det2(M: Mobius2): Complex {
  return csub(cmul(M.a, M.d), cmul(M.b, M.c));
}

/** Scale matrix so det ≈ 1 (avoids drift from floating error). */
function normalizeDeterminant(M: Mobius2): Mobius2 {
  const Δ = det2(M);
  const r = Math.hypot(Δ.re, Δ.im);
  if (r < 1e-20) return identityMobius();
  const invSqrt = 1 / Math.sqrt(r);
  const ang = -0.5 * Math.atan2(Δ.im, Δ.re);
  const sr = invSqrt * Math.cos(ang);
  const si = invSqrt * Math.sin(ang);
  const s: Complex = { re: sr, im: si };
  return {
    a: cmul(s, M.a),
    b: cmul(s, M.b),
    c: cmul(s, M.c),
    d: cmul(s, M.d),
  };
}

export function identityMobius(): Mobius2 {
  return { a: { ...C1 }, b: { ...C0 }, c: { ...C0 }, d: { ...C1 } };
}

/** True if M is (numerically) the identity map. */
export function isIdentityMobius(M: Mobius2, eps = 1e-8): boolean {
  return (
    Math.abs(M.a.re - 1) < eps &&
    Math.abs(M.a.im) < eps &&
    Math.abs(M.b.re) < eps &&
    Math.abs(M.b.im) < eps &&
    Math.abs(M.c.re) < eps &&
    Math.abs(M.c.im) < eps &&
    Math.abs(M.d.re - 1) < eps &&
    Math.abs(M.d.im) < eps
  );
}

/** Compose: w ↦ M2(M1(w)); matrix product M2 · M1. */
export function composeMobius(M2: Mobius2, M1: Mobius2): Mobius2 {
  const out: Mobius2 = {
    a: cadd(cmul(M2.a, M1.a), cmul(M2.b, M1.c)),
    b: cadd(cmul(M2.a, M1.b), cmul(M2.b, M1.d)),
    c: cadd(cmul(M2.c, M1.a), cmul(M2.d, M1.c)),
    d: cadd(cmul(M2.c, M1.b), cmul(M2.d, M1.d)),
  };
  return normalizeDeterminant(out);
}

export function applyMobiusToW(wre: number, wim: number, M: Mobius2): { re: number; im: number } {
  const w: Complex = { re: wre, im: wim };
  const num = cadd(cmul(M.a, w), M.b);
  const den = cadd(cmul(M.c, w), M.d);
  const z = cdiv(num, den);
  return { re: z.re, im: z.im };
}

/**
 * Small disk automorphism close to identity: w' = (w + ε) / (1 + ε̄ w), |ε| small,
 * normalized to det 1 (SU(1,1) form with δ = conj(α), γ = conj(β)).
 */
function mobiusSmallTranslate(eps: Complex): Mobius2 {
  const e2 = eps.re * eps.re + eps.im * eps.im;
  const k = Math.sqrt(Math.max(1e-10, 1 - e2));
  const a: Complex = { re: 1 / k, im: 0 };
  const b: Complex = { re: eps.re / k, im: eps.im / k };
  return { a, b, c: cconj(b), d: cconj(a) };
}

/** Max |ε| per drag sample to stay inside disk and stable. */
const EPS_DRAG_CAP = 0.06;
/** Scale: screen pixels → disk ε (tune for “natural” pan speed). */
const DRAG_TO_DISK = 0.0028;

/**
 * Increment accumulated viewer Möbius from Shift+drag pixel delta (screen coords).
 * Composes on the left: M ← M_delta · M (new motion after current view).
 */
export function incrementMobiusFromDrag(M: Mobius2, dxPx: number, dyPx: number, canvasW: number, canvasH: number): Mobius2 {
  const w = Math.max(1, canvasW);
  const h = Math.max(1, canvasH);
  let er = DRAG_TO_DISK * dxPx * (2 / w);
  let ei = -DRAG_TO_DISK * dyPx * (2 / h);
  const m = Math.hypot(er, ei);
  if (m > EPS_DRAG_CAP) {
    const s = EPS_DRAG_CAP / m;
    er *= s;
    ei *= s;
  }
  const Mdelta = mobiusSmallTranslate({ re: er, im: ei });
  return composeMobius(Mdelta, M);
}
