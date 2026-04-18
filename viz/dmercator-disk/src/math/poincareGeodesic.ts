/** Poincaré disk ↔ upper half-plane and geodesic sampling (ported from `02d_disk_focus_mobius.ipynb`). */

function diskToUpper(zre: number, zim: number): { re: number; im: number } {
  const dr = 1 - zre;
  const di = -zim;
  const d2 = dr * dr + di * di;
  const invr = dr / d2;
  const invi = -di / d2;
  const onePr = 1 + zre;
  const onePi = zim;
  const numr = -(onePi * invr + onePr * invi);
  const numi = onePr * invr - onePi * invi;
  return { re: numr, im: numi };
}

function upperToDisk(wre: number, wim: number): { re: number; im: number } {
  const numr = wre;
  const numi = wim - 1;
  const denr = wre;
  const deni = wim + 1;
  const d2 = denr * denr + deni * deni;
  if (d2 < 1e-20) return { re: 0, im: 0 };
  return {
    re: (numr * denr + numi * deni) / d2,
    im: (numi * denr - numr * deni) / d2,
  };
}

function halfplaneGeodesicPoints(
  u1r: number,
  u1i: number,
  u2r: number,
  u2i: number,
  n: number,
  outR: Float32Array,
  outI: Float32Array,
): void {
  const d = Math.hypot(u1r - u2r, u1i - u2i);
  if (d < 1e-14) {
    for (let k = 0; k < n; k++) {
      outR[k] = u1r;
      outI[k] = u1i;
    }
    return;
  }
  const scale = 1e-12 * (1 + Math.abs(u1i) + Math.abs(u2i));
  if (Math.abs(u1r - u2r) < scale) {
    for (let k = 0; k < n; k++) {
      const t = k / (n - 1 || 1);
      outR[k] = u1r;
      outI[k] = u1i + t * (u2i - u1i);
    }
    return;
  }
  const denom = 2 * (u2r - u1r);
  const cr = ((u2r * u2r + u2i * u2i) - (u1r * u1r + u1i * u1i)) / denom;
  const ci = 0;
  const r = Math.hypot(u1r - cr, u1i - ci);
  const t1 = Math.atan2(u1i - ci, u1r - cr);
  const t2 = Math.atan2(u2i - ci, u2r - cr);
  const deltaShort = ((((t2 - t1 + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const pickDelta = (): number => {
    if (Math.abs(deltaShort) < 1e-14) return deltaShort;
    const deltaLong = deltaShort - Math.sign(deltaShort) * 2 * Math.PI;
    for (const delta of [deltaShort, deltaLong]) {
      let ok = true;
      for (let k = 0; k < n; k++) {
        const ang = t1 + (k / (n - 1)) * delta;
        const im = ci + r * Math.sin(ang);
        if (im < -1e-10) {
          ok = false;
          break;
        }
      }
      if (ok) return delta;
    }
    return deltaShort;
  };
  const delta = Math.abs(deltaShort) < 1e-14 ? deltaShort : pickDelta();
  for (let k = 0; k < n; k++) {
    const ang = t1 + (k / (n - 1 || 1)) * delta;
    outR[k] = cr + r * Math.cos(ang);
    outI[k] = ci + r * Math.sin(ang);
  }
}

/** Cartesian samples of the Poincaré-disk geodesic between z1 and z2 (disk coords). */
export function poincareGeodesicXY(
  z1re: number,
  z1im: number,
  z2re: number,
  z2im: number,
  n: number,
  gx: Float32Array,
  gy: Float32Array,
): void {
  const u1 = diskToUpper(z1re, z1im);
  const u2 = diskToUpper(z2re, z2im);
  const hr = new Float32Array(n);
  const hi = new Float32Array(n);
  halfplaneGeodesicPoints(u1.re, u1.im, u2.re, u2.im, n, hr, hi);
  for (let k = 0; k < n; k++) {
    const d = upperToDisk(hr[k], hi[k]);
    gx[k] = d.re;
    gy[k] = d.im;
  }
}
