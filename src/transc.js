// Exact model for the "Fast math" widgets (bbe-transc-*), and for
// scripts/check-transc.mjs.  Pure functions, no DOM, importable under node.
//
// ex2Emu() is a bit-exact JS emulation of FlashAttention 4's software 2^x
// (flash_attn/cute/utils.py, Dao-AILab/flash-attention, BSD):
//
//   ex2_emulation(x, poly_degree)      Horner as out = out*x + c: mul.f32 then
//                                      add.f32, TWO roundings per step
//                                      (evaluate_polynomial)
//   ex2_emulation_2(x, y, poly_degree) Horner as fma_packed_f32x2: ONE
//                                      rounding per step (evaluate_polynomial_2);
//                                      this is the one FA4's softmax calls,
//                                      with the default poly_degree = 3
//
// Every step below is one PTX instruction, emulated exactly:
//   fmax.f32                    exact (no rounding)
//   add.rm.ftz.f32 x, 1.5*2^23  see addRoundDown below
//   sub.f32 (rn)                Math.fround(a - b): correctly rounded
//   mul.f32 / add.f32 (rn)      Math.fround(a * b), Math.fround(a + b)
//   fma.rn.f32                  fma32 from fp.js (BigInt, one rounding)
//   shl.b32 / add.s32           32-bit integer ops on the bit pattern
import { f32, bits32, fromBits32, fma32, mulberry32 } from './fp.js';

// Sollya fpminimax(2^x, d, [|1,24...|], [0;1], relative): every coefficient
// has 24 significant bits, so each is exactly an fp32 (checked in
// scripts/check-transc.mjs).  POLY_EX2[d][i] multiplies f^i.
export const POLY_EX2 = {
  1: [1.0, 0.922497093677520751953125],
  2: [1.0, 0.6657850742340087890625, 0.330107033252716064453125],
  3: [1.0, 0.695146143436431884765625, 0.227564394474029541015625, 0.077119089663028717041015625],
  4: [1.0, 0.693042695522308349609375, 0.2412912547588348388671875, 5.2225358784198760986328125e-2,
    1.3434938155114650726318359375e-2],
  5: [1.0, 0.693151414394378662109375, 0.24016360938549041748046875, 5.5802188813686370849609375e-2,
    9.01452265679836273193359375e-3, 1.86810153536498546600341796875e-3],
};

export const ROUND_INT = 2 ** 23 + 2 ** 22;   // 1.5 * 2^23 = 12582912, bits 0x4B400000

// add.rm.f32(x, 1.5*2^23) for fp32 x in [-127, 127].  The exact sum lies in
// [2^23 + 2^22 - 127, 2^23 + 2^22 + 128) which is inside [2^23, 2^24), where
// consecutive fp32 values are exactly 1 apart (the integers).  Rounding toward
// minus infinity there is therefore floor(), and since 1.5*2^23 is an integer,
// floor(x + 1.5*2^23) = floor(x) + 1.5*2^23, computed exactly in doubles.
export function addRoundDown(x) {
  if (!(x >= -127 && x <= 127)) throw new RangeError('addRoundDown: x outside [-127, 127]');
  return Math.floor(x) + ROUND_INT;
}

// Horner: out = c[d]; for i = d-1 .. 0: out = out*f + c[i]
export function horner(f, deg, fused) {
  const c = POLY_EX2[deg];
  let out = c[deg];
  for (let i = deg - 1; i >= 0; i--) out = fused ? fma32(out, f, c[i]) : f32(f32(out * f) + c[i]);
  return out;
}

// FA4's 2^x, every intermediate returned for the bits picture.
// x: an fp32 value <= 127.  fused: true = ex2_emulation_2 (fma Horner).
export function ex2Emu(x, deg = 3, fused = true) {
  x = f32(x);
  const xc = Math.max(x, -127);                 // fmax(x, -127.0)
  const xr = addRoundDown(xc);                  // floor(x) now in the low significand bits
  const n = f32(xr - ROUND_INT);                // x_rounded_back = floor(x), exact
  const frac = f32(xc - n);                     // x_frac: rounds only for some x in (-1, 0)
  const poly = horner(frac, deg, fused);        // ~2^frac in [1, 2]
  // combine_int_frac_ex2: shl.b32 by 23, then add.s32 onto poly's bit pattern.
  // The low 9 bits of bits(xr) are floor(x) mod 512 (0x4B400000 has zeros
  // there), so the shift drops floor(x) into the exponent field.
  const shifted = (bits32(xr) << 23) >>> 0;
  const outBits = (shifted + bits32(poly)) >>> 0;
  return { x, xc, xr, n, frac, poly, shifted, outBits, out: fromBits32(outBits) };
}
export const ex2 = (x, deg = 3, fused = true) => ex2Emu(x, deg, fused).out;

// Reference: 2^x in double (V8's pow, error well under one double ulp), then
// rounded once to fp32.  Since a double carries 29 more bits than an fp32,
// this is the correctly rounded fp32 2^x unless 2^x falls within ~2^-52
// (relative) of a midpoint between two fp32 values; check-transc.mjs scans
// the plotted grid for such near-ties and finds none.
export const ref2 = (x) => f32(2 ** x);

// signed distance between two positive fp32 values, in units in the last
// place, counted as the difference of their bit patterns (so it stays an
// integer when the two straddle a power of two)
export const ulpsApart = (y, r) => bits32(y) - bits32(r);

// PTX ISA: ex2.approx.f32 max relative error 2^-22.5 (primary range).  The
// fp32 values that satisfy |y - 2^x| <= 2^-22.5 * 2^x, as an integer range of
// ulps relative to ref2(x).  This is the "permissible set" for one x.
export const EX2_REL = 2 ** -22.5;
export function permissible(x, rel = EX2_REL) {
  const t = 2 ** x, r = ref2(x), rb = bits32(r);
  const lo = t * (1 - rel), hi = t * (1 + rel);
  let a = bits32(f32(lo)), b = bits32(f32(hi));
  if (fromBits32(a) < lo) a++;                 // smallest fp32 >= lo
  if (fromBits32(b) > hi) b--;                 // largest fp32 <= hi
  return [a - rb, b - rb];
}

// Error of the emulation over a uniform grid of fp32 x in [x0, x1).
export function errorScan(deg, fused, N = 1024, x0 = -1, x1 = 1) {
  const pts = [];
  for (let k = 0; k < N; k++) {
    const x = f32(x0 + (x1 - x0) * (k + 0.5) / N);
    pts.push({ x, u: ulpsApart(ex2(x, deg, fused), ref2(x)) });
  }
  return pts;
}

// ---- division: three PTX instructions that all mean x / y -----------------

// fp32 neighbours: step by k ulps (positive values only)
export const stepUlps = (v, k) => fromBits32(bits32(v) + k);

// The fp32 values within `u` ulps of the exact real t > 0, where an ulp is
// measured at t (the spacing of the binade containing t).  Exact for the
// inputs used (t = x / y computed in double, see divSets).
function withinUlps(t, u) {
  const r = f32(t), e = Math.floor(Math.log2(t)), ulp = 2 ** (Math.max(e, -126) - 23);
  const out = [];
  for (let k = -4 * u - 2; k <= 4 * u + 2; k++) {
    const v = stepUlps(r, k);
    if (Math.abs(v - t) <= u * ulp) out.push(v);
  }
  return out;
}

// For fp32 x, y > 0: the documented result sets.
//   div.rn.f32      IEEE, correctly rounded: one value
//   div.full.f32    max 2 ulp error: every fp32 within 2 ulp of x/y
//   rcp.approx.f32  max 1 ulp error, then mul.f32 rounds x * r:
//                   { fround(x * r) : r within 1 ulp of 1/y }
// x/y and 1/y are computed in double: these sets only need comparisons of
// the exact quotient against fp32 grid points 1 or 2 ulps away, and a double
// quotient is within 2^-53 relative of exact, far below that spacing.
export function divSets(x, y) {
  const t = x / y;
  const rn = f32(t);
  const full = withinUlps(t, 2);
  const rcps = withinUlps(1 / y, 1);
  const pairs = rcps.map((r) => [r, f32(x * r)]);           // each allowed r and the x*r it gives
  const viaRcp = [...new Set(pairs.map(([, q]) => q))].sort((a, b) => a - b);
  return { t, rn, full, rcps, pairs, viaRcp, rcpRn: f32(1 / y) };
}

// ---- numbers shown by bbe-transc-err ---------------------------------------
// The inputs: 2^20 fp32 values, uniform in [-1, 1], from a fixed seed.
export function statInputs(n = 1 << 20, seed = 2025) {
  const rnd = mulberry32(seed), xs = new Float32Array(n);
  for (let k = 0; k < n; k++) xs[k] = f32(-1 + 2 * rnd());
  return xs;
}
// Per variant ('3f' = degree 3, fma Horner; '3m' = degree 3, mul+add): ulps
// from the correctly rounded 2^x (min lo, max hi), and counts of inputs (out
// of STATS_N) bitwise equal to it (same), outside the ex2.approx bound
// (outside), whose bf16 rounding differs from bf16 of the correctly rounded
// value (bf16), and that differ from the other Horner variant of the same
// degree (vsOther).  Recomputed and asserted by scripts/check-transc.mjs.
export const STATS_N = 1 << 20;
export const STATS = {
  '3f': { lo: -1429, hi: 1268, same: 318, outside: 1047238, bf16: 10429, vsOther: 173115 },
  '3m': { lo: -1430, hi: 1268, same: 338, outside: 1047248, bf16: 10431, vsOther: 173115 },
  '5f': { lo: -2, hi: 2, same: 413515, outside: 0, bf16: 5, vsOther: 173947 },
  '5m': { lo: -3, hi: 3, same: 416579, outside: 8, bf16: 7, vsOther: 173947 },
};

export const DIV_PRESETS = [[1, 3], [3, 7], [7, 10]];
