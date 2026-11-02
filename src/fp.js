// Exact floating-point helpers shared by the group-A widgets and by
// scripts/check-fp.mjs.  Pure functions, no DOM, importable under node.
//
// Conventions: an fp32 value is a JS number that satisfies
// Math.fround(x) === x; a bf16 value is an fp32 value whose low 16 bits are 0.
// Exact values are carried as { m: BigInt, e: int } meaning m * 2^e.

const F = new Float32Array(1), U = new Uint32Array(F.buffer);

export const f32 = Math.fround;
export function bits32(x) { F[0] = x; return U[0]; }
export function fromBits32(u) { U[0] = u >>> 0; return F[0]; }
export const hex32 = (x) => '0x' + bits32(x).toString(16).toUpperCase().padStart(8, '0');
export const hex16 = (x) => '0x' + (bits32(x) >>> 16).toString(16).toUpperCase().padStart(4, '0');

// fp32 add / mul: one double op is exact enough that fround of it is the
// correctly rounded fp32 result (53 >= 2*24 + 2).
export const add32 = (a, b) => f32(a + b);
export const mul32 = (a, b) => f32(a * b);
// bf16 + bf16 rounded once to bf16 (8 significand bits, min normal 2^-126)
export const addBf16 = (a, b) => roundExact(exactSum([exact32(a), exact32(b)]), 8, -133);

// fp32 -> bf16, round to nearest even, by bit manipulation (what torch does).
export function bf16(x) {
  let u = bits32(f32(x));
  if ((u & 0x7f800000) === 0x7f800000 && (u & 0x7fffff)) return NaN;
  u = (u + 0x7fff + ((u >>> 16) & 1)) >>> 0;
  return fromBits32(u & 0xffff0000);
}

// ---- exact values ---------------------------------------------------------

// fp32 (or bf16) value -> exact { m, e }.  Zero gives m = 0n.
export function exact32(x) {
  const u = bits32(x), s = u >>> 31, ex = (u >>> 23) & 0xff, fr = u & 0x7fffff;
  let m = BigInt(ex ? fr | 0x800000 : fr);
  if (s) m = -m;
  return { m, e: ex ? ex - 150 : -149 };
}
// exact sum of a list of { m, e }
export function exactSum(list) {
  const nz = list.filter((v) => v.m !== 0n);
  if (!nz.length) return { m: 0n, e: 0 };
  const e = Math.min(...nz.map((v) => v.e));
  let m = 0n;
  for (const v of nz) m += v.m << BigInt(v.e - e);
  return { m, e };
}
export const exactMul = (a, b) => ({ m: a.m * b.m, e: a.e + b.e });
const bitlen = (m) => (m === 0n ? 0 : m.toString(2).length);
const absB = (m) => (m < 0n ? -m : m);

// Round an exact value to a binary format with `p` significand bits and
// minimum quantum 2^qmin (subnormals), mode 'rne' or 'rz'.  Returns a double
// (always exactly representable; the caller frounds to catch overflow).
export function roundExact({ m, e }, p = 24, qmin = -149, mode = 'rne') {
  if (m === 0n) return 0;
  const neg = m < 0n; let a = absB(m);
  const top = bitlen(a) - 1 + e;                 // exponent of the leading bit
  const q = Math.max(top - p + 1, qmin);         // exponent of one ulp
  const sh = q - e;
  let r;
  if (sh <= 0) r = a << BigInt(-sh);
  else {
    r = a >> BigInt(sh);
    const rem = a - (r << BigInt(sh)), half = 1n << BigInt(sh - 1);
    if (mode === 'rne' && (rem > half || (rem === half && (r & 1n)))) r += 1n;
  }
  const v = Number(r) * 2 ** q;
  return neg ? -v : v;
}
export const roundExactF32 = (v, mode = 'rne') => f32(roundExact(v, 24, -149, mode));
export const exactToNumber = ({ m, e }) => Number(m) * 2 ** e;   // display only

// Decimal string of an exact value, `digits` significant digits (for display).
export function exactDecimal({ m, e }, digits = 12) {
  if (m === 0n) return '0';
  const neg = m < 0n, a = absB(m);
  const num = e >= 0 ? a << BigInt(e) : a, den = e >= 0 ? 1n : 1n << BigInt(-e);
  // n = round(value * 10^k), half up
  const at = (k) => {
    const [p, q] = k >= 0 ? [num * 10n ** BigInt(k), den] : [num, den * 10n ** BigInt(-k)];
    return (2n * p + q) / (2n * q);
  };
  let k = digits - 1 - Math.floor(Math.log10(exactToNumber({ m: a, e })));
  let n = at(k);
  if (n >= 10n ** BigInt(digits)) n = at(--k);
  else if (n < 10n ** BigInt(digits - 1)) n = at(++k);
  let s = n.toString();
  if (k <= 0) s += '0'.repeat(-k);
  else if (k >= s.length) s = '0.' + '0'.repeat(k - s.length) + s;
  else s = s.slice(0, s.length - k) + '.' + s.slice(s.length - k);
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return (neg ? '-' : '') + s;
}

// Shortest decimal that rounds back to the same fp32 value.
export function fmt32(x, maxDigits = 9) {
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  for (let p = 1; p <= maxDigits; p++) {
    const s = x.toPrecision(p);
    if (f32(parseFloat(s)) === x) return String(parseFloat(s)).replace('e+', 'e');
  }
  return String(x);
}

// ---- fused multiply-add ---------------------------------------------------

// fp32 fma(a, b, c) = round(a*b + c), ONE rounding, via exact BigInt.
export function fma32(a, b, c) {
  const v = exactSum([exactMul(exact32(a), exact32(b)), exact32(c)]);
  if (v.m === 0n) return (a * b < 0 || Object.is(a * b, -0)) && Object.is(c, -0) ? -0 : 0;
  return roundExactF32(v);
}

// ---- tensor-core block FMA model -------------------------------------------
// Khattak & Mikaitis, "Accurate Models of NVIDIA Tensor Cores"
// (arXiv:2512.07004), bf16 inputs, fp32 accumulator, K products per block:
//  products exact; the K products and c are aligned to the largest exponent
//  (the exponent of a product is ea + eb, so its significand is in [1, 4));
//  each aligned addend keeps 2 integer + 25 fraction bits, the rest is
//  truncated; the truncated addends are summed exactly and the sum is
//  normalised once and truncated (round toward zero) to fp32.
export const BLOCK_FRAC = 25;
export const BLOCK_EMIN = -133;

function expOf(x) {                      // unbiased exponent field (subnormal: -126)
  const ex = (bits32(x) >>> 23) & 0xff;
  return ex ? ex - 127 : -126;
}

// Returns everything a picture needs: per addend its exact value, its
// exponent, the kept (truncated) value, and the final result.
export function blockFma(as, bs, c) {
  const terms = as.map((a, i) => ({
    kind: 'prod', exact: exactMul(exact32(a), exact32(bs[i])),
    E: expOf(a) + expOf(bs[i]), zero: a === 0 || bs[i] === 0,
  }));
  terms.push({ kind: 'acc', exact: exact32(c), E: expOf(c), zero: c === 0 });
  const live = terms.filter((t) => !t.zero);
  const emax = Math.max(BLOCK_EMIN, ...live.map((t) => t.E));
  const qe = emax - BLOCK_FRAC;          // exponent of the last kept bit
  let sum = 0n;
  for (const t of terms) {
    const { m, e } = t.exact;
    const sh = qe - e;                   // shift right by sh to land on 2^qe units
    const a = absB(m);
    let kept = sh <= 0 ? a << BigInt(-sh) : a >> BigInt(sh);
    t.dropped = sh > 0 && (kept << BigInt(sh)) !== a;
    if (m < 0n) kept = -kept;
    t.kept = { m: kept, e: qe };         // the truncated addend
    sum += kept;
  }
  const keptSum = { m: sum, e: qe };
  return { terms, emax, qe, keptSum, result: roundExactF32(keptSum, 'rz') };
}

// The fp32 fma chain over the same terms: acc = c; acc = fma(a_k, b_k, acc).
export function fmaChain(as, bs, c) {
  let acc = c;
  for (let k = 0; k < as.length; k++) acc = fma32(as[k], bs[k], acc);
  return acc;
}

// distance between two fp32 values in units in the last place of `ref`
export function ulpDiff32(x, ref) {
  const u = bits32(Math.abs(ref)) & 0x7f800000;
  const ulp = u ? 2 ** (((u >>> 23) - 127) - 23) : 2 ** -149;
  return (x - ref) / ulp;
}

// ---- deterministic PRNG -----------------------------------------------------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- reductions -------------------------------------------------------------
// Each order is a binary tree over leaf indices: a leaf is a number, an
// internal node is [left, right].  evalTree does one fp32 add per node.
export const REDUCE_ORDERS = [
  { key: 'seq', label: 'left to right', tree: (n) => seqTree([...Array(n).keys()]) },
  { key: 'pair', label: 'pairwise tree', tree: (n) => pairTree([...Array(n).keys()]) },
  { key: 'lanes8', label: '8 lanes, then tree', tree: (n) => lanes8Tree(n) },
  { key: 'span2', label: '2 spans, then add', tree: (n) => [seqTree(range(0, n / 2)), seqTree(range(n / 2, n))] },
  { key: 'rev', label: 'right to left', tree: (n) => seqTree([...Array(n).keys()].reverse()) },
];
const range = (a, b) => [...Array(b - a).keys()].map((i) => i + a);
export function seqTree(ix) { return ix.reduce((acc, i) => (acc === null ? i : [acc, i]), null); }
export function pairTree(ix) {
  if (ix.length === 1) return ix[0];
  const h = ix.length / 2;
  return [pairTree(ix.slice(0, h)), pairTree(ix.slice(h))];
}
// element i -> lane i mod 8, each lane sequential; lanes (0+2)+(4+6) and
// (1+3)+(5+7), then those two added.
export function lanes8Tree(n) {
  const lane = (j) => seqTree(range(0, n).filter((i) => i % 8 === j));
  return [[[lane(0), lane(2)], [lane(4), lane(6)]], [[lane(1), lane(3)], [lane(5), lane(7)]]];
}
export function evalTree(t, xs) {
  if (typeof t === 'number') return xs[t];
  return add32(evalTree(t[0], xs), evalTree(t[1], xs));
}
