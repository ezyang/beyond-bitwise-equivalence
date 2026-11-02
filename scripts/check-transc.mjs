// Exactness checks for src/transc.js (the "Fast math" widgets).
//   node scripts/check-transc.mjs          (~15 s: 2^20 inputs x 4 variants)
// Exits non-zero on any failed assertion.  Also recomputes STATS, the
// numbers the widgets display, and checks they match.
import {
  POLY_EX2, ROUND_INT, addRoundDown, ex2, ex2Emu, horner, ref2, ulpsApart, permissible,
  statInputs, STATS, divSets, DIV_PRESETS,
} from '../src/transc.js';
import { f32, bits32, fromBits32, bf16, exact32, exactSum, roundExactF32, fma32 } from '../src/fp.js';

let fails = 0;
const ok = (cond, msg) => { if (!cond) { fails++; console.log('FAIL', msg); } };
const eq = (a, b, msg) => ok(Object.is(a, b), `${msg}: got ${a}, want ${b}`);

// coefficients are fp32 values (24-bit, as Sollya was told)
for (const [d, cs] of Object.entries(POLY_EX2)) cs.forEach((c, i) => eq(f32(c), c, `POLY_EX2[${d}][${i}] is fp32`));
eq(bits32(ROUND_INT), 0x4B400000, 'bits(1.5*2^23)');

// add.rm via floor agrees with an independent exact rounding.  The sum is
// positive, so round-down = round-toward-zero, which fp.js implements on
// exact BigInt values.
{
  const xs = [-127, -126.5, -3.7, -2.3, -1, -0.75, -(2 ** -30), -0, 0, 2 ** -40, 0.3, 1, 1.5, 7.999, 126.99, 127]
    .map(f32);
  for (let k = 0; k < 20000; k++) xs.push(f32(-127 + 254 * ((k * 0.6180339887) % 1)));
  for (const x of xs) {
    const viaBig = roundExactF32(exactSum([exact32(x), exact32(ROUND_INT)]), 'rz');
    eq(addRoundDown(x), viaBig, `addRoundDown(${x})`);
  }
}

// the worked example in the bits widget
{
  const s = ex2Emu(-2.3, 3, true);
  eq(s.n, -3, 'floor(-2.3) via the magic add');
  eq(s.xr, ROUND_INT - 3, 'x_rounded for -2.3');
  eq(s.frac, f32(f32(-2.3) + 3), 'frac for -2.3');
  eq(s.shifted, ((-3 << 23) >>> 0), 'floor(x) << 23 for -2.3');
}

// 2^0 = 1 exactly, and integer x gives exact powers of two (poly(0) = c0 = 1)
for (const d of [1, 2, 3, 4, 5]) for (const fu of [true, false]) {
  eq(ex2(0, d, fu), 1, `ex2(0) deg ${d} ${fu ? 'fma' : 'mul+add'}`);
  for (let n = -126; n <= 127; n++) eq(ex2(n, d, fu), 2 ** n, `ex2(${n}) deg ${d}`);
}
eq(ex2(-127, 3, true), 0, 'ex2(-127): 2^-127 is subnormal, the integer trick gives +0');

// fma Horner really is one rounding per step: spot-check against an
// independent double-double evaluation of a single step
{
  const c = POLY_EX2[3];
  for (const f of [0.1, 0.7, 0.999].map(f32)) {
    const a = c[3], step = fma32(a, f, c[2]);
    const exact = a * f + c[2];          // a*f exact in double (24x24 bits), + c[2] may round...
    // ...so compare via BigInt instead: fma32 is round(a*f + c) by definition
    const big = roundExactF32(exactSum([{ m: exact32(a).m * exact32(f).m, e: exact32(a).e + exact32(f).e }, exact32(c[2])]));
    eq(step, big, `fma step at f=${f}`);
    ok(Math.abs(step - exact) < 1e-7, 'fma step sane');
  }
  // mul+add Horner, written out: two roundings per step
  const f = f32(0.5), r1 = f32(f32(c[3] * f) + c[2]), r2 = f32(f32(r1 * f) + c[1]), r3 = f32(f32(r2 * f) + c[0]);
  eq(horner(f, 3, false), r3, 'mul+add Horner at f=0.5');
}

// reference: no input in the stat set has 2^x within 2^-50 (relative) of a
// midpoint between two fp32 values, so f32(2**x) is the correctly rounded
// fp32 2^x for every input below (V8's pow error is far below 2^-50).
const X = statInputs();
{
  let near = 0;
  for (const x of X) {
    const t = 2 ** x, r = f32(t), nb = t > r ? fromBits32(bits32(r) + 1) : fromBits32(bits32(r) - 1);
    if (Math.abs(t - (r + nb) / 2) <= 2 ** -50 * t) near++;
  }
  eq(near, 0, 'near-ties in the reference');
}

// recompute STATS over the same inputs
const res = {};
for (const d of [3, 5]) for (const fu of [true, false]) {
  let lo = 0, hi = 0, same = 0, inside = 0, bfDiff = 0;
  const outs = new Float32Array(X.length);
  X.forEach((x, k) => {
    const y = ex2(x, d, fu), r = ref2(x), u = ulpsApart(y, r), [a, b] = permissible(x);
    outs[k] = y;
    lo = Math.min(lo, u); hi = Math.max(hi, u);
    if (u === 0) same++;
    if (u >= a && u <= b) inside++;
    if (bf16(y) !== bf16(r)) bfDiff++;
    ok(a <= 0 && b >= 0, 'the correctly rounded value satisfies the bound');
  });
  res[`${d}${fu ? 'f' : 'm'}`] = { lo, hi, same: same / X.length, inside: inside / X.length, bf16: bfDiff / X.length, outs };
}
for (const d of [3, 5]) {
  const a = res[`${d}f`].outs, b = res[`${d}m`].outs;
  let diff = 0; for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) diff++;
  res[`${d}f`].vsOther = res[`${d}m`].vsOther = diff / a.length;
}
console.log(`STATS over ${X.length} random fp32 x in [-1, 1] (counts):`);
for (const [k, r] of Object.entries(res)) {
  const N = X.length, cnt = (v) => Math.round(v * N);
  const want = { lo: r.lo, hi: r.hi, same: cnt(r.same), outside: N - cnt(r.inside), bf16: cnt(r.bf16), vsOther: cnt(r.vsOther) };
  console.log(`  '${k}': ${JSON.stringify(want).replace(/"/g, '')},`);
  for (const f of Object.keys(want)) eq(String(STATS[k]?.[f]), String(want[f]), `STATS.${k}.${f}`);
}
eq(res['5f'].inside, 1, 'degree 5 (fma) always inside the ex2.approx bound');

// division sets
for (const [x, y] of DIV_PRESETS) {
  const s = divSets(x, y);
  eq(s.rn, f32(x / y), `div.rn ${x}/${y}`);
  ok(s.full.includes(s.rn) && s.full.length >= 4, `div.full set ${x}/${y}`);
  console.log(`  ${x}/${y}: div.full ${s.full.map((v) => bits32(v) - bits32(s.rn))}  rcp*x ${s.viaRcp.map((v) => bits32(v) - bits32(s.rn))}`);
}
ok(!divSets(3, 7).viaRcp.includes(f32(3 / 7)), '3/7: x*rcp(y) can never be the correctly rounded quotient');

// <bbe-lse-heatmap>: the widget draws data/div-measured.json, measured on a
// GPU by scripts/measure-div.py (no exact model of rcp.approx / lg2.approx
// exists to recompute it).  Check the file's structure and the identities it
// records, and that every number the widget prints matches.
import { shown } from '../src/lse-heatmap.js';
import { readFileSync } from 'node:fs';
{
  const d = JSON.parse(readFileSync(new URL('../data/div-measured.json', import.meta.url), 'utf8'));
  const RCP = 'rcp.approx.ftz.f32 then mul.rn.f32', FULL = 'div.full.f32';
  ok(d && d.div && d.log2, 'div-measured.json holds a measurement');
  if (d?.div) {
    eq(d.device, 'NVIDIA B200', 'measured device');
    for (const u of Object.values(d.div)) eq(u.length, d.n * d.n, 'div grid size');
    for (const u of Object.values(d.log2)) eq(u.length, d.m, 'log2 sweep size');
    ok(d.div['div.rn.f32'].every((v) => v === 0), 'div.rn.f32 is the correctly rounded quotient on every cell');
    ok(d.div[FULL].every((v, k) => v === d.div[RCP][k]), 'div.full.f32 = rcp.approx.ftz then mul.rn on every cell');
    ok(d.div['triton x / y'].every((v, k) => v === d.div[FULL][k]), 'Triton x / y = div.full.f32');
    ok(d.log2['triton tl.log2'].every((v, k) => v === d.log2.log2f[k]), 'Triton tl.log2 = CUDA log2f');
    const L = d.lse, n = L.n;
    for (const k of ['FA4-style', 'flex']) eq(L[k].length, n * n, `lse grid size ${k}`);
    L.M.forEach((m, j) => eq(m, f32(-16 + 64 * (j + (0.6180339887 * (j + 1)) % 1) / n), `lse grid M[${j}]`));
    L.l.forEach((l, i) => eq(l, f32(2 ** (15 * (i + 0.5) / n)), `lse grid l[${i}]`));
    eq(JSON.stringify(shown(d)), JSON.stringify({
      lse: [['1.1%', '8.6%', '82.5%', '7.8%', '<0.1%', 6], ['0%', '7.1%', '84.5%', '8.4%', '0%', 1]],
      closer: ['0%', '0.3%', '97.0%', '0.4%', '1.4%', '1.0%'],
      grey: '4.5%',
    }), 'numbers shown by bbe-lse-heatmap');
    console.log(`  lse-heatmap (${d.device}): ${JSON.stringify(shown(d))}`);
  }
}

console.log(fails ? `${fails} failure(s)` : 'all checks pass');
process.exit(fails ? 1 : 0);
