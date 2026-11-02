// Self-check for the group-A widgets: the arithmetic in src/fp.js against
// independent double-precision reimplementations, and the key numbers each
// widget displays.   node scripts/check-fp.mjs
import * as fp from '../src/fp.js';
import * as rounding from '../src/rounding.js';
import * as fma from '../src/fma.js';
import * as reduce from '../src/reduce.js';
import * as block from '../src/blockfma.js';
import * as online from '../src/online.js';
import * as scaleplace from '../src/scaleplace.js';

let fails = 0, passes = 0;
function check(name, cond, detail = '') {
  if (cond) passes++;
  else { fails++; console.log(`FAIL ${name}${detail ? '  (' + detail + ')' : ''}`); }
}
const eq = (name, got, want) => check(name, Object.is(got, want) || got === want, `got ${got}, want ${want}`);
const near = (name, got, want, digits) => check(name, Number(got).toPrecision(digits) === Number(want).toPrecision(digits), `got ${got}, want ${want} to ${digits} digits`);
const { f32, bf16, hex32 } = fp;
const rnd = fp.mulberry32(12345);

// ---- fp.js against independent implementations ----------------------------
// random fp32 with exponent in [lo, hi]
const r32 = (lo, hi) => f32((rnd() < 0.5 ? -1 : 1) * (1 + rnd()) * 2 ** (lo + Math.floor(rnd() * (hi - lo + 1))));
const nextUp32 = (x) => fp.fromBits32(x >= 0 ? fp.bits32(x) + 1 : fp.bits32(x) - 1);

// bf16: against "nearest of the two bf16 neighbours, ties to even" by value
for (let i = 0; i < 20000; i++) {
  const x = r32(-20, 20), u = fp.bits32(x) & 0xffff0000;
  const lo = fp.fromBits32(u), hi = fp.fromBits32(u + 0x10000);   // same sign, |lo| <= |x| < |hi|
  const dl = Math.abs(x - lo), dh = Math.abs(hi - x);
  const want = dl < dh ? lo : dh < dl ? hi : ((u >>> 16) & 1 ? hi : lo);
  if (bf16(x) !== want) { check('bf16 rne', false, `${x}: ${bf16(x)} vs ${want}`); break; }
}
check('bf16 rne (20000 random)', true);

// addBf16: when a + b is exact in fp32, bf16(f32(a + b)) has a single rounding
for (let i = 0; i < 20000; i++) {
  const a = bf16(r32(-3, 3)), b = bf16(r32(-3, 3));
  if (f32(a + b) !== a + b) continue;
  if (fp.addBf16(a, b) !== bf16(a + b)) { check('addBf16', false, `${a} + ${b}`); break; }
}
check('addBf16 (random)', true);

// fma32: a*b is exact in double (48 bits); TwoSum gives a*b + c = s + err
// exactly; f32(s) is the correctly rounded result unless s is an fp32 tie
// and err breaks it.
function fmaRef(a, b, c) {
  const p = a * b, s = p + c, bb = s - p, err = (p - (s - bb)) + (c - bb);
  let r = f32(s);
  const d = s - r;
  if (err !== 0 && d !== 0) {
    const ulp = Math.abs(nextUp32(Math.abs(r)) - Math.abs(r));
    if (Math.abs(d) === ulp / 2 && Math.sign(err) === Math.sign(d)) r = f32(r + 2 * d);
  } else if (err !== 0 && d === 0) { /* s is an fp32; the true value is within half a double ulp: still rounds to s */ }
  return r;
}
let fmaBad = 0;
for (let i = 0; i < 50000; i++) {
  const a = r32(-10, 10), b = r32(-10, 10);
  const c = i % 3 === 0 ? f32(-f32(a * b) * (1 + (rnd() - 0.5) * 2 ** -20)) : r32(-30, 30);
  if (fp.fma32(a, b, c) !== fmaRef(a, b, c)) { if (!fmaBad++) console.log('  fma mismatch', a, b, c, fp.fma32(a, b, c), fmaRef(a, b, c)); }
}
check('fma32 vs TwoSum reference (50000 random, incl. near-cancellation)', fmaBad === 0, `${fmaBad} mismatches`);
{ // double rounding: a*b + c rounds (in double) onto an fp32 tie that the 2^-60 should break
  const a = f32(1 + 2 ** -12), c = 2 ** -60;
  eq('fma32 breaks the tie fround(a*b+c) cannot see', fp.fma32(a, a, c), 1 + 2 ** -11 + 2 ** -23);
  eq('fround(a*b+c) double-rounds', f32(a * a + c), 1 + 2 ** -11);
}

// roundExact rne/rz on random exact values against doubles
for (let i = 0; i < 20000; i++) {
  const x = rnd() * 2 ** (Math.floor(rnd() * 40) - 20) * (rnd() < 0.5 ? -1 : 1);
  const m = BigInt(Math.round(x * 2 ** 30)), v = Number(m) * 2 ** -30;   // exact in double (< 2^53 ulps)
  const rne = fp.roundExact({ m, e: -30 }, 24, -149, 'rne'), rz = fp.roundExact({ m, e: -30 }, 24, -149, 'rz');
  let rzRef = f32(v); if (Math.abs(rzRef) > Math.abs(v)) rzRef = fp.fromBits32(fp.bits32(rzRef) - 1);
  if (rne !== f32(v) || rz !== rzRef) { check('roundExact', false, `${v}: ${rne}/${f32(v)} ${rz}/${rzRef}`); break; }
}
check('roundExact rne / rz (20000 random)', true);
eq('exactDecimal(0.1f)', fp.exactDecimal(fp.exact32(f32(0.1)), 20), '0.10000000149011611938');

// ---- bbe-rounding ------------------------------------------------------------
{
  const E = rounding.EXAMPLES.bf16, M = rounding.model(E);
  eq('rounding bf16: 1 + 2^-8 -> 1', M.left[0].r, 1);
  check('rounding bf16: first add rounds', M.left[0].rounded);
  eq('rounding bf16: exact 1 + 2^-8', M.left[0].exactStr, '1.00390625');
  eq('rounding bf16: (a + b) + c', M.resLeft, 1);
  eq('rounding bf16: b + c', M.right[0].r, 2 ** -7);
  eq('rounding bf16: a + (b + c)', M.resRight, 1.0078125);
  for (const t of E.ticks) eq(`rounding bf16 tick ${t} representable`, bf16(t), t);
  for (let i = 0; i + 1 < E.ticks.length; i++) {   // neighbours: nothing representable strictly between
    eq(`rounding bf16 ticks ${i},${i + 1} adjacent`, fp.fromBits32(fp.bits32(E.ticks[i]) + 0x10000), E.ticks[i + 1]);
  }
  const F = rounding.EXAMPLES.fp32, N = rounding.model(F);
  eq('rounding fp32: (1e8 + 4) + 4', N.resLeft, 1e8);
  eq('rounding fp32: 1e8 + (4 + 4)', N.resRight, 1e8 + 8);
  for (const t of F.ticks) eq(`rounding fp32 tick ${t}`, f32(t), t);
  eq('rounding fp32: gap at 1e8 is 8', nextUp32(1e8) - 1e8, 8);
}

// ---- bbe-fma -----------------------------------------------------------------
{
  const M = fma.model();
  eq('fma: a = 1 + 2^-12', M.a, 1 + 2 ** -12);
  eq('fma: a hex', hex32(M.a), '0x3F800800');
  eq('fma: round(a*b) = 1 + 2^-11', M.p, 1 + 2 ** -11);
  eq('fma: c hex', hex32(M.c), '0xBF801000');
  eq('fma: exact a*b', fp.exactToNumber(M.prod), 1 + 2 ** -11 + 2 ** -24);
  eq('fma: unfused', M.unfused, 0);
  eq('fma: fused', M.fused, 2 ** -24);
  eq('fma: fused hex', hex32(M.fused), '0x33800000');
  eq('fma: bits shown', M.bits.join(''), '1' + '0'.repeat(10) + '1' + '0'.repeat(12) + '100');
}

// ---- bbe-reduce ----------------------------------------------------------------
{
  for (const [i, v] of reduce.VALUES.entries()) eq(`reduce value ${i} prints as written`, fp.fmt32(v), String([5080, 0.732, 0.0442, 8.35, 7980, 0.0298, 91.9, 0.0995, 0.701, 0.581, 0.524, 592, 0.0389, 9.95, 0.0313, 0.772][i]));
  const M = reduce.model();
  const want = { seq: '0x46571707', pair: '0x46571704', lanes8: '0x46571705', span2: '0x46571706', rev: '0x46571703' };
  for (const r of M.rows) eq(`reduce ${r.key}`, hex32(r.r), want[r.key]);
  eq('reduce: distinct results', new Set(M.rows.map((r) => r.r)).size, 5);
  eq('reduce: correctly rounded', hex32(M.cr), '0x46571704');
  // tree shapes, by the leaves each subtree covers
  const leaves = (t) => (typeof t === 'number' ? [t] : [...leaves(t[0]), ...leaves(t[1])]);
  const L8 = fp.lanes8Tree(16);
  eq('reduce lanes8: left half lanes 0,2,4,6', JSON.stringify(leaves(L8[0])), JSON.stringify([0, 8, 2, 10, 4, 12, 6, 14]));
  eq('reduce lanes8: lane 3 = elements 3, 11', JSON.stringify(L8[1][0][1]), JSON.stringify([3, 11]));
  // the 8-lane order computed by hand
  const x = reduce.VALUES, A = fp.add32, lane = (j) => A(x[j], x[j + 8]);
  eq('reduce lanes8 by hand', fp.evalTree(L8, x), A(A(A(lane(0), lane(2)), A(lane(4), lane(6))), A(A(lane(1), lane(3)), A(lane(5), lane(7)))));
  let acc0 = x[0], acc1 = x[8];
  for (let i = 1; i < 8; i++) { acc0 = A(acc0, x[i]); acc1 = A(acc1, x[8 + i]); }
  eq('reduce span2 by hand', M.rows.find((r) => r.key === 'span2').r, A(acc0, acc1));
}

// ---- bbe-blockfma --------------------------------------------------------------
{
  // independent block-FMA model in doubles: every addend is a*b (exact in a
  // double) or c; truncate each to a multiple of 2^qe (trunc of an exact
  // power-of-two scaling is exact), sum as integers (< 2^53), RZ to fp32
  const expOf = (x) => { const e = (fp.bits32(x) >>> 23) & 0xff; return e ? e - 127 : -126; };
  function blockRef(as, bs, c) {
    const terms = as.map((a, i) => [a * bs[i], expOf(a) + expOf(bs[i]), a !== 0 && bs[i] !== 0]);
    terms.push([c, expOf(c), c !== 0]);
    const emax = Math.max(-133, ...terms.filter((t) => t[2]).map((t) => t[1])), qe = emax - 25;
    let S = 0;
    for (const [v] of terms) S += Math.trunc(v * 2 ** -qe);
    const x = S * 2 ** qe;
    let r = f32(x);
    if (Math.abs(r) > Math.abs(x)) r = fp.fromBits32(fp.bits32(r) - 1);
    return r;
  }
  let bad = 0, differ = 0, n = 0;
  for (const big of [false, true]) for (let s = 1; s <= 300; s++) {
    const { as, bs, c } = block.inputs(s, big);
    const B = fp.blockFma(as, bs, c);
    if (B.result !== blockRef(as, bs, c)) bad++;
    if (B.result !== fp.fmaChain(as, bs, c)) differ++;
    n++;
  }
  check('blockFma vs double reference (600 input sets)', bad === 0, `${bad} mismatches`);
  console.log(`  block FMA differs from the fma chain on ${differ} of ${n} input sets`);
  // inputs really are bf16, c fp32
  const I = block.inputs(1);
  check('blockfma inputs are bf16', I.as.concat(I.bs).every((v) => bf16(v) === v));
  // the default view: results distinct
  const M = block.model(1, false);
  check('blockfma default: block != chain', M.B.result !== M.chain, `${hex32(M.B.result)} vs ${hex32(M.chain)}`);
  eq('blockfma default block', hex32(M.B.result), '0x40BBA617');
  eq('blockfma default chain', hex32(M.chain), '0x40BBA612');
  eq('blockfma default exact rounded', hex32(M.cr), '0x40BBA616');
  check('blockfma default: some bits truncated', M.B.terms.some((t) => t.dropped));
  // hand case: c = 1, one product 2^-26 (below the cut) -> block ignores it, fma keeps it
  const z = Array(16).fill(0), a1 = [2 ** -13, ...z.slice(1)], b1 = [2 ** -13, ...z.slice(1)];
  eq('blockfma hand: 1 + 2^-26 truncated to 1', fp.blockFma(a1, b1, 1).result, 1);
  // 1 + 2^-24 + 2^-24: the chain rounds each tie to even (1, 1); block keeps both (1 + 2^-23)
  const a2 = [2 ** -12, 2 ** -12, ...z.slice(2)], b2 = [2 ** -12, 2 ** -12, ...z.slice(2)];
  eq('blockfma hand: block 1 + 2*2^-24', fp.blockFma(a2, b2, 1).result, 1 + 2 ** -23);
  eq('blockfma hand: chain', fp.fmaChain(a2, b2, 1), 1);
}

// ---- bbe-online ----------------------------------------------------------------
{
  const { TILES, run, SCENARIOS: S } = online;
  const K = run(true), E = run(false), J = run(true, TILES.s2jump);
  // policy decisions: 5 is 2.89 log2 units above 3 (keep under lazy), 15 is 17.3 (move)
  eq('online keep: m stays 3', K.T2.m, 3);
  eq('online every: m moves to 5', E.T2.m, 5);
  eq('online jump: m moves to 15', J.T2.m, 15);
  near('online: 2 nats in log2 units', 2 * Math.LOG2E, 2.89, 3);
  near('online: 12 nats in log2 units', 12 * Math.LOG2E, 17.3, 3);
  // first tile identical for every run; bf16(P) departs from p; l stays exact to 7 digits, O does not
  for (const R of [E, J]) eq('online T1 shared', JSON.stringify(R.T1), JSON.stringify(K.T1));
  K.T1.pR.forEach((p, j) => { eq(`online T1 p_${j} = fround(exp)`, p, f32(Math.exp(TILES.s1[j] - 3))); eq(`online T1 P_${j} = bf16(p)`, K.T1.P[j], bf16(p)); });
  check('online T1: bf16(P) differs from p', K.T1.P.some((P, j) => P !== K.T1.pR[j]));
  eq('online T1: rounded l = exact l to 7 digits', S.keep.T1.l[1], S.keep.T1.l[0]);
  check('online T1: rounded O differs from exact O', S.keep.T1.O[1] !== S.keep.T1.O[0]);
  // the rounded lazy run by hand
  {
    let l = 0, O = 0;
    for (const [s, v, m] of [[TILES.s1, TILES.v1, 3], [TILES.s2, TILES.v2, 3]])
      s.forEach((x, j) => { const p = f32(Math.exp(x - m)); l = fp.add32(l, p); O = fp.add32(O, fp.mul32(bf16(p), v[j])); });
    eq('online keep rounded l by hand', K.T2.l[1], l);
    eq('online keep rounded O by hand', K.T2.O[1], O);
    eq('online keep out by hand', K.epi.out, bf16(f32(O / l)));
  }
  check('online: p > 1 under lazy rescaling', K.T2.pR[0] > 1, K.T2.pR[0]);
  // the invariant: exact O/l agrees across policies; the rounded one does not
  check('online: exact O/l lazy == every (to 1e-12)', Math.abs(K.T2.Ol[0] - E.T2.Ol[0]) < 1e-12 * K.T2.Ol[0]);
  eq('online: exact O/l lazy == every as displayed', S.keep.T2.Ol[0], S.move.T2.Ol[0]);
  check('online: rounded O/l lazy != every', K.T2.Ol[1] !== E.T2.Ol[1]);
  check('online: out lazy != every', K.epi.out !== E.epi.out);
  // exact O/l equals the softmax-weighted mean of v over all keys
  const all = (s2) => { const s = [...TILES.s1, ...s2], v = [...TILES.v1, ...TILES.v2], w = s.map(Math.exp), Z = w.reduce((a, b) => a + b);
    return { mean: w.reduce((a, x, i) => a + x * v[i], 0) / Z, lse: Math.log(Z) }; };
  near('online: exact O/l = softmax . v', K.T2.Ol[0], all(TILES.s2).mean, 12);
  near('online jump: exact O/l = softmax . v', J.T2.Ol[0], all(TILES.s2jump).mean, 12);
  near('online: exact lse', K.epi.lseEx, all(TILES.s2).lse, 12);
  console.log(`  online: v1 = [${TILES.v1}], v2 = [${TILES.v2}]; T1 bf16 P = [${K.T1.P}]`);
  for (const [n, R] of [['lazy', K], ['every', E], ['jump', J]])
    console.log(`  online ${n}: exact O/l ${R.T2.Ol[0]}, rounded O/l ${R.T2.Ol[1]}, out ${R.epi.out} (bf16 of exact ${R.epi.outEx}), lse ${R.epi.lse} (exact ${R.epi.lseEx})`);
}

// ---- scaleplace: the ex2 argument in three pipelines ----------------------------
{
  const { A, B, C, S, MAX, fmt9, fmtExact } = scaleplace, e = scaleplace.example();
  eq('scaleplace a bits', hex32(A), '0x3DB504F3');
  eq('scaleplace b bits', hex32(B), '0x3FB8AA3B');
  eq('scaleplace c bits', hex32(C), '0x3E0293EE');
  check('scaleplace: 1/sqrt(128) rounds', A !== 1 / Math.sqrt(128));
  check('scaleplace: log2 e rounds', B !== Math.LOG2E);
  check('scaleplace: c = a*b rounds', C !== A * B);
  eq('scaleplace c exact shown', fmtExact(Math.LOG2E / Math.sqrt(128)), '0.1275174308');
  eq('scaleplace c = fp32(exact)', C, f32(Math.LOG2E / Math.sqrt(128)));
  eq('scaleplace a exact shown', fmtExact(1 / Math.sqrt(128)), '0.08838834765');
  eq('scaleplace b exact shown', fmtExact(Math.LOG2E), '1.442695041');
  // each pipeline by hand; the fmas checked against double arithmetic that is
  // exact here (19 * c and t1 * b fit in 53 bits with the subtrahend aligned)
  const mc = f32(MAX * C), mf = f32(f32(MAX * A) * B), t1 = f32(S * A), t2 = f32(t1 * B);
  const xa = f32(S * C - mc), xp = f32(t2 - mf), xf = f32(t1 * B - mf);
  check('scaleplace: S*C - mc exact in double', S * C - mc - (S * C) === -mc);
  check('scaleplace: t1*B - mf exact in double', t1 * B - mf - (t1 * B) === -mf);
  eq('scaleplace fa4 x = fma', e.x.fa4, fp.fma32(S, C, -mc)); eq('scaleplace fa4 x by double', e.x.fa4, xa);
  eq('scaleplace part x', e.x.part, xp);
  eq('scaleplace full x = fma', e.x.full, fp.fma32(t1, B, -mf)); eq('scaleplace full x by double', e.x.full, xf);
  check('scaleplace: three x differ', xa !== xp && xa !== xf && xp !== xf);
  // displayed strings
  for (const [n, v, want] of [['m*c', e.mc, '3.06041837'], ['m flex', e.mf, '3.06041813'], ['t1', e.t1, '1.67937863'], ['t2', e.t2, '2.42283130'],
    ['a', A, '0.0883883461'], ['b', B, '1.44269502'], ['c', C, '0.127517432'],
    ['x fa4', e.x.fa4, '−0.637587190'], ['x part', e.x.part, '−0.637586832'], ['x full', e.x.full, '−0.637586951']]) eq(`scaleplace shown ${n}`, fmt9(v), want);
  eq('scaleplace hex fa4', hex32(e.x.fa4), '0xBF2338EA');
  eq('scaleplace hex part', hex32(e.x.part), '0xBF2338E4');
  eq('scaleplace hex full', hex32(e.x.full), '0xBF2338E6');
  // exact x = (s - max) log2 e / sqrt(128) = -5 log2 e sqrt 2 / 16, in 39-digit fixed point
  const P = 10n ** 39n, LOG2E = 1442695040888963407359924681001892137426n, SQRT2 = 1414213562373095048801688724209698078570n;
  const exBig = Number((BigInt(S - MAX) * LOG2E * SQRT2) / (16n * P)) / 1e39;
  check('scaleplace exact x', Math.abs(e.exact - exBig) < 1e-15, `${e.exact} vs ${exBig}`);
  eq('scaleplace exact shown', fmtExact(e.exact), '−0.6375871541');
  eq('scaleplace fp32(exact) bits', hex32(f32(e.exact)), '0xBF2338E9');
  eq('scaleplace ulps fa4', e.ulps.fa4, -1); eq('scaleplace ulps part', e.ulps.part, 5); eq('scaleplace ulps full', e.ulps.full, 3);
  console.log(`  scaleplace: s = ${S}, row max = ${MAX}; x fa4 ${fmt9(e.x.fa4)}, flex partial ${fmt9(e.x.part)}, flex full ${fmt9(e.x.full)}, exact ${fmtExact(e.exact)}`);
}

// ---- bbe-rowsum ----------------------------------------------------------------
{
  const rowsum = await import('../src/rowsum.js');
  const { P, L0, ALPHA } = rowsum, M = rowsum.model();
  eq('rowsum: 128 elements', P.length, 128);
  eq('rowsum: alpha = 1', ALPHA, 1);
  eq('rowsum: previous l prints as 41.7', fp.fmt32(L0), '41.7');
  for (const p of P) check('rowsum: p is a 3-digit decimal in [0.01, 1]', p === f32(p) && p >= 0.01 && p <= 1 && fp.fmt32(p) === String(+p.toPrecision(3)), p);
  // FA4-style by hand: lanes j = k mod 8, l into lane 0, tree ((0+2)+(4+6)) + ((1+3)+(5+7))
  const ln = Array.from({ length: 8 }, (_, j) => P.filter((_, k) => k % 8 === j).reduce((s, x) => f32(s + x), j ? 0 : L0));
  const t = (i, j) => f32(ln[i] + ln[j]);
  eq('rowsum FA4 by hand', M.fa4, f32(f32(t(0, 2) + t(4, 6)) + f32(t(1, 3) + t(5, 7))));
  // flex by hand: two halves left to right, a + b, then l + s rounded once (fma with alpha = 1)
  const half = (lo) => P.slice(lo, lo + 64).reduce((s, x) => f32(s + x), 0);
  eq('rowsum flex by hand', M.flex, f32(L0 + f32(half(0) + half(64))));
  // exact sum: every value is a multiple of 2^-30
  const big = [L0, ...P].reduce((s, x) => s + BigInt(x * 2 ** 30), 0n);
  eq('rowsum exact sum rounded once', M.cr, f32(Number(big) / 2 ** 30));
  // displayed numbers
  eq('rowsum FA4 hex', hex32(M.fa4), '0x427EF9C4');
  eq('rowsum flex hex', hex32(M.flex), '0x427EF9C0');
  eq('rowsum exact rounded hex', hex32(M.cr), '0x427EF9C1');
  eq('rowsum FA4 value', fp.fmt32(M.fa4), '63.74391');
  eq('rowsum flex value', fp.fmt32(M.flex), '63.743896');
  eq('rowsum exact rounded value', fp.fmt32(M.cr), '63.7439');
  eq('rowsum exact decimal', fp.exactDecimal(M.exact, 13), '63.7439007014');
  eq('rowsum FA4 ulps', fp.ulpDiff32(M.fa4, M.cr), 3);
  eq('rowsum flex ulps', fp.ulpDiff32(M.flex, M.cr), -1);
  console.log(`  rowsum: l = ${fp.fmt32(L0)}; FA4 ${fp.fmt32(M.fa4)} (${hex32(M.fa4)}), flex ${fp.fmt32(M.flex)} (${hex32(M.flex)}), exact ${fp.exactDecimal(M.exact, 13)} -> ${hex32(M.cr)}`);
}

// ---- bbe-visitorder (slide deck): fp32 fold of one query row's l ----------
{
  const vo = await import('../src/visitorder.js');
  // independent: query 959 sees doc 3 keys 560..959, i.e. 80, 128, 128, 64 keys of blocks 4..7
  const span = { 4: [560, 640], 5: [640, 768], 6: [768, 896], 7: [896, 960] };
  const rnd = fp.mulberry32(vo.SEED), acc = new Float32Array(1), leaf = {};
  for (const kb of [4, 5, 6, 7]) { acc[0] = 0; for (let k = span[kb][0]; k < span[kb][1]; k++) acc[0] += Math.fround(rnd()); leaf[kb] = acc[0]; }
  const fold = (ks) => { acc[0] = 0; return ks.map((kb) => (acc[0] += leaf[kb], acc[0])); };
  const want = { fa4: fold([7, 6, 5, 4]), flex: fold([4, 7, 5, 6]) };
  eq('visitorder QROW', vo.QROW, 959);
  for (const kb of [4, 5, 6, 7]) { eq(`visitorder terms ${kb}`, vo.TERMS[kb], span[kb][1] - span[kb][0]); eq(`visitorder leaf ${kb}`, vo.LEAF[kb], leaf[kb]); }
  for (const who of ['fa4', 'flex']) want[who].forEach((v, j) => eq(`visitorder ${who} step ${j}`, vo.FOLD[who][j], v));
  // displayed strings
  const show = { 4: '40.428387', 5: '59.12498', 6: '62.85602', 7: '29.583445' };
  for (const kb in show) eq(`visitorder leaf ${kb} prints`, fp.fmt32(vo.LEAF[kb]), show[kb]);
  eq('visitorder fa4 l1 prints', fp.fmt32(vo.FOLD.fa4[1]), '92.43947');
  eq('visitorder fa4 l2 prints', fp.fmt32(vo.FOLD.fa4[2]), '151.56445');
  eq('visitorder flex l1 prints', fp.fmt32(vo.FOLD.flex[1]), '70.01183');
  eq('visitorder flex l2 prints', fp.fmt32(vo.FOLD.flex[2]), '129.13681');
  eq('visitorder fa4 l prints', fp.fmt32(vo.FOLD.fa4[3]), '191.99284');
  eq('visitorder flex l prints', fp.fmt32(vo.FOLD.flex[3]), '191.99283');
  eq('visitorder fa4 l hex', hex32(vo.FOLD.fa4[3]), '0x433FFE2B');
  eq('visitorder flex l hex', hex32(vo.FOLD.flex[3]), '0x433FFE2A');
  eq('visitorder roots differ by 1 ulp', fp.ulpDiff32(vo.FOLD.fa4[3], vo.FOLD.flex[3]), 1);
}

// ---- offline-online (and slide prose): overflow, m, alpha, offline = online ------
{
  const R = await import('../src/offline-online.js'), FMAX = fp.fromBits32(0x7f7fffff);
  check('offline-online prose: 88.72 is the fp32 exp overflow point', Math.log(2 ** 128).toFixed(2) === '88.72' && f32(Math.exp(88.72)) <= FMAX && f32(Math.exp(88.73)) === Infinity);
  check('offline-online prose: every raw e^s is fp32 inf', R.TILES.flat().every((s) => f32(Math.exp(s)) === Infinity));
  eq('offline-online: raw out = inf / inf = NaN', Infinity / Infinity, NaN);
  eq('offline-online: offline m', R.M, 95);
  eq('offline-online: online m after each tile', R.REFS.join(), '93,95,95');
  // the stepper: m and alpha cells per step; alpha = e^(m_old - m_new) where m moves
  eq('offline-online steps: m cells', R.STEPS.map((s) => s.m).join(), '93,95,95,95,95');
  eq('offline-online steps: alpha cells', R.STEPS.map((s) => s.alpha).join(), `–,e^${R.REFS[0] - R.REFS[1]},e^-2,e^-2,1`);
  check('offline-online steps: weights <= 1 once rescaled', R.STEPS.every((s) => s.ref >= Math.max(...R.TILES.slice(0, s.added).flat())));
  near('offline-online: l after tile 1 = e^-2 + 1 + e^-1', R.STEPS[0].sums[0], Math.exp(-2) + 1 + Math.exp(-1), 12);
  near('offline-online: rescale multiplies l by e^-2', R.STEPS[2].sums[0], R.STEPS[1].sums[0] * Math.exp(-2), 12);
  near('offline-online: rescale multiplies O by e^-2', R.STEPS[2].sums[1], R.STEPS[1].sums[1] * Math.exp(-2), 12);
  near('offline-online: final online l = offline l', R.STEPS[4].sums[0], R.OFFLINE[0], 12);
  near('offline-online: final online O = offline O', R.STEPS[4].sums[1], R.OFFLINE[1], 12);
  check('offline-online: earlier steps stop short of the guide lines', R.STEPS.slice(0, 4).every((s) => s.sums[0] < R.OFFLINE[0] && s.sums[1] < R.OFFLINE[1]));
}

// ---- unitgrid (slide "Rescale policy"): toy 3-bit grid, held values, out bits ----
{
  const U = await import('../src/unitgrid.js');
  const { TILES } = online;
  // toy rounding against "nearest of the enumerated 3-bit grid values, ties to even significand"
  const G = [];
  for (let e = -12; e <= 8; e++) for (let k = 4; k < 8; k++) G.push({ g: k * 2 ** (e - 2), even: k % 2 === 0 });
  const nearest = (x) => G.reduce((b, c) => { const d = Math.abs(c.g - x) - Math.abs(b.g - x); return d < 0 || (d === 0 && c.even) ? c : b; }).g;
  let bad = 0;
  for (let i = 0; i < 5000; i++) { const x = 2 ** (-8 + 14 * rnd()); if (U.toy(x) !== nearest(x)) bad++; }
  for (const x of [1.125, 1.375, 2.25, 0.1640625]) if (U.toy(x) !== nearest(x)) bad++;   // exact ties
  eq('unitgrid toy round = nearest 3-bit value', bad, 0);
  // inputs that are doubles of transcendental values must not sit within 1e-9 of a toy tie
  const C = Object.fromEntries(U.CS.map(([k, c]) => [k, c]));
  const tieSafe = (x) => { const q = 2 ** (Math.floor(Math.log2(x)) - 2), f = x / q % 1; return Math.abs(f - 0.5) > 1e-9; };
  const fed = [Math.exp(-2) + 1 + Math.exp(-1), U.EXACT[1], U.EXACT[2], ...Object.values(C).flatMap((c) => [c * U.L1, c * U.EXACT[1], c * U.EXACT[2]])];
  check('unitgrid: no transcendental input near a toy tie', fed.every(tieSafe), fed.filter((x) => !tieSafe(x)).join());
  eq('unitgrid l after tile 1', U.L1, 1.5);
  eq('unitgrid lazy held', U.HELD.e.lazy.join(), '1.5,2.5,7,10');
  eq('unitgrid every held, c = e^-2', U.HELD.e.every.join(), '0.1875,0.375,1,1.5');
  eq('unitgrid every held, c = 2^-3', U.HELD.two.every.join(), '0.1875,0.3125,0.875,1.25');
  eq('unitgrid lazy held independent of c', U.HELD.two.lazy.join(), U.HELD.e.lazy.join());
  eq('unitgrid: c = e^-2, no held value corresponds', U.HELD.e.same.join(), 'false,false,false,false');
  eq('unitgrid: c = 2^-3, every held value corresponds', U.HELD.two.same.join(), 'true,true,true,true');
  check('unitgrid: c = 2^-3 held values are exactly lazy / 8', U.HELD.two.every.every((v, i) => v * 8 === U.HELD.two.lazy[i]));
  eq('unitgrid: lazy l after tile 1 is on a tick', U.toy(U.L1), U.L1);
  check('unitgrid: c * l1 is between ticks for c = e^-2', U.toy(C.e * U.L1) !== C.e * U.L1);
  eq('unitgrid: p for s = 5 is a tick in every-tile units', U.HELD.e.every[2], 1);
  // out bits: online.js's run(), and the policy run by hand: fp32 p, fp32 l, bf16 P for P.V, fp32 O; alpha before tile 2 when m moves
  const byHand = (m2) => {
    let l = 0, O = 0;
    for (const [s, v, m] of [[TILES.s1, TILES.v1, 3], [TILES.s2, TILES.v2, m2]]) {
      const a = f32(Math.exp(3 - m)); l = fp.mul32(l, a); O = fp.mul32(O, a);
      s.forEach((x, j) => { const p = f32(Math.exp(x - m)); l = fp.add32(l, p); O = fp.add32(O, fp.mul32(bf16(p), v[j])); });
    }
    return bf16(f32(O / l));
  };
  eq('unitgrid out bits', U.OUT.join(), '0x408A,0x408B');
  eq('unitgrid out bits by hand', [byHand(3), byHand(5)].map(fp.hex16).join(), U.OUT.join());
  check('unitgrid: online.js lazy keeps m = 3, every tile moves to 5 (c = e^-2)', online.run(true).T2.m === 3 && online.run(false).T2.m === 5);
}

// ---- rescalepolicy (slide "Rescale policy"): the aligned rows it shows ----------
{
  const P = await import('../src/rescalepolicy.js');
  const { TILES } = online;
  // the policy run by hand: fp32 p, fp32 l, bf16 P for P.V, fp32 O; alpha applied before tile 2 when m moves
  const byHand = (m2) => {
    let l = 0, O = 0;
    for (const [s, v, m] of [[TILES.s1, TILES.v1, 3], [TILES.s2, TILES.v2, m2]]) {
      const a = f32(Math.exp(3 - m)); l = fp.mul32(l, a); O = fp.mul32(O, a);
      s.forEach((x, j) => { const p = f32(Math.exp(x - m)); l = fp.add32(l, p); O = fp.add32(O, fp.mul32(bf16(p), v[j])); });
    }
    return { Ol: f32(O / l), out: bf16(f32(O / l)) };
  };
  const H = [byHand(3), byHand(5)];
  check('rescalepolicy: 5 is under m + 8 ln 2 = 8.55', 5 < 3 + P.BAND && (3 + P.BAND).toFixed(2) === '8.55');
  const rows = Object.fromEntries(P.ROWS.map(([k, v]) => [k, v.join(' | ')]));
  eq('rescalepolicy m', rows.m, '3 | 5');
  eq('rescalepolicy alpha', rows['α'], '1 | e^-2');
  eq('rescalepolicy O/l', rows['O / ℓ (exact 4.331542)'], `${+H[0].Ol.toPrecision(7)} | ${+H[1].Ol.toPrecision(7)}`);
  eq('rescalepolicy O/l values', rows['O / ℓ (exact 4.331542)'], '4.324174 | 4.330957');
  eq('rescalepolicy out', rows['out = bf16(O / ℓ)'], `${H[0].out} | ${H[1].out}`);
  eq('rescalepolicy out values', rows['out = bf16(O / ℓ)'], '4.3125 | 4.34375');
  eq('rescalepolicy out bits', rows['out bits'], '0x408A | 0x408B');
  near('rescalepolicy exact O/l', P.RUNS[0].T2.Ol[0], 4.331542, 7);
}

console.log(`check-fp: ${passes} passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
