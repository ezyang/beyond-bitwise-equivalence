// <bbe-blockfma>: one tensor-core MMA step for one output element, K = 16
// bf16 products plus the fp32 accumulator c, after Khattak & Mikaitis
// (arXiv:2512.07004).  The 17 addends are drawn as bit strips aligned at
// the largest exponent; bits past 25 fraction bits are truncated (orange).
// Below: the block-FMA result vs an fp32 fma chain over the same terms.
// All arithmetic is exact BigInt, in src/fp.js (blockFma, fmaChain).
import { bf16, f32, mulberry32, blockFma, fmaChain, exactSum, exactDecimal, exactToNumber, roundExactF32,
  hex32, fmt32, ulpDiff32, BLOCK_FRAC } from './fp.js';

// deterministic inputs #n: a_i, b_i bf16 with exponents -5..2; c an fp32
// with a full 24-bit significand; `big` makes c ~2^11..2^14 so the products
// lose many bits (as when the accumulator has grown over earlier k-blocks)
export function inputs(n, big = false) {
  const r = mulberry32(0x5eed + (n + 2) * 7919);
  const val = (lo, hi) => (r() < 0.5 ? -1 : 1) * (1 + r()) * 2 ** (lo + Math.floor(r() * (hi - lo + 1)));
  const as = [], bs = [];
  for (let i = 0; i < 16; i++) { as.push(bf16(val(-5, 2))); bs.push(bf16(val(-5, 2))); }
  let c = f32(val(0, 3));
  if (big) c = f32(Math.abs(c) * 2 ** 11);
  return { as, bs, c };
}

export function model(n, big) {
  const { as, bs, c } = inputs(n, big);
  const B = blockFma(as, bs, c), chain = fmaChain(as, bs, c);
  const exact = exactSum(B.terms.map((t) => t.exact));
  return { as, bs, c, B, chain, exact, cr: roundExactF32(exact) };
}

const SUB = (n) => String(n).split('').map((d) => '₀₁₂₃₄₅₆₇₈₉'[d]).join('');
const LEFT = 3, DROP = 14, NCOL = LEFT + 2 + BLOCK_FRAC + DROP;   // columns: carry, 2 integer, 25 fraction, dropped
const LW = 132, CW = 12.4, RH = 13, GY = 46;                      // label width, column pitch, row pitch, grid top
const W = 688;

const CSS = `
bbe-blockfma { display: block; height: 440px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.4 var(--sans); color: var(--text); }
bbe-blockfma .top { display: flex; align-items: center; gap: 10px; height: 26px; font-size: 12px; color: var(--muted); }
bbe-blockfma .seg { display: inline-flex; }
bbe-blockfma .seg button, bbe-blockfma .btn { font: 12px var(--sans); padding: 2px 9px; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-blockfma .btn { border-radius: 4px; }
bbe-blockfma .seg button + button { border-left: none; }
bbe-blockfma .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-blockfma .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-blockfma .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-blockfma .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
bbe-blockfma svg { display: block; margin-top: 4px; font: 10.5px var(--mono); font-variant-numeric: tabular-nums; }
bbe-blockfma svg text { fill: var(--text); }
bbe-blockfma .lab { fill: var(--muted); }
bbe-blockfma .hd { font: 11px var(--sans); fill: var(--muted); }
bbe-blockfma .hd.o { fill: var(--orange); }
bbe-blockfma .k1 { fill: var(--blue); }
bbe-blockfma .k0 { fill: var(--blue-soft); }
bbe-blockfma .d1 { fill: var(--orange); }
bbe-blockfma .d0 { fill: var(--orange-soft); }
bbe-blockfma .zone { fill: var(--orange-soft); opacity: .35; }
bbe-blockfma .cut { stroke: var(--ink); stroke-width: 1.5; }
bbe-blockfma .pt { stroke: var(--rule-strong); stroke-dasharray: 2 2; }
bbe-blockfma .sep { stroke: var(--rule-strong); }
bbe-blockfma .more { fill: var(--orange); font-size: 10px; }
bbe-blockfma .s1 { fill: var(--ink); }
bbe-blockfma .s0 { fill: var(--rule); }
bbe-blockfma .t1 { fill: var(--orange); }
bbe-blockfma .t0 { fill: var(--orange-soft); }
bbe-blockfma table { border-collapse: collapse; font-size: 12.5px; margin-top: 4px; width: 100%; table-layout: fixed; }
bbe-blockfma th:nth-child(1) { width: 34%; } bbe-blockfma th:nth-child(2) { width: 18%; } bbe-blockfma th:nth-child(3) { width: 17%; }
bbe-blockfma td, bbe-blockfma th { padding: 2px 10px 2px 0; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }
bbe-blockfma th { font-weight: 600; color: var(--muted); font-size: 12px; }
bbe-blockfma td.m { font-family: var(--mono); font-variant-numeric: tabular-nums; }
bbe-blockfma tr.ref td { color: var(--muted); }
bbe-blockfma .dg { color: var(--orange); font-weight: 700; }
bbe-blockfma .ok { color: var(--green); } bbe-blockfma .bad { color: var(--orange); }
`;

// bits of |m| * 2^e, as a map exponent -> 0/1, over [lo, hi]
function bitsOf({ m, e }, lo, hi) {
  const a = m < 0n ? -m : m, out = new Map();
  for (let k = hi; k >= lo; k--) out.set(k, k - e < 0 ? 0 : Number((a >> BigInt(k - e)) & 1n));
  return out;
}

function svg(M) {
  const { B } = M, emax = B.emax, qe = B.qe;
  const X = (k) => LW + (emax + LEFT + 1 - k) * CW;   // left edge of the column for exponent k
  const cutX = X(qe) + CW, ptX = X(emax) + CW;
  const rows = [B.terms[16], ...B.terms.slice(0, 16)];
  const sumY = GY + rows.length * RH + 10, H = sumY + RH + 26;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  // headers
  s += `<rect class="zone" x="${cutX}" y="${GY - 4}" width="${X(qe - DROP) + CW - cutX}" height="${rows.length * RH + 4}"/>`;
  const p2 = (k) => '2' + String(k).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
  for (const k of [emax, qe]) s += `<text class="hd" x="${X(k) + CW / 2}" y="${GY - 22}" text-anchor="middle">${p2(k)}</text>`;
  s += `<text class="hd" x="${(X(emax + 1) + ptX) / 2}" y="${GY - 8}" text-anchor="middle">int</text>`;
  s += `<text class="hd" x="${(ptX + cutX) / 2}" y="${GY - 8}" text-anchor="middle">${BLOCK_FRAC} fraction bits</text>`;
  s += `<text class="hd o" x="${cutX + 4}" y="${GY - 8}">truncated</text>`;
  s += `<line class="pt" x1="${ptX}" x2="${ptX}" y1="${GY - 18}" y2="${sumY + RH}"/>`;
  s += `<line class="cut" x1="${cutX}" x2="${cutX}" y1="${GY - 18}" y2="${sumY - 6}"/>`;
  rows.forEach((t, r) => {
    const y = GY + r * RH, isC = t.kind === 'acc', i = r - 1;
    const v = exactToNumber(t.exact);
    const name = isC ? 'c (acc)' : `a${SUB(i)}·b${SUB(i)}`;
    s += `<text class="lab" x="0" y="${y + 9}">${name}</text><text x="${LW - 8}" y="${y + 9}" text-anchor="end">${v < 0 ? '−' : '+'}${Math.abs(v).toPrecision(5)}</text>`;
    // significand span: products 2 integer + 14 fraction bits from 2^(E+1); c 24 bits from 2^E
    const hi = isC ? t.E : t.E + 1, lo = isC ? t.E - 23 : t.E - 14;
    const bits = bitsOf(t.exact, lo, hi);
    let more = 0;
    for (const [k, b] of bits) {
      if (k < qe - DROP) { more++; continue; }
      if (k > emax + LEFT + 1) continue;
      const kept = k >= qe;
      s += `<rect class="${kept ? 'k' : 'd'}${b}" x="${X(k) + 1.5}" y="${y + 1.5}" width="${CW - 3}" height="${RH - 3}" rx="1.5"/>`;
    }
    if (more) s += `<text class="more" x="${X(qe - DROP) + CW + 3}" y="${y + 9}">+${more}</text>`;
  });
  // the exact sum of the kept addends, then truncated to 24 bits
  const sum = B.keptSum, sv = exactToNumber(sum);
  s += `<line class="sep" x1="0" x2="${X(qe - DROP) + CW}" y1="${sumY - 5}" y2="${sumY - 5}"/>`;
  s += `<text class="lab" x="0" y="${sumY + 9}">sum, exact</text><text x="${LW - 8}" y="${sumY + 9}" text-anchor="end">${sv < 0 ? '−' : '+'}${Math.abs(sv).toPrecision(5)}</text>`;
  if (sum.m !== 0n) {
    const a = sum.m < 0n ? -sum.m : sum.m, top = a.toString(2).length - 1 + sum.e;
    for (const [k, b] of bitsOf(sum, qe, top)) {
      const kept = k > top - 24;
      s += `<rect class="${kept ? 's' : 't'}${b}" x="${X(k) + 1.5}" y="${sumY + 1.5}" width="${CW - 3}" height="${RH - 3}" rx="1.5"/>`;
    }
  }
  const ly = H - 12, sw = (x, cls, label) => `<rect class="${cls}" x="${x}" y="${ly}" width="${CW - 3}" height="${RH - 3}" rx="1.5"/><text class="hd" x="${x + CW + 2}" y="${ly + 9}">${label}</text>`;
  s += sw(0, 'k1', 'kept') + sw(64, 'd1', 'truncated') + sw(158, 's1', 'fp32 result bits');
  return s + '</svg>';
}

class BbeBlockfma extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-blockfma-css')) {
      const st = document.createElement('style'); st.id = 'bbe-blockfma-css'; st.textContent = CSS; document.head.append(st);
    }
    this.n = 1; this.big = false;
    this.render();
    this.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.a === 'next') this.n++;
      else if (b.dataset.v) this.big = b.dataset.v === 'big';
      else return;
      this.render();
    });
  }
  render() {
    const M = model(this.n, this.big), crHex = hex32(M.cr);
    const hexDiff = (x) => [...hex32(x)].map((ch, i) => (ch !== crHex[i] ? `<span class="dg">${ch}</span>` : ch)).join('');
    const ul = (u) => (u === 0 ? '<span class="ok">0</span>' : `<span class="bad">${u > 0 ? '+' : '−'}${Math.abs(u)}</span>`);
    this.innerHTML = `<div class="top"><button class="btn" data-a="next">new inputs</button><span class="num">#${this.n}</span>
      <span class="seg"><button data-v="typ" class="${this.big ? '' : 'on'}">typical c</button><button data-v="big" class="${this.big ? 'on' : ''}">large c</button></span>
</div>
      ${svg(M)}
      <table><tr><th>17 terms summed by</th><th>fp32 bits</th><th>value</th><th>ulps from exact</th></tr>
      <tr><td>tensor core (block FMA)</td><td class="m">${hexDiff(M.B.result)}</td><td class="m">${fmt32(M.B.result)}</td><td class="m">${ul(ulpDiff32(M.B.result, M.cr))}</td></tr>
      <tr><td>fp32 fma chain, k = 0…15</td><td class="m">${hexDiff(M.chain)}</td><td class="m">${fmt32(M.chain)}</td><td class="m">${ul(ulpDiff32(M.chain, M.cr))}</td></tr>
      <tr class="ref"><td>exact, rounded once</td><td class="m">${crHex}</td><td class="m">${fmt32(M.cr)}</td><td class="m">exact: ${exactDecimal(M.exact, 12)}</td></tr>
      </table>`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-blockfma', BbeBlockfma);
