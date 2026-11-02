// <bbe-scaleplace>: where 1/√d and log2 e enter the ex2 argument x, for one
// score s and row max, in the three pipelines: FA4-style (log2 e/√d folded into
// one constant, one fma on raw scores), flex on a partial block (mul, mul, mask, sub; the
// mask after both muls, as LLVM leaves it) and
// flex on a full block (mul, then the compiler's fma).  Static.
import { f32, fma32, hex32, ulpDiff32 } from './fp.js';

export const D = 128, S = 19, MAX = 24;               // head_dim, score, row max (fp32)
export const A = f32(1 / Math.sqrt(D)), B = f32(Math.LOG2E), C = f32(A * B);
export function example(s = S, max = MAX) {
  const mc = f32(max * C), mf = f32(f32(max * A) * B);   // each kernel's m
  const t1 = f32(s * A), t2 = f32(t1 * B);
  const exact = ((s - max) * Math.LOG2E) / Math.sqrt(D);
  const x = { fa4: fma32(s, C, -mc), part: f32(t2 - mf), full: fma32(t1, B, -mf) };
  const ulps = {};
  for (const k in x) ulps[k] = ulpDiff32(x[k], f32(exact));
  return { mc, mf, t1, t2, x, exact, ulps };
}
export const fmt9 = (v) => v.toPrecision(9).replace('-', '−');
export const fmtExact = (v) => v.toPrecision(10).replace('-', '−');

const W = 820, X0 = 160, BW = 90, GAP = 24, BH = 28, RH = 84;
const col = (i) => X0 + i * (BW + GAP);
const XR = col(3) + BW, OX = 620;                       // flow right edge, output column
const HY = 96, R0 = HY + 22;                            // column headers, first row top
const EY = R0 + 3 * RH, H = EY + 70;

const CSS = `
bbe-scaleplace { display: block; height: ${H + 4}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-scaleplace svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-scaleplace text { fill: var(--muted); }
bbe-scaleplace .th { font-size: 10.5px; fill: var(--faint); }
bbe-scaleplace .v { font: 11.5px var(--mono); font-variant-numeric: tabular-nums; fill: var(--ink); }
bbe-scaleplace .hx { font: 10.5px var(--mono); fill: var(--faint); }
bbe-scaleplace .ex { fill: var(--blue); }
bbe-scaleplace .ag { fill: var(--green); }
bbe-scaleplace .df { fill: var(--orange); }
bbe-scaleplace .num { text-anchor: end; }
bbe-scaleplace .rt { font-weight: 600; font-size: 12px; fill: var(--text); }
bbe-scaleplace .rt.fa4 { fill: var(--fa4-text, var(--text)); }
bbe-scaleplace .rt.flex { fill: var(--flex-text, var(--text)); }
bbe-scaleplace .rs { font-size: 10.5px; fill: var(--faint); }
bbe-scaleplace .box { fill: var(--card); stroke: var(--rule-strong); }
bbe-scaleplace .box.cmp { fill: var(--purple-soft); stroke: var(--purple); }
bbe-scaleplace .op { font: 11.5px var(--mono); fill: var(--ink); text-anchor: middle; }
bbe-scaleplace .op.cmp { fill: var(--purple); }
bbe-scaleplace .wire { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-sp-ah); }
bbe-scaleplace #bbe-sp-ah path { fill: var(--faint); }
bbe-scaleplace .rnd { fill: var(--orange); }
bbe-scaleplace .rule { stroke: var(--rule); }
`;

const dot = (x, y) => `<circle class="rnd" cx="${x}" cy="${y}" r="4"/>`;
const T = (x, y, cls, s) => `<text class="${cls}" x="${x}" y="${y}">${s}</text>`;
// digits of v coloured against the exact decimal: green while they agree, orange after
function digits(v, ref, x, y) {
  let k = 0;
  while (k < v.length && v[k] === ref[k]) k++;
  return `<text class="v" x="${x}" y="${y}"><tspan class="ag">${v.slice(0, k)}</tspan><tspan class="df">${v.slice(k)}</tspan></text>`;
}

function svg() {
  const e = example(), ref = fmtExact(e.exact);
  const o = [`<defs><marker id="bbe-sp-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>`];
  // constants: exact value, the fp32 the kernel uses, its bits
  const k0 = 0, k1 = 120, k2 = 250, k3 = 370;
  o.push(T(k0, 12, 'th', 'constant'), T(k1, 12, 'th', 'exact'), T(k2 + 12, 12, 'th', 'fp32'), T(k3, 12, 'th', 'bits'));
  const consts = [
    ['1/√d', fmtExact(1 / Math.sqrt(D)), A],
    ['log2 e', fmtExact(Math.LOG2E), B],
    ['log2 e / √d', fmtExact(Math.LOG2E / Math.sqrt(D)), C],
  ];
  consts.forEach(([n, ex, v], i) => {
    const y = 32 + i * 18;
    o.push(T(k0, y, 'v', n), T(k1, y, 'v ex', ex), dot(k2 + 4, y - 4), T(k2 + 12, y, 'v', fmt9(v)), T(k3, y, 'hx', hex32(v)));
  });
  // inputs
  o.push(T(OX, 12, 'th', 'input (fp32)'));
  o.push(T(OX, 32, 'v', `s = ${S}`), T(OX, 50, 'v', `row max = ${MAX}`));
  // column headers
  o.push(`<path class="rule" d="M0 ${HY - 16} H${W}"/>`);
  o.push(T(OX, HY, 'th', 'x'), `<text class="th num" x="${W - 2}" y="${HY}">ulps from fp32(exact)</text>`);
  // rows: [title, subtitle, boxes, m tag, intermediate values, x key]
  // a box is [first col, last col, label, rounds, compiler]
  const rows = [
    ['FA4-style', 'raw units', [[0, 3, 'fma(s, log2 e/√d, −m·log2 e/√d)', true]],
      ['m·log2 e/√d', e.mc, 1], [], 'fa4'],
    ['flex attention', 'partial block',
      [[0, 0, '× 1/√d', true], [1, 1, '× log2 e', true], [2, 2, 'mask', false], [3, 3, '− m', true]],
      ['m', e.mf, 2], [e.t1, e.t2, e.t2], 'part'],
    ['flex attention', 'full block', [[0, 0, '× 1/√d', true], [1, 3, 'fma(·, log2 e, −m)', true, true]],
      ['m', e.mf, 2], [e.t1], 'full'],
  ];
  rows.forEach(([title, sub, boxes, [mn, mv, md], mids, key], r) => {
    const y0 = R0 + r * RH, by = y0 + 22, cy = by + BH / 2;
    o.push(T(0, cy - 2, key === 'fa4' ? 'rt fa4' : 'rt flex', title), T(0, cy + 13, 'rs', sub));
    o.push(T(X0 - 34, cy + 4, 'v', 's'));
    let px = X0 - 22;                                    // where the incoming wire starts
    boxes.forEach(([c0, c1, label, rounds, cmp], i) => {
      const bx = col(c0), bw = col(c1) + BW - bx;
      o.push(`<path class="wire" d="M${px} ${cy} H${bx}"/>`);
      o.push(`<rect class="box${cmp ? ' cmp' : ''}" x="${bx}" y="${by}" width="${bw}" height="${BH}" rx="4"/>`);
      o.push(`<text class="op${cmp ? ' cmp' : ''}" x="${bx + bw / 2}" y="${cy + 4}">${label}</text>`);
      if (i < mids.length) o.push(`<text class="v" x="${bx + bw / 2}" y="${by + BH + 17}" text-anchor="middle">${fmt9(mids[i])}</text>`);
      if (rounds) o.push(dot(bx + bw + GAP / 2, cy));
      px = bx + bw;
    });
    o.push(`<path class="wire" d="M${px} ${cy} H${OX - 10}"/>`);
    // the m operand, above the box that takes it
    const tag = `${mn} = ${fmt9(mv)}`;
    o.push(`<text class="v" x="${XR}" y="${by - 8}" text-anchor="end">${tag}</text>`);
    for (let k = 0; k < md; k++) o.push(dot(XR - tag.length * 6.95 - 10 - k * 11, by - 12));
    // the result
    o.push(digits(fmt9(e.x[key]), ref, OX, cy + 1), T(OX, cy + 17, 'hx', hex32(e.x[key])));
    const u = e.ulps[key];
    o.push(`<text class="v num" x="${W - 2}" y="${cy + 1}">${u > 0 ? '+' : u < 0 ? '−' : ''}${Math.abs(u)}</text>`);
  });
  // the exact value
  o.push(`<path class="rule" d="M0 ${EY + 4} H${W}"/>`);
  const ey = EY + 24;
  o.push(T(0, ey, 'rt', 'exact'), T(X0, ey, 'v ex', '(s − row max) · log2 e / √d'), T(OX, ey, 'v ex', ref));
  // legend
  const ly = H - 8;
  o.push(dot(4, ly - 4), T(14, ly, '', 'rounds to fp32'));
  o.push(`<rect class="box cmp" x="120" y="${ly - 10}" width="16" height="12" rx="2"/>`, T(142, ly, '', 'fma made by the compiler'));
  o.push(`<text x="310" y="${ly}"><tspan class="ag">digits</tspan> agree, <tspan class="df">digits</tspan> differ from exact</text>`);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Three ways to compute the ex2 argument from one score: FA4-style folds 1/sqrt(d) and log2 e into one constant and uses one fma; flex multiplies twice and subtracts on partial blocks, and multiplies then uses a compiler-made fma on full blocks. The three results differ in the last bits.">${o.join('')}</svg>`;
}

class BbeScaleplace extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-scaleplace-css')) {
      const s = document.createElement('style'); s.id = 'bbe-scaleplace-css'; s.textContent = CSS; document.head.append(s);
    }
    this.innerHTML = svg();
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('bbe-scaleplace')) customElements.define('bbe-scaleplace', BbeScaleplace);
