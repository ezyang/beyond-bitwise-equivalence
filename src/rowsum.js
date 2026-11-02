// <bbe-rowsum>: one row of a 128-wide tile, l <- l·alpha + sum(p), summed in
// the FA4-style order (8 interleaved accumulators, previous l into
// accumulator 0, then a fixed tree) and the flex attention order (two
// contiguous halves of 64, added, then one fma with l).  Static.
import { f32, add32, fma32, exact32, exactSum, exactMul, roundExactF32, fmt32, mulberry32 } from './fp.js';

// 128 probabilities 2^-x, x uniform in [0, 6.6], as 3-digit decimals rounded
// to fp32 (each prints back as written); seed chosen so the two orders land
// on opposite sides of the exact sum
const rnd = mulberry32(469);
export const P = Array.from({ length: 128 }, () => f32(+(2 ** (-6.6 * rnd())).toPrecision(3)));
export const L0 = f32(41.7);                 // previous row sum; alpha = 1
export const ALPHA = 1;
const TREE = [[[0, 2], [4, 6]], [[1, 3], [5, 7]]];

export function model(p = P, l = L0) {
  const acc = [f32(l * ALPHA), 0, 0, 0, 0, 0, 0, 0];
  p.forEach((x, k) => { acc[k % 8] = add32(acc[k % 8], x); });
  const ev = (t) => (typeof t === 'number' ? acc[t] : add32(ev(t[0]), ev(t[1])));
  let a = 0, b = 0;
  for (let k = 0; k < 64; k++) { a = add32(a, p[k]); b = add32(b, p[k + 64]); }
  const exact = exactSum([exactMul(exact32(l), exact32(ALPHA)), ...p.map(exact32)]);
  const cr = roundExactF32(exact);
  return { acc, fa4: ev(TREE), a, b, flex: fma32(l, ALPHA, add32(a, b)), exact, cr };
}

// colours: accumulator j is blue (even) or purple (odd), shade by j >> 1
// (0/1 toward ink, 4-7 toward the background);
// a merge node is the mix of what it merges; l is amber
const shade = (j) => `color-mix(in srgb, var(--${j % 2 ? 'purple' : 'blue'}) ${[68, 100, 72, 46][j >> 1]}%, var(--${j < 2 ? 'ink' : 'card'}))`;
const mix = (a, b) => `color-mix(in srgb, ${a}, ${b})`;
const ACC = [0, 1, 2, 3, 4, 5, 6, 7].map(shade), T0 = shade(2), T1 = shade(3);

const W = 720, X0 = 40, CW = 5.25, SY = 30, SH = 18;    // strip: left, cell width, top, height
const BT = 82, BW = 14, BH = 4;                         // FA4 block: top, column width, cell height (+1 gap)
const NY = BT + 16 * (BH + 1) + 14, D = 30;             // FA4 accumulator row, level spacing
const SEC = NY + 3 * D + 24, H = SEC + 214;             // flex starts at SEC
const NX = (k) => 222 + 44 * k;                         // node slots, tree order (centred under the strip)
const RX = 560;                                         // result boxes
const CSS = `
bbe-rowsum { display: block; height: ${H}px; max-width: ${W}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-rowsum svg { display: block; font-size: 11px; }
bbe-rowsum text { fill: var(--muted); }
bbe-rowsum .pt { font-weight: 600; font-size: 13px; fill: var(--ink); }
bbe-rowsum .pt.fa4 { fill: var(--fa4-text, var(--ink)); }
bbe-rowsum .pt.flex { fill: var(--flex-text, var(--ink)); }
bbe-rowsum .ps { fill: var(--faint); }
bbe-rowsum .nl { paint-order: stroke; stroke: var(--bg); stroke-width: 3px; stroke-linejoin: round; }
bbe-rowsum .ix { font: 10px var(--mono); fill: var(--faint); }
bbe-rowsum .e { stroke: var(--rule-strong); stroke-width: 1.5; fill: none; }
bbe-rowsum .ah { fill: var(--rule-strong); }
bbe-rowsum .nd { stroke: var(--card); stroke-width: 1.5; }
bbe-rowsum .lc { fill: var(--amber); }
bbe-rowsum .box { fill: var(--card); stroke: var(--rule-strong); }
bbe-rowsum .v { font: 600 14px var(--mono); }
bbe-rowsum .ok { fill: var(--green); } bbe-rowsum .dg { fill: var(--orange); }
bbe-rowsum .ref { font: 12px var(--mono); fill: var(--faint); }
`;

const line = (x1, y1, x2, y2, col = '') => `<line class="e" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"${col ? ` stroke="${col}"` : ''}/>`;
const node = (x, y, col, r = 6) => `<circle class="nd" cx="${x}" cy="${y}" r="${r}" fill="${col}"/>`;
const label = (x, y, t, anchor = 'end') => `<text class="nl" x="${x}" y="${y + 4}" text-anchor="${anchor}">${t}</text>`;
const rect = (x, y, w, h, fill) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}"/>`;
// a square bracket under [x1, x2] at y, with a stem from its middle down to (x, y2)
const bracket = (x1, x2, y, x, y2, col = '') => `<path class="e" d="M${x1} ${y} v5 H${x2} v-5 M${(x1 + x2) / 2} ${y + 5} L${x} ${y2}"${col ? ` stroke="${col}"` : ''}/>`;

// the 128-cell strip, coloured by owner
function strip(y0, col) {
  let s = '';
  for (let k = 0; k < 128; k++) s += rect(X0 + CW * k, y0 + SY, CW + 0.3, SH, col(k));
  return s + `<text class="ix" x="${X0}" y="${y0 + SY - 5}">p[0]</text><text class="ix" x="${X0 + 128 * CW}" y="${y0 + SY - 5}" text-anchor="end">p[127]</text>`;
}
const title = (y0, t, sub, who) => `<text class="pt ${who}" x="0" y="${y0 + 14}">${t}</text><text class="ps" x="${t.length * 7.6 + 12}" y="${y0 + 14}">${sub}</text>`;

// final value, digits that agree with the exact sum rounded once in green, the rest orange
function result(x, y, v, ref) {
  const s = fmt32(v), r = fmt32(ref), k = [...s].findIndex((ch, i) => ch !== r[i]);
  const cut = k < 0 ? s.length : k;
  return line(x, y, RX, y) + `<rect class="box" x="${RX}" y="${y - 14}" width="110" height="28" rx="4"/>` +
    `<text class="v" x="${RX + 55}" y="${y + 5}" text-anchor="middle"><tspan class="ok">${s.slice(0, cut)}</tspan><tspan class="dg">${s.slice(cut)}</tspan></text>`;
}

function fa4(M) {
  // the strip folded into 8 columns of 16, one per accumulator, in tree order;
  // l sits on top of accumulator 0's column
  const ord = [0, 2, 4, 6, 1, 3, 5, 7], cx = X0 + 64 * CW;
  let o = title(0, 'FA4-style', '1 thread · 8 interleaved accumulators', 'fa4') + strip(0, (k) => ACC[k % 8]);
  o += bracket(X0, X0 + 128 * CW, SY + SH + 3, cx, BT - 12) + `<path class="ah" d="M${cx - 4} ${BT - 13} h8 l-4 6 z"/>`;
  ord.forEach((j, k) => {
    for (let i = 0; i < 16; i++) o += rect(NX(k) - BW / 2, BT + i * (BH + 1), BW, BH, ACC[j]);
    o += line(NX(k), BT + 16 * (BH + 1), NX(k), NY - 6);
  });
  o += `<rect class="lc" x="${NX(0) - BW / 2}" y="${BT - 7}" width="${BW}" height="5"/>` + label(NX(0) - 12, BT - 6, 'ℓ');
  // ((0+2)+(4+6)) + ((1+3)+(5+7)); in tree order the leaves are adjacent
  let lv = ord.map((j, k) => ({ x: NX(k), col: ACC[j] })), y = NY, n = '', e = '';
  for (let k = 0; k < 8; k++) n += node(lv[k].x, y, lv[k].col) + label(lv[k].x - 10, y, ord[k]);
  while (lv.length > 1) {
    const up = [];
    for (let k = 0; k < lv.length; k += 2) {
      const a = lv[k], b = lv[k + 1], p = { x: (a.x + b.x) / 2, col: mix(a.col, b.col) };
      e += line(a.x, y, p.x, y + D) + line(b.x, y, p.x, y + D);
      n += node(p.x, y + D, p.col, 5);
      up.push(p);
    }
    lv = up; y += D;
  }
  return o + label(NX(0) - 24, NY, 'acc') + result(lv[0].x, y, M.fa4, M.cr) + e + n;
}

function flex(M) {
  const y0 = SEC, ya = y0 + 92, mid = X0 + 64 * CW, xa = (X0 + mid - 2) / 2, xb = (mid + 2 + X0 + 128 * CW) / 2, xs = mid;
  const yp = ya + D + 6, yf = yp + D + 6, cs = mix(T0, T1), yb = y0 + SY + SH + 3;
  let o = title(y0, 'flex attention', '2 threads · 64 contiguous each', 'flex') + strip(y0, (k) => (k < 64 ? T0 : T1));
  o += bracket(X0, mid - 2, yb, xa, ya - 6, T0) + bracket(mid + 2, X0 + 128 * CW, yb, xb, ya - 6, T1);
  o += line(xa, ya, xs, yp) + line(xb, ya, xs, yp) + line(xs, yp, xs, yf) + line(xs - 110 + 14, yf, xs, yf) + result(xs, yf, M.flex, M.cr);
  o += `<rect class="lc" x="${xs - 110}" y="${yf - SH / 2}" width="14" height="${SH}" rx="2"/>` + label(xs - 103, yf - SH / 2 - 9, 'ℓ', 'middle');
  o += node(xa, ya, T0) + node(xb, ya, T1) + node(xs, yp, cs, 5) + node(xs, yf, mix(cs, 'var(--amber)'), 5);
  o += label(xa - 10, ya, 'thread 0') + label(xb + 10, ya, 'thread 1', 'start') + label(xs - 10, yp, '+') + label(xs - 10, yf - 12, 'fma');
  o += `<text class="ref" x="${RX + 55}" y="${yf + 34}" text-anchor="middle">exact: ${fmt32(M.cr)}</text>`;
  return o;
}

class BbeRowsum extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-rowsum-css')) {
      const st = document.createElement('style'); st.id = 'bbe-rowsum-css'; st.textContent = CSS; document.head.append(st);
    }
    const M = model();
    this.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${fa4(M)}${flex(M)}</svg>`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-rowsum', BbeRowsum);
