// <bbe-lse-heatmap>: the fp32 natural-log lse the FA4-style kernel and flex
// attention write, measured on a GPU over a grid of (M, l), M the running max
// in log2 units.  One panel per kernel, its ulps from the correctly rounded
// lse, and a third panel saying which kernel is closer and by how many ulps
// (|FA4 error| - |flex error|).  Cells whose
// exact lse = (M + log2 l) ln 2 has |lse| < 1 are greyed and not counted:
// near lse = 0 an ulp of lse is tiny.
//
// Data: the lse entry of data/div-measured.json, written by
// scripts/measure-div.py, whose header documents the grid and the exact lse
// expressions.  lg2.approx has no published exact definition, so there is
// nothing to emulate.  If the file is missing or `null`, the panels show
// "no data".  Every number shown is asserted against the JSON by
// scripts/check-transc.mjs.

const KERNELS = ['FA4-style', 'flex'], COLS = [...KERNELS, 'closer to exact'];
const FN = { 'FA4-style': 'lg2.approx(ℓ)', flex: 'log2f(ℓ)', 'closer to exact': 'FA4-style vs flex' };

// counts per bin [<=-2, -1, 0, +1, >=+2]
export const bins = (u) => u.reduce((c, v) => (c[Math.max(-2, Math.min(2, v)) + 2]++, c), [0, 0, 0, 0, 0]);
const pct = (c, n) => (c === 0 ? '0%' : 100 * c / n < 0.1 ? '<0.1%' : (100 * c / n).toFixed(1) + '%');
// per kernel: [5 bin percentages, max |ulps|] over the counted cells
const stats = (u) => [...bins(u).map((c) => pct(c, u.length)), Math.max(...u.map(Math.abs))];
// third panel, per cell: 0/1 FA4 closer by >=2 / 1 ulp, 2 same bits, 3 same
// |error| on opposite sides, 4/5 flex closer by 1 / >=2 ulps
const CLOSER = ['f2', 'f1', 'c0', 'tie', 'x1', 'x2'];
const closer = (a, b) => (a === b ? 2 : Math.abs(a) === Math.abs(b) ? 3 : Math.abs(a) < Math.abs(b)
  ? (Math.abs(b) - Math.abs(a) >= 2 ? 0 : 1) : (Math.abs(a) - Math.abs(b) >= 2 ? 5 : 4));
// lse cells drawn grey: exact |lse| < 1
const nearZero = (L) => L.M.flatMap((m) => L.l.map((l) => Math.abs((m + Math.log2(l)) * Math.LN2) < 1));
// the third panel's classes, one per cell
const closers = (L) => L[KERNELS[0]].map((a, c) => closer(a, L[KERNELS[1]][c]));
// every number the widget prints, from the JSON
export const shown = (d) => {
  const near = nearZero(d.lse), far = (u) => u.filter((_, c) => !near[c]), w = far(closers(d.lse));
  return {
    lse: KERNELS.map((k) => stats(far(d.lse[k]))),
    closer: CLOSER.map((_, q) => pct(w.filter((v) => v === q).length, w.length)),
    grey: pct(near.filter(Boolean).length, near.length),
  };
};

const S = 128, TW = 118, X = [44, 360, 676], Y = 64, W = 936, H = 228;
const BIN_LABEL = ['≤−2', '−1', '0', '+1', '≥+2'];
const cls = (v) => (v <= -2 || v >= 2 ? 'c2' : v ? (v < 0 ? 'cm' : 'cp') : 'c0');
const BIN_CLS = ['c2', 'cm', 'c0', 'cp', 'c2'];
const CSS = `
bbe-lse-heatmap { display: block; height: ${H}px; max-width: ${W}px; margin: 18px 0 26px; }
bbe-lse-heatmap svg { display: block; font: 11px var(--mono); font-variant-numeric: tabular-nums; }
bbe-lse-heatmap text { fill: var(--muted); }
bbe-lse-heatmap .k { fill: var(--ink); font: 600 13px var(--sans); }
bbe-lse-heatmap .fn { fill: var(--muted); font: 11.5px var(--mono); }
bbe-lse-heatmap text.ax { font: 11.5px var(--sans); }
bbe-lse-heatmap text.tag { fill: var(--faint); font: 11.5px var(--sans); }
bbe-lse-heatmap text.none { fill: var(--faint); font: 12px var(--sans); }
bbe-lse-heatmap .c0 { fill: var(--panel); }
bbe-lse-heatmap .cm { fill: color-mix(in srgb, var(--orange) 40%, var(--card)); }
bbe-lse-heatmap .cp { fill: var(--orange); }
bbe-lse-heatmap .c2 { fill: var(--red); }
bbe-lse-heatmap .cx { fill: var(--rule-strong); }
bbe-lse-heatmap .f2 { fill: var(--fa4, var(--amber)); }
bbe-lse-heatmap .f1 { fill: color-mix(in srgb, var(--fa4, var(--amber)) 40%, var(--card)); }
bbe-lse-heatmap .x2 { fill: var(--flex, var(--blue)); }
bbe-lse-heatmap .x1 { fill: color-mix(in srgb, var(--flex, var(--blue)) 40%, var(--card)); }
bbe-lse-heatmap .tie { fill: var(--orange); }
bbe-lse-heatmap .fr { fill: none; stroke: var(--rule-strong); stroke-width: 1; }
bbe-lse-heatmap .fr.none { fill: var(--card); stroke-dasharray: 3 3; }
bbe-lse-heatmap .cells { shape-rendering: crispEdges; }
bbe-lse-heatmap .rule { stroke: var(--rule); stroke-width: 1; }
`;

const t = (x, y, s, c = '', a = 'start') => `<text x="${x}" y="${y}" text-anchor="${a}"${c ? ` class="${c}"` : ''}>${s}</text>`;
const none = (x, y) => `<rect class="fr none" x="${x}" y="${y}" width="${S}" height="${S}"/>` + t(x + S / 2, y + S / 2 + 4, 'no data', 'none', 'middle');
const pow2 = (e) => (e ? `2<tspan dy="-4" font-size="9">${e}</tspan>` : '1');

// heatmap at (x0, y0) of n x n cells with class cl(cell index), row 0 at the
// bottom, runs of equal classes merged; grey[c] marks a cell not counted
function heat(x0, y0, cl, n, grey) {
  const c = S / n, k = (i) => (grey[i] ? 'cx' : cl(i));
  let s = `<g class="cells"><rect class="c0" x="${x0}" y="${y0}" width="${S}" height="${S}"/>`;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n;) {
      const q = k(j * n + i);
      let e = i + 1;
      while (e < n && k(j * n + e) === q) e++;
      if (q !== 'c0') s += `<rect class="${q}" x="${x0 + i * c}" y="${y0 + S - (j + 1) * c}" width="${(e - i) * c}" height="${c}"/>`;
      i = e;
    }
  }
  return s + `</g><rect class="fr" x="${x0}" y="${y0}" width="${S}" height="${S}"/>`;
}
// axes: ticks as [fraction, label]
function axes(x0, y0, xt, xl, yt, yl) {
  let s = xt.map(([f, lab]) => t(x0 + S * f, y0 + S + 14, lab, '', 'middle')).join('');
  s += yt.map(([f, lab]) => t(x0 - 6, y0 + S - S * f + 4, lab, '', 'end')).join('');
  s += t(x0 + S / 2, y0 + S + 28, xl, 'ax', 'middle');
  return s + `<text class="ax" transform="translate(${x0 - 34} ${y0 + S / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
}
// table to the right of a heatmap: rows [class or '', label, value], a rule
// above row `rule`
function table(x0, y0, title, rows, rule) {
  const x = x0 + S + 10, xr = x + TW;
  let s = t(x, y0 + 9, title, 'ax');
  rows.forEach(([k, lab, v], r) => {
    const y = y0 + 26 + 15 * r;
    if (r === rule) s += `<line class="rule" x1="${x}" x2="${xr}" y1="${y - 11}" y2="${y - 11}"/>`;
    s += (k ? `<rect class="${k}" x="${x}" y="${y - 9}" width="10" height="10"/>` + t(x + 14, y, lab) : t(x, y, lab)) + t(xr, y, v, '', 'end');
  });
  return s;
}
const CLOSER_LABEL = ['FA4 by ≥2', 'FA4 by 1', 'same bits', 'same |err|', 'flex by 1', 'flex by ≥2'];

class BbeLseHeatmap extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-lse-heatmap-css')) {
      const st = document.createElement('style'); st.id = 'bbe-lse-heatmap-css'; st.textContent = CSS; document.head.append(st);
    }
    this.render(null);
    fetch(new URL('../data/div-measured.json', import.meta.url)).then((r) => (r.ok ? r.json() : null))
      .then((d) => d?.lse && this.render(d), () => {});
  }
  render(d) {
    const sh = d && shown(d), L = d?.lse, near = d && nearZero(L), v = d?.versions;
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    if (d) s += t(X[0], 12, `measured: ${d.device}, driver ${v.driver}, CUDA ${v.cuda}`, 'tag');
    COLS.forEach((k, c) => {
      const x0 = X[c], grey = ['cx', '|lse| < 1', sh?.grey];
      s += t(x0, Y - 26, k, 'k') + t(x0, Y - 8, FN[k], 'fn');
      if (!d) s += none(x0, Y);
      else if (c < 2) s += heat(x0, Y, (i) => cls(L[k][i]), L.n, near)
        + table(x0, Y, 'ulps', [...BIN_LABEL.map((lab, b) => [BIN_CLS[b], lab, sh.lse[c][b]]), ['', 'max |ulps|', sh.lse[c][5]], grey], 5);
      else {
        const w = closers(L);
        s += heat(x0, Y, (i) => (w[i] === 2 ? 'c0' : CLOSER[w[i]]), L.n, near)
          + table(x0, Y, 'closer', [...CLOSER_LABEL.map((lab, q) => [CLOSER[q], lab, sh.closer[q]]), grey], 6);
      }
      s += axes(x0, Y, [0, 5, 10, 15].map((e) => [e / 15, pow2(e)]), 'ℓ', [-16, 0, 16, 32, 48].map((m) => [(m + 16) / 64, m]), 'M (log2 units)');
    });
    this.innerHTML = s + '</svg>';
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-lse-heatmap', BbeLseHeatmap);
