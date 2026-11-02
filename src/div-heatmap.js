// <bbe-div-heatmap>: the two "function choice" sites, FA4-style kernel next
// to flex attention, measured on a GPU.  Columns are the kernels, rows the
// ops; each panel shows that kernel's ulps from the correctly rounded result.
//   out = O / l   FA4-style O * rcp.approx(l), flex div.full(O, l), over a
//                 grid of fp32 (O, l)
//   lse           the fp32 natural-log lse each kernel writes, over a grid of
//                 (M, l), M the running max in log2 units.  Cells whose exact
//                 lse = (M + log2 l) ln 2 has |lse| < 1 are greyed and not
//                 counted: near lse = 0 an ulp of lse is tiny.
//
// Data: data/div-measured.json, written by scripts/measure-div.py, whose
// header documents the grids and the exact lse expressions.  rcp.approx and
// lg2.approx have no published exact definition, so there is nothing to
// emulate.  If the file is missing or `null`, the panels show "no data".
// Every number shown is asserted against the JSON by scripts/check-transc.mjs.
//
// Division grid: column i is O = fp32(2^((i + 0.5) / n)), row j is
// l = fp32(2^((j + 0.5) / n)): one binade each, log-spaced so every input has
// a full significand; scaling O or l by a power of two scales the quotient
// exactly, so every binade gives the same picture.

const KERNELS = ['FA4-style', 'flex'];
const DIV = { 'FA4-style': 'rcp.approx.ftz.f32 then mul.rn.f32', flex: 'div.full.f32' };

// counts per bin [<=-2, -1, 0, +1, >=+2]
export const bins = (u) => u.reduce((c, v) => (c[Math.max(-2, Math.min(2, v)) + 2]++, c), [0, 0, 0, 0, 0]);
const pct = (c, n) => (c === 0 ? '0%' : 100 * c / n < 0.1 ? '<0.1%' : (100 * c / n).toFixed(1) + '%');
// per kernel: [5 bin percentages, max |ulps|] over the counted cells
const stats = (u) => [...bins(u).map((c) => pct(c, u.length)), Math.max(...u.map(Math.abs))];
// lse cells drawn grey: exact |lse| < 1
const nearZero = (L) => L.M.flatMap((m) => L.l.map((l) => Math.abs((m + Math.log2(l)) * Math.LN2) < 1));
// every number the widget prints, from the JSON
export const shown = (d) => {
  const L = d.lse, near = nearZero(L);
  return {
    out: KERNELS.map((k) => stats(d.div[DIV[k]])),
    lse: KERNELS.map((k) => stats(L[k].filter((_, c) => !near[c]))),
    lseGrey: pct(near.filter(Boolean).length, near.length),
  };
};

const S = 160, TW = 112, X = [44, 396], ROW = [84, 336], W = 684, H = 548;
const BIN_LABEL = ['≤−2', '−1', '0', '+1', '≥+2'];
const cls = (v) => (v <= -2 || v >= 2 ? 'c2' : v ? (v < 0 ? 'cm' : 'cp') : 'c0');
const BIN_CLS = ['c2', 'cm', 'c0', 'cp', 'c2'];
const CSS = `
bbe-div-heatmap { display: block; height: ${H}px; max-width: ${W}px; margin: 18px 0 26px; }
bbe-div-heatmap svg { display: block; font: 11px var(--mono); font-variant-numeric: tabular-nums; }
bbe-div-heatmap text { fill: var(--muted); }
bbe-div-heatmap .k { fill: var(--ink); font: 600 13px var(--sans); }
bbe-div-heatmap .op { fill: var(--ink); font: 600 12px var(--mono); }
bbe-div-heatmap .fn { fill: var(--muted); font: 11.5px var(--mono); }
bbe-div-heatmap text.ax { font: 11.5px var(--sans); }
bbe-div-heatmap text.tag { fill: var(--faint); font: 11.5px var(--sans); }
bbe-div-heatmap text.none { fill: var(--faint); font: 12px var(--sans); }
bbe-div-heatmap .c0 { fill: var(--panel); }
bbe-div-heatmap .cm { fill: color-mix(in srgb, var(--orange) 40%, var(--card)); }
bbe-div-heatmap .cp { fill: var(--orange); }
bbe-div-heatmap .c2 { fill: var(--red); }
bbe-div-heatmap .cx { fill: var(--rule-strong); }
bbe-div-heatmap .fr { fill: none; stroke: var(--rule-strong); stroke-width: 1; }
bbe-div-heatmap .fr.none { fill: var(--card); stroke-dasharray: 3 3; }
bbe-div-heatmap .cells { shape-rendering: crispEdges; }
bbe-div-heatmap .rule { stroke: var(--rule); stroke-width: 1; }
`;

const t = (x, y, s, c = '', a = 'start') => `<text x="${x}" y="${y}" text-anchor="${a}"${c ? ` class="${c}"` : ''}>${s}</text>`;
const none = (x, y) => `<rect class="fr none" x="${x}" y="${y}" width="${S}" height="${S}"/>` + t(x + S / 2, y + S / 2 + 4, 'no data', 'none', 'middle');
const pow2 = (e) => (e ? `2<tspan dy="-4" font-size="9">${e}</tspan>` : '1');

// heatmap at (x0, y0), row 0 at the bottom, runs of equal classes merged;
// grey[c] marks a cell drawn as not counted
function heat(x0, y0, u, n, grey) {
  const c = S / n, k = (i) => (grey?.[i] ? 'cx' : cls(u[i]));
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
// ulp bins and max |ulps| to the right of a heatmap; grey adds a row for the
// cells not counted
function binTable(x0, y0, st, grey) {
  const x = x0 + S + 12, xr = x + TW, rows = BIN_LABEL.map((lab, b) =>
    [`<rect class="${BIN_CLS[b]}" x="${x}" y="__-9" width="10" height="10"/>` + t(x + 14, '__', lab), st[b]]);
  rows.push([t(x, '__', 'max |ulps|'), st[5]]);
  if (grey) rows.push([`<rect class="cx" x="${x}" y="__-9" width="10" height="10"/>` + t(x + 14, '__', '|lse| < 1'), grey]);
  let s = t(x, y0 + 9, 'ulps', 'ax');
  rows.forEach(([lab, v], r) => {
    const y = y0 + 30 + 17 * r;
    if (r === 5) s += `<line class="rule" x1="${x}" x2="${xr}" y1="${y - 12}" y2="${y - 12}"/>`;
    s += lab.replace(/"__-9"/g, `"${y - 9}"`).replace(/"__"/g, `"${y}"`) + t(xr, y, v, '', 'end');
  });
  return s;
}

const OPS = [
  { key: 'out', title: 'out = O / ℓ', fn: { 'FA4-style': 'O · rcp.approx(ℓ)', flex: 'div.full(O, ℓ)' } },
  { key: 'lse', title: 'lse = (M + log2 ℓ) · ln 2', fn: { 'FA4-style': 'lg2.approx(ℓ)', flex: 'log2f(ℓ)' } },
];

class BbeDivHeatmap extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-div-heatmap-css')) {
      const st = document.createElement('style'); st.id = 'bbe-div-heatmap-css'; st.textContent = CSS; document.head.append(st);
    }
    this.render(null);
    fetch(new URL('../data/div-measured.json', import.meta.url)).then((r) => (r.ok ? r.json() : null))
      .then((d) => d?.div && this.render(d), () => {});
  }
  render(d) {
    const sh = d && shown(d), v = d?.versions;
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    if (d) s += t(X[0], 12, `measured: ${d.device}, driver ${v.driver}, CUDA ${v.cuda}`, 'tag');
    KERNELS.forEach((k, c) => (s += t(X[c], 40, k, 'k')));
    const half = [[0, '1'], [Math.log2(1.5), '1.5'], [1, '2']];
    OPS.forEach((op, r) => {
      const y0 = ROW[r];
      s += t(X[0] - 40, y0 - 26, op.title, 'op');
      KERNELS.forEach((k, c) => {
        const x0 = X[c];
        s += t(x0, y0 - 8, op.fn[k], 'fn');
        if (!d) s += none(x0, y0);
        else if (op.key === 'out') s += heat(x0, y0, d.div[DIV[k]], d.n) + binTable(x0, y0, sh.out[c]);
        else s += heat(x0, y0, d.lse[k], d.lse.n, nearZero(d.lse)) + binTable(x0, y0, sh.lse[c], sh.lseGrey);
        s += op.key === 'out'
          ? axes(x0, y0, half, 'O (mantissa)', half, 'ℓ (mantissa)')
          : axes(x0, y0, [0, 5, 10, 15].map((e) => [e / 15, pow2(e)]), 'ℓ', [-16, 0, 16, 32, 48].map((m) => [(m + 16) / 64, m]), 'M (log2 units)');
      });
    });
    this.innerHTML = s + '</svg>';
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-div-heatmap', BbeDivHeatmap);
