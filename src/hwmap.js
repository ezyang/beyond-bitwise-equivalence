// <bbe-hwmap>: how one row's elements are dealt out to threads, and the
// order each thread then sums its share in registers, then the warp
// shuffle to one total; a table compares the three mappings' fp32 totals.  Toggle the mapping
// (contiguous spans / strided / vectorised strided); hover or click a cell
// or a thread to highlight that thread.  8 threads x 64 elements, reduced
// for legibility.
import { f32, add32, hex32, fmt32, exact32, exactSum, exactDecimal, exactToNumber, ulpDiff32 } from './fp.js';

const NE = 64, T = 8, VW = 4;
const W = 720, CX = 40, CW = 10;                  // element row: left edge, cell width
const RY = 44, TY = 98, TH = 19;                 // element row top, thread table top, row pitch
const RES = TY + T * TH + 30;                     // results block top
const H = RES + 104;

// 64 fixed fp32 values, mixed magnitudes (0.02 … 9930; 3 significant digits,
// drawn once from a seeded PRNG and frozen here).  Chosen so the three
// mappings give three different fp32 totals.
const XS = [8230, 0.0531, 7880, 4.92, 0.552, 345, 82.5, 0.0415, 278, 8440, 161, 13.7, 1.88, 579, 76.2, 0.412,
  60.8, 5.32, 0.201, 5.87, 0.0544, 766, 158, 1790, 2720, 0.752, 9580, 75.5, 0.737, 186, 0.0407, 9010,
  36.9, 175, 7350, 127, 0.0222, 0.0492, 2.87, 5.26, 0.0302, 0.0524, 0.454, 0.0662, 71.8, 126, 3560, 6880,
  5.34, 2.11, 327, 75.5, 5.07, 598, 8680, 0.095, 0.0297, 0.681, 0.324, 0.822, 317, 975, 2320, 9930].map(f32);
const EXSUM = exactSum(XS.map(exact32)), EXACT = exactDecimal(EXSUM, 13);

const MAPS = {
  contig: { label: 'contiguous spans', owner: (e) => [Math.floor(e / 8), e % 8] },
  strided: { label: 'strided', owner: (e) => [e % T, Math.floor(e / T)] },
  vec: { label: 'vectorised (4-wide)', owner: (e) => { const v = Math.floor(e / VW); return [v % T, Math.floor(v / T) * VW + (e % VW)]; } },
};
// per mapping: each thread's elements in accumulation order, its partial
// (left-to-right fp32 adds), and the total from the 8-lane butterfly
const lane0 = (p) => add32(add32(add32(p[0], p[4]), add32(p[2], p[6])), add32(add32(p[1], p[5]), add32(p[3], p[7])));
const ORDERS = {}, PART = {}, TOTAL = {};
for (const [k, m] of Object.entries(MAPS)) {
  const per = [...Array(T)].map(() => []);
  for (let e = 0; e < NE; e++) { const [t, pos] = m.owner(e); per[t][pos] = e; }
  ORDERS[k] = per;
  PART[k] = per.map((es) => es.map((e) => XS[e]).reduce((a, x) => add32(a, x)));
  TOTAL[k] = lane0(PART[k]);
}

const CSS = `
bbe-hwmap { display: block; height: ${H + 34}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-hwmap .ctl { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); height: 26px; }
bbe-hwmap .seg { display: inline-flex; }
bbe-hwmap .seg button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong); background: var(--card);
  color: var(--muted); cursor: pointer; }
bbe-hwmap .seg button + button { border-left: none; }
bbe-hwmap .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-hwmap .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-hwmap .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-hwmap svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-hwmap text { fill: var(--muted); }
bbe-hwmap .hd { font-weight: 600; font-size: 11.5px; fill: var(--text); }
bbe-hwmap .ax { font: 9.5px var(--mono); fill: var(--faint); font-variant-numeric: tabular-nums; }
bbe-hwmap .c rect { fill: var(--card); stroke: var(--rule-strong); stroke-width: 0.6; }
bbe-hwmap .c.odd rect { fill: var(--blue-soft); }
bbe-hwmap .c.on rect { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.2; }
bbe-hwmap .c text { font: 8.5px var(--mono); text-anchor: middle; fill: var(--muted); }
bbe-hwmap .c.on text { fill: var(--ink); font-weight: 600; }
bbe-hwmap .vec { fill: none; stroke: var(--muted); stroke-width: 1.2; }
bbe-hwmap .tr { cursor: pointer; }
bbe-hwmap .tr .bg { fill: transparent; }
bbe-hwmap .tr.on .bg { fill: var(--amber-soft); }
bbe-hwmap .tl { font: 600 10.5px var(--mono); fill: var(--text); }
bbe-hwmap .term { font: 10px var(--mono); fill: var(--text); font-variant-numeric: tabular-nums; }
bbe-hwmap .op { font: 10px var(--mono); fill: var(--faint); }
bbe-hwmap .res { font: 600 10px var(--mono); fill: var(--blue); }
bbe-hwmap .note { font-size: 10.5px; }
bbe-hwmap .hit { fill: transparent; cursor: pointer; }
bbe-hwmap .pv { font: 10px var(--mono); fill: var(--text); font-variant-numeric: tabular-nums; }
bbe-hwmap .sl { font: 9.5px var(--mono); fill: var(--faint); }
bbe-hwmap .bf { stroke: var(--rule-strong); stroke-width: 1; fill: none; }
bbe-hwmap .bf.hot { stroke: var(--amber); stroke-width: 1.5; }
bbe-hwmap .add circle { fill: var(--card); stroke: var(--faint); }
bbe-hwmap .add text { font: 600 9px var(--mono); fill: var(--muted); text-anchor: middle; }
bbe-hwmap .add.hot circle { fill: var(--amber-soft); stroke: var(--amber); }
bbe-hwmap .add.hot text { fill: var(--ink); }
bbe-hwmap .cell { fill: var(--card); stroke: var(--rule-strong); }
bbe-hwmap .ct { font: 9.5px var(--mono); fill: var(--text); text-anchor: middle; }
bbe-hwmap .th { font-size: 10px; fill: var(--faint); }
bbe-hwmap .rw { font: 11px var(--mono); fill: var(--text); font-variant-numeric: tabular-nums; }
bbe-hwmap .rl { font-size: 11px; fill: var(--text); }
bbe-hwmap .rsel { fill: var(--amber-soft); }
bbe-hwmap .dd { fill: var(--orange); font-weight: 600; }
bbe-hwmap .rule { stroke: var(--rule); }
bbe-hwmap .ex { font: 11px var(--mono); fill: var(--blue); }
bbe-hwmap .msg { font-weight: 600; font-size: 11.5px; fill: var(--text); }
`;

const t = (x, y, cls, str, a) => `<text class="${cls}" x="${x}" y="${y}"${a ? ` text-anchor="${a}"` : ''}>${str}</text>`;

function svg(map, sel) {
  const o = [], own = (e) => MAPS[map].owner(e)[0];
  o.push(t(CX, 12, 'hd', `1 · Element → thread (cell = owning thread), ${NE} fp32 values`));
  for (let e = 0; e < NE; e += 8) o.push(t(CX + e * CW + 1, RY - 6, 'ax', e));
  for (let e = 0; e < NE; e++) {
    const th = own(e);
    o.push(`<g class="c${th % 2 ? ' odd' : ''}${th === sel ? ' on' : ''}"><rect x="${CX + e * CW}" y="${RY}" width="${CW}" height="16"/>` +
      `<text x="${CX + e * CW + CW / 2}" y="${RY + 11.5}">${th}</text></g>`);
  }
  if (map === 'vec') for (let v = 0; v < NE / VW; v++) o.push(`<rect class="vec" x="${CX + v * VW * CW + 1}" y="${RY + 1}" width="${VW * CW - 2}" height="14" rx="2"/>`);
  for (let e = 0; e < NE; e++) o.push(`<rect class="hit" data-t="${own(e)}" x="${CX + e * CW}" y="${RY}" width="${CW}" height="16"><title>x${e} = ${fmt32(XS[e])}</title></rect>`);

  // 2: per-thread sums
  o.push(t(CX, TY - 12, 'hd', '2 · Per-thread sums, in registers'));
  const TX = CX + 50, step = 38;
  ORDERS[map].forEach((es, th) => {
    const y = TY + th * TH;
    const g = [`<g class="tr${th === sel ? ' on' : ''}" data-t="${th}"><rect class="bg" x="${CX - 6}" y="${y}" width="${TX + T * step + 118 - CX}" height="${TH - 2}" rx="3"/>`,
      t(CX, y + 12.5, 'tl', `thread ${th}`)];
    es.forEach((e, i) => {
      const x = TX + 12 + i * step;
      if (i) g.push(t(x - 9, y + 12.5, 'op', '+', 'middle'));
      g.push(`<text class="term" x="${x + 9}" y="${y + 12.5}" text-anchor="middle">x${e}<title>x${e} = ${fmt32(XS[e])}</title></text>`);
    });
    const ex = TX + 12 + T * step;
    g.push(t(ex - 9, y + 12.5, 'op', '→', 'middle'), t(ex + 2, y + 12.5, 'res', `p${th}`), t(ex + 20, y + 12.5, 'pv', `= ${fmt32(PART[map][th])}`), '</g>');
    o.push(g.join(''));
  });

  // 3: the 8-lane butterfly (shfl.xor 4, 2, 1), as in <bbe-hwtree>
  const BX = (i) => 548 + i * 22, BY = (s) => TY + 8 + s * 38;
  o.push(t(BX(0) - 10, TY - 12, 'hd', '3 · Warp shuffle'));
  const hot = new Set(['3,0']);
  for (let s = 3; s >= 1; s--) for (let i = 0; i < 8; i++) if (hot.has(`${s},${i}`)) { const off = 4 >> (s - 1); hot.add(`${s - 1},${i}`); hot.add(`${s - 1},${i ^ off}`); }
  for (let s = 1; s <= 3; s++) {
    const off = 4 >> (s - 1);
    for (let i = 0; i < 8; i++) for (const j of [i, i ^ off]) {
      const h = hot.has(`${s},${i}`) && hot.has(`${s - 1},${j}`);
      o.push(`<path class="bf${h ? ' hot' : ''}" d="M${BX(j)} ${BY(s - 1) + (s === 1 ? 8 : 6.5)} L${BX(i)} ${BY(s) - 6.5}"/>`);
    }
  }
  for (let i = 0; i < 8; i++) o.push(`<rect class="cell" x="${BX(i) - 9.5}" y="${BY(0) - 8}" width="19" height="16" rx="2"/>` + t(BX(i), BY(0) + 3.5, 'ct', `p${i}`));
  for (let s = 1; s <= 3; s++) {
    for (let i = 0; i < 8; i++) o.push(`<g class="add${hot.has(`${s},${i}`) ? ' hot' : ''}"><circle cx="${BX(i)}" cy="${BY(s)}" r="6.5"/>${t(BX(i), BY(s) + 3, '', '+')}</g>`);
  }
  o.push(t(BX(0) - 10, BY(3) + 22, 'sl', 'shfl.xor 4, 2, 1'), t(BX(0) - 10, BY(3) + 36, 'sl', 'lane 0 = ((p0+p4)+(p2+p6))'), t(BX(0) + 26, BY(3) + 48, 'sl', '+ ((p1+p5)+(p3+p7))'));

  // results: all three mappings
  const keys = Object.keys(MAPS), hx = keys.map((k) => hex32(TOTAL[k]));
  o.push(t(CX, RES, 'msg', '4 · Total, by mapping'));
  const c1 = CX, c2 = CX + 170, c3 = CX + 330, ry0 = RES + 20;
  const c4 = c3 + 120;
  o.push(t(c1, ry0, 'th', 'mapping'), t(c2, ry0, 'th', 'fp32 total (exact decimal)'), t(c3, ry0, 'th', 'bits'), t(c4, ry0, 'th', 'error vs exact (fp32 ulps)'));
  keys.forEach((k, r) => {
    const y = ry0 + 18 + r * 18;
    if (k === map) o.push(`<rect class="rsel" x="${CX - 6}" y="${y - 13}" width="${c4 + 150 - CX}" height="17" rx="2"/>`);
    const digits = [...hx[r]].map((ch, i) => (i > 1 && hx.some((h) => h[i] !== ch) ? `<tspan class="dd">${ch}</tspan>` : ch)).join('');
    o.push(t(c1, y, 'rl', MAPS[k].label), t(c2, y, 'rw', exactDecimal(exact32(TOTAL[k]), 13)), `<text class="rw" x="${c3}" y="${y}">${digits}</text>`,
      t(c4, y, 'rw', (d => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(2))(ulpDiff32(TOTAL[k], exactToNumber(EXSUM)))));
  });
  const ye = ry0 + 18 + 3 * 18;
  o.push(`<path class="rule" d="M${CX - 6} ${ye - 13} H${c4 + 144}"/>`, t(c1, ye + 2, 'rl', 'exact real sum'), t(c2, ye + 2, 'ex', EXACT));
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="64 fp32 values dealt out to 8 threads under three mappings, each thread's left-to-right sum, the warp shuffle that combines them, and the three different fp32 totals that result.">${o.join('')}</svg>`;
}

class BbeHwmap extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-hwmap-css')) {
      const s = document.createElement('style'); s.id = 'bbe-hwmap-css'; s.textContent = CSS; document.head.append(s);
    }
    this.map = 'strided'; this.sel = 2;
    this.innerHTML = `<div class="ctl"><span>element → thread:</span><span class="seg">${Object.entries(MAPS).map(([k, m]) =>
      `<button data-m="${k}"${k === this.map ? ' class="on"' : ''}>${m.label}</button>`).join('')}</span><span>· hover a cell or thread</span></div><div class="fig"></div>`;
    this.fig = this.querySelector('.fig');
    this.querySelectorAll('.seg button').forEach((b) => b.onclick = () => {
      this.map = b.dataset.m;
      this.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
      this.draw();
    });
    const pick = (e) => {
      const n = e.target.closest?.('[data-t]');
      if (n && +n.dataset.t !== this.sel) { this.sel = +n.dataset.t; this.draw(); }
    };
    this.fig.addEventListener('mouseover', pick);
    this.fig.addEventListener('click', pick);
    this.draw();
  }
  draw() { this.fig.innerHTML = svg(this.map, this.sel); }
}
if (!customElements.get('bbe-hwmap')) customElements.define('bbe-hwmap', BbeHwmap);
