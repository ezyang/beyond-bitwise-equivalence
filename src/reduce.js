// <bbe-reduce>: one fixed vector of 16 fp32 values summed in five orders.
// The SVG draws the chosen order's tree (leaves on top, each internal node
// one fp32 add: blue if that add was exact, orange if it rounded); the
// table lists all five results at once against the exact sum.
import { f32, add32, exact32, exactSum, exactDecimal, roundExactF32, hex32, fmt32, ulpDiff32, REDUCE_ORDERS } from './fp.js';

// short decimals, rounded to fp32 (each prints back as written); chosen so
// that all five orders give five different fp32 sums
export const VALUES = [5080, 0.732, 0.0442, 8.35, 7980, 0.0298, 91.9, 0.0995,
  0.701, 0.581, 0.524, 592, 0.0389, 9.95, 0.0313, 0.772].map(f32);

const LABELS = { seq: 'left to right', pair: 'pairwise', lanes8: '8 interleaved lanes, then a tree', span2: '2 contiguous spans, then add', rev: 'right to left' };

// exact sum, its correctly rounded fp32 value, and each order's result
export function model(xs = VALUES) {
  const exact = exactSum(xs.map(exact32));
  const cr = roundExactF32(exact);
  const rows = REDUCE_ORDERS.map((o) => {
    const tree = o.tree(xs.length);
    const r = evalNodes(tree, xs).value;
    return { key: o.key, tree, r, ulps: ulpDiff32(r, cr) };
  });
  return { exact, cr, rows };
}

// evaluate with per-node records: value, exactness, level (leaves 0)
function evalNodes(t, xs, out = []) {
  if (typeof t === 'number') return { leaf: t, value: xs[t], level: 0 };
  const l = evalNodes(t[0], xs, out), r = evalNodes(t[1], xs, out);
  const value = add32(l.value, r.value);
  const ex = exactSum([exact32(l.value), exact32(r.value)]);
  const back = exactSum([ex, exact32(-value)]);
  const n = { l, r, value, rounded: back.m !== 0n, level: 1 + Math.max(l.level, r.level) };
  out.push(n);
  n.all = out;
  return n;
}

const W = 688, PITCH = 43, LX = (i) => 22 + PITCH * i, TOPY = 30, ROOTY = 262, H = 292;
const CSS = `
bbe-reduce { display: block; height: 488px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.4 var(--sans); color: var(--text); }
bbe-reduce .seg { display: inline-flex; }
bbe-reduce .seg button { font: 12px var(--sans); padding: 2px 9px; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-reduce .seg button + button { border-left: none; }
bbe-reduce .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-reduce .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-reduce .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-reduce .ln { fill: var(--muted); font-size: 9px; paint-order: stroke; stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round; }
bbe-reduce svg { display: block; margin-top: 6px; font: 10px var(--mono); font-variant-numeric: tabular-nums; }
bbe-reduce svg text { fill: var(--text); }
bbe-reduce .lf rect { fill: var(--panel); stroke: var(--rule); }
bbe-reduce .ix { fill: var(--faint); font-size: 9px; }
bbe-reduce .e { stroke: var(--rule-strong); stroke-width: 1; fill: none; }
bbe-reduce .n { stroke: var(--card); stroke-width: 1; }
bbe-reduce .n.ex { fill: var(--blue); }
bbe-reduce .n.rd { fill: var(--orange); }
bbe-reduce .n:hover, bbe-reduce .n.hov { stroke: var(--amber); stroke-width: 3; }
bbe-reduce .root { fill: var(--amber-soft); stroke: var(--amber); }
bbe-reduce .rt { font-weight: 600; fill: var(--ink); }
bbe-reduce .key { font: 11px var(--sans); fill: var(--muted); }
bbe-reduce .kex { fill: var(--blue); } bbe-reduce .krd { fill: var(--orange); }
bbe-reduce .hovl { font: 11px var(--mono); fill: var(--muted); }
bbe-reduce table { border-collapse: collapse; font-size: 12.5px; margin-top: 4px; width: 100%; }
bbe-reduce td, bbe-reduce th { padding: 2px 10px 2px 4px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }
bbe-reduce th { font-weight: 600; color: var(--muted); font-size: 12px; }
bbe-reduce td.m { font-family: var(--mono); font-variant-numeric: tabular-nums; }
bbe-reduce tr.o { cursor: pointer; }
bbe-reduce tr.sel td { background: var(--amber-soft); }
bbe-reduce tr.ref td { color: var(--muted); }
bbe-reduce .dg { color: var(--orange); font-weight: 700; }
bbe-reduce .ok { color: var(--green); } bbe-reduce .bad { color: var(--orange); }
`;

function svgTree(key, xs) {
  const o = REDUCE_ORDERS.find((q) => q.key === key);
  const nodes = [], root = evalNodes(o.tree(xs.length), xs, nodes);
  const depth = root.level, dy = (ROOTY - 24 - TOPY - 22) / depth;
  const pos = new Map();
  const place = (n) => {
    if (n.leaf !== undefined) { pos.set(n, [LX(n.leaf), TOPY + 14]); return; }
    place(n.l); place(n.r);
    const x = (pos.get(n.l)[0] + pos.get(n.r)[0]) / 2;
    pos.set(n, [x, TOPY + 14 + 8 + n.level * dy]);
  };
  place(root);
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for (const n of nodes) for (const c of [n.l, n.r]) {
    const [x1, y1] = pos.get(c), [x2, y2] = pos.get(n);
    s += `<line class="e" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  }
  xs.forEach((v, i) => {
    s += `<g class="lf"><rect x="${LX(i) - 20.5}" y="${TOPY - 12}" width="41" height="26" rx="3"/>` +
      `<text x="${LX(i)}" y="${TOPY - 15}" class="ix" text-anchor="middle">${i}</text>` +
      `<text x="${LX(i)}" y="${TOPY + 5}" text-anchor="middle">${fmt32(v)}</text></g>`;
  });
  nodes.forEach((n, k) => {
    const [x, y] = pos.get(n);
    s += `<circle class="n ${n.rounded ? 'rd' : 'ex'}" data-k="${k}" cx="${x}" cy="${y}" r="4"/>`;
    if (key === 'lanes8' && n.level === 1) s += `<text class="ln" x="${x}" y="${y + 14}" text-anchor="middle">lane ${n.l.leaf}</text>`;
  });
  const [px, py] = pos.get(root), rx = Math.min(Math.max(px, 52), W - 52), ry = ROOTY;
  s = s.replace('<g class="lf">', `<line class="e" x1="${px}" y1="${py}" x2="${rx}" y2="${ry}"/><g class="lf">`);
  s += `<rect class="root" x="${rx - 52}" y="${ry - 11}" width="104" height="22" rx="4"/>` +
    `<text class="rt" x="${rx}" y="${ry + 4}" text-anchor="middle">${fmt32(root.value)}</text>`;
  s += `<text class="key" x="0" y="${H - 4}"><tspan class="kex">●</tspan> exact add   <tspan class="krd">●</tspan> rounded add (${nodes.filter((n) => n.rounded).length} of ${nodes.length})</text>`;
  s += `<text class="hovl" x="${W}" y="${H - 4}" text-anchor="end"></text>`;
  return { svg: s + '</svg>', nodes };
}

class BbeReduce extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-reduce-css')) {
      const st = document.createElement('style'); st.id = 'bbe-reduce-css'; st.textContent = CSS; document.head.append(st);
    }
    this.key = 'seq';
    this.render();
    this.addEventListener('click', (e) => {
      const b = e.target.closest('[data-o]');
      if (b && b.dataset.o !== this.key) { this.key = b.dataset.o; this.render(); }
    });
    this.addEventListener('mouseover', (e) => {
      const c = e.target.closest('circle.n'), lab = this.querySelector('.hovl');
      if (!c) { lab.textContent = ''; return; }
      const n = this.nodes[+c.dataset.k];
      lab.textContent = `${fmt32(n.l.value)} + ${fmt32(n.r.value)} = ${fmt32(n.value)}${n.rounded ? ' (rounded)' : ''}`;
    });
  }
  render() {
    const M = model(), crHex = hex32(M.cr);
    const { svg, nodes } = svgTree(this.key, VALUES);
    this.nodes = nodes;
    // hex digits that differ from the correctly rounded result, in orange
    const hexDiff = (x) => [...hex32(x)].map((ch, i) => (ch !== crHex[i] ? `<span class="dg">${ch}</span>` : ch)).join('');
    const ul = (u) => (u === 0 ? '<span class="ok">0</span>' : `<span class="bad">${u > 0 ? '+' : '−'}${Math.abs(u)}</span>`);
    this.innerHTML = `<div class="seg">${REDUCE_ORDERS.map((o) =>
      `<button data-o="${o.key}" class="${o.key === this.key ? 'on' : ''}">${LABELS[o.key]}</button>`).join('')}</div>
      ${svg}
      <table><tr><th>order</th><th>fp32 bits</th><th>value</th><th>ulps from exact</th></tr>
      ${M.rows.map((r) => `<tr class="o${r.key === this.key ? ' sel' : ''}" data-o="${r.key}"><td>${LABELS[r.key]}</td>` +
        `<td class="m">${hexDiff(r.r)}</td><td class="m">${fmt32(r.r)}</td><td class="m">${ul(r.ulps)}</td></tr>`).join('')}
      <tr class="ref"><td>exact sum, rounded once</td><td class="m">${crHex}</td><td class="m">${fmt32(M.cr)}</td><td class="m">exact: ${exactDecimal(M.exact, 13)}</td></tr>
      </table>`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-reduce', BbeReduce);
