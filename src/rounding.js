// <bbe-rounding>: "Every operation rounds".  A number line of representable
// values near a, and the two evaluation orders of a + b + c as lanes: each
// add jumps to its exact sum (hollow dot) and snaps to the nearest
// representable value (orange arrow when it moves, blue when exact).
import { add32, addBf16, exact32, exactSum, exactDecimal } from './fp.js';

export const EXAMPLES = {
  bf16: {
    label: 'bf16: 1 + 2⁻⁸ + 2⁻⁸', fmt: 'bf16', add: addBf16,
    a: 1, b: 2 ** -8, c: 2 ** -8, sym: { a: '1', b: '2⁻⁸', c: '2⁻⁸', bc: '2⁻⁷' },
    ticks: [1 - 2 * 2 ** -8, 1 - 2 ** -8, 1, 1 + 2 ** -7, 1 + 2 * 2 ** -7, 1 + 3 * 2 ** -7],
    lo: 1 - 1.7 * 2 ** -8, hi: 1 + 2.45 * 2 ** -7, tickLab: String,
    gaps: ['2⁻⁸', '2⁻⁷'],
    fine: true,
  },
  fp32: {
    label: 'fp32: 10⁸ + 4 + 4', fmt: 'fp32', add: add32,
    a: 1e8, b: 4, c: 4, sym: { a: '10⁸', b: '4', c: '4', bc: '8' },
    ticks: [1e8 - 16, 1e8 - 8, 1e8, 1e8 + 8, 1e8 + 16],
    lo: 1e8 - 13.6, hi: 1e8 + 19.6,
    tickLab: (v) => (v === 1e8 ? '10⁸' : '10⁸' + (v > 1e8 ? '+' : '−') + Math.abs(v - 1e8)),
    gaps: ['8', '8'],
    fine: false,
  },
};

const dec = (...xs) => exactDecimal(exactSum(xs.map(exact32)), 17);

// the numbers the widget draws (also asserted by scripts/check-fp.mjs)
export function model(ex) {
  const step = (x, y) => { const r = ex.add(x, y); return { from: x, by: y, exact: x + y, exactStr: dec(x, y), r, rounded: r !== x + y }; };
  const l1a = step(ex.a, ex.b), l1b = step(l1a.r, ex.c);
  const l2a = step(ex.b, ex.c), l2b = step(ex.a, l2a.r);
  return { left: [l1a, l1b], right: [l2a, l2b], resLeft: l1b.r, resRight: l2b.r };
}

const W = 688, X0 = 250, X1 = 672, H = 274;
const CSS = `
bbe-rounding { display: block; height: 306px; max-width: 720px; margin: 18px 0 26px;
  font: 13px/1.4 var(--sans); color: var(--text); }
bbe-rounding .top { display: flex; align-items: center; gap: 12px; height: 26px; }
bbe-rounding .seg { display: inline-flex; }
bbe-rounding .seg button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-rounding .seg button + button { border-left: none; }
bbe-rounding .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-rounding .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-rounding .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-rounding svg { display: block; margin-top: 6px; font: 11.5px var(--sans); }
bbe-rounding svg text.m { font-size: 12px; }
bbe-rounding svg text { fill: var(--text); }
bbe-rounding .m { font-family: var(--mono); font-variant-numeric: tabular-nums; }
bbe-rounding .mut { fill: var(--muted); }
bbe-rounding .fnt { fill: var(--faint); }
bbe-rounding .or { fill: var(--orange); }
bbe-rounding .bl { fill: var(--blue); }
bbe-rounding .b { font-weight: 600; }
bbe-rounding .axis { stroke: var(--rule-strong); }
bbe-rounding .tk { stroke: var(--muted); }
bbe-rounding .comb { fill: url(#bbe-r-comb); }
bbe-rounding #bbe-r-comb line { stroke: var(--rule-strong); }
bbe-rounding .arc { fill: none; stroke: var(--faint); stroke-width: 1.2; }
bbe-rounding .snap { fill: none; stroke: var(--orange); stroke-width: 1.8; }
bbe-rounding .hollow { fill: var(--card); stroke: var(--orange); stroke-width: 1.5; }
bbe-rounding .dot-b { fill: var(--blue); }
bbe-rounding .dot-o { fill: var(--orange); }
bbe-rounding .dot-s { fill: var(--muted); }
bbe-rounding .ah-o { fill: var(--orange); }
bbe-rounding .ah-g { fill: var(--faint); }
bbe-rounding .bar { stroke: var(--rule); }
`;

function svg(ex) {
  const M = model(ex);
  const X = (v) => X0 + (v - ex.lo) / (ex.hi - ex.lo) * (X1 - X0);
  const t = (x, y, s, cls = '', anchor = 'start') => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${s}</text>`;
  let o = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs><pattern id="bbe-r-comb" width="2.5" height="8" patternUnits="userSpaceOnUse"><line x1="0.5" y1="0" x2="0.5" y2="8"/></pattern>
  <marker id="bbe-r-ao" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path class="ah-o" d="M0 0 L8 4 L0 8 Z"/></marker>
  <marker id="bbe-r-ag" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path class="ah-g" d="M0 0 L8 4 L0 8 Z"/></marker></defs>`;
  // the axis of representable values
  const ay = 30;
  o += t(0, ay - 12, `${ex.fmt} values`, 'mut');
  if (ex.fine) o += t(0, ay + 4, 'fp32 values (not to scale)', 'fnt');
  if (ex.fine) o += `<rect class="comb" x="${X0}" y="${ay - 4}" width="${X1 - X0}" height="8"/>`;
  o += `<line class="axis" x1="${X0 - 6}" y1="${ay}" x2="${X1 + 6}" y2="${ay}"/>`;
  for (const v of ex.ticks.filter((v) => v >= ex.lo && v <= ex.hi)) o += `<line class="tk" x1="${X(v)}" y1="${ay - 7}" x2="${X(v)}" y2="${ay + 7}"/>` + (v >= ex.lo && v <= ex.hi ? t(X(v), ay - 12, ex.tickLab(v), 'm', 'middle') : '');
  // spacing brackets either side of a
  const by = ay + 20;
  [[ex.ticks[1], ex.ticks[2], ex.gaps[0]], [ex.ticks[2], ex.ticks[3], ex.gaps[1]]].forEach(([u, v, g]) => {
    o += `<path class="arc" d="M${X(u) + 2} ${by - 4} V${by} H${X(v) - 2} V${by - 4}"/>` + t((X(u) + X(v)) / 2, by + 12, 'gap ' + g, 'mut m', 'middle');
  });
  // lanes
  const lane = (y0, title, steps, res, texts) => {
    const ly = y0 + 46;
    let s = `<line class="bar" x1="0" y1="${y0 - 6}" x2="${W}" y2="${y0 - 6}"/>`;
    s += t(0, y0 + 12, title, 'm b');
    texts.forEach((tx, i) => { s += `<text x="0" y="${y0 + 30 + 16 * i}" class="m">${tx}</text>`; });
    s += `<line class="axis" x1="${X0 - 6}" y1="${ly}" x2="${X1 + 6}" y2="${ly}"/>`;
    for (const v of ex.ticks.filter((v) => v >= ex.lo && v <= ex.hi)) s += `<line class="tk" x1="${X(v)}" y1="${ly - 5}" x2="${X(v)}" y2="${ly + 5}"/>`;
    steps.forEach((st, i) => {
      if (st.from < ex.lo) return;   // b + c happens far to the left of this line
      const up = i === 0 || steps[0].from < ex.lo ? -1 : 1;   // second jump of the left lane goes below
      const x1 = X(st.from), x2 = X(st.exact), xm = (x1 + x2) / 2, h = 30 * up;
      s += `<circle class="dot-s" cx="${x1}" cy="${ly}" r="3"/>`;
      s += `<path class="arc" marker-end="url(#bbe-r-ag)" d="M${x1} ${ly + 3 * up} Q${xm} ${ly + h} ${x2} ${ly + 5 * up}"/>`;
      s += t(xm, ly + h * 0.62 + (up < 0 ? -2 : 10), '+' + (i === 0 ? ex.sym.b : (steps[0].from < ex.lo ? ex.sym.bc : ex.sym.c)), 'mut m', 'middle');
      if (st.rounded) {
        const xr = X(st.r);
        s += `<circle class="hollow" cx="${x2}" cy="${ly}" r="3.5"/>`;
        s += `<path class="snap" marker-end="url(#bbe-r-ao)" d="M${x2 - 4} ${ly - 1 * up} Q${(x2 + xr) / 2} ${ly + 9 * up} ${xr + 5} ${ly + 3 * up}"/>`;
      } else s += `<circle class="dot-b" cx="${x2}" cy="${ly}" r="3.5"/>`;
    });
    s += `<circle class="${res.cls}" cx="${X(res.v)}" cy="${ly}" r="4.5"/>`;
    return s;
  };
  const [p, q] = M.left, [u, v] = M.right, S = ex.sym;
  const res = (st, exact) => st.rounded ? `<tspan class="or">${exact}</tspan> → ${st.r === ex.a ? S.a : st.r}` : `<tspan class="bl">${exact}</tspan>`;
  o += lane(88, '(a + b) + c', M.left, { v: M.resLeft, cls: 'dot-o' }, [
    `① ${S.a} + ${S.b} = ${res(p, p.exactStr)}`,
    `② ${p.r === ex.a ? S.a : p.r} + ${S.c} = ${res(q, q.exactStr)}`,
    `<tspan class="mut">result</tspan> <tspan class="b">${M.resLeft}</tspan>`]);
  o += lane(174, 'a + (b + c)', M.right, { v: M.resRight, cls: 'dot-b' }, [
    `① ${S.b} + ${S.c} = ${res(u, S.bc)}`,
    `② ${S.a} + ${S.bc} = ${res(v, v.exactStr)}`,
    `<tspan class="mut">result</tspan> <tspan class="b">${M.resRight}</tspan>`]);
  // legend
  const gy = H - 8;
  o += `<circle class="hollow" cx="5" cy="${gy - 4}" r="3.5"/>` + t(14, gy, 'exact sum, not representable', 'mut');
  o += `<path class="snap" marker-end="url(#bbe-r-ao)" d="M192 ${gy - 4} H214"/>` + t(222, gy, 'rounded to nearest (ties to even)', 'mut');
  o += `<circle class="dot-b" cx="420" cy="${gy - 4}" r="3.5"/>` + t(429, gy, 'exact sum, representable', 'mut');
  return o + '</svg>';
}

class BbeRounding extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-rounding-css')) {
      const st = document.createElement('style'); st.id = 'bbe-rounding-css'; st.textContent = CSS; document.head.append(st);
    }
    this.key = 'bf16';
    this.render();
    this.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-k]');
      if (b && b.dataset.k !== this.key) { this.key = b.dataset.k; this.render(); }
    });
  }
  render() {
    const ex = EXAMPLES[this.key];
    this.innerHTML = `<div class="top"><span class="seg">${Object.entries(EXAMPLES).map(([k, e]) =>
      `<button data-k="${k}" class="${k === this.key ? 'on' : ''}">${e.label}</button>`).join('')}</span>
</div>${svg(ex)}`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-rounding', BbeRounding);
