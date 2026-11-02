// <bbe-unitgrid>: the tile-2 state of <bbe-online>'s example held by the two
// rescale policies.  Lazy keeps m = 3; every tile moves m to 5, so each of
// its values is the lazy one times c = e^(3 - 5) = e^-2 in exact arithmetic.
// Two rulers share one log axis in lazy units.  Ticks are a toy binary grid
// with 3 significant bits (fp32 ticks would be invisible): the top ruler is
// the grid itself, the bottom ruler is the grid in every-tile units, drawn at
// g / c, so an exact quantity sits at the same x on both.  Each kernel holds
// its own nearest tick (round to nearest even), exact via fp.js roundExact.
// Quantities: l after tile 1 (the same bits in both kernels; lazy keeps it,
// every tile rounds c * l), the two tile-2 weights, and l after tile 2 summed
// in run()'s order in each kernel's units.  The toggle c = 2^-3 makes the
// bottom grid the top grid shifted by three binades, so every held value
// corresponds; one button flips c and animates between the two.  Last row: the real fp32 / bf16 out bits of the example
// (c = e^-2), from online.js's run().
import { roundExact, hex16 } from './fp.js';
import { run } from './online.js';

export const BITS = 3;
// a double as an exact { m, e }, then rounded to BITS significant bits
const exactD = (x) => { let e = 0; while (!Number.isInteger(x)) { x *= 2; e--; } return { m: BigInt(x), e }; };
export const toy = (x) => roundExact(exactD(x), BITS, -1074);
export const CS = [['e', Math.exp(-2), 'c = e<tspan dy="-5" font-size="9">−2</tspan>'], ['two', 1 / 8, 'c = 2<tspan dy="-5" font-size="9">−3</tspan>']];
export const L1 = toy(Math.exp(-2) + 1 + Math.exp(-1));                  // l after tile 1, both kernels
export const EXACT = [L1, Math.E, Math.exp(2), L1 + Math.exp(2) + Math.E]; // lazy units (m = 3)
export const NAMES = ['ℓ after tile 1', 'p for s = 4', 'p for s = 5', 'ℓ after tile 2'];
// held values: [l1, p(s=4), p(s=5), l]; tile 2 adds s = 5 first, then s = 4
export function held(c) {
  const hold = (v) => [v[0], v[1], v[2], toy(toy(v[0] + v[2]) + v[1])];
  const lazy = hold([L1, toy(EXACT[1]), toy(EXACT[2])]);
  const every = hold([toy(c * L1), toy(c * EXACT[1]), toy(c * EXACT[2])]);
  return { lazy, every, same: lazy.map((v, i) => Math.abs(every[i] / c - v) < 1e-12 * v) };
}
export const HELD = Object.fromEntries(CS.map(([k, c]) => [k, held(c)]));
export const OUT = [run(true), run(false)].map((r) => hex16(r.epi.out));

const W = 720, H = 262, X0 = 120, PX = 146, YT = 104, YB = 172, R = 4.5;
const X = (v) => X0 + Math.log2(v) * PX;                 // v in lazy units, 1 .. 16
// toy grid values in [lo, hi]
function grid(lo, hi) {
  const out = [];
  for (let e = -8; e <= 6; e++) for (let k = 4; k < 8; k++) { const g = k * 2 ** (e - 2); if (g >= lo * (1 - 1e-9) && g <= hi * (1 + 1e-9)) out.push(g); }
  return out;
}
const pow2 = (g) => Number.isInteger(Math.log2(g));
const frac = (g) => (g >= 1 ? String(g) : `1/${1 / g}`);
const CSS = `
bbe-unitgrid { display: block; position: relative; height: ${H}px; max-width: ${W}px; margin: 18px 0 26px; }
bbe-unitgrid svg { display: block; font: 11px var(--sans); }
bbe-unitgrid text { fill: var(--muted); }
bbe-unitgrid .h { font-weight: 600; fill: var(--text); font-size: 12px; }
bbe-unitgrid .t { font-weight: 600; fill: var(--ink); font-size: 13px; }
bbe-unitgrid .sub { fill: var(--faint); }
bbe-unitgrid .num { font: 11px var(--mono); font-variant-numeric: tabular-nums; }
bbe-unitgrid .ax { stroke: var(--rule-strong); }
bbe-unitgrid .tk { stroke: var(--muted); }
bbe-unitgrid .guide { stroke: var(--faint); stroke-dasharray: 2 3; }
bbe-unitgrid .ex { fill: none; stroke: var(--muted); stroke-width: 1.2; }
bbe-unitgrid .dot { fill: var(--blue); }
bbe-unitgrid .dot.d { fill: var(--orange); }
bbe-unitgrid .link { stroke: var(--blue); stroke-width: 1.5; }
bbe-unitgrid .link.d { stroke: var(--orange); }
bbe-unitgrid .rule { stroke: var(--rule); }
bbe-unitgrid .bits { font: 600 15px var(--mono); fill: var(--ink); }
bbe-unitgrid .bits tspan.d { fill: var(--orange); }
bbe-unitgrid .ctl { position: absolute; left: 0; top: 0; }
bbe-unitgrid .tg { font: 12px var(--sans); padding: 2px 9px; min-width: 76px; border: 1px solid var(--rule-strong); border-radius: 4px;
  background: var(--card); color: var(--ink); cursor: pointer; white-space: nowrap; }
`;

// t in [0, 1]: 0 is c = e^-2, 1 is c = 2^-3; positions and colours in between are interpolated
const [SA, SB] = CS.map(([k, c]) => ({ sh: -Math.log2(c), ...HELD[k] }));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (p) => (p < 0.5 ? 4 * p ** 3 : 1 - (2 - 2 * p) ** 3 / 2);
const DUR = 500;
const inRange = (x) => x >= X(1) - 0.5 && x <= X(16) + 0.5;

// a ruler is the toy grid shifted by sh binades (sh = -log2 c); ticks outside 1 .. 16 are not drawn,
// power-of-two labels fade by whether they are in range at each end
function ruler(y, labelY, name, sub, shA, shB, t) {
  let o = `<text class="t" x="0" y="${y + 4}">${name}</text><text class="sub" x="0" y="${y + 19}">${sub}</text>`;
  o += `<line class="ax" x1="${X(1)}" x2="${X(16)}" y1="${y}" y2="${y}"/>`;
  const at = (g, sh) => X0 + (Math.log2(g) + sh) * PX;
  for (const g of grid(1 / 32, 32)) {
    const x = at(g, lerp(shA, shB, t)), big = pow2(g);
    if (inRange(x)) o += `<line class="tk" x1="${x}" x2="${x}" y1="${y - (big ? 8 : 5)}" y2="${y + (big ? 8 : 5)}"/>`;
    const op = lerp(+inRange(at(g, shA)), +inRange(at(g, shB)), t);
    if (big && op > 0) o += `<text class="num" x="${x}" y="${labelY}" opacity="${op}" text-anchor="middle">${frac(g)}</text>`;
  }
  return o;
}

function body(t) {
  let o = '';
  // legend
  const lx = 360;
  o += `<circle class="ex" cx="${lx}" cy="10" r="${R}"/><text x="${lx + 9}" y="14">exact</text>`;
  o += `<circle class="dot" cx="${lx + 62}" cy="10" r="${R}"/><text x="${lx + 71}" y="14">rounded, same point</text>`;
  o += `<circle class="dot d" cx="${lx + 200}" cy="10" r="${R}"/><text x="${lx + 209}" y="14">rounded, differs</text>`;
  EXACT.forEach((v, i) => (o += `<text class="h" x="${X(v)}" y="52" text-anchor="middle">${NAMES[i]}</text>` +
    `<line class="guide" x1="${X(v)}" x2="${X(v)}" y1="58" y2="${YB}"/>`));
  o += ruler(YT, YT - 13, 'lazy', 'm = 3', 0, 0, t) + ruler(YB, YB + 22, 'every tile', 'values × c', SA.sh, SB.sh, t);
  o += `<text class="sub" x="${X(16)}" y="${YB + 40}" text-anchor="end">toy grid: 3 significant bits, log scale</text>`;
  EXACT.forEach((v, i) => {
    // lazy holds the same values for both c; the every-tile dot slides, orange fades in over blue where they differ
    const xt = X(SA.lazy[i]), xA = X(SA.every[i] * 2 ** SA.sh), xB = X(SB.every[i] * 2 ** SB.sh), xb = lerp(xA, xB, t);
    const u = lerp(+!SA.same[i], +!SB.same[i], t), off = (x) => +(Math.abs(x - X(v)) > 1);
    const mix = (f) => f('', 1) + (u > 0 ? f(' d', u) : '');
    o += mix((d, op) => `<line class="link${d}" opacity="${op}" x1="${xt}" x2="${xb}" y1="${YT}" y2="${YB}"/>`);
    [[off(xt), off(xt), YT], [off(xA), off(xB), YB]].forEach(([a, b, y]) => {
      const op = lerp(a, b, t);
      if (op > 0) o += `<circle class="ex" opacity="${op}" cx="${X(v)}" cy="${y}" r="${R}"/>`;
    });
    o += mix((d, op) => `<circle class="dot${d}" opacity="${op}" cx="${xt}" cy="${YT}" r="${R}"/><circle class="dot${d}" opacity="${op}" cx="${xb}" cy="${YB}" r="${R}"/>`);
  });
  // the real recipe's out bits for c = e^-2: the differing hex digits in orange; dimmed for c = 2^-3
  let i = 0;
  while (OUT[0][i] === OUT[1][i]) i++;
  const bits = (s) => `${s.slice(0, i)}<tspan class="d">${s.slice(i)}</tspan>`;
  o += `<line class="rule" x1="0" x2="${W}" y1="${H - 40}" y2="${H - 40}"/>`;
  o += `<g opacity="${lerp(1, 0.35, t)}"><text class="h" x="0" y="${H - 12}">out, fp32 / bf16, c = e<tspan dy="-5" font-size="9">−2</tspan></text>`;
  o += `<text x="${X(2)}" y="${H - 12}">lazy</text><text class="bits" x="${X(2) + 34}" y="${H - 11}">${bits(OUT[0])}</text>`;
  o += `<text x="${X(6)}" y="${H - 12}">every tile</text><text class="bits" x="${X(6) + 64}" y="${H - 11}">${bits(OUT[1])}</text></g>`;
  return o;
}

class BbeUnitgrid extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-unitgrid-css')) {
      const st = document.createElement('style'); st.id = 'bbe-unitgrid-css'; st.textContent = CSS; document.head.append(st);
    }
    this.p = 0; this.goal = 0;
    this.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"></svg><div class="ctl"><button class="tg"></button></div>`;
    this.svg = this.querySelector('svg'); this.btn = this.querySelector('button');
    this.btn.onclick = () => { this.goal = 1 - this.goal; this.label(); this.animate(); };
    // Space and Enter press the button; keep them from the deck's page-level keys
    this.btn.onkeydown = (e) => { if (e.key === ' ' || e.key === 'Enter') e.stopPropagation(); };
    this.label(); this.draw();
  }
  label() { this.btn.innerHTML = CS[this.goal][2].replace(/<tspan[^>]*>(.*?)<\/tspan>/, '<sup>$1</sup>'); }
  draw() { this.svg.innerHTML = body(ease(this.p)); }
  // p moves toward the goal at 1 / DUR per ms, so a click mid-way reverses smoothly
  animate() {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { this.p = this.goal; this.draw(); return; }
    if (this.raf) return;
    let last = performance.now();
    const step = (now) => {
      const d = (now - last) / DUR; last = now;
      this.p = this.goal ? Math.min(1, this.p + d) : Math.max(0, this.p - d);
      this.draw();
      this.raf = this.p === this.goal ? 0 : requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-unitgrid', BbeUnitgrid);
