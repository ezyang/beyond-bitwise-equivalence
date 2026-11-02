// <bbe-transc-div>: three PTX instructions that all mean x / y, with their
// documented guarantees, drawn as the set of fp32 results each one allows on
// the fp32 grid around the exact quotient.  Sets from divSets() in
// src/transc.js (exact; see the comment there).
import { divSets, stepUlps, DIV_PRESETS } from './transc.js';
import { bits32, fmt32 } from './fp.js';

const W = 720, H = 222, LX = 236, GAP = 58, C = LX + 26 + 3 * GAP, K = 3;
const ROWS = [
  { key: 'rn', op: 'div.rn.f32', what: 'correctly rounded', who: 'CUDA /' },
  { key: 'full', op: 'div.full.f32', what: '≤ 2 ulp from x / y', who: 'Triton /' },
  { key: 'rcp', op: 'rcp.approx.f32', what: 'r ≤ 1 ulp from 1 / y', who: 'FA4: O · rcp(ℓ)' },
  { key: 'mul', op: '  then mul.f32', what: 'x · r, rounded', who: '' },
];
const RY = (i) => 64 + i * 46;

const CSS = `
bbe-transc-div { display: block; height: 264px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.4 var(--sans); color: var(--text); }
bbe-transc-div .ctl { display: flex; align-items: center; gap: 12px; height: 26px; }
bbe-transc-div .seg { display: inline-flex; }
bbe-transc-div .seg button { font: 12px var(--mono); padding: 2px 10px; white-space: nowrap; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-transc-div .seg button + button { border-left: none; }
bbe-transc-div .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-transc-div .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-transc-div .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-transc-div .key { display: flex; gap: 16px; margin-left: auto; font-size: 12px; color: var(--muted); white-space: nowrap; }
bbe-transc-div .key i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
bbe-transc-div .key i.eq { background: var(--green); } bbe-transc-div .key i.ne { background: var(--orange); }
bbe-transc-div .key i.ex { width: 2px; height: 13px; border-radius: 0; background: var(--blue); vertical-align: -2px; }
bbe-transc-div .m { font-family: var(--mono); font-variant-numeric: tabular-nums; }
bbe-transc-div svg { display: block; margin-top: 6px; font: 11px var(--mono); font-variant-numeric: tabular-nums; }
bbe-transc-div svg text { fill: var(--muted); }
bbe-transc-div .tick { stroke: var(--rule-strong); stroke-width: 1; }
bbe-transc-div .grid { stroke: var(--rule); stroke-width: 1; stroke-dasharray: 2 3; }
bbe-transc-div .lane { stroke: var(--rule-strong); stroke-width: 1; }
bbe-transc-div .ex { stroke: var(--blue); stroke-width: 1.5; }
bbe-transc-div text.op { fill: var(--ink); font: 600 12px var(--mono); }
bbe-transc-div text.what { fill: var(--muted); font: 11.5px var(--sans); }
bbe-transc-div text.cnt { fill: var(--text); font: 11.5px var(--sans); }
bbe-transc-div text.dec { fill: var(--text); }
bbe-transc-div .eq { fill: var(--green); stroke: var(--card); stroke-width: 1.5; }
bbe-transc-div .ne { fill: var(--orange); stroke: var(--card); stroke-width: 1.5; }
bbe-transc-div .br { fill: none; stroke: var(--rule-strong); stroke-width: 1; }
bbe-transc-div text.who { fill: var(--faint); font: 11.5px var(--sans); }
bbe-transc-div .arr { stroke: var(--muted); stroke-width: 1; fill: none; }
bbe-transc-div .ah { fill: var(--muted); }
`;

const dec = (v) => v.toPrecision(9);
const sgn = (u) => (u > 0 ? '+' : u < 0 ? '−' : '') + Math.abs(u);

class BbeTranscDiv extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-transc-div-css')) {
      const st = document.createElement('style'); st.id = 'bbe-transc-div-css'; st.textContent = CSS; document.head.append(st);
    }
    this.sel = 1;
    this.innerHTML = `<div class="ctl"><span class="seg">${DIV_PRESETS.map(([x, y], i) =>
      `<button data-i="${i}">${x} / ${y}</button>`).join('')}</span>
      <span class="key"><span><i class="ex"></i>exact</span><span><i class="eq"></i>correctly rounded</span><span><i class="ne"></i>other allowed result</span></span></div><div class="fig"></div>`;
    this.addEventListener('click', (e) => {
      const b = e.target.closest('[data-i]');
      if (b && +b.dataset.i !== this.sel) { this.sel = +b.dataset.i; this.render(); }
    });
    this.render();
  }
  render() {
    const [x, y] = DIV_PRESETS[this.sel], S = divSets(x, y), rb = bits32(S.rn), rrb = bits32(S.rcpRn);
    this.querySelectorAll('[data-i]').forEach((b) => b.classList.toggle('on', +b.dataset.i === this.sel));
    const off = (v) => bits32(v) - rb, roff = (v) => bits32(v) - rrb, PX = (k) => C + k * GAP;
    // position of an exact real t on a lane whose 0 is the fp32 value z
    const at = (t, z) => PX(t >= z ? (t - z) / (stepUlps(z, 1) - z) : (t - z) / (z - stepUlps(z, -1)));
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    s += `<text x="${LX - 10}" y="16" text-anchor="end" style="font:11.5px var(--sans)">correctly rounded x / y</text>`;
    s += `<text x="${LX - 10}" y="32" text-anchor="end" style="font:11.5px var(--sans)">ulps from correctly rounded</text>`;
    for (let k = -K; k <= K; k++) {
      s += `<line class="grid" x1="${PX(k)}" x2="${PX(k)}" y1="38" y2="${H - 4}"/>` +
        `<text x="${PX(k)}" y="32" text-anchor="middle">${sgn(k)}</text>`;
    }
    s += `<text class="dec" x="${PX(0)}" y="16" text-anchor="middle">${dec(S.rn)}</text>`;
    ROWS.forEach((r, i) => {
      const yy = RY(i), rc = r.key === 'rcp';
      const ks = r.key === 'rn' ? [0] : r.key === 'full' ? S.full.map(off) : rc ? S.rcps.map(roff) : S.viaRcp.map(off);
      s += `<text class="op" x="0" y="${yy - 2}" xml:space="preserve">${r.op}</text><text class="what" x="0" y="${yy + 13}">${r.what}</text>`;
      if (r.who) s += `<text class="who" x="${W}" y="${(rc ? yy + 23 : yy) + 4}" text-anchor="end">${r.who}</text>`;
      s += `<line class="lane" x1="${PX(-K) - 18}" x2="${PX(K) + 18}" y1="${yy}" y2="${yy}"/>`;
      const ex = rc ? at(1 / y, S.rcpRn) : at(S.t, S.rn);
      s += `<line class="ex" x1="${ex}" x2="${ex}" y1="${yy - 13}" y2="${yy + 13}"/>`;
      const k0 = Math.min(...ks), k1 = Math.max(...ks);
      if (ks.length > 1) s += `<path class="br" d="M${PX(k0) - 11},${yy - 4} v-8 H${PX(k1) + 11} v8"/>`;
      for (const k of ks) s += `<circle class="${k === 0 ? 'eq' : 'ne'}" cx="${PX(k)}" cy="${yy}" r="6"/>`;
    });
    // each allowed r and the x*r it produces
    for (const [r, q] of S.pairs) {
      const x1 = PX(roff(r)), x2 = PX(off(q)), y1 = RY(2) + 8, y2 = RY(3) - 15;
      s += `<path class="arr" d="M${x1},${y1} L${x2},${y2}" marker-end="url(#bbe-div-ah)"/>`;
    }
    s += `<defs><marker id="bbe-div-ah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L6,3 L0,6 z" class="ah"/></marker></defs>`;
    this.querySelector('.fig').innerHTML = s + '</svg>';
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-transc-div', BbeTranscDiv);
