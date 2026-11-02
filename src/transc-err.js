// <bbe-transc-err>: two valid approximations of 2^x, different bits.  The
// error of FA4's software 2^x (degree 3 or 5, Horner with fma or with a
// separate mul and add) over x in [-1, 1], in ulps of 2^x measured from the
// exact value (2^x in double: 29 bits beyond fp32, exact at plot scale).
// The band is the documented ex2.approx.f32 bound, |y - 2^x| <= 2^-22.5 2^x;
// the inner strip, +-1/2 ulp, is where the correctly rounded value lives.
// Dot colour: bitwise equal to the correctly rounded fp32 2^x or not.  Every
// dot is an exact emulation (src/transc.js).
import { ex2, ref2, ulpsApart, EX2_REL, STATS, STATS_N } from './transc.js';
import { bits32, f32, mulberry32, hex32, fmt32 } from './fp.js';

const N = 900, W = 720, H = 236, L = 60, R = 84, T = 10, B = 24, PW = W - L - R, PH = H - T - B;
const SCALE = { 3: { max: 1500, ticks: [-1500, -1000, -500, 0, 500, 1000, 1500] }, 5: { max: 3.3, ticks: [-3, -2, -1, 0, 1, 2, 3] } };

// generic fp32 inputs (full 24-bit significands), one per 1/450 of [-1, 1]
const XS = (() => {
  const rnd = mulberry32(11), xs = [];
  for (let k = 0; k < N; k++) xs.push(f32(-1 + 2 * (k + rnd()) / N));
  return xs;
})();
// one ulp of 2^x: the fp32 spacing in the binade containing 2^x
const ulpAt = (t) => 2 ** (Math.floor(Math.log2(t)) - 23);
// band edge (in ulps) on a fine uniform grid, so the sawtooth is drawn sharp
const BAND = Array.from({ length: 721 }, (_, i) => {
  const x = -1 + i / 360, t = 2 ** x;
  return [x, (EX2_REL * t) / ulpAt(t)];
});
const cache = {};
const points = (deg, fused) => (cache[`${deg}${fused}`] ??= XS.map((x) => {
  const y = ex2(x, deg, fused), r = ref2(x), t = 2 ** x;
  return { x, y, r, u: ulpsApart(y, r), e: (y - t) / ulpAt(t) };
}));

const CSS = `
bbe-transc-err { display: block; height: 392px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.4 var(--sans); color: var(--text); }
bbe-transc-err .ctl { display: flex; gap: 12px; align-items: center; flex-wrap: nowrap; height: 26px; }
bbe-transc-err .seg { display: inline-flex; }
bbe-transc-err .seg button { font: 12px var(--sans); padding: 2px 9px; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; white-space: nowrap; }
bbe-transc-err .seg button + button { border-left: none; }
bbe-transc-err .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-transc-err .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-transc-err .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-transc-err .key { display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin: 8px 0 2px; height: 18px; white-space: nowrap; }
bbe-transc-err .sw { display: inline-block; width: 14px; height: 9px; vertical-align: -1px; margin-right: 5px; background: var(--blue-soft); border: 1px solid var(--blue); box-sizing: border-box; }
bbe-transc-err .sw.cr { background: var(--green-soft); border-color: var(--green); }
bbe-transc-err .dt { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; vertical-align: 0; }
bbe-transc-err .dt.eq { background: var(--green); } bbe-transc-err .dt.ne { background: var(--orange); }
bbe-transc-err svg { display: block; font: 10.5px var(--mono); font-variant-numeric: tabular-nums; }
bbe-transc-err svg text { fill: var(--muted); }
bbe-transc-err .ax { stroke: var(--rule-strong); stroke-width: 1; }
bbe-transc-err .gr { stroke: var(--rule); stroke-width: 1; }
bbe-transc-err .zero { stroke: var(--blue); stroke-width: 1.2; }
bbe-transc-err .band { fill: var(--blue-soft); stroke: var(--blue); stroke-width: 1; }
bbe-transc-err .crb { fill: var(--green-soft); stroke: var(--green); stroke-width: .8; stroke-dasharray: 3 2; }
bbe-transc-err .eq { fill: var(--green); } bbe-transc-err .ne { fill: var(--orange); }
bbe-transc-err .hl { fill: none; stroke: var(--ink); stroke-width: 1.5; }
bbe-transc-err .cur { stroke: var(--amber); stroke-width: 1; }
bbe-transc-err text.lv { fill: var(--text); }
bbe-transc-err text.lvp { fill: var(--faint); font-size: 10px; }
bbe-transc-err text.ttl { font: 11px var(--sans); fill: var(--muted); }
bbe-transc-err text.bl { font: 11px var(--sans); fill: var(--blue); }
bbe-transc-err text.gl { font: 11px var(--sans); fill: var(--green); }
bbe-transc-err .ro { font: 12px var(--mono); font-variant-numeric: tabular-nums; color: var(--text); height: 18px; margin: 2px 0 0 ${L}px; white-space: nowrap; }
bbe-transc-err .ro .p { color: var(--faint); font-family: var(--sans); }
bbe-transc-err .ro .o { color: var(--orange); } bbe-transc-err .ro .g { color: var(--green); }
bbe-transc-err .ro .sel { font-weight: 700; text-decoration: underline; text-underline-offset: 3px; }
bbe-transc-err .stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 2px; }
bbe-transc-err .st { border-top: 2px solid var(--rule); padding-top: 4px; }
bbe-transc-err .st b { display: block; font: 600 15px var(--mono); font-variant-numeric: tabular-nums; color: var(--ink); }
bbe-transc-err .st span { display: block; font-size: 11.5px; color: var(--muted); line-height: 1.3; }
bbe-transc-err .sh { font-size: 11.5px; color: var(--faint); margin-top: 8px; }
bbe-transc-err sup { line-height: 0; }
`;

const pct = (c) => {
  const v = (100 * c) / STATS_N;
  return v === 0 ? '0%' : v === 100 ? '100%' : v < 0.01 ? v.toFixed(4) + '%' : v > 99.99 ? v.toFixed(4) + '%' : v < 1 ? v.toFixed(2) + '%' : v.toFixed(1) + '%';
};
const sgn = (u) => (u > 0 ? '+' : u < 0 ? '−' : '') + Math.abs(u);

class BbeTranscErr extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-transc-err-css')) {
      const st = document.createElement('style'); st.id = 'bbe-transc-err-css'; st.textContent = CSS; document.head.append(st);
    }
    this.deg = 5; this.fused = true;
    this.innerHTML = `<div class="ctl">
        <span class="seg">${[3, 5].map((d) => `<button data-d="${d}">degree ${d}</button>`).join('')}</span>
        <span class="seg"><button data-f="1">Horner with fma (1 rounding per step)</button><button data-f="0">mul, then add (2 roundings)</button></span>
      </div>
      <div class="key"><span><span class="sw"></span>ex2.approx bound</span><span><span class="sw cr"></span>correctly rounded (±½ ulp)</span><span><span class="dt eq"></span>emulation: same bits as correctly rounded</span><span><span class="dt ne"></span>different bits</span></div>
      <div class="plot"></div><div class="ro"></div><div class="sh">2<sup>20</sup> random fp32 x in [−1, 1]</div><div class="stats"></div>`;
    this.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.d) this.deg = +b.dataset.d;
      if (b.dataset.f) this.fused = b.dataset.f === '1';
      this.render();
    });
    this.addEventListener('mousemove', (e) => {
      const svg = this.querySelector('svg');
      if (!svg || !svg.contains(e.target)) return;
      const r = svg.getBoundingClientRect(), px = e.clientX - r.left;
      const k = Math.max(0, Math.min(N - 1, Math.round(((px - L) / PW) * N - 0.5)));
      this.hover(k);
    });
    this.render();
  }
  render() {
    const { deg, fused } = this, P = points(deg, fused), S = SCALE[deg];
    this.querySelectorAll('[data-d]').forEach((b) => b.classList.toggle('on', +b.dataset.d === deg));
    this.querySelectorAll('[data-f]').forEach((b) => b.classList.toggle('on', (b.dataset.f === '1') === fused));
    const X = (x) => L + ((x + 1) / 2) * PW, Y = (u) => T + PH / 2 - (u / S.max) * (PH / 2);
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    for (const t of S.ticks) {
      s += `<line class="${t === 0 ? 'ax' : 'gr'}" x1="${L}" x2="${L + PW}" y1="${Y(t)}" y2="${Y(t)}"/>` +
        `<text x="${L - 6}" y="${Y(t) + 3.5}" text-anchor="end">${sgn(t)}</text>`;
    }
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      s += `<line class="ax" x1="${X(t)}" x2="${X(t)}" y1="${T + PH}" y2="${T + PH + 4}"/>` +
        `<text x="${X(t)}" y="${H - 6}" text-anchor="middle">${sgn(t)}</text>`;
    }
    s += `<text class="ttl" x="${L + PW}" y="${H - 6}" text-anchor="end" dx="${R - 2}">x</text>`;
    s += `<text class="ttl" transform="translate(13 ${T + PH / 2}) rotate(-90)" text-anchor="middle">error (ulps of 2<tspan dy="-3" font-size="8">x</tspan><tspan dy="3">)</tspan></text>`;

    // the bound, and the +-1/2 ulp strip (each at least 1px from the centre
    // line, so both stay visible at degree 3's scale)
    const px1 = S.max / (PH / 2);
    const strip = (hw) => {
      const top = BAND.map(([x, h]) => `${X(x).toFixed(1)},${Y(Math.max(hw(h), px1)).toFixed(1)}`);
      const bot = BAND.map(([x, h]) => `${X(x).toFixed(1)},${Y(-Math.max(hw(h), px1)).toFixed(1)}`).reverse();
      return `${top.join(' ')} ${bot.join(' ')}`;
    };
    s += `<polygon class="band" points="${strip((h) => h)}"/>`;
    s += `<polygon class="crb" points="${strip(() => 0.5)}"/>`;
    for (const p of P) s += `<circle class="${p.u === 0 ? 'eq' : 'ne'}" cx="${X(p.x).toFixed(1)}" cy="${Y(p.e).toFixed(1)}" r="1.7"/>`;
    // right margin: what the band edges are
    const st = STATS[`${deg}${fused ? 'f' : 'm'}`], bx = L + PW + 8;
    if (deg === 5) {
      s += `<text class="bl" x="${bx}" y="${Y(2.83) + 4}">bound</text>`;
      s += `<text class="gl" x="${bx}" y="${Y(0) + 4}">±½ ulp</text>`;
    } else {
      s += `<text class="bl" x="${bx}" y="${Y(0) + 4}">bound</text>`;
    }
    s += `<line class="cur" x1="0" x2="0" y1="${T}" y2="${T + PH}" visibility="hidden"/><circle class="hl" r="4" visibility="hidden"/>`;
    this.querySelector('.plot').innerHTML = s + '</svg>';
    this.X = X; this.Y = Y;
    const other = fused ? 'mul + add' : 'fma';
    const tiles = [
      [`${sgn(st.lo)} … ${sgn(st.hi)}`, 'ulps from correctly rounded'],
      [pct(st.same), 'same bits as correctly rounded'],
      [pct(STATS_N - st.outside), 'within ex2.approx bound'],
      [pct(st.vsOther), `differ from ${other}`],
      [pct(st.bf16), 'differ after rounding to bf16'],
    ];
    this.querySelector('.stats').innerHTML = tiles.map(([v, l]) => `<div class="st"><b>${v}</b><span>${l}</span></div>`).join('');
    const A = points(deg, true), Bm = points(deg, false);
    let k0 = Math.round(N * 0.3);
    while (k0 < N - 1 && A[k0].y === Bm[k0].y) k0++;
    this.hover(k0);
  }
  hover(k) {
    const p = points(this.deg, this.fused)[k], svg = this.querySelector('svg');
    const cx = this.X(p.x), cy = this.Y(p.e);
    const cur = svg.querySelector('.cur'), hl = svg.querySelector('.hl');
    cur.setAttribute('x1', cx); cur.setAttribute('x2', cx); cur.setAttribute('visibility', 'visible');
    hl.setAttribute('cx', cx); hl.setAttribute('cy', cy); hl.setAttribute('visibility', 'visible');
    // both Horner variants at this x: their bits can differ from each other
    const q = (fu) => points(this.deg, fu)[k], hx = (pt, fu) =>
      `<span class="p">${fu ? 'fma' : 'mul+add'}</span> <span class="${pt.u ? 'o' : 'g'}${fu === this.fused ? ' sel' : ''}">${hex32(pt.y)}</span>`;
    this.querySelector('.ro').innerHTML = `x = ${fmt32(p.x).replace('-', '−').padEnd(10, ' ').replace(/ /g, '&nbsp;')}` +
      ` <span class="p">correctly rounded</span> ${hex32(p.r)}&nbsp; ${hx(q(true), true)}&nbsp; ${hx(q(false), false)}` +
      `&nbsp; <span class="p">(${p.u ? sgn(p.u) + ' ulp' + (Math.abs(p.u) > 1 ? 's' : '') : 'same bits'})</span>`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-transc-err', BbeTranscErr);
