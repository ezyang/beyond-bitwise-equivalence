// <bbe-online>: one query row of attention over two key tiles (scale 1),
// stepping the running state (m, l, O) under three runs: lazy rescaling
// (keep the reference m until a tile max exceeds it by more than 8 log2
// units), rescaling on every tile, and a lazy run whose second tile jumps
// past the threshold.  Everything is computed here, twice:
//  exact:    real arithmetic in doubles;
//  rounded:  p = fround(Math.exp(s - m)) -- the double exp rounded once to
//            fp32 stands in for the kernel's exp; this is an illustration,
//            not a model of ex2.approx --; l = fp32 sum in order after
//            l = fround(l * alpha); P = bf16(p); O = fround(O * alpha), then
//            O = add32(O, mul32(P_j, v_j)) (P_j * v_j is exact in fp32 for
//            these small integer v, so this equals an fma); out =
//            bf16(fround(O / l)); lse = fround(m + log(l)).
import { f32, add32, mul32, bf16 } from './fp.js';

export const TILES = { s1: [1, 3, 2], v1: [1, 2, 3], s2: [5, 4], v2: [6, 1], s2jump: [15, 14] };
export const THRESHOLD = 8;   // log2 units

// one run: per tile { scores, v, m, moved, alpha, pEx, pR, P, l: [exact, rounded], O, Ol }, then the epilogue
export function run(lazy, s2 = TILES.s2) {
  let m = -Infinity, lE = 0, OE = 0, lR = 0, OR = 0;
  const tiles = [[TILES.s1, TILES.v1], [s2, TILES.v2]].map(([s, v]) => {
    const mt = Math.max(...s), first = m === -Infinity;
    const mNew = first ? mt : !lazy || (mt - m) * Math.LOG2E > THRESHOLD ? Math.max(m, mt) : m;
    const alpha = first ? 0 : Math.exp(m - mNew), alphaR = f32(alpha);
    lE *= alpha; OE *= alpha; lR = mul32(lR, alphaR); OR = mul32(OR, alphaR);
    const pEx = s.map((x) => Math.exp(x - mNew)), pR = pEx.map(f32), P = pR.map(bf16);
    s.forEach((_, j) => { lE += pEx[j]; OE += pEx[j] * v[j]; lR = add32(lR, pR[j]); OR = add32(OR, mul32(P[j], v[j])); });
    const t = { scores: s, v, m: mNew, moved: !first && mNew !== m, alpha, pEx, pR, P, l: [lE, lR], O: [OE, OR], Ol: [OE / lE, f32(OR / lR)] };
    m = mNew;
    return t;
  });
  const epi = { outEx: bf16(OE / lE), out: bf16(f32(OR / lR)), lseEx: m + Math.log(lE), lse: f32(m + Math.log(lR)) };
  return { T1: tiles[0], T2: tiles[1], epi };
}

// display strings (orange marks a rounded value that differs from the exact one at this precision)
const g = (x, d) => String(+x.toPrecision(d));
function show(r) {
  const tile = (t) => ({
    scores: t.scores, v: t.v, m: t.m, moved: t.moved, alpha: t.alpha ? g(t.alpha, 6) : '0', p: t.pR,
    pEx: t.pEx.map((x) => g(x, 6)), pSp: t.pR.map((x) => g(x, 6)), P: t.P.map((x) => g(x, 6)),
    l: t.l.map((x) => g(x, 7)), O: t.O.map((x) => g(x, 7)), Ol: t.Ol.map((x) => g(x, 7)),
  });
  const e = r.epi;
  return { T1: tile(r.T1), T2: tile(r.T2), epi: { out: String(e.out), outEx: String(e.outEx), lse: g(e.lse, 7), lseEx: g(e.lseEx, 7) } };
}
export const SCENARIOS = {
  keep: { label: 'lazy rescaling (threshold 8)', name: 'lazy', lazy: true, ...show(run(true)) },
  move: { label: 'rescale every tile', name: 'every tile', lazy: false, ...show(run(false)) },
  jump: { label: 'lazy rescaling: big jump', name: 'lazy, big jump', lazy: true, ...show(run(true, TILES.s2jump)) },
};
const STEPS = ['tile 1', 'tile 2', 'epilogue'];

const CW = 278, CH = 270, BX0 = 30, LX1 = 214;   // chart geometry
const CSS = `
bbe-online { display: block; height: 344px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.4 var(--sans); color: var(--text); }
bbe-online .top { display: flex; justify-content: space-between; align-items: center; height: 26px; }
bbe-online .seg { display: inline-flex; }
bbe-online .seg button { font: 12px var(--sans); padding: 2px 9px; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-online .seg button + button { border-left: none; }
bbe-online .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-online .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-online .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-online .nx { font: 12px var(--sans); padding: 2px 9px; margin-left: 6px; border: 1px solid var(--rule-strong); border-radius: 4px;
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-online .nx:disabled { opacity: .4; cursor: default; }
bbe-online .main { display: flex; gap: 14px; margin-top: 8px; }
bbe-online svg { flex: none; font: 10.5px var(--mono); font-variant-numeric: tabular-nums; }
bbe-online svg text { fill: var(--text); }
bbe-online .ax { stroke: var(--rule-strong); }
bbe-online .grid { stroke: var(--rule); }
bbe-online .tl { fill: var(--faint); }
bbe-online .bar { fill: var(--blue-soft); stroke: var(--blue); }
bbe-online .bar.over { fill: var(--amber-soft); stroke: var(--amber); }
bbe-online .bar.off { fill: none; stroke: var(--rule-strong); stroke-dasharray: 3 2; }
bbe-online .ref { transition: transform 200ms ease; }
bbe-online .ref line { stroke: var(--amber); stroke-width: 2; }
bbe-online .ref text { fill: var(--ink); font-weight: 600; }
bbe-online .ghost line { stroke: var(--faint); stroke-dasharray: 3 3; }
bbe-online .ghost text { fill: var(--faint); }
bbe-online .pl { font-size: 10.5px; }
bbe-online .pl.big { fill: var(--ink); font-weight: 700; }
bbe-online .cap { font: 11px var(--sans); fill: var(--muted); }
bbe-online table { border-collapse: collapse; font-size: 12.5px; flex: 1; align-self: flex-start; }
bbe-online td, bbe-online th { padding: 2.5px 8px 2.5px 0; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: baseline; white-space: nowrap; }
bbe-online th { font-weight: 600; color: var(--muted); font-size: 12px; }
bbe-online td.m { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 11.5px; }
bbe-online td.q { color: var(--muted); }
bbe-online .d { color: var(--orange); }
bbe-online .hi { background: var(--amber-soft); }
bbe-online table.sum { margin-top: 10px; width: 100%; }
bbe-online tr.cur td { background: var(--amber-soft); }
bbe-online .thr line { stroke: var(--muted); stroke-dasharray: 5 3; }
bbe-online .thr text { fill: var(--muted); font-family: var(--sans); }
`;

class BbeOnline extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-online-css')) {
      const st = document.createElement('style'); st.id = 'bbe-online-css'; st.textContent = CSS; document.head.append(st);
    }
    this.sc = 'keep'; this.step = 0;
    this.render(null);
    this.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const prevM = this.curM();
      if (b.dataset.p && b.dataset.p !== this.sc) { this.sc = b.dataset.p; this.render(null); }
      else if (b.dataset.s !== undefined && +b.dataset.s !== this.step) { this.step = +b.dataset.s; this.render(prevM); }
      else if (b.classList.contains('nx') && this.step < 2) { this.step++; this.render(prevM); }
    });
  }
  curM() { const S = SCENARIOS[this.sc]; return this.step === 0 ? S.T1.m : S.T2.m; }

  chart(S, prevM) {
    const top = S.T2.m > 10 ? 16 : 9, tick = top > 10 ? 4 : 1, ymax = top > 10 ? 17.5 : 9.6;
    const Y = (s) => CH - 64 - (s / ymax) * (CH - 84);
    const m = this.curM(), t2 = this.step >= 1;
    const bars = [...S.T1.scores.map((s, j) => ({ s, t: 1, j })), ...S.T2.scores.map((s, j) => ({ s, t: 2, j }))];
    const BX = (i) => BX0 + 26 + i * 33 + (i >= 3 ? 14 : 0);
    let o = `<svg width="${CW}" height="${CH}" viewBox="0 0 ${CW} ${CH}">`;
    for (let s = 0; s <= top; s += tick) o += `<line class="grid" x1="${BX0}" x2="${LX1}" y1="${Y(s)}" y2="${Y(s)}"/><text class="tl" x="${BX0 - 6}" y="${Y(s) + 3.5}" text-anchor="end">${s}</text>`;
    o += `<text class="cap" x="0" y="11">score s</text>`;
    // lazy policies keep the reference until a score exceeds it by 8 log2 units = 8 ln 2 nats
    if (S.lazy) { const th = Y(S.T1.m + 8 * Math.LN2); o += `<g class="thr"><line x1="${BX0}" x2="${LX1}" y1="${th}" y2="${th}"/><text x="${LX1 + 5}" y="${th + 4}">threshold</text></g>`; }
    o += `<line class="ax" x1="${BX0}" x2="${LX1}" y1="${Y(0)}" y2="${Y(0)}"/>`;
    bars.forEach((b, i) => {
      const live = b.t === 1 || t2, cur = (b.t === 1 && this.step === 0) || (b.t === 2 && t2);
      const T = b.t === 1 ? S.T1 : S.T2;
      const cls = !live ? 'off' : b.s > m && cur ? 'over' : '';
      o += `<rect class="bar ${cls}" x="${BX(i) - 11}" y="${Y(b.s)}" width="22" height="${Y(0) - Y(b.s)}"/>`;
      o += `<text x="${BX(i)}" y="${Y(0) + 17}" text-anchor="middle" class="tl">${b.s}</text>`;
      if (cur && this.step < 2) o += `<text class="pl${b.s > m ? ' big' : ''}" x="${BX(i)}" y="${Y(0) + 30}" text-anchor="middle">${+T.p[b.j].toPrecision(3)}</text>`;
      o += `<text class="tl" x="${BX(i)}" y="${Y(0) + 43}" text-anchor="middle">${T.v[b.j]}</text>`;
    });
    o += `<text class="tl" x="${BX0 - 6}" y="${Y(0) + 17}" text-anchor="end">s</text><text class="tl" x="${BX0 - 6}" y="${Y(0) + 30}" text-anchor="end">p</text><text class="tl" x="${BX0 - 6}" y="${Y(0) + 43}" text-anchor="end">v</text>`;
    o += `<text class="cap" x="${(BX(0) + BX(2)) / 2}" y="${CH - 4}" text-anchor="middle">tile 1</text><text class="cap" x="${(BX(3) + BX(4)) / 2}" y="${CH - 4}" text-anchor="middle">tile 2</text>`;
    // old reference (ghost) when it moved, then the live reference line
    if (t2 && S.T2.m !== S.T1.m) o += `<g class="ghost"><line x1="${BX0}" x2="${LX1}" y1="${Y(S.T1.m)}" y2="${Y(S.T1.m)}"/><text x="${LX1 + 5}" y="${Y(S.T1.m) + 4}">old m = ${S.T1.m}</text></g>`;
    const from = prevM ?? m;
    o += `<g class="ref" data-dy="${Y(m) - Y(from)}" style="transform: translateY(${Y(from) - Y(m)}px)"><line x1="${BX0}" x2="${LX1}" y1="${Y(m)}" y2="${Y(m)}"/>` +
      `<text x="${LX1 + 5}" y="${Y(m) + 4}">m = ${m}</text></g>`;
    return o + '</svg>';
  }

  table(S) {
    const T = this.step === 0 ? S.T1 : S.T2, step = this.step;
    const list = (a, ref) => a.map((v, i) => (ref && v !== ref[i] ? `<span class="d">${v}</span>` : v)).join(', ');
    const pair = (name, [ex, sp], hi = '') => `<tr class="${hi}"><td class="q">${name}</td><td class="m">${ex}</td><td class="m${ex !== sp ? ' d' : ''}">${sp}</td></tr>`;
    if (step === 2) {
      const E = S.epi, T2 = S.T2;
      return `<table><tr><th>epilogue</th><th>exact</th><th>fp32 / bf16</th></tr>
        ${pair('ℓ', T2.l)}${pair('O', T2.O)}${pair('O / ℓ', T2.Ol)}
        <tr class="hi"><td class="q">out = bf16(O / ℓ)</td><td class="m">${E.outEx}</td><td class="m${E.out !== E.outEx ? ' d' : ''}">${E.out}</td></tr>
        <tr><td class="q">lse = m + log ℓ</td><td class="m">${E.lseEx}</td><td class="m${E.lse !== E.lseEx ? ' d' : ''}">${E.lse}</td></tr>
        </table>`;
    }
    return `<table><tr><th>after ${STEPS[step]}</th><th>exact</th><th>fp32 / bf16</th></tr>
      <tr><td class="q">reference m</td><td class="m" colspan="2">${T.m}${step === 1 && S.T2.m !== S.T1.m ? ` (was ${S.T1.m})` : ''}</td></tr>
      <tr><td class="q">α (rescale)</td><td class="m" colspan="2">${T.alpha}</td></tr>
      <tr><td class="q">p = exp(s − m)</td><td class="m">${list(T.pEx)}</td><td class="m">${list(T.pSp, T.pEx)}</td></tr>
      <tr class="hi"><td class="q">P̃, bf16, to the MMA</td><td class="m"></td><td class="m">${list(T.P, T.pSp)}</td></tr>
      ${pair('ℓ', T.l)}${pair('O', T.O)}${pair('O / ℓ', T.Ol, 'hi')}
      </table>`;
  }

  summary() {
    const K = SCENARIOS, d = (v, ref) => `<td class="m${v !== ref ? ' d' : ''}">${v}</td>`;
    return `<table class="sum"><tr><th>policy</th><th>exact O / ℓ</th><th>rounded O / ℓ</th><th>out (bf16)</th></tr>` +
      Object.entries(K).map(([k, v]) => `<tr class="${k === this.sc ? 'cur' : ''}"><td class="q">${v.name}</td><td class="m">${v.T2.Ol[0]}</td>` +
        d(v.T2.Ol[1], v.T2.Ol[0]) + d(v.epi.out, v.epi.outEx) + '</tr>').join('') + '</table>';
  }

  render(prevM) {
    const S = SCENARIOS[this.sc];
    this.innerHTML = `<div class="top"><span class="seg">${Object.entries(SCENARIOS).map(([k, v]) =>
      `<button data-p="${k}" class="${k === this.sc ? 'on' : ''}">${v.label}</button>`).join('')}</span>
      <span><span class="seg">${STEPS.map((s, i) => `<button data-s="${i}" class="${i === this.step ? 'on' : ''}">${s}</button>`).join('')}</span><button class="nx"${this.step === 2 ? ' disabled' : ''}>step ›</button></span></div>
      <div class="main">${this.chart(S, prevM)}<div style="flex:1">${this.table(S)}${this.summary()}</div></div>`;
    const g = this.querySelector('.ref');
    if (g && g.dataset.dy !== '0') { g.getBoundingClientRect(); g.style.transform = 'translateY(0px)'; }
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-online', BbeOnline);
