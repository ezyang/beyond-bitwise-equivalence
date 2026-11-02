// <bbe-offline-online>: softmax of one query row computed offline (whole
// row at once) and online (three key tiles in order), in two parallel halves
// on the same key axis and scales.  Scores are 91..95 (every fp32 e^s would
// be inf); subtracting a reference m gives the weights p = e^(s - m).
// Offline: m is the row max, all weights at once.  Online: a stepper walks
// the kernel loop -- add tile 1; tile 2 brings a new max; rescale (earlier
// weights, l and O shrink by alpha = e^(m_old - m_new), dashed: before); add
// tile 2; add tile 3.  The offline l and O ends are guide lines that turn
// green when the online sums land on them.  Bar lengths are real arithmetic;
// the displayed values are symbolic.  Default: the last step.
export const TILES = [[91, 93, 92], [95, 92, 94], [93, 94, 91]];
export const V = [[2, 1, 3], [1, 2, 1], [3, 1, 2]];
export const REFS = TILES.map((_, t) => Math.max(...TILES.slice(0, t + 1).flat()));   // online m after each tile
export const M = REFS[2];                                 // offline m
const w = (s, m) => Math.exp(s - m);
const part = (u, m) => TILES[u].reduce(([l, O], s, j) => [l + w(s, m), O + w(s, m) * V[u][j]], [0, 0]);
const sums = (t, m) => [0, 1].map((k) => TILES.slice(0, t + 1).reduce((a, _, u) => a + part(u, m)[k], 0));
export const OFFLINE = sums(2, M);
// online steps: tiles seen (scores shown), tiles added to l and O, the m the
// added weights are measured against, the tile just added, dashed outlines, alpha
export const STEPS = [
  { name: 'tile 1', seen: 1, added: 1, ref: 93, cur: 0, ghost: false, alpha: '–' },
  { name: 'new max', seen: 2, added: 1, ref: 93, cur: -1, ghost: false, alpha: 'e^-2' },
  { name: 'rescale', seen: 2, added: 1, ref: 95, cur: -1, ghost: true, alpha: 'e^-2' },
  { name: 'tile 2', seen: 2, added: 2, ref: 95, cur: 1, ghost: true, alpha: 'e^-2' },
  { name: 'tile 3', seen: 3, added: 3, ref: 95, cur: 2, ghost: false, alpha: '1' },
].map((s) => ({ ...s, m: REFS[s.seen - 1], sums: sums(s.added - 1, s.ref) }));
export const GHOST = sums(0, 93);                         // l and O before x alpha

const W = 720, H = 470, LX = 128, PITCH = 24, TW = 88, BW = 18, G = (TW - 2 * PITCH) / 2;
const MX = 398, AX = 440, BAR = [[472, 54], [604, 42]];   // l and O bars: x0, px per unit
const S0 = 90, SU = 13, PU = 46;                           // score axis floor, px per score unit, px per unit weight
const SB = [122, 336], PB = [212, 426];                    // score and weight baselines: offline, online
const kx = (t, j) => LX + t * TW + j * PITCH, XL = LX - G, XE = kx(2, 2) + G;
const CSS = `
bbe-offline-online { display: block; position: relative; height: ${H}px; max-width: ${W}px; margin: 18px 0 26px; outline: none; }
bbe-offline-online:focus-visible { outline: 1px dashed var(--rule-strong); outline-offset: 4px; }
bbe-offline-online svg { display: block; font: 11px var(--sans); }
bbe-offline-online text { fill: var(--muted); }
bbe-offline-online .h { font-weight: 600; fill: var(--text); font-size: 12px; }
bbe-offline-online .t { font-weight: 600; fill: var(--ink); font-size: 13px; }
bbe-offline-online .sub { fill: var(--faint); }
bbe-offline-online .num { font: 11px var(--mono); font-variant-numeric: tabular-nums; fill: var(--text); }
bbe-offline-online .big { font: 600 13px var(--mono); fill: var(--ink); }
bbe-offline-online .big.mv { fill: var(--orange); }
bbe-offline-online .ax { stroke: var(--rule-strong); }
bbe-offline-online .rule { stroke: var(--rule-strong); }
bbe-offline-online .one { stroke: var(--rule); stroke-dasharray: 3 3; }
bbe-offline-online rect { vector-effect: non-scaling-stroke; transition: transform 300ms ease, opacity 300ms ease, fill 300ms, stroke 300ms; }
bbe-offline-online .sc { fill: var(--panel); stroke: var(--rule-strong); }
bbe-offline-online .sc.off { fill: none; stroke-dasharray: 2 2; }
bbe-offline-online .old { fill: color-mix(in srgb, var(--muted) 35%, var(--card)); stroke: var(--muted); }
bbe-offline-online .cur { fill: color-mix(in srgb, var(--ink) 65%, var(--card)); stroke: var(--ink); }
bbe-offline-online .was { fill: none; stroke: var(--orange); stroke-dasharray: 3 2; }
bbe-offline-online .ref { stroke: var(--ink); stroke-width: 2.5; fill: none; }
bbe-offline-online .refl { fill: var(--ink); font-weight: 600; }
bbe-offline-online .mv-g { transition: transform 300ms ease; }
bbe-offline-online .eq { stroke: var(--rule-strong); stroke-width: 1; stroke-dasharray: 2 3; transition: stroke 300ms; }
bbe-offline-online .eq.on { stroke: var(--green); stroke-width: 2; stroke-dasharray: none; }
bbe-offline-online .hide { opacity: 0; }
bbe-offline-online .ctl { position: absolute; left: 200px; top: ${PB[0] + 24}px; display: flex; gap: 6px; }
bbe-offline-online .seg { display: inline-flex; }
bbe-offline-online .ctl button { font: 12px var(--sans); padding: 2px 9px; border: 1px solid var(--rule-strong); background: var(--card); color: var(--muted); cursor: pointer; white-space: nowrap; }
bbe-offline-online .seg button + button { border-left: none; }
bbe-offline-online .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-offline-online .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-offline-online .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-offline-online .nx { border-radius: 4px; }
bbe-offline-online .nx:disabled { opacity: .4; cursor: default; }
@media (prefers-reduced-motion: reduce) { bbe-offline-online rect, bbe-offline-online .mv-g, bbe-offline-online .eq { transition: none; } }
`;
const sup = (b, e) => `${b}<tspan dy="-5" font-size="9">${e}</tspan><tspan dy="5"></tspan>`;
const alphaText = (a) => (a.startsWith('e^') ? sup('e', a.slice(2).replace('-', '−')) : a);
// bars drawn as unit rects scaled by a CSS transform, so steps can animate them
const vbar = (cls, x, y, h, attr = '') => `<rect class="${cls}" ${attr} x="${x - BW / 2}" y="-1" width="${BW}" height="1" style="transform: translate(0px, ${y}px) scaleY(${h})"/>`;
const hbar = (cls, x, y, len, hgt, attr = '') => `<rect class="${cls}" ${attr} x="0" y="${y}" width="1" height="${hgt}" style="transform: translate(${x}px, 0px) scaleX(${len})"/>`;
const htr = (x, len) => `translate(${x}px, 0px) scaleX(${len})`;

function scores(y, half) {
  const Y = (s) => y - (s - S0) * SU;
  let o = `<text class="h" x="0" y="${y - 30}">scores</text><line class="ax" x1="${XL}" x2="${XE}" y1="${y}" y2="${y}"/>`;
  o += `<text class="num sub" x="${XL - 4}" y="${y + 4}" text-anchor="end">${S0}</text>`;
  o += `<text class="sub" x="${XL - 4}" y="${y + 16}" text-anchor="end">s</text><text class="sub" x="${XL - 4}" y="${y + 29}" text-anchor="end">v</text>`;
  TILES.forEach((ss, t) => ss.forEach((s, j) => {
    o += `<rect class="sc" data-${half}-tile="${t}" x="${kx(t, j) - BW / 2}" y="${Y(s)}" width="${BW}" height="${(s - S0) * SU}"/>`;
    o += `<text class="num" x="${kx(t, j)}" y="${y + 16}" text-anchor="middle">${s}</text><text class="num sub" x="${kx(t, j)}" y="${y + 29}" text-anchor="middle">${V[t][j]}</text>`;
  }));
  // the reference: a flat line that moves with m (online) or sits at the row max (offline)
  return o + `<g class="mv-g" data-${half}-m><line class="ref" x1="${XL}" x2="${XE}" y1="${Y(M)}" y2="${Y(M)}"/><text class="refl" x="${XE + 5}" y="${Y(M) + 4}">m</text></g>`;
}

function weightAxes(y) {
  return `<text class="h" x="0" y="${y - 18}">${sup('p = e', 's − m')}</text>` +
    `<line class="one" x1="${XL}" x2="${XE}" y1="${y - PU}" y2="${y - PU}"/><text class="num sub" x="${XL - 4}" y="${y - PU + 4}" text-anchor="end">1</text>` +
    `<line class="ax" x1="${XL}" x2="${XE}" y1="${y}" y2="${y}"/>`;
}

function svg() {
  let o = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  TILES.forEach((_, t) => (o += `<text class="h" x="${kx(t, 1)}" y="14" text-anchor="middle">tile ${t + 1}</text>`));
  o += `<text class="h" x="${MX}" y="14" text-anchor="middle">m</text><text class="h" x="${AX}" y="14" text-anchor="middle">α</text>`;
  o += `<text class="h" x="${BAR[0][0]}" y="14">ℓ = Σ p</text><text class="h" x="${BAR[1][0]}" y="14">O = Σ p · v</text>`;
  // offline: whole row at once
  o += `<text class="t" x="0" y="38">offline: whole row</text>` + scores(SB[0], 'off') + weightAxes(PB[0]);
  TILES.forEach((ss, u) => ss.forEach((s, j) => (o += vbar('cur', kx(u, j), PB[0], w(s, M) * PU))));
  o += `<text class="big" x="${MX}" y="${PB[0] - 18}" text-anchor="middle">${M}</text><text class="big" x="${AX}" y="${PB[0] - 18}" text-anchor="middle">–</text>`;
  BAR.forEach(([x0, sc], k) => {
    let x = x0;
    TILES.forEach((_, u) => { const len = part(u, M)[k] * sc; o += hbar('cur', x, PB[0] - 30, len, 16); x += len; });
  });
  // online: one tile at a time, one row driven by the stepper
  o += `<line class="rule" x1="0" x2="${W}" y1="${PB[0] + 16}" y2="${PB[0] + 16}"/>`;
  o += `<text class="t" x="0" y="${PB[0] + 40}">online: one tile at a time</text>` + scores(SB[1], 'on') + weightAxes(PB[1]);
  TILES.forEach((ss, u) => ss.forEach((s, j) => {
    if (u === 0) o += vbar('was', kx(u, j), PB[1], w(s, 93) * PU, 'data-ghost');
    o += vbar('old', kx(u, j), PB[1], 0, `data-w="${u},${j}"`);
  }));
  o += `<text class="big" data-mcell x="${MX}" y="${PB[1] - 18}" text-anchor="middle"></text><text class="big" data-acell x="${AX}" y="${PB[1] - 18}" text-anchor="middle"></text>`;
  BAR.forEach(([x0, sc], k) => {
    o += hbar('was', x0, PB[1] - 46, GHOST[k] * sc, 12, 'data-ghost');
    TILES.forEach((_, u) => (o += hbar('old', x0, PB[1] - 30, 0, 16, `data-seg="${k},${u}"`)));
    // offline end: a guide line that turns green when the online sum lands on it
    const x = x0 + OFFLINE[k] * sc;
    o += `<line class="eq" data-eq x1="${x}" x2="${x}" y1="${PB[0] - 36}" y2="${PB[1] - 8}"/>`;
  });
  // legend
  const ly = H - 6;
  [['cur', 'just added', 0], ['old', 'added earlier', 100], ['was', 'before × α', 208]].forEach(([c, t, x]) =>
    (o += `<rect class="${c}" x="${XL + x}" y="${ly - 10}" width="14" height="11"/><text x="${XL + x + 20}" y="${ly}">${t}</text>`));
  o += `<line class="ref" x1="${XL + 302}" x2="${XL + 320}" y1="${ly - 4}" y2="${ly - 4}"/><text x="${XL + 326}" y="${ly}">reference m</text>`;
  o += `<line class="eq on" x1="${XL + 412}" x2="${XL + 430}" y1="${ly - 4}" y2="${ly - 4}"/><text x="${XL + 436}" y="${ly}">offline = online</text>`;
  return o + '</svg>';
}

class BbeOfflineOnline extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-offline-online-css')) {
      const st = document.createElement('style'); st.id = 'bbe-offline-online-css'; st.textContent = CSS; document.head.append(st);
    }
    this.tabIndex = 0;
    this.innerHTML = svg() + `<div class="ctl"><span class="seg">${STEPS.map((s, i) => `<button data-s="${i}">${s.name}</button>`).join('')}</span><button class="nx">step ›</button></div>`;
    this.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (b) this.show(b.dataset.s !== undefined ? +b.dataset.s : this.step + 1);
    });
    // arrow keys step only while the widget has focus; the deck uses them otherwise
    this.addEventListener('keydown', (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!d) return;
      e.preventDefault(); e.stopPropagation();
      this.show(this.step + d);
    });
    this.show(STEPS.length - 1);
  }

  show(i) {
    this.step = Math.max(0, Math.min(STEPS.length - 1, i));
    const S = STEPS[this.step], $$ = (q) => this.querySelectorAll(q);
    $$('.seg button').forEach((b) => b.classList.toggle('on', +b.dataset.s === this.step));
    this.querySelector('.nx').disabled = this.step === STEPS.length - 1;
    $$('[data-on-tile]').forEach((r) => r.classList.toggle('off', +r.dataset.onTile >= S.seen));
    this.querySelector('[data-on-m]').style.transform = `translateY(${(M - S.m) * SU}px)`;
    $$('[data-w]').forEach((r) => {
      const [u, j] = r.dataset.w.split(',').map(Number), on = u < S.added;
      r.setAttribute('class', `${u === S.cur ? 'cur' : 'old'}${on ? '' : ' hide'}`);
      r.style.transform = `translate(0px, ${PB[1]}px) scaleY(${on ? w(TILES[u][j], S.ref) * PU : 0})`;
    });
    $$('[data-ghost]').forEach((r) => r.classList.toggle('hide', !S.ghost));
    $$('[data-seg]').forEach((r) => {
      const [k, u] = r.dataset.seg.split(',').map(Number), [x0, sc] = BAR[k], on = u < S.added;
      const x = x0 + TILES.slice(0, u).reduce((a, _, v) => a + (v < S.added ? part(v, S.ref)[k] * sc : 0), 0);
      r.setAttribute('class', `${u === S.cur ? 'cur' : 'old'}${on ? '' : ' hide'}`);
      r.style.transform = htr(x, on ? part(u, S.ref)[k] * sc : 0);
    });
    this.querySelector('[data-mcell]').textContent = S.m;
    const a = this.querySelector('[data-acell]');
    a.innerHTML = alphaText(S.alpha);
    a.classList.toggle('mv', S.alpha.startsWith('e^'));
    $$('[data-eq]').forEach((l) => l.classList.toggle('on', this.step === STEPS.length - 1));
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-offline-online', BbeOfflineOnline);
