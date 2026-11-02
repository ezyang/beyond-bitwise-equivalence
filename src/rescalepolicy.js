// <bbe-rescalepolicy>: the two rescale policies side by side at the second
// key tile of <bbe-online>'s example (scale 1, natural exp, as in
// <bbe-offline-online>).  Lazy (FA4-style) keeps m = 3 because the tile max 5
// is under m + 8 ln 2 (8 log2 units); every-tile (flex) moves m to 5 and
// multiplies by alpha = e^-2.  Weights are drawn against the current m.  The
// numbers are online.js's run(): fp32 p, fp32 l, bf16 P for P.V, fp32 O.
// Orange marks the digits where the two policies differ.  Static.
import { run, TILES, THRESHOLD } from './online.js';
import { hex16 } from './fp.js';

export const RUNS = [run(true), run(false)];
const g = (x, d) => String(+x.toPrecision(d));
export const BAND = THRESHOLD * Math.LN2;                     // 8 log2 units in score units (5.545)
export const ROWS = [
  ['m', RUNS.map((r) => String(r.T2.m))],
  ['α', RUNS.map((r) => (r.T2.moved ? `e^${r.T1.m - r.T2.m}` : '1'))],
  [`O / ℓ (exact ${g(RUNS[0].T2.Ol[0], 7)})`, RUNS.map((r) => g(r.T2.Ol[1], 7))],
  ['out = bf16(O / ℓ)', RUNS.map((r) => String(r.epi.out))],
  ['out bits', RUNS.map((r) => hex16(r.epi.out))],
];
const NAMES = ['lazy (FA4-style)', 'every tile (flex)'];
const S = [TILES.s1, TILES.s2];

const W = 720, H = 458, CX = [118, 418], SB = 148, SU = 11.5, PB = 312, PU = 16, BW = 18;
const kx = (c, i) => CX[c] + 30 + i * 30 + (i >= 3 ? 20 : 0);   // 3 keys of tile 1, 2 of tile 2
const tileX = (c, t) => [kx(c, t ? 3 : 0) - 15, kx(c, t ? 4 : 2) + 15];
const XL = (c) => CX[c] + 5, XR = (c) => kx(c, 4) + 25;
const CSS = `
bbe-rescalepolicy { display: block; height: ${H}px; max-width: ${W}px; margin: 18px 0 26px; }
bbe-rescalepolicy svg { display: block; font: 11px var(--sans); }
bbe-rescalepolicy text { fill: var(--muted); }
bbe-rescalepolicy .h { font-weight: 600; fill: var(--text); font-size: 12px; }
bbe-rescalepolicy .t { font-weight: 600; fill: var(--ink); font-size: 13px; }
bbe-rescalepolicy .t.fa4 { fill: var(--fa4-text, var(--ink)); }
bbe-rescalepolicy .t.flex { fill: var(--flex-text, var(--ink)); }
bbe-rescalepolicy .sub { fill: var(--faint); }
bbe-rescalepolicy .num { font: 11px var(--mono); font-variant-numeric: tabular-nums; fill: var(--text); }
bbe-rescalepolicy .val { font: 600 13px var(--mono); font-variant-numeric: tabular-nums; fill: var(--ink); }
bbe-rescalepolicy .d { fill: var(--orange); }
bbe-rescalepolicy .ax { stroke: var(--rule-strong); }
bbe-rescalepolicy .rule { stroke: var(--rule); }
bbe-rescalepolicy .one { stroke: var(--rule); stroke-dasharray: 3 3; }
bbe-rescalepolicy .sc { fill: var(--panel); stroke: var(--rule-strong); }
bbe-rescalepolicy .band { fill: color-mix(in srgb, var(--ink) 8%, transparent); }
bbe-rescalepolicy .old { fill: color-mix(in srgb, var(--muted) 35%, var(--card)); stroke: var(--muted); }
bbe-rescalepolicy .cur { fill: color-mix(in srgb, var(--ink) 65%, var(--card)); stroke: var(--ink); }
bbe-rescalepolicy .was { fill: none; stroke: var(--orange); stroke-dasharray: 3 2; }
bbe-rescalepolicy .ref { stroke: var(--ink); stroke-width: 2.5; fill: none; }
bbe-rescalepolicy .refl { fill: var(--ink); font-weight: 600; }
`;
const sup = (b, e) => `${b}<tspan dy="-5" font-size="9">${e}</tspan><tspan dy="5"></tspan>`;
// the two strings with the part after their common prefix in orange
function split(a, b) {
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  return [a, b].map((s) => `${s.slice(0, i)}<tspan class="d">${s.slice(i)}</tspan>`);
}

function svg() {
  let o = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  o += `<text class="h" x="0" y="${SB - 50}">scores</text><text class="h" x="0" y="${PB - 60}">${sup('p = e', 's − m')}</text>`;
  RUNS.forEach((r, c) => {
    o += `<text class="t ${c ? 'flex' : 'fa4'}" x="${(XL(c) + XR(c)) / 2}" y="14" text-anchor="middle">${NAMES[c]}</text>`;
    [0, 1].forEach((t) => o += `<text class="sub" x="${(tileX(c, t)[0] + tileX(c, t)[1]) / 2}" y="34" text-anchor="middle">tile ${t + 1}</text>`);
    const Y = (s) => SB - s * SU, refs = [r.T1.m, r.T2.m];
    // scores, the lazy threshold band, and the reference over each tile
    if (c === 0) {
      o += `<rect class="band" x="${XL(c)}" y="${Y(r.T1.m + BAND)}" width="${XR(c) - XL(c)}" height="${BAND * SU}"/>`;
      o += `<text class="num" x="${(XL(c) + XR(c)) / 2}" y="${Y(r.T1.m + BAND) + 14}" text-anchor="middle">m + 8 ln 2 = ${(r.T1.m + BAND).toFixed(2)}</text>`;
    }
    o += `<line class="ax" x1="${XL(c)}" x2="${XR(c)}" y1="${SB}" y2="${SB}"/>`;
    S.forEach((ss, t) => ss.forEach((s, j) => {
      const i = t * 3 + j;
      o += `<rect class="sc" x="${kx(c, i) - BW / 2}" y="${Y(s)}" width="${BW}" height="${s * SU}"/><text class="num" x="${kx(c, i)}" y="${SB + 14}" text-anchor="middle">${s}</text>`;
    }));
    o += `<path class="ref" d="M${XL(c)} ${Y(refs[0])} L${(tileX(c, 0)[1] + tileX(c, 1)[0]) / 2} ${Y(refs[0])} L${(tileX(c, 0)[1] + tileX(c, 1)[0]) / 2} ${Y(refs[1])} L${XR(c)} ${Y(refs[1])}"/>`;
    o += `<text class="refl" x="${XR(c) + 5}" y="${Y(refs[1]) + 4}">m = ${refs[1]}</text>`;
    // weights at tile 2 against the current m; dashed: tile-1 weights before x alpha
    o += `<line class="one" x1="${XL(c)}" x2="${XR(c)}" y1="${PB - PU}" y2="${PB - PU}"/><text class="num sub" x="${XL(c) - 4}" y="${PB - PU + 4}" text-anchor="end">1</text>`;
    o += `<line class="ax" x1="${XL(c)}" x2="${XR(c)}" y1="${PB}" y2="${PB}"/>`;
    [r.T1, r.T2].forEach((T, t) => T.scores.forEach((s, j) => {
      const x = kx(c, t * 3 + j) - BW / 2, h = Math.exp(s - r.T2.m) * PU;
      o += `<rect class="${t ? 'cur' : 'old'}" x="${x}" y="${PB - h}" width="${BW}" height="${h}"/>`;
      if (!t && r.T2.moved) { const h0 = Math.exp(s - r.T1.m) * PU; o += `<rect class="was" x="${x}" y="${PB - h0}" width="${BW}" height="${h0}"/>`; }
    }));
  });
  [['cur', 'tile 2', 0], ['old', 'tile 1', 80], ['was', 'tile 1 before × α', 160], ['band', 'm kept', 310]].forEach(([c, t, x]) =>
    (o += `<rect class="${c}" x="${XL(0) + x}" y="${PB + 14}" width="14" height="11"/><text x="${XL(0) + x + 20}" y="${PB + 24}">${t}</text>`));
  o += `<line class="ref" x1="${XL(0) + 390}" x2="${XL(0) + 408}" y1="${PB + 20}" y2="${PB + 20}"/><text x="${XL(0) + 414}" y="${PB + 24}">reference m</text>`;
  // aligned rows of the values; orange where the policies differ
  ROWS.forEach(([name, [a, b]], i) => {
    const y = 376 + i * 19;
    o += `<line class="rule" x1="0" x2="${W}" y1="${y - 13}" y2="${y - 13}"/><text class="h" x="0" y="${y}">${name}</text>`;
    const [sa, sb] = name === 'α' || name === 'm' ? [a, b] : split(a, b);
    [sa, sb].forEach((s, c) => (o += `<text class="val" x="${(XL(c) + XR(c)) / 2}" y="${y}" text-anchor="middle">${s.replace(/e\^(-?\d+)/, (_, e) => sup('e', e.replace('-', '−')))}</text>`));
  });
  return o + '</svg>';
}

class BbeRescalepolicy extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-rescalepolicy-css')) {
      const st = document.createElement('style'); st.id = 'bbe-rescalepolicy-css'; st.textContent = CSS; document.head.append(st);
    }
    this.innerHTML = svg();
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-rescalepolicy', BbeRescalepolicy);
