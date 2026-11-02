// <bbe-sdpa>: the scaled dot-product attention block diagram (Vaswani et al.
// 2017, Fig. 2) beside the flash-attention loop, with a wire from each block
// to the lines of the loop that compute it.  Hovering a block highlights them.
import { LOOP } from './variants.js';

// [label, LOOP keys]
const BLOCKS = [
  ['MatMul', ['S']],
  ['Scale', ['x']],
  ['Mask (opt.)', ['mask']],
  ['SoftMax', ['m', 'alpha', 'x', 'P', 'l', 'out']],
  ['MatMul', ['O']],
];
const BX = 32, BW = 104, BH = 24, B0 = 64, BP = 40;   // blocks: left, width, height, first centre, pitch
const PX = 200, PW = 270, R0 = 40, RP = 20;           // listing: left, width, first row centre, pitch
const W = PX + PW, H = 262;
const row = (k) => R0 + RP * LOOP.findIndex(([key]) => key === k);
const by = (i) => B0 + BP * i;

const CSS = `
bbe-sdpa { display: block; height: ${H}px; font-family: var(--sans); }
bbe-sdpa svg { display: block; width: ${W}px; height: ${H}px; }
bbe-sdpa .hd { font-size: 11.5px; font-weight: 600; fill: var(--text); }
bbe-sdpa .box { fill: var(--card); stroke: var(--rule-strong); }
bbe-sdpa .bt { font-size: 12px; fill: var(--ink); text-anchor: middle; }
bbe-sdpa .in { font: 12px var(--mono); fill: var(--ink); text-anchor: middle; }
bbe-sdpa .arr { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-sd-ah); }
bbe-sdpa #bbe-sd-ah path { fill: var(--faint); }
bbe-sdpa .code { fill: var(--panel); }
bbe-sdpa .t { font: 12px var(--mono); fill: var(--ink); white-space: pre; }
bbe-sdpa .w { stroke: var(--rule-strong); stroke-width: 1.2; fill: none; }
bbe-sdpa .hl { fill: var(--amber-soft); opacity: 0; }
bbe-sdpa g.b { cursor: default; }
bbe-sdpa g.b.on .box { stroke: var(--amber); }
bbe-sdpa .w.on { stroke: var(--amber); stroke-width: 1.8; }
bbe-sdpa .hl.on { opacity: 1; }
`;

function svg() {
  const o = [`<defs><marker id="bbe-sd-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>`];
  const cx = BX + BW / 2;
  o.push(`<text class="hd" x="${BX - 8}" y="14">scaled dot-product attention</text><text class="hd" x="${PX}" y="14">flash-attention loop</text>`);
  // listing, with a hidden highlight band per row
  o.push(`<rect class="code" x="${PX}" y="${R0 - 16}" width="${PW}" height="${RP * LOOP.length + 12}" rx="6"/>`);
  LOOP.forEach(([key, code], k) => {
    const y = R0 + RP * k;
    o.push(`<rect class="hl" data-k="${key}" x="${PX}" y="${y - RP / 2}" width="${PW}" height="${RP}"/>`);
    o.push(`<text class="t" x="${PX + 12}" y="${y + 4}">${code.replace(/<sup>(.*?)<\/sup>/g, '<tspan dy="-5" font-size="9">$1</tspan>')}</text>`);
  });
  // inputs and the block chain
  o.push(`<text class="in" x="${cx - 18}" y="${B0 - 30}">Q</text><text class="in" x="${cx + 18}" y="${B0 - 30}">K</text>`);
  o.push(`<path class="arr" d="M${cx - 18} ${B0 - 24} V${B0 - BH / 2}"/><path class="arr" d="M${cx + 18} ${B0 - 24} V${B0 - BH / 2}"/>`);
  o.push(`<text class="in" x="${BX - 22}" y="${by(4) + 4}">V</text><path class="arr" d="M${BX - 13} ${by(4)} H${BX}"/>`);
  BLOCKS.forEach(([label, keys], i) => {
    const y = by(i), x0 = BX + BW;
    if (i) o.push(`<path class="arr" d="M${cx} ${by(i - 1) + BH / 2} V${y - BH / 2}"/>`);
    o.push(`<g class="b" data-i="${i}"><rect class="box" x="${BX}" y="${y - BH / 2}" width="${BW}" height="${BH}" rx="4"/>` +
      `<text class="bt" x="${cx}" y="${y + 4}">${label}</text></g>`);
    for (const k of keys) {
      const ry = row(k), m = (x0 + PX) / 2;
      o.push(`<path class="w" data-i="${i}" d="M${x0} ${y} C${m} ${y} ${m} ${ry} ${PX} ${ry}"/>`);
    }
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The blocks of scaled dot-product attention, each wired to the lines of the flash-attention loop that compute it.">${o.join('')}</svg>`;
}

class BbeSdpa extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-sdpa-css')) {
      const s = document.createElement('style'); s.id = 'bbe-sdpa-css'; s.textContent = CSS; document.head.append(s);
    }
    this.innerHTML = svg();
    this.querySelectorAll('g.b').forEach((g) => {
      const keys = BLOCKS[g.dataset.i][1];
      const set = (on) => {
        g.classList.toggle('on', on);
        this.querySelectorAll(`.w[data-i="${g.dataset.i}"]`).forEach((w) => w.classList.toggle('on', on));
        this.querySelectorAll('.hl').forEach((r) => r.classList.toggle('on', on && keys.includes(r.dataset.k)));
      };
      g.onmouseenter = () => set(true);
      g.onmouseleave = () => set(false);
    });
  }
}
if (!customElements.get('bbe-sdpa')) customElements.define('bbe-sdpa', BbeSdpa);
