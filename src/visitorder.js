// <bbe-visitorder>: FA4-style and flex attention side by side on the same
// block-causal mask (3 packed documents, 8 x 8 blocks of 128 x 128); partial
// blocks are filled only where visible.  FA4-style: neutral fill, one arrow
// per row in the visit direction.  Flex: one blue scale whose position is
// visit order, partial list at the strong end, full list at the faint end.
// Below each matrix, the fold of the row sum l for one query row of block 7,
// one lane per step merging into an l stream, with exact fp32 values.
import { f32, add32, fmt32, hex32, mulberry32 } from './fp.js';

const N = 8, B = 128, C = 34, Q = 7;               // blocks per side, tokens per block, cell px, spelled-out query block
const DOCS = [[0, 300], [300, 560], [560, 1024]];  // packed documents [start, end)
export const QROW = 959, SEED = 1;                 // the query row whose l is folded; weight seed
const W = 940, GY = 120, GX = [110, 560], LT = GY + N * C + 60, LH = 40, H = LT + 4 * LH + 66;

const docOf = (t) => DOCS.findIndex(([s, e]) => t >= s && t < e);
// block kind: 'full', 'part', or null (empty).  Query q sees key k iff same document and k <= q.
function kind(qb, kb) {
  let any = false, all = true;
  for (let q = qb * B; q < qb * B + B; q++) for (let k = kb * B; k < kb * B + B; k++) {
    const v = k <= q && docOf(k) === docOf(q);
    any ||= v; all &&= v;
  }
  return all ? 'full' : any ? 'part' : null;
}
export const KIND = [...Array(N)].map((_, qb) => [...Array(N)].map((_, kb) => kind(qb, kb)));
const visible = (qb) => [...Array(N).keys()].filter((kb) => KIND[qb][kb]);
export const ORDER = {
  fa4: (qb) => visible(qb).reverse(),
  flex: (qb) => [...visible(qb).filter((kb) => KIND[qb][kb] === 'part'), ...visible(qb).filter((kb) => KIND[qb][kb] === 'full')],
};

// Row sum for query QROW, with a fixed reference m (no rescaling): one fp32
// weight p in [0, 1) per visible key, drawn in ascending key order from
// mulberry32(SEED).  A block's contribution is its weights summed left to
// right in fp32; the fold adds contributions to l in visit order.  Both
// kernels do identical arithmetic per step; only the order differs.
export const TERMS = {}, LEAF = {};
{
  const rnd = mulberry32(SEED);
  for (const kb of visible(Q)) {
    let s = 0, n = 0;
    for (let k = kb * B; k < kb * B + B; k++) if (k <= QROW && docOf(k) === docOf(QROW)) { s = add32(s, f32(rnd())); n++; }
    TERMS[kb] = n; LEAF[kb] = s;
  }
}
export const FOLD = Object.fromEntries(Object.keys(ORDER).map((who) => {
  const steps = []; let l = 0;
  for (const kb of ORDER[who](Q)) steps.push(l = add32(l, LEAF[kb]));
  return [who, steps];                             // steps[0] = first leaf, steps.at(-1) = final l
}));

// FA4-style: one neutral fill.  Flex: one scale, strong (blue toward ink) to faint (blue toward bg);
// partial list = steps 0-2 by rank in that list, full list = steps 3-4, with a gap between.
const NEUTRAL = 'color-mix(in srgb, var(--faint) 40%, var(--bg))';
const mix = (pct, to) => `color-mix(in srgb, var(--blue) ${pct}%, var(${to}))`;
const SCALE = { part: [mix(42, '--ink'), mix(70, '--ink'), 'var(--blue)'], full: [mix(52, '--bg'), mix(26, '--bg')] };
const colour = (who, qb, kb) => who === 'fa4' ? NEUTRAL :
  SCALE[KIND[qb][kb]][ORDER.flex(qb).filter((k) => KIND[qb][k] === KIND[qb][kb]).indexOf(kb)];

const CSS = `
bbe-visitorder { display: block; height: ${H}px; margin: 14px 0 0; font-family: var(--sans); }
bbe-visitorder svg { display: block; font-size: 12px; }
bbe-visitorder text { fill: var(--muted); }
bbe-visitorder .ax { font-size: 11.5px; fill: var(--faint); }
bbe-visitorder .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
bbe-visitorder .h { font-weight: 600; fill: var(--text); font-size: 15px; }
bbe-visitorder .h.fa4 { fill: var(--fa4-text, var(--text)); }
bbe-visitorder .h.flex { fill: var(--flex-text, var(--text)); }
bbe-visitorder .grid { fill: none; stroke: var(--rule); }
bbe-visitorder .blk { fill: none; stroke: var(--rule-strong); }
bbe-visitorder .dir { fill: none; stroke: var(--faint); stroke-width: 1.2; marker-end: url(#bbe-vo-ah); }
bbe-visitorder #bbe-vo-ah path { fill: var(--faint); }
bbe-visitorder .sel { fill: none; stroke: var(--ink); stroke-width: 1.5; }
bbe-visitorder .br { fill: none; stroke: var(--rule-strong); }
bbe-visitorder .mg { fill: none; stroke: var(--faint); stroke-width: 1.2; marker-end: url(#bbe-vo-ah); }
bbe-visitorder .op circle { fill: var(--card); stroke: var(--ink); stroke-width: 1.5; }
bbe-visitorder .op path { stroke: var(--ink); stroke-width: 1.5; }
bbe-visitorder .bnd { fill: none; stroke: var(--faint); }
bbe-visitorder .v { font: 11.5px var(--mono); font-variant-numeric: tabular-nums; fill: var(--text); }
bbe-visitorder .root rect { fill: var(--panel); stroke: var(--ink); }
bbe-visitorder .root text { font: 600 12px var(--mono); fill: var(--ink); text-anchor: middle; }
bbe-visitorder .root .hx { font-weight: 400; fill: var(--muted); }
bbe-visitorder .root .dg { fill: var(--orange); font-weight: 700; }
`;

// one block in matrix-local coordinates, filled in its visit-order colour, clipped to the visible region;
// its boundary is drawn darker where it runs through a fill
const fill = (who, qb, kb) => `<rect x="${kb * C}" y="${qb * C}" width="${C}" height="${C}" style="fill:${colour(who, qb, kb)}" clip-path="url(#bbe-vo-vis)"/>`;
const edge = (qb, kb) => `<rect class="blk" x="${kb * C}" y="${qb * C}" width="${C}" height="${C}"/>` +
  `<rect class="bnd" x="${kb * C}" y="${qb * C}" width="${C}" height="${C}" clip-path="url(#bbe-vo-vis)"/>`;
const cell = (who, qb, kb) => fill(who, qb, kb) + edge(qb, kb);

function matrix(who, label) {
  const o = [`<text class="h ${who}" x="${N * C / 2}" y="-44" text-anchor="middle">${label}</text>`,
    `<text class="ax" x="${N * C / 2}" y="-24" text-anchor="middle">key block →</text>`,
    `<text class="ax" transform="translate(-30 ${N * C / 2}) rotate(90)" text-anchor="middle">query block →</text>`];
  for (let i = 0; i < N; i++) o.push(`<text class="ax num" x="${i * C + C / 2}" y="-8" text-anchor="middle">${i}</text>`,
    `<text class="ax num" x="-8" y="${i * C + C / 2 + 4}" text-anchor="end">${i}</text>`);
  for (let i = 0; i <= N; i++) o.push(`<path class="grid" d="M0 ${i * C} H${N * C} M${i * C} 0 V${N * C}"/>`);
  for (let r = 0; r < N; r++) for (const k of visible(r)) o.push(fill(who, r, k));
  for (let r = 0; r < N; r++) for (const k of visible(r)) o.push(edge(r, k));
  // FA4-style: kinds interleave, so mark each row's visit direction
  if (who === 'fa4') for (let r = 0; r < N; r++) {
    const ks = ORDER.fa4(r);
    if (ks.length > 1) o.push(`<path class="dir" d="M${ks[0] * C + C * 0.8} ${r * C + C / 2} H${ks.at(-1) * C + C * 0.2}"/>`);
  }
  o.push(`<rect class="sel" x="-3" y="${Q * C - 3}" width="${N * C + 6}" height="${C + 6}" rx="3"/>`);
  DOCS.forEach(([s, e], i) => {
    const x0 = s * C / B + 1, x1 = e * C / B - 1;
    o.push(`<path class="br" d="M${x0} ${N * C + 6} v4 H${x1} v-4"/><text class="ax" x="${(x0 + x1) / 2}" y="${N * C + 24}" text-anchor="middle">doc ${i + 1}</text>`);
  });
  return o.join('');
}

// characters of s that differ from other at the same position, in orange
const diff = (s, other) => [...s].map((ch, i) => (ch !== other[i] ? `<tspan class="dg">${ch}</tspan>` : ch)).join('');

// the fold of l over query block Q's visible key blocks: one lane per step,
// the block added at that step in its own key column; the first lane turns
// down into the running sum and each later lane merges into it
function fold(who, gx) {
  const ks = ORDER[who](Q), steps = FOLD[who], o = [], sx = gx + N * C + 30;
  const other = FOLD[who === 'fa4' ? 'flex' : 'fa4'].at(-1), ly = (j) => LT + j * LH + LH / 2;
  const end = ly(ks.length - 1) + 22;
  o.push(`<text class="ax" x="${gx + 4 * C - 8}" y="${LT - 4}" text-anchor="end">Σ p, query ${QROW}</text>`,
    `<path class="mg" d="M${gx + ks[0] * C + C + 3} ${ly(0)} H${sx} V${end - 2}"/>`);
  ks.forEach((kb, j) => {
    const y = ly(j), x = gx + kb * C;
    o.push(`<text class="ax" x="${gx - 8}" y="${y + 4}" text-anchor="end">step ${j + 1}</text>`,
      `<g transform="translate(${gx} ${y - C / 2 - Q * C})">${cell(who, Q, kb)}</g>`,
      `<text class="v" x="${x - 6}" y="${y + 4}" text-anchor="end">${fmt32(LEAF[kb])}</text>`,
      j && `<path class="mg" d="M${x + C + 3} ${y} H${sx - 11}"/>` +
        `<g class="op"><circle cx="${sx}" cy="${y}" r="9"/><path d="M${sx - 5} ${y} h10 M${sx} ${y - 5} v10"/></g>`);
    if (j && j < ks.length - 1) o.push(`<text class="v" x="${sx + 15}" y="${y + 4}">${fmt32(steps[j])}</text>`);
  });
  o.push(`<g class="root"><rect x="${sx - 62}" y="${end}" width="124" height="36" rx="4"/>` +
    `<text x="${sx}" y="${end + 15}">${diff(fmt32(steps.at(-1)), fmt32(other))}</text>` +
    `<text class="hx" x="${sx}" y="${end + 30}">${diff(hex32(steps.at(-1)), hex32(other))}</text></g>`);
  return o.join('');
}

// FA4-style key: block shapes and the arrow; flex key: the scale itself
function legend() {
  const x0 = GX[0] - 20, sq = (x, y, inner) => inner + `<rect class="blk" x="${x}" y="${y}" width="14" height="14"/>`;
  const fa4 = sq(x0, 4, `<rect x="${x0}" y="4" width="14" height="14" style="fill:${NEUTRAL}"/>`) + `<text x="${x0 + 20}" y="15">full block</text>` +
    sq(x0 + 100, 4, `<path d="M${x0 + 100} 4 V18 H${x0 + 114} Z" style="fill:${NEUTRAL}"/>`) + `<text x="${x0 + 120}" y="15">partial block</text>` +
    sq(x0 + 220, 4, '') + `<text x="${x0 + 240}" y="15">empty block</text>` +
    `<path class="dir" d="M${x0 + 30} 36 H${x0}"/><text x="${x0 + 40}" y="40">visit direction</text>`;
  const SW = 28, GAP = 14, bx = GX[1] + N * C / 2 - (5 * SW + GAP) / 2, by = 28;
  const sw = [...SCALE.part, null, ...SCALE.full].map((c, i) => c && `<rect x="${bx + i * SW - (i > 3 ? SW - GAP : 0)}" y="${by}" width="${SW}" height="14" style="fill:${c}"/>`);
  const fx = bx + 3 * SW + GAP, br = (x0, x1, txt) => `<path class="br" d="M${x0} ${by - 4} v-4 H${x1} v4"/>` +
    `<text class="ax" x="${(x0 + x1) / 2}" y="${by - 12}" text-anchor="middle">${txt}</text>`;
  return fa4 + sw.join('') + br(bx, bx + 3 * SW, 'partial list') + br(fx, fx + 2 * SW, 'full list') +
    `<text x="${bx - 6}" y="${by + 11}" text-anchor="end">first</text><text x="${fx + 2 * SW + 6}" y="${by + 11}">last</text>`;
}

function svg() {
  const vis = DOCS.map(([s, e]) => `M${s * C / B} ${s * C / B} V${e * C / B} H${e * C / B} Z`).join(' ');
  const panels = [['fa4', 'FA4-style'], ['flex', 'flex attention']];
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Block-causal mask over three packed documents, twice: FA4-style with an arrow per row in its visit direction, and flex attention coloured by visit order, partial list then full list; below each, the fp32 fold of one query row's sum.">` +
    `<defs><clipPath id="bbe-vo-vis"><path d="${vis}"/></clipPath>` +
    `<marker id="bbe-vo-ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>` +
    legend() +
    panels.map(([who, label], p) => `<g transform="translate(${GX[p]} ${GY})">${matrix(who, label)}</g>` + fold(who, GX[p])).join('') +
    '</svg>';
}

class BbeVisitorder extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-visitorder-css')) {
      const s = document.createElement('style'); s.id = 'bbe-visitorder-css'; s.textContent = CSS; document.head.append(s);
    }
    this.innerHTML = svg();
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('bbe-visitorder')) customElements.define('bbe-visitorder', BbeVisitorder);
