// <bbe-variants>: the online-softmax loop as a source listing; each decision
// site carries a badge with one kernel's choice.  Toggle FA4-style / flex; the
// sites where the two kernels differ are marked orange, the compiler's
// contraction (not in flex's source) purple.
// The loop, one [key, code, comment] per line; dev/slides.html shows the same listing.
export const LOOP = [
  ['for', 'for j in blocks(i):', 'visit order'],
  ['S', '  S  = Q K[j]<sup>T</sup>', 'scores'],
  ['mask', '  S  = where(visible, S, −∞)', 'partial blocks only'],
  ['m', '  m′ = max(m, rowmax S)', 'running max'],
  ['alpha', '  α  = ex2((m − m′) · log2 e / √d)', 'rescale'],
  ['x', '  x  = (S − m′) · log2 e / √d'],
  ['P', '  P̃  = ex2(x)', 'weights'],
  ['l', '  ℓ  = α ℓ + rowsum P̃', 'denominator'],
  ['O', '  O  = α O + P̃ V[j]', 'numerator'],
  ['out', 'out = O / ℓ', 'normalize'],
  ['lse', 'lse = m / √d + log2(ℓ) · ln 2', 'log-sum-exp'],
];
// Badges, keyed by LOOP line: [text, cls] lines, cls 'cmp' = compiler decision.
const SITES = {
  for: {
    fa4: ['128 keys per tile', 'visible blocks in descending order (diagonal first)'],
    flex: ['BLOCK_N = 128 keys (64 when head_dim is 64)', 'partial blocks ascending, then full blocks ascending'] },
  S: { fa4: ['Q Kᵀ on the tensor core'], flex: ['Q Kᵀ on the tensor core'] },
  m: {
    fa4: ['rescale only when the max grows > 8 log2 units', 'otherwise reference m kept, α = 1'],
    flex: ['rescale on every tile'] },
  x: {
    fa4: ['p = ex2(fma(s, c, −m·c))', 'c = log2 e / √d, one precomputed fp32 constant'],
    flex: ['s *= 1/√d;  mask;  s *= log2 e;  p = ex2(s − m)', ['full blocks: p = ex2(fma(s, log2 e, −m))', 'cmp']] },
  l: {
    fa4: ['interleaved per-thread lanes, then a fixed tree', '(tree shape kernel-specific)'],
    flex: ['2 lanes × 64 contiguous columns, left to right', 'lane sums added, then ℓ = fma(ℓ, α, sum)'] },
  O: { fa4: ['P̃ rounded to bf16', 'O·α as the MMA accumulator'], flex: ['P̃ rounded to bf16', 'O·α as the MMA accumulator'] },
  out: { fa4: ['out = O · rcp.approx(ℓ)'], flex: ['out = div.full(O, ℓ)'] },
  lse: { fa4: ['lse = m + log ℓ  (lg2.approx)'], flex: ['lse = m + log2(ℓ)'] },
};
const CX = 18, CW = 264, TX = 290, TW = 422, LH = 15, W = 720;
const lines = (k) => k.map((l) => (typeof l === 'string' ? [l, ''] : l));
const nl = (s) => Math.max(SITES[s].fa4.length, SITES[s].flex.length);
const differs = (s) => JSON.stringify(SITES[s].fa4) !== JSON.stringify(SITES[s].flex);

// layout (independent of the kernel: badges reserve the worst-case height)
const ROWS = [];
let Y = 30;
for (const [key, code] of LOOP) {
  const bh = SITES[key] ? nl(key) * LH + 12 : 0;
  const rh = Math.max(20, bh + 8);   // code line centred on its badge
  ROWS.push({ key, code, y: Y, rh, bh });
  Y += rh;
}
const H = Y + 26, EH = H + 30;

const CSS = `
bbe-variants { display: block; height: ${EH}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-variants .ctl { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); height: 26px; }
bbe-variants .seg { display: inline-flex; }
bbe-variants .seg button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong); background: var(--card);
  color: var(--muted); cursor: pointer; }
bbe-variants .seg button + button { border-left: none; }
bbe-variants .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-variants .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-variants .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-variants svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-variants text { fill: var(--muted); }
bbe-variants .code { fill: var(--panel); }
bbe-variants .t { font: 12px var(--mono); fill: var(--ink); white-space: pre; }
bbe-variants .lead { stroke: var(--rule-strong); stroke-dasharray: 2 2; }
bbe-variants .bdg { fill: var(--panel); stroke: none; }
bbe-variants .bar { stroke: var(--green); stroke-width: 3; }
bbe-variants .bar.dif { stroke: var(--orange); }
bbe-variants .bar.cmpb { stroke: var(--purple); }
bbe-variants .bl { font: 11px var(--mono); fill: var(--text); }
bbe-variants .bl.cmp { fill: var(--purple); }
bbe-variants .tag { font-size: 10px; text-anchor: end; font-weight: 600; }
bbe-variants .tag.same { fill: var(--green); }
bbe-variants .tag.dif { fill: var(--orange); }
bbe-variants .tag.cmp { fill: var(--purple); }
bbe-variants .hd { font-weight: 600; font-size: 11.5px; fill: var(--text); }
bbe-variants .hd.fa4 { fill: var(--fa4-text, var(--text)); }
bbe-variants .hd.flex { fill: var(--flex-text, var(--text)); }
bbe-variants .lg { font-size: 10.5px; }
`;

function svg(who) {
  const o = [];
  o.push(`<text class="hd ${who}" x="${TX}" y="14">${who === 'fa4' ? 'FA4-style (lazy rescaling)' : 'flex attention (head_dim 128)'}</text>`);
  o.push(`<rect class="code" x="${CX - 10}" y="${ROWS[0].y - 4}" width="${CW}" height="${Y - ROWS[0].y + 8}" rx="6"/>`);
  ROWS.forEach(({ key, code, y: y0, rh, bh }) => {
    const c = y0 + rh / 2, y = c - bh / 2;
    const t = code.replace(/<sup>(.*?)<\/sup>/g, '<tspan dy="-5" font-size="9">$1</tspan>');
    o.push(`<text class="t" x="${CX}" y="${c + 4}">${t}</text>`);
    if (bh) {
      const ln = lines(SITES[key][who]), h = ln.length * LH + 12, d = differs(key);
      o.push(`<path class="lead" d="M${CX - 10 + CW} ${c} H${TX}"/>`);
      o.push(`<rect class="bdg" x="${TX}" y="${y}" width="${TW}" height="${h}" rx="3"/>` +
        `<path class="bar${d ? ' dif' : ''}" d="M${TX + 1.5} ${y} V${y + h}"/>`);
      o.push(`<text class="tag ${d ? 'dif' : 'same'}" x="${TX + TW - 8}" y="${y + 14}">${d ? 'differs' : 'same'}</text>`);
      ln.forEach(([l, c], k) => o.push(`<text class="bl ${c}" x="${TX + 12}" y="${y + 16 + k * LH}">${l}</text>`));
      if (who === 'flex' && key === 'x') o.push(`<text class="tag cmp" x="${TX + TW - 8}" y="${y + 16 + LH}">compiler</text>`);
    }
  });
  // legend
  const ly2 = H - 2;
  o.push(`<path class="bar" d="M${TX} ${ly2 - 8} V${ly2 + 2}"/><text class="lg" x="${TX + 7}" y="${ly2}">same in both</text>`);
  o.push(`<path class="bar dif" d="M${TX + 100} ${ly2 - 8} V${ly2 + 2}"/><text class="lg" x="${TX + 107}" y="${ly2}">differs</text>`);
  o.push(`<path class="bar cmpb" d="M${TX + 170} ${ly2 - 8} V${ly2 + 2}"/><text class="lg" x="${TX + 177}" y="${ly2}">compiler decision (not in source)</text>`);
  return `<svg viewBox="0 0 ${W} ${H + 4}" role="img" aria-label="The online-softmax loop of one attention kernel, with the choice an FA4-style kernel or flex attention makes at each step.">${o.join('')}</svg>`;
}

class BbeVariants extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-variants-css')) {
      const s = document.createElement('style'); s.id = 'bbe-variants-css'; s.textContent = CSS; document.head.append(s);
    }
    this.innerHTML = `<div class="ctl"><span class="seg"><button data-w="fa4" class="on">FA4-style (lazy rescaling)</button><button data-w="flex">flex attention</button></span></div><div class="fig"></div>`;
    const fig = this.querySelector('.fig');
    this.querySelectorAll('.seg button').forEach((b) => b.onclick = () => {
      this.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
      fig.innerHTML = svg(b.dataset.w);
    });
    fig.innerHTML = svg('fa4');
  }
}
if (!customElements.get('bbe-variants')) customElements.define('bbe-variants', BbeVariants);
