// <bbe-tileorder>: the order in which an FA4-style kernel and flex attention visit one query block's
// key blocks under a document-causal mask (3 packed documents, 8 x 8 blocks
// of 128 x 128).  Click or hover a query-block row; toggle the kernel.
const N = 8, B = 128;                          // blocks per side, tokens per block
const DOCS = [[0, 300], [300, 560], [560, 1024]];   // packed documents [start, end)
const C = 36, GX = 64, GY = 54;                // cell size, grid top-left
const PX = 396;                                // right panel x
const H = 412;

const docOf = (t) => DOCS.findIndex(([s, e]) => t >= s && t < e);
// block kind: 'full' (every query sees every key), 'part', or null (empty).
// Query q sees key k iff same document and k <= q.
function kind(qb, kb) {
  let any = false, all = true;
  for (let q = qb * B; q < qb * B + B; q++) for (let k = kb * B; k < kb * B + B; k++) {
    const v = k <= q && docOf(k) === docOf(q);
    any ||= v; all &&= v;
  }
  return all ? 'full' : any ? 'part' : null;
}
const KIND = [...Array(N)].map((_, qb) => [...Array(N)].map((_, kb) => kind(qb, kb)));
const visible = (qb) => [...Array(N).keys()].filter((kb) => KIND[qb][kb]);
const ORDER = {
  fa4: (qb) => visible(qb).reverse(),
  flex: (qb) => [...visible(qb).filter((kb) => KIND[qb][kb] === 'part'), ...visible(qb).filter((kb) => KIND[qb][kb] === 'full')],
};

const CSS = `
bbe-tileorder { display: block; height: ${H}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-tileorder .ctl { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); height: 26px; }
bbe-tileorder .seg { display: inline-flex; }
bbe-tileorder .seg button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong); background: var(--card);
  color: var(--muted); cursor: pointer; }
bbe-tileorder .seg button + button { border-left: none; }
bbe-tileorder .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-tileorder .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-tileorder .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-tileorder svg { display: block; width: 100%; max-width: 720px; height: auto; font-size: 11px; }
bbe-tileorder text { fill: var(--muted); }
bbe-tileorder .ax { font-size: 10.5px; fill: var(--faint); }
bbe-tileorder .num { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 10.5px; }
bbe-tileorder .grid { fill: none; stroke: var(--rule); stroke-width: 1; }
bbe-tileorder .vis { fill: var(--blue-soft); }
bbe-tileorder .blk { fill: none; stroke: var(--rule-strong); stroke-width: 1; }
bbe-tileorder .row { fill: transparent; cursor: pointer; }
bbe-tileorder .sel { fill: none; stroke: var(--amber); stroke-width: 2; rx: 3; }
bbe-tileorder .bdg circle { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.2; }
bbe-tileorder .bdg text { fill: var(--ink); font: 600 11px var(--mono); text-anchor: middle; }
bbe-tileorder .h { font-weight: 600; fill: var(--text); font-size: 11.5px; }
bbe-tileorder .chip rect { fill: var(--card); stroke: var(--rule-strong); }
bbe-tileorder .chip.full rect { fill: var(--blue-soft); }
bbe-tileorder .chip .k { font: 600 12px var(--mono); fill: var(--ink); text-anchor: middle; }
bbe-tileorder .chip .t { font-size: 9.5px; text-anchor: middle; }
bbe-tileorder .seq.on .chip rect { stroke: var(--amber); stroke-width: 1.5; }
bbe-tileorder .seq:not(.on) { opacity: 0.55; }
bbe-tileorder .seq .who { font-weight: 600; fill: var(--text); }
bbe-tileorder .arr { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-to-ah); }
bbe-tileorder #bbe-to-ah path { fill: var(--faint); }
bbe-tileorder .pass { stroke: var(--rule-strong); fill: none; }
`;

const sx = (t) => GX + t * C / B, sy = (t) => GY + t * C / B;

function svg(qb, who) {
  const o = [];
  o.push(`<defs><marker id="bbe-to-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>`);
  // axes
  o.push(`<text class="ax" x="${GX + N * C / 2}" y="${GY - 24}" text-anchor="middle">key block →</text>`);
  for (let i = 0; i < N; i++) {
    o.push(`<text class="ax num" x="${GX + i * C + C / 2}" y="${GY - 8}" text-anchor="middle">${i}</text>`);
    o.push(`<text class="ax num" x="${GX - 8}" y="${GY + i * C + C / 2 + 4}" text-anchor="end">${i}</text>`);
  }
  o.push(`<text class="ax" transform="translate(${GX - 30} ${GY + N * C / 2}) rotate(90)" text-anchor="middle">query block →</text>`);
  // exact visibility: one triangle per document (k <= q within the doc)
  for (const [s, e] of DOCS) o.push(`<path class="vis" d="M${sx(s)} ${sy(s)} L${sx(s)} ${sy(e)} L${sx(e)} ${sy(e)} Z"/>`);
  for (let i = 0; i <= N; i++) o.push(`<path class="grid" d="M${GX} ${GY + i * C} H${GX + N * C} M${GX + i * C} ${GY} V${GY + N * C}"/>`);
  for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) if (KIND[r][k]) o.push(`<rect class="blk" x="${GX + k * C}" y="${GY + r * C}" width="${C}" height="${C}"/>`);
  // selected row and its visit order
  const ord = ORDER[who](qb);
  o.push(`<rect class="sel" x="${GX - 2}" y="${GY + qb * C - 2}" width="${N * C + 4}" height="${C + 4}"/>`);
  ord.forEach((kb, i) => o.push(`<g class="bdg"><circle cx="${GX + kb * C + C / 2}" cy="${GY + qb * C + C / 2}" r="9"/>` +
    `<text x="${GX + kb * C + C / 2}" y="${GY + qb * C + C / 2 + 4}">${i + 1}</text></g>`));
  for (let r = 0; r < N; r++) o.push(`<rect class="row" data-r="${r}" x="${GX - 24}" y="${GY + r * C}" width="${N * C + 24}" height="${C}"/>`);
  // the packed documents, as spans under the key axis
  DOCS.forEach(([s, e], i) => o.push(`<path class="pass" d="M${sx(s) + 1} ${GY + N * C + 6} v4 H${sx(e) - 1} v-4"/>` +
    `<text class="ax" x="${(sx(s) + sx(e)) / 2}" y="${GY + N * C + 24}" text-anchor="middle">doc ${i + 1} · ${e - s}</text>`));

  // right panel: legend, then both visit sequences for the selected row
  let y = GY - 8;
  o.push(`<text class="h" x="${PX}" y="${y}">block = 128 queries × 128 keys</text>`);
  y += 12;
  const sw = (x, cls, tri) => (cls ? `<rect x="${x}" y="${y}" width="14" height="14" class="${cls}"/><rect class="blk" x="${x}" y="${y}" width="14" height="14"/>` : '') +
    (tri ? `<path class="vis" d="M${x} ${y} L${x} ${y + 14} L${x + 14} ${y + 14} Z"/><rect class="blk" x="${x}" y="${y}" width="14" height="14"/>` : '');
  o.push(sw(PX, 'vis') + `<text x="${PX + 20}" y="${y + 11}">full</text>`);
  y += 20;
  o.push(sw(PX, '', true) + `<text x="${PX + 20}" y="${y + 11}">partial</text>`);
  y += 20;
  o.push(`<rect x="${PX}" y="${y}" width="14" height="14" class="grid"/><text x="${PX + 20}" y="${y + 11}">empty (skipped)</text>`);
  y += 36;
  o.push(`<text class="h" x="${PX}" y="${y}">visit order, query block ${qb}</text>`);
  y += 14;
  const CW = 40, CG = 18;
  const seq = (k, label) => {
    const ks = ORDER[k](qb), g = [`<g class="seq${k === who ? ' on' : ''}"><text class="who" x="${PX}" y="${y + 20}">${label}</text>`];
    ks.forEach((kb, i) => {
      const x = PX + 66 + i * (CW + CG);
      g.push(`<g class="chip ${KIND[qb][kb]}"><rect x="${x}" y="${y}" width="${CW}" height="32" rx="3"/>` +
        `<text class="k" x="${x + CW / 2}" y="${y + 15}">${kb}</text><text class="t" x="${x + CW / 2}" y="${y + 27}">${KIND[qb][kb] === 'full' ? 'full' : 'partial'}</text></g>`);
      if (i) g.push(`<path class="arr" d="M${x - CG + 3} ${y + 16} H${x}"/>`);
    });
    // flex: bracket the two passes
    if (k === 'flex') {
      const np = ks.filter((kb) => KIND[qb][kb] === 'part').length, nf = ks.length - np;
      const br = (i0, n, txt) => {
        const x0 = PX + 66 + i0 * (CW + CG), x1 = x0 + n * CW + (n - 1) * CG;
        return `<path class="pass" d="M${x0} ${y + 37} v4 H${x1} v-4"/><text class="ax" x="${(x0 + x1) / 2}" y="${y + 53}" text-anchor="middle">${txt}</text>`;
      };
      if (np) g.push(br(0, np, 'partial list'));
      if (nf) g.push(br(np, nf, 'full list'));
    }
    g.push('</g>');
    o.push(g.join(''));
  };
  seq('fa4', 'FA4-style');
  y += 56;
  seq('flex', 'flex');
  return `<svg viewBox="0 0 720 ${H - 34}" role="img" aria-label="Block-causal attention mask over three packed documents, with the order in which an FA4-style kernel and flex attention visit one query block's key blocks.">${o.join('')}</svg>`;
}

class BbeTileorder extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-tileorder-css')) {
      const s = document.createElement('style'); s.id = 'bbe-tileorder-css'; s.textContent = CSS; document.head.append(s);
    }
    this.qb = 7; this.who = 'fa4';
    this.innerHTML = `<div class="ctl"><span class="seg"><button data-w="fa4" class="on">FA4-style</button><button data-w="flex">flex</button></span>` +
      `<span>hover a query-block row</span></div><div class="fig"></div>`;
    this.fig = this.querySelector('.fig');
    this.querySelectorAll('.seg button').forEach((b) => b.onclick = () => {
      this.who = b.dataset.w;
      this.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
      this.draw();
    });
    const pick = (e) => {
      const r = e.target.closest?.('.row');
      if (r && +r.dataset.r !== this.qb) { this.qb = +r.dataset.r; this.draw(); }
    };
    this.fig.addEventListener('click', pick);
    this.fig.addEventListener('mouseover', pick);
    this.draw();
  }
  draw() { this.fig.innerHTML = svg(this.qb, this.who); }
}
if (!customElements.get('bbe-tileorder')) customElements.define('bbe-tileorder', BbeTileorder);
