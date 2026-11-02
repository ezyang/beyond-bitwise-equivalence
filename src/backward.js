// <bbe-backward>: a flash-attention backward (FA4-style) as a dataflow diagram, with the two
// accumulation orders drawn out on the right: dK/dV (query blocks ascending,
// in the MMA accumulator) and dQ (one partial per key tile, TMA reduce-added
// into an fp32 buffer, in a fixed order; descending is drawn).  Toggle deterministic /
// atomics: in atomic mode the dQ arrival order is reshuffled on each click
// (a fixed seed sequence, so the page is deterministic).
const W = 720, LW = 452, RX = 472, RW = W - RX;   // left dataflow width, right panel
const NQ = 5, NK = 6;                               // query blocks / key tiles in the order sketches
const H = 474;

const CSS = `
bbe-backward { display: block; height: ${H + 30}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-backward .ctl { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); height: 26px; }
bbe-backward .seg { display: inline-flex; }
bbe-backward .ctl button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong); background: var(--card);
  color: var(--muted); cursor: pointer; }
bbe-backward .seg button + button { border-left: none; }
bbe-backward .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-backward .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-backward .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; }
bbe-backward .again { border-radius: 4px; }
bbe-backward .ttl { font-weight: 600; color: var(--text); }
bbe-backward .again:disabled { visibility: hidden; }
bbe-backward svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-backward text { fill: var(--muted); }
bbe-backward .box { fill: var(--card); stroke: var(--rule-strong); }
bbe-backward .box.tc { fill: var(--panel); }
bbe-backward .t { font: 500 11.5px var(--mono); fill: var(--ink); }
bbe-backward .st { font-size: 10px; fill: var(--faint); }
bbe-backward .grp { fill: none; stroke: var(--rule-strong); stroke-dasharray: 4 3; }
bbe-backward .gl { font-style: italic; font-size: 10.5px; fill: var(--faint); }
bbe-backward .wire { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-bw-ah); }
bbe-backward #bbe-bw-ah path { fill: var(--faint); }
bbe-backward .dot { fill: var(--faint); }
bbe-backward .chip { fill: var(--blue-soft); stroke: none; }
bbe-backward .chip.no { fill: none; stroke: var(--orange); stroke-dasharray: 3 2; }
bbe-backward .ct { font: 11px var(--mono); fill: var(--ink); }
bbe-backward .ct.no { fill: var(--orange); text-decoration: line-through; }
bbe-backward .wl { font: 10px var(--mono); fill: var(--muted); }
bbe-backward .note { font-size: 10.5px; }
bbe-backward .hd { font-weight: 600; font-size: 11.5px; fill: var(--text); }
bbe-backward .cell { fill: var(--card); stroke: var(--rule-strong); }
bbe-backward .ci { font: 10.5px var(--mono); fill: var(--muted); text-anchor: middle; }
bbe-backward .acc { fill: var(--blue-soft); stroke: var(--blue); }
bbe-backward .acc.at { fill: var(--orange-soft); stroke: var(--orange); }
bbe-backward .stem { stroke: var(--faint); stroke-width: 1.2; }
bbe-backward .at .wire, bbe-backward .at .stem { stroke: var(--orange); }
bbe-backward .at .wire { marker-end: url(#bbe-bw-ah-at); }
bbe-backward #bbe-bw-ah-at path { fill: var(--orange); }
bbe-backward .bdg circle { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.2; }
bbe-backward .bdg text { fill: var(--ink); font: 600 10.5px var(--mono); text-anchor: middle; }
bbe-backward .warn { font-size: 11px; fill: var(--orange); }
bbe-backward .ok { font-size: 10.5px; fill: var(--green); }
bbe-backward .arrive { animation: bbe-bw-in 150ms ease-out both; }
@keyframes bbe-bw-in { from { opacity: 0; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { bbe-backward .arrive { animation: none; } }
`;

// deterministic PRNG (mulberry32) for the atomic-mode arrival orders
function rng(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const DESC = [...Array(NK).keys()].reverse();
function shuffled(seed) {
  const r = rng(seed), a = [...Array(NK).keys()];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a.join() === DESC.join() ? shuffled(seed + 1000) : a;
}

const box = (x, y, w, h, t, st, tc) => `<rect class="box${tc ? ' tc' : ''}" x="${x}" y="${y}" width="${w}" height="${h}" rx="4"/>` +
  `<text class="t" x="${x + 9}" y="${y + (st ? 15 : h / 2 + 4)}">${t}</text>` + (st ? `<text class="st" x="${x + 9}" y="${y + 28}">${st}</text>` : '');
const wire = (d) => `<path class="wire" d="${d}"/>`;
const dot = (x, y) => `<circle class="dot" cx="${x}" cy="${y}" r="2.5"/>`;
const text = (x, y, cls, s, anchor) => `<text class="${cls}" x="${x}" y="${y}"${anchor ? ` text-anchor="${anchor}"` : ''}>${s}</text>`;
const lines = (x, y, cls, ls) => ls.map((l, i) => text(x, y + i * 13, cls, l)).join('');

function left() {
  const o = [];
  // saved from the forward
  o.push(text(0, 17, 'hd', 'Saved by the forward'));
  const chip = (x, w, s, no) => `<rect class="chip${no ? ' no' : ''}" x="${x}" y="4" width="${w}" height="18" rx="3"/>` + text(x + 7, 17, `ct${no ? ' no' : ''}`, s);
  o.push(chip(140, 68, 'out bf16'), chip(214, 68, 'lse fp32'), chip(288, 18, 'P', true), text(312, 17, 'note', 'not saved'));
  // preprocess
  o.push(`<rect class="grp" x="0.5" y="34" width="${LW - 1}" height="96" rx="6"/>`, text(10, 49, 'gl', 'preprocess, once per query row'));
  o.push(box(10, 58, 196, 28, 'D = rowsum(out ∘ dO)'), box(10, 94, 196, 28, 'lse2 = lse · log2 e'));
  o.push(text(218, 76, 'note', 'per-thread sums, then a butterfly'));
  // the loop body
  const LY = 144;
  o.push(`<rect class="grp" x="0.5" y="${LY}" width="${LW - 1}" height="258" rx="6"/>`, text(10, LY + 15, 'gl', 'for each 128-row query block × 128-key tile'));
  const S = [10, LY + 26, 196, 34], DP = [246, LY + 26, 196, 34];
  const P = [10, LY + 88, 196, 34], DS = [246, LY + 148, 196, 34];
  o.push(box(...S, 'S = Q Kᵀ', 'tensor core', 1), box(...DP, 'dP = dO Vᵀ', 'tensor core', 1));
  o.push(box(...P, 'P = ex2(fma(S, c, −lse2))', 'recomputed; c = scale · log2 e'), box(...DS, 'dS = P ∘ (dP − D)', 'fp32'));
  o.push(wire(`M108 ${S[1] + 34} V${P[1]}`));
  o.push(wire(`M${P[0] + 196} ${P[1] + 17} H310 V${DS[1]}`));
  o.push(wire(`M380 ${DP[1] + 34} V${DS[1]}`));
  // MMAs fed by bf16(P) and bf16(dS)
  const MY = LY + 212, MW = 138;
  const MX = [10, 157, 304];
  o.push(box(MX[0], MY, MW, 34, 'dV += Pᵀ dO', 'in the MMA accumulator', 1), box(MX[1], MY, MW, 34, 'dK += dSᵀ Q', 'in the MMA accumulator', 1),
    box(MX[2], MY, MW, 34, 'dQⱼ = dS K', 'TMA reduce into dQ_acc', 1));
  o.push(wire(`M79 ${P[1] + 34} V${MY}`), text(85, MY - 22, 'wl', 'bf16(P)'));
  const fy = DS[1] + 34 + 8;
  o.push(wire(`M344 ${DS[1] + 34} V${fy} H226 V${MY}`), wire(`M344 ${fy} H373 V${MY}`), dot(344, fy), text(232, MY - 4, 'wl', 'bf16(dS)'));
  // epilogue
  // epilogue: one box per gradient, under its MMA (sized to the text, so
  // the row runs past the loop's width; the right panel ends above it)
  const EY = LY + 280;
  o.push(text(10, EY - 4, 'gl', 'epilogue'));
  const EB = [[10, 136, 'dV = bf16(dV_acc)'], [157, 190, 'dK = bf16(dK_acc · scale)'], [357, 190, 'dQ = bf16(dQ_acc · scale)']];
  EB.forEach(([x, w, t], i) => {
    o.push(box(x, EY + 8, w, 28, t));
    o.push(wire(`M${MX[i] + MW / 2} ${MY + 34} V${EY + 8}`));
  });
  return o.join('');
}

function right(atomic, order) {
  const o = [], x0 = RX;
  o.push(text(x0, 17, 'hd', 'Accumulation orders'));
  // dK, dV: one key tile's CTA walks the query blocks ascending
  const cw = 32, gap = 8;
  let y = 44;
  o.push(text(x0, y, 'note', 'dK, dV for key tile j: query blocks ascending'));
  y += 14;
  for (let i = 0; i < NQ; i++) {
    const x = x0 + i * (cw + gap);
    o.push(`<rect class="cell" x="${x}" y="${y}" width="${cw}" height="22" rx="3"/>` + text(x + cw / 2, y + 15, 'ci', `q${i}`));
    o.push(`<path class="stem" d="M${x + cw / 2} ${y + 22} V${y + 32}"/>` + wire(`M${x + cw / 2} ${y + 48} V${y + 58}`));
    o.push(`<g class="bdg"><circle cx="${x + cw / 2}" cy="${y + 40}" r="8"/>${text(x + cw / 2, y + 44, '', i + 1)}</g>`);
  }
  const aw = NK * (cw + gap) - gap;   // as wide as dQ_acc below
  o.push(`<rect class="acc" x="${x0}" y="${y + 58}" width="${aw}" height="22" rx="3"/>` + text(x0 + aw / 2, y + 73, 'ci', 'dK_acc, dV_acc (MMA accumulator)'));

  // dQ: one partial per key tile into the fp32 buffer
  y = 170;
  o.push(text(x0, y, 'note', atomic ? 'dQ for query block i: arrival order' : 'dQ for query block i: fixed order'));
  y += 14;
  const kw = 32, kg = 8, arr = [...Array(NK)];
  order.forEach((k, n) => { arr[k] = n; });
  for (let k = 0; k < NK; k++) {
    const x = x0 + k * (kw + kg), n = arr[k], dl = `style="animation-delay:${n * 50}ms"`;
    o.push(`<rect class="cell" x="${x}" y="${y}" width="${kw}" height="22" rx="3"/>` + text(x + kw / 2, y + 15, 'ci', `k${k}`));
    o.push(`<g class="arrive${atomic ? ' at' : ''}" ${dl}><path class="stem" d="M${x + kw / 2} ${y + 22} V${y + 32}"/><path class="wire" d="M${x + kw / 2} ${y + 48} V${y + 58}"/>` +
      `<g class="bdg"><circle cx="${x + kw / 2}" cy="${y + 40}" r="8"/>${text(x + kw / 2, y + 44, '', n + 1)}</g></g>`);
  }
  const bw = NK * (kw + kg) - kg;
  o.push(`<rect class="acc${atomic ? ' at' : ''}" x="${x0}" y="${y + 58}" width="${bw}" height="22" rx="3"/>` + text(x0 + bw / 2, y + 73, 'ci', atomic ? 'dQ_acc (fp32, atomic adds)' : 'dQ_acc (fp32, TMA reduce-add)'));
  y += 98;
  o.push(`<g class="bdg"><circle cx="${x0 + 8}" cy="${y - 4}" r="8"/>${text(x0 + 8, y, '', 'n')}</g>`, text(x0 + 22, y, 'note', 'order of arrival'));
  return o.join('');
}

class BbeBackward extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-backward-css')) {
      const s = document.createElement('style'); s.id = 'bbe-backward-css'; s.textContent = CSS; document.head.append(s);
    }
    this.atomic = false; this.seed = 1;
    this.innerHTML = `<div class="ctl"><span class="ttl">A flash-attention backward (FA4-style)</span><span>· dQ accumulation:</span><span class="seg"><button data-m="det" class="on">deterministic</button><button data-m="atomic">atomics</button></span>` +
      `<button class="again" disabled>run again</button></div><div class="fig"></div>`;
    this.fig = this.querySelector('.fig');
    this.again = this.querySelector('.again');
    this.querySelectorAll('.seg button').forEach((b) => b.onclick = () => {
      const a = b.dataset.m === 'atomic';
      if (a) this.seed++;            // each click on "atomics" is a new run
      this.atomic = a;
      this.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
      this.again.disabled = !a;
      this.draw();
    });
    this.again.onclick = () => { this.seed++; this.draw(); };
    this.draw();
  }
  draw() {
    const order = this.atomic ? shuffled(this.seed) : DESC;
    this.fig.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="A flash-attention backward pass (FA4-style) as a dataflow diagram, with the order in which dK, dV and dQ are accumulated.">` +
      `<defs><marker id="bbe-bw-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker>` +
      `<marker id="bbe-bw-ah-at" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>` +
      left() + right(this.atomic, order) + '</svg>';
  }
}
if (!customElements.get('bbe-backward')) customElements.define('bbe-backward', BbeBackward);
