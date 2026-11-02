// <bbe-hwtree>: the hardware hierarchy that turns per-thread partials into
// one sum: lanes within a warp (register shuffles), warps within a block
// (shared memory), blocks across SMs (global memory).  A small GPU sketch
// frames the three levels.  Toggle atomicAdd / fixed order for the last
// level; atomic arrivals reshuffle per click from a fixed seed sequence.
const W = 720, H = 400;
const PY = 164;                                    // panels' top
const Y = (s) => PY + 44 + s * 40;                 // stage rows in the warp panel

const CSS = `
bbe-hwtree { display: block; height: ${H + 34}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-hwtree .ctl { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); height: 26px; }
bbe-hwtree .seg { display: inline-flex; }
bbe-hwtree .ctl button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong); background: var(--card);
  color: var(--muted); cursor: pointer; }
bbe-hwtree .seg button + button { border-left: none; }
bbe-hwtree .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-hwtree .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-hwtree .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; }
bbe-hwtree .again { border-radius: 4px; }
bbe-hwtree .again:disabled { visibility: hidden; }
bbe-hwtree svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-hwtree text { fill: var(--muted); }
bbe-hwtree .frame { fill: none; stroke: var(--rule-strong); }
bbe-hwtree .sm { fill: var(--card); stroke: var(--rule-strong); }
bbe-hwtree .warp { fill: var(--panel); stroke: var(--rule-strong); }
bbe-hwtree .lane { fill: var(--blue-soft); stroke: var(--blue); stroke-width: 0.6; }
bbe-hwtree .smem { fill: var(--green-soft); stroke: var(--green); stroke-width: 0.8; }
bbe-hwtree .gmem { fill: var(--purple-soft); stroke: var(--purple); stroke-width: 0.8; }
bbe-hwtree .fl { font-size: 10px; }
bbe-hwtree .fh { font-weight: 600; font-size: 11px; fill: var(--text); }
bbe-hwtree .hd { font-weight: 600; font-size: 11.5px; fill: var(--text); }
bbe-hwtree .sub { font-size: 10px; fill: var(--faint); }
bbe-hwtree .num circle { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.2; }
bbe-hwtree .num text { fill: var(--ink); font: 600 10px var(--mono); text-anchor: middle; }
bbe-hwtree .cell { fill: var(--card); stroke: var(--rule-strong); }
bbe-hwtree .ct { font: 10px var(--mono); fill: var(--text); text-anchor: middle; }
bbe-hwtree .ax { font: 9.5px var(--mono); fill: var(--faint); text-anchor: middle; }
bbe-hwtree .sl { font: 9.5px var(--mono); fill: var(--faint); }
bbe-hwtree .add circle { fill: var(--card); stroke: var(--faint); }
bbe-hwtree .add text { font: 600 10px var(--mono); fill: var(--muted); text-anchor: middle; }
bbe-hwtree .add.hot circle { fill: var(--amber-soft); stroke: var(--amber); }
bbe-hwtree .add.hot text { fill: var(--ink); }
bbe-hwtree .bf { stroke: var(--rule-strong); stroke-width: 1; fill: none; }
bbe-hwtree .bf.hot { stroke: var(--amber); stroke-width: 1.5; }
bbe-hwtree .wire { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-ht-ah); }
bbe-hwtree .stem { stroke: var(--faint); stroke-width: 1.2; }
bbe-hwtree #bbe-ht-ah path { fill: var(--faint); }
bbe-hwtree .at .wire, bbe-hwtree .at .stem { stroke: var(--orange); }
bbe-hwtree .at .wire { marker-end: url(#bbe-ht-ah-at); }
bbe-hwtree #bbe-ht-ah-at path { fill: var(--orange); }
bbe-hwtree .bdg circle { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.2; }
bbe-hwtree .bdg text { fill: var(--ink); font: 600 10px var(--mono); text-anchor: middle; }
bbe-hwtree .sync { stroke: var(--muted); stroke-dasharray: 3 3; }
bbe-hwtree .res { font: 10px var(--mono); fill: var(--text); }
bbe-hwtree .note { font-size: 10.5px; }
bbe-hwtree .warn { font-size: 10.5px; fill: var(--orange); }
bbe-hwtree .ok { font-size: 10.5px; fill: var(--green); }
bbe-hwtree .arrive { animation: bbe-ht-in 150ms ease-out both; }
@keyframes bbe-ht-in { from { opacity: 0; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { bbe-hwtree .arrive { animation: none; } }
`;

function rng(seed) {   // mulberry32
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const NB = 6;
function shuffled(seed) {
  const r = rng(seed), a = [...Array(NB).keys()];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a.join() === [...Array(NB).keys()].join() ? shuffled(seed + 1000) : a;
}

const t = (x, y, cls, s, a) => `<text class="${cls}" x="${x}" y="${y}"${a ? ` text-anchor="${a}"` : ''}>${s}</text>`;
const num = (x, y, n) => `<g class="num"><circle cx="${x}" cy="${y}" r="7"/>${t(x, y + 3.5, '', n)}</g>`;
const wire = (d) => `<path class="wire" d="${d}"/>`;

// the GPU sketch: SMs -> warps -> lanes, shared memory per SM, global memory
function frame() {
  const o = [`<rect class="frame" x="0.5" y="0.5" width="${W - 1}" height="146" rx="6"/>`, t(10, 15, 'fh', 'GPU'),
    `<rect class="lane" x="240" y="6" width="9" height="11" rx="1"/>`, t(254, 15, 'fl', 'lane (thread) and its registers')];
  o.push(`<rect class="sm" x="10" y="22" width="490" height="86" rx="4"/>`, t(18, 35, 'fl', 'SM 0 (one block)'));
  for (let k = 0; k < 4; k++) {
    const x = 18 + k * 110;
    o.push(`<rect class="warp" x="${x}" y="41" width="102" height="34" rx="3"/>`, t(x + 6, 53, 'fl', `warp ${k}`));
    for (let i = 0; i < 8; i++) o.push(`<rect class="lane" x="${x + 6 + i * 11}" y="58" width="9" height="11" rx="1"/>`);
  }
  o.push(num(474, 58, 1));
  o.push(`<rect class="smem" x="18" y="82" width="432" height="18" rx="2"/>`, t(234, 95, 'fl', 'shared memory (per block)', 'middle'), num(474, 91, 2));
  for (let k = 1; k < 3; k++) {
    const x = 512 + (k - 1) * 90;
    o.push(`<rect class="sm" x="${x}" y="22" width="82" height="86" rx="4"/>`, t(x + 8, 35, 'fl', `SM ${k}`));
    for (let w = 0; w < 4; w++) o.push(`<rect class="warp" x="${x + 8 + w * 17}" y="41" width="13" height="34" rx="2"/>`);
    o.push(`<rect class="smem" x="${x + 8}" y="82" width="64" height="18" rx="2"/>`);
  }
  o.push(t(700, 70, 'fh', '…', 'middle'));
  o.push(`<rect class="gmem" x="10" y="116" width="${W - 20}" height="20" rx="2"/>`, t(W / 2, 130, 'fl', 'global memory (HBM)', 'middle'), num(W - 24, 126, 3));
  return o.join('');
}

// level 1: an 8-lane butterfly (shfl.xor 4, 2, 1); lane 0's tree highlighted
function warpPanel() {
  const x0 = 0, LX = (i) => x0 + 50 + i * 24, o = [];
  o.push(num(x0 + 7, PY + 2, 1), t(x0 + 19, PY + 6, 'hd', 'Within a warp: shuffles'));
  o.push(t(x0, PY + 19, 'sub', '8 of 32 lanes, one partial each'));
  o.push(t(x0, PY + 31, 'sl', 'lane'));
  for (let i = 0; i < 8; i++) o.push(t(LX(i), PY + 31, 'ax', i));
  // which nodes feed lane 0's final value
  const hot = new Set(['3,0']);
  for (let s = 3; s >= 1; s--) for (let i = 0; i < 8; i++) if (hot.has(`${s},${i}`)) { const off = 4 >> (s - 1); hot.add(`${s - 1},${i}`); hot.add(`${s - 1},${i ^ off}`); }
  for (let s = 1; s <= 3; s++) {
    const off = 4 >> (s - 1);
    o.push(t(x0, Y(s) + 3.5, 'sl', `xor ${off}`));
    for (let i = 0; i < 8; i++) for (const j of [i, i ^ off]) {
      const h = hot.has(`${s},${i}`) && hot.has(`${s - 1},${j}`);
      o.push(`<path class="bf${h ? ' hot' : ''}" d="M${LX(j)} ${Y(s - 1) + (s === 1 ? 8 : 7)} L${LX(i)} ${Y(s) - 7}"/>`);
    }
  }
  for (let i = 0; i < 8; i++) o.push(`<rect class="cell" x="${LX(i) - 10}" y="${Y(0) - 8}" width="20" height="16" rx="2"/>` + t(LX(i), Y(0) + 3.5, 'ct', `p${i}`));
  for (let s = 1; s <= 3; s++) for (let i = 0; i < 8; i++) {
    o.push(`<g class="add${hot.has(`${s},${i}`) ? ' hot' : ''}"><circle cx="${LX(i)}" cy="${Y(s)}" r="7"/>${t(LX(i), Y(s) + 3.5, '', '+')}</g>`);
  }
  o.push(t(x0, Y(3) + 26, 'res', 'lane 0 = ((p0+p4)+(p2+p6))'), t(x0 + 48, Y(3) + 39, 'res', '+ ((p1+p5)+(p3+p7))'));
  return o.join('');
}

// level 2: lane 0 of each warp writes to shared memory; warp 0 reduces those
function blockPanel() {
  const x0 = 250, CX = (k) => x0 + 56 + k * 50, o = [];
  o.push(num(x0 + 7, PY + 2, 2), t(x0 + 19, PY + 6, 'hd', 'Across warps: shared memory'));
  o.push(t(x0, PY + 19, 'sub', '4 of up to 32 warps'));
  o.push(t(x0, PY + 31, 'sl', 'warp results (lane 0)'));
  const y0 = Y(0) - 8;
  for (let k = 0; k < 4; k++) {
    o.push(`<rect class="cell" x="${CX(k) - 16}" y="${y0}" width="32" height="16" rx="2"/>` + t(CX(k), y0 + 11.5, 'ct', `w${k}`));
    o.push(wire(`M${CX(k)} ${y0 + 16} V${y0 + 36}`));
  }
  const sy = y0 + 36;
  o.push(`<rect class="smem" x="${CX(0) - 22}" y="${sy}" width="${CX(3) - CX(0) + 44}" height="18" rx="2"/>`);
  o.push(t((CX(1) + CX(2)) / 2, sy + 12.5, 'fl', 'shared memory', 'middle'));
  const by = sy + 34;
  o.push(`<path class="sync" d="M${CX(0) - 22} ${by} H${CX(3) + 22}"/>`, t(x0, by + 3.5, 'sl', 'sync'));
  const ly = by + 14;
  for (let k = 0; k < 4; k++) {
    o.push(wire(`M${CX(k)} ${sy + 18} V${ly}`));
    o.push(`<rect class="cell" x="${CX(k) - 16}" y="${ly}" width="32" height="16" rx="2"/>` + t(CX(k), ly + 11.5, 'ct', `w${k}`));
  }
  o.push(t(x0, ly + 11.5, 'sl', 'warp 0'));
  const ny = [ly + 50, ly + 84];
  [2, 1].forEach((off, s) => {
    const yp = s ? ny[0] + 7 : ly + 16;
    for (let k = 0; k < 4; k++) for (const j of [k, k ^ off]) {
      const h = s ? k === 0 : k === 0 || k === 1;
      o.push(`<path class="bf${h ? ' hot' : ''}" d="M${CX(j)} ${yp} L${CX(k)} ${ny[s] - 7}"/>`);
    }
    for (let k = 0; k < 4; k++) o.push(`<g class="add${(s ? k === 0 : k <= 1) ? ' hot' : ''}"><circle cx="${CX(k)}" cy="${ny[s]}" r="7"/>${t(CX(k), ny[s] + 3.5, '', '+')}</g>`);
    o.push(t(x0, ny[s] + 3.5, 'sl', `xor ${off}`));
  });
  o.push(t(x0, ny[1] + 26, 'res', 'block = (w0+w2) + (w1+w3)'));
  return o.join('');
}

// level 3: one partial per block, combined in global memory
function gridPanel(atomic, order) {
  const x0 = 494, CX = (k) => x0 + 19 + k * 38, o = [];
  o.push(num(x0 + 7, PY + 2, 3), t(x0 + 19, PY + 6, 'hd', 'Across blocks: global memory'));
  o.push(t(x0, PY + 19, 'sub', '6 blocks, on different SMs'));
  o.push(t(x0, PY + 31, 'sl', 'block results'));
  const y0 = Y(0) - 8, gy = y0 + 58, arr = [];
  order.forEach((b, n) => { arr[b] = n; });
  for (let k = 0; k < NB; k++) {
    o.push(`<rect class="cell" x="${CX(k) - 15}" y="${y0}" width="30" height="16" rx="2"/>` + t(CX(k), y0 + 11.5, 'ct', `b${k}`));
    if (atomic) {
      o.push(`<g class="arrive at" style="animation-delay:${arr[k] * 50}ms"><path class="stem" d="M${CX(k)} ${y0 + 16} V${y0 + 22}"/>` +
        `<g class="bdg"><circle cx="${CX(k)}" cy="${y0 + 30}" r="8"/>${t(CX(k), y0 + 33.5, '', arr[k] + 1)}</g>` +
        `<path class="wire" d="M${CX(k)} ${y0 + 38} V${gy}"/></g>`);
    } else o.push(wire(`M${CX(k)} ${y0 + 16} V${gy}`));
  }
  const gx = CX(0) - 15, gw = CX(NB - 1) - CX(0) + 30;
  if (atomic) {
    o.push(`<rect class="gmem" x="${gx}" y="${gy}" width="${gw}" height="20" rx="2"/>` + t(gx + gw / 2, gy + 14, 'ct', 'one fp32 total (atomicAdd)'));
    o.push(`<g class="bdg"><circle cx="${x0 + 8}" cy="${gy + 40}" r="8"/>${t(x0 + 8, gy + 43.5, '', 'n')}</g>`, t(x0 + 22, gy + 44, 'note', 'order of arrival'));
  } else {
    o.push(`<rect class="gmem" x="${gx}" y="${gy}" width="${gw}" height="20" rx="2"/>`);
    for (let k = 0; k < NB; k++) o.push(t(CX(k), gy + 14, 'ct', `[${k}]`));
    o.push(wire(`M${gx + gw / 2} ${gy + 20} V${gy + 40}`));
    o.push(`<rect class="cell" x="${gx}" y="${gy + 40}" width="${gw}" height="20" rx="2"/>` + t(gx + gw / 2, gy + 54, 'ct', '((b0+b1)+b2)+…+b5'));
    o.push(t(gx + gw, gy + 74, 'sl', 'second pass, index order', 'end'));
  }
  return o.join('');
}

class BbeHwtree extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-hwtree-css')) {
      const s = document.createElement('style'); s.id = 'bbe-hwtree-css'; s.textContent = CSS; document.head.append(s);
    }
    this.atomic = true; this.seed = 1;
    this.innerHTML = `<div class="ctl"><span>across blocks:</span><span class="seg"><button data-m="atomic" class="on">atomicAdd</button><button data-m="fixed">fixed order</button></span>` +
      `<button class="again">run again</button></div><div class="fig"></div>`;
    this.fig = this.querySelector('.fig');
    this.again = this.querySelector('.again');
    this.querySelectorAll('.seg button').forEach((b) => b.onclick = () => {
      this.atomic = b.dataset.m === 'atomic';
      if (this.atomic) this.seed++;
      this.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b));
      this.again.disabled = !this.atomic;
      this.draw();
    });
    this.again.onclick = () => { this.seed++; this.draw(); };
    this.draw();
  }
  draw() {
    this.fig.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="How a GPU combines partial sums: lanes within a warp by register shuffles, warps within a block through shared memory, blocks through global memory, either by atomic adds in arrival order or in a fixed order.">` +
      `<defs><marker id="bbe-ht-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker>` +
      `<marker id="bbe-ht-ah-at" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>` +
      frame() + warpPanel() + blockPanel() + gridPanel(this.atomic, this.atomic ? shuffled(this.seed) : [...Array(NB).keys()]) + '</svg>';
  }
}
if (!customElements.get('bbe-hwtree')) customElements.define('bbe-hwtree', BbeHwtree);
