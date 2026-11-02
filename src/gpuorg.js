// <bbe-gpuorg>: the programming model (grid -> CTAs -> warps -> lanes) next
// to the hardware it runs on (SMs with warp schedulers, a register file and
// shared memory; L2 and HBM below).  Hover a CTA to see which SM holds it.
// The ①②③ badges match <bbe-hwtree>'s three reduction levels.
const W = 720, H = 372;
const NC = 8, NSM = 3;                    // CTAs drawn, SMs drawn
const smOf = (c) => (c < 6 ? c % NSM : -1);   // CTAs 6, 7 wait for room

const CSS = `
bbe-gpuorg { display: block; height: ${H + 26}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-gpuorg .ctl { font-size: 12px; color: var(--muted); height: 22px; }
bbe-gpuorg svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-gpuorg text { fill: var(--muted); }
bbe-gpuorg .hd { font-weight: 600; font-size: 12px; fill: var(--text); }
bbe-gpuorg .grp { fill: none; stroke: var(--rule-strong); }
bbe-gpuorg .gl { font-size: 10.5px; fill: var(--text); }
bbe-gpuorg .sl { font-size: 10px; fill: var(--faint); }
bbe-gpuorg .cta rect { fill: var(--card); stroke: var(--rule-strong); }
bbe-gpuorg .cta text { font: 10px var(--mono); fill: var(--text); text-anchor: middle; }
bbe-gpuorg .cta { cursor: pointer; }
bbe-gpuorg .cta.wait rect { stroke-dasharray: 3 2; fill: none; }
bbe-gpuorg .cta.wait text { fill: var(--faint); }
bbe-gpuorg .zoomb { fill: none; stroke: var(--rule-strong); }
bbe-gpuorg .cta.on rect { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.5; }
bbe-gpuorg .warp { fill: var(--panel); stroke: var(--rule-strong); }
bbe-gpuorg .lane { fill: var(--blue-soft); stroke: var(--blue); stroke-width: 0.5; }
bbe-gpuorg .sm { fill: var(--card); stroke: var(--rule-strong); }
bbe-gpuorg .sm.on { stroke: var(--amber); stroke-width: 1.5; }
bbe-gpuorg .sched { fill: var(--panel); stroke: var(--rule-strong); }
bbe-gpuorg .rf { fill: var(--blue-soft); stroke: var(--blue); stroke-width: 0.8; }
bbe-gpuorg .reg { stroke: var(--blue); stroke-width: 0.4; opacity: 0.5; }
bbe-gpuorg .smem { fill: var(--green-soft); stroke: var(--green); stroke-width: 0.8; }
bbe-gpuorg .l2 { fill: var(--panel); stroke: var(--rule-strong); }
bbe-gpuorg .gmem { fill: var(--purple-soft); stroke: var(--purple); stroke-width: 0.8; }
bbe-gpuorg .in { font-size: 10px; fill: var(--text); }
bbe-gpuorg .mono { font: 10px var(--mono); }
bbe-gpuorg .fact { font-size: 10.5px; fill: var(--text); }
bbe-gpuorg .num circle { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.2; }
bbe-gpuorg .num text { fill: var(--ink); font: 600 10px var(--mono); text-anchor: middle; }
bbe-gpuorg .wire { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-go-ah); }
bbe-gpuorg #bbe-go-ah path { fill: var(--faint); }
bbe-gpuorg .bus { stroke: var(--faint); stroke-width: 1.2; fill: none; }
bbe-gpuorg .chip rect { fill: var(--card); stroke: var(--rule-strong); }
bbe-gpuorg .chip text { font: 9.5px var(--mono); fill: var(--text); text-anchor: middle; }
bbe-gpuorg .chip.on rect { fill: var(--amber-soft); stroke: var(--amber); stroke-width: 1.5; }
bbe-gpuorg .hit { fill: transparent; cursor: pointer; }
`;

const t = (x, y, cls, s, a) => `<text class="${cls}" x="${x}" y="${y}"${a ? ` text-anchor="${a}"` : ''}>${s}</text>`;
const num = (x, y, n) => `<g class="num"><circle cx="${x}" cy="${y}" r="7"/>${t(x, y + 3.5, '', n)}</g>`;

function svg(sel) {
  const o = [`<defs><marker id="bbe-go-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>`];
  // ---- software view
  o.push(t(0, 13, 'hd', 'Software: what a kernel launches'));
  o.push(`<rect class="grp" x="0.5" y="24" width="280" height="96" rx="5"/>`, t(10, 39, 'gl', 'grid of thread blocks (CTAs)'));
  const ctaXY = (c) => [12 + (c % 4) * 66, 48 + Math.floor(c / 4) * 34];
  o.push(`<g class="cta wait"><rect x="0.5" y="128" width="16" height="10" rx="2"/></g>`, t(22, 137, 'sl', 'waiting for an SM'));
  for (let c = 0; c < NC; c++) {
    const [x, y] = ctaXY(c);
    o.push(`<g class="cta${c === sel ? ' on' : ''}${smOf(c) < 0 ? ' wait' : ''}" data-c="${c}"><rect x="${x}" y="${y}" width="58" height="26" rx="3"/>${t(x + 29, y + 17, '', `CTA ${c}`)}</g>`);
  }
  // CTA 0, opened up: warps of 32 lanes
  o.push(`<rect class="grp" x="0.5" y="150" width="280" height="122" rx="5"/>`, t(10, 165, 'gl', 'CTA 0: warps (4 of up to 32)'));
  for (let w = 0; w < 4; w++) {
    const y = 176 + w * 22;
    o.push(`<rect class="warp" x="8" y="${y}" width="265" height="17" rx="2"/>`, t(14, y + 12, 'in', `warp ${w}`));
    for (let i = 0; i < 32; i++) o.push(`<rect class="lane" x="${58 + i * 6.7}" y="${y + 3}" width="5.2" height="11" rx="0.5"/>`);
  }
  o.push(`<path class="zoomb" d="M58 276 v4 H${58 + 31 * 6.7 + 5.2} v-4"/>`, t(58 + 31 * 3.35 + 2.6, 294, 'sl', '32 lanes (threads)', 'middle'));

  // ---- hardware view
  const HX = 336;
  o.push(t(HX, 13, 'hd', 'Hardware: where it runs'));
  o.push(`<path class="wire" d="M281 72 H${HX}"/>`, t(308, 66, 'sl', 'runs on', 'middle'));
  const s0 = smOf(sel) === 0;
  o.push(`<rect class="sm${s0 ? ' on' : ''}" x="${HX}" y="24" width="270" height="248" rx="5"/>`, t(HX + 10, 39, 'gl', 'SM 0'));
  // resident CTAs as chips
  const chip = (x, y, c) => `<g class="chip${c === sel ? ' on' : ''}"><rect x="${x}" y="${y}" width="40" height="15" rx="2"/>${t(x + 20, y + 11, '', `CTA ${c}`)}</g>`;
  o.push(t(HX + 10, 58, 'sl', 'resident'), chip(HX + 56, 47, 0), chip(HX + 100, 47, 3));
  // warp schedulers, one instruction broadcast to 32 lanes
  for (let k = 0; k < 4; k++) o.push(`<rect class="sched" x="${HX + 10 + k * 64}" y="70" width="58" height="18" rx="2"/>`, t(HX + 39 + k * 64, 82, 'sl', 'scheduler', 'middle'));
  const bx0 = HX + 12, bx1 = HX + 12 + 31 * 7.9;
  o.push(`<path class="wire" d="M${HX + 39} 88 V114"/>`, t(HX + 46, 104, 'in', 'one instruction (e.g. FADD)'));
  o.push(`<path class="bus" d="M${bx0} 114 H${bx1}"/>`);
  for (let i = 0; i < 32; i++) {
    const x = bx0 + i * 7.9;
    o.push(`<path class="bus" d="M${x} 114 V119"/><rect class="lane" x="${x - 2.8}" y="119" width="5.6" height="10" rx="0.5"/>`);
  }
  o.push(t(HX + 10, 145, 'fact', 'one warp’s 32 lanes, in lockstep (SIMT)'));
  // register file
  o.push(`<rect class="rf" x="${HX + 10}" y="166" width="250" height="20" rx="2"/>`);
  for (let i = 1; i < 32; i++) o.push(`<path class="reg" d="M${HX + 10 + i * 250 / 32} 166 V186"/>`);
  o.push(t(HX + 10, 198, 'sl', 'register file: a private column per lane'));
  o.push(num(HX + 17, 214, 1), t(HX + 28, 218, 'fact', 'shuffles: lane ↔ lane, within a warp'));
  o.push(`<rect class="smem" x="${HX + 10}" y="228" width="250" height="18" rx="2"/>`, t(HX + 135, 241, 'in', 'shared memory (per CTA)', 'middle'));
  o.push(num(HX + 17, 259, 2), t(HX + 28, 263, 'fact', 'shared memory + barrier: warps of one CTA'));
  // more SMs
  [1, 2].forEach((k) => {
    const x = HX + 270 + 10 + (k - 1) * 52;
    o.push(`<rect class="sm${smOf(sel) === k ? ' on' : ''}" x="${x}" y="24" width="46" height="248" rx="5"/>`, t(x + 23, 39, 'gl', `SM ${k}`, 'middle'));
    o.push(chip(x + 3, 47, k), chip(x + 3, 66, k + 3));
    o.push(`<rect class="sched" x="${x + 6}" y="90" width="34" height="18" rx="2"/>`, `<rect class="rf" x="${x + 6}" y="166" width="34" height="20" rx="2"/>`,
      `<rect class="smem" x="${x + 6}" y="228" width="34" height="18" rx="2"/>`);
  });
  o.push(t(W - 1, 290, 'sl', '… (132 SMs on H100)', 'end'));
  // memory hierarchy below
  o.push(`<rect class="l2" x="${HX}" y="298" width="${W - HX - 0.5}" height="18" rx="2"/>`, t((HX + W) / 2, 311, 'in', 'L2 cache', 'middle'));
  o.push(`<rect class="gmem" x="${HX}" y="322" width="${W - HX - 0.5}" height="20" rx="2"/>`, t((HX + W) / 2, 336, 'in', 'HBM (global memory)', 'middle'));
  o.push(num(HX + 7, 360, 3), t(HX + 18, 364, 'fact', 'global memory: the only path between SMs'));
  // hit areas for the CTAs (tiles and chips)
  for (let c = 0; c < NC; c++) { const [x, y] = ctaXY(c); o.push(`<rect class="hit" data-c="${c}" x="${x}" y="${y}" width="58" height="26"/>`); }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="A kernel launch is a grid of thread blocks; each block is a set of warps of 32 lanes. Each block runs on one streaming multiprocessor, which has warp schedulers, a register file and shared memory; global memory is shared by all.">${o.join('')}</svg>`;
}

class BbeGpuorg extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-gpuorg-css')) {
      const s = document.createElement('style'); s.id = 'bbe-gpuorg-css'; s.textContent = CSS; document.head.append(s);
    }
    this.sel = 0;
    this.innerHTML = `<div class="ctl">hover a CTA</div><div class="fig"></div>`;
    const fig = this.querySelector('.fig');
    const pick = (e) => {
      const n = e.target.closest?.('[data-c]');
      if (n && +n.dataset.c !== this.sel) { this.sel = +n.dataset.c; fig.innerHTML = svg(this.sel); }
    };
    fig.addEventListener('mouseover', pick);
    fig.addEventListener('click', pick);
    fig.innerHTML = svg(this.sel);
  }
}
if (!customElements.get('bbe-gpuorg')) customElements.define('bbe-gpuorg', BbeGpuorg);
