// <bbe-coalesce>: one warp-wide fp32 load (iteration k of every thread's
// loop) under three element->thread mappings, against memory drawn as
// 32-byte sectors grouped into 128-byte lines.  Row: n = 256 fp32 (1 KB);
// block size T = 32 (one warp), so contiguous spans have S = n / T = 8.
const N = 256, T = 32, S = N / T;
const W = 720, H = 236;
const LX = (l) => 9 + l * 22 + 10;                      // lane centres
const SW = 20.5, LG = 6, MX = 11;                       // sector width, gap between lines, strip left
const fx = (f) => MX + Math.floor(f / 32) * LG + (f / 8) * SW;   // x of fp32 element f (8 per sector, 32 per line)
const LY = 18, MY = 114;                                // lane row top, memory strip top

const MODES = {
  contig: { label: 'contiguous spans', kmax: 2, w: 1, addr: (l, k) => l * S + k,
    addrLabel: (k) => `lane t → x[${S}t + ${k}]` },
  strided: { label: 'strided', kmax: 2, w: 1, addr: (l, k) => k * T + l,
    addrLabel: (k) => `lane t → x[${k}·T + t], T = 32` },
  vec: { label: 'vectorised (float4)', kmax: 1, w: 4, addr: (l, k) => 4 * T * k + 4 * l,
    addrLabel: (k) => { const b = 4 * T * k ? 4 * T * k + ' + ' : ''; return `lane t → x[${b}4t … ${b}4t+3]`; } },
};

const CSS = `
bbe-coalesce { display: block; height: ${H + 32}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-coalesce .ctl { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); height: 26px; }
bbe-coalesce .seg { display: inline-flex; }
bbe-coalesce .seg button { font: 12px var(--sans); padding: 2px 10px; border: 1px solid var(--rule-strong); background: var(--card);
  color: var(--muted); cursor: pointer; }
bbe-coalesce .seg button + button { border-left: none; }
bbe-coalesce .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-coalesce .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-coalesce .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-coalesce .seg button:disabled { color: var(--rule-strong); cursor: default; }
bbe-coalesce .k { font-family: var(--mono); }
bbe-coalesce svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-coalesce text { fill: var(--muted); }
bbe-coalesce .lane { fill: var(--panel); stroke: var(--rule-strong); }
bbe-coalesce .ln { font: 8.5px var(--mono); fill: var(--muted); text-anchor: middle; }
bbe-coalesce .ray { stroke: var(--blue); stroke-width: 0.8; opacity: 0.8; }
bbe-coalesce .sec { fill: var(--card); stroke: var(--rule-strong); stroke-width: 0.7; }
bbe-coalesce .sec.hit { fill: var(--orange-soft); stroke: var(--orange); }
bbe-coalesce .used { fill: var(--blue); }
bbe-coalesce .past { fill: var(--blue-soft); }
bbe-coalesce .br { stroke: var(--rule-strong); fill: none; }
bbe-coalesce .ax { font: 9px var(--mono); fill: var(--faint); text-anchor: middle; }
bbe-coalesce .sl { font-size: 10px; fill: var(--faint); }
bbe-coalesce .cnt { font: 11.5px var(--mono); fill: var(--text); font-variant-numeric: tabular-nums; }
bbe-coalesce .bad { fill: var(--orange); font-weight: 600; }
bbe-coalesce .good { fill: var(--green); font-weight: 600; }
bbe-coalesce .note { font-size: 10.5px; }
`;

const t = (x, y, cls, s, a) => `<text class="${cls}" x="${x}" y="${y}"${a ? ` text-anchor="${a}"` : ''}>${s}</text>`;

function svg(mode, k) {
  const m = MODES[mode], o = [];
  o.push(t(0, 11, 'sl', `one warp-wide load, iteration k = ${k}`));
  for (let l = 0; l < 32; l++) o.push(`<rect class="lane" x="${LX(l) - 10}" y="${LY}" width="20" height="16" rx="2"/>` + t(LX(l), LY + 11.5, 'ln', l));
  // which elements this load reads, and which were read at earlier k
  const now = new Set(), past = new Set();
  for (let l = 0; l < 32; l++) for (let j = 0; j < m.w; j++) {
    now.add(m.addr(l, k) + j);
    for (let q = 0; q < k; q++) past.add(m.addr(l, q) + j);
  }
  const hit = new Set([...now].map((f) => Math.floor(f / 8)));
  for (let s = 0; s < N / 8; s++) o.push(`<rect class="sec${hit.has(s) ? ' hit' : ''}" x="${fx(s * 8)}" y="${MY}" width="${SW}" height="22"/>`);
  for (const f of past) if (!now.has(f)) o.push(`<rect class="past" x="${fx(f) + 0.3}" y="${MY + 1}" width="${SW / 8 - 0.6}" height="20"/>`);
  for (const f of now) o.push(`<rect class="used" x="${fx(f) + 0.3}" y="${MY + 1}" width="${SW / 8 - 0.6}" height="20"/>`);
  // lane -> address rays
  for (let l = 0; l < 32; l++) {
    const a = m.addr(l, k), x = fx(a) + (m.w * SW / 8) / 2;
    o.push(`<path class="ray wire" d="M${LX(l)} ${LY + 16} L${x} ${MY}"/>`);
  }
  // 128-byte lines under the strip
  for (let L = 0; L < N / 32; L++) {
    const x0 = fx(L * 32), x1 = x0 + 4 * SW;
    o.push(`<path class="br" d="M${x0} ${MY + 26} v4 H${x1} v-4"/>`, t((x0 + x1) / 2, MY + 41, 'ax', `${L * 128}–${L * 128 + 127} B`));
  }
  o.push(t(0, MY + 58, 'sl', 'row of 256 fp32 (1 KB): 32-byte sectors in 128-byte lines'));
  // counters
  const sec = hit.size, used = now.size * 4, fetched = sec * 32, eff = used / fetched;
  const cy = MY + 84;
  o.push(`<text class="cnt" x="0" y="${cy}">sectors fetched: ${sec} · bytes used: ${used} of ${fetched} · <tspan class="${eff < 1 ? 'bad' : 'good'}">${(eff * 100).toFixed(1).replace('.0', '')}% efficiency</tspan></text>`);
  o.push(t(W - 1, cy, 'cnt', m.addrLabel(k), 'end'));
  // legend
  const ly = cy + 24;
  o.push(`<rect class="used" x="0" y="${ly - 8}" width="8" height="10"/>`, t(12, ly, 'note', 'bytes used by this load'));
  o.push(`<rect class="sec hit" x="140" y="${ly - 8}" width="14" height="10"/>`, t(158, ly, 'note', 'sector fetched (bytes unused)'));
  o.push(`<rect class="past" x="318" y="${ly - 8}" width="8" height="10"/>`, t(330, ly, 'note', 'read at an earlier k'));
  o.push(t(460, ly, 'note', 'counts: no help from caches'));
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="One warp-wide load: with contiguous spans each lane hits a different 32-byte sector; with a strided loop the 32 lanes read 128 consecutive bytes.">${o.join('')}</svg>`;
}

class BbeCoalesce extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-coalesce-css')) {
      const s = document.createElement('style'); s.id = 'bbe-coalesce-css'; s.textContent = CSS; document.head.append(s);
    }
    this.mode = 'contig'; this.k = 0;
    this.innerHTML = `<div class="ctl"><span class="seg mode">${Object.entries(MODES).map(([key, m]) =>
      `<button data-m="${key}"${key === this.mode ? ' class="on"' : ''}>${m.label}</button>`).join('')}</span>` +
      `<span class="seg step">${[0, 1, 2].map((k) => `<button data-k="${k}"${k ? '' : ' class="on"'}><span class="k">k = ${k}</span></button>`).join('')}</span></div><div class="fig"></div>`;
    this.fig = this.querySelector('.fig');
    this.querySelectorAll('.mode button').forEach((b) => b.onclick = () => { this.mode = b.dataset.m; this.draw(); });
    this.querySelectorAll('.step button').forEach((b) => b.onclick = () => { this.k = +b.dataset.k; this.draw(); });
    this.draw();
  }
  draw() {
    const m = MODES[this.mode];
    this.k = Math.min(this.k, m.kmax);     // float4: 256 floats take two iterations
    this.querySelectorAll('.mode button').forEach((b) => b.classList.toggle('on', b.dataset.m === this.mode));
    this.querySelectorAll('.step button').forEach((b) => { b.classList.toggle('on', +b.dataset.k === this.k); b.disabled = +b.dataset.k > m.kmax; });
    this.fig.innerHTML = svg(this.mode, this.k);
  }
}
if (!customElements.get('bbe-coalesce')) customElements.define('bbe-coalesce', BbeCoalesce);
