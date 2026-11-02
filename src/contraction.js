// <bbe-contraction>: one Triton source line, two instantiations.  On the
// partial-block path InstCombine has pushed the multiply into the mask's
// select, so DAGCombiner's fsub(fmul) -> fma pattern cannot fire; on the
// full-block path it does.  Static diagram plus the PTX census.
const W = 720, PW = 352, GAP = 16;
const PY = 52;                        // panel top
const BW = 164, BH = 28;              // op boxes
const R = {                           // row tops, relative to the panel top
  fmul: 50, sel: 112, fork: 162, fsub: 184, ex2: 246, p: 300, ptx: 314,
};
const PH = 412, H = PY + PH + 92;

const CSS = `
bbe-contraction { display: block; height: ${H + 4}px; margin: 18px 0 26px; font-family: var(--sans); }
bbe-contraction svg { display: block; width: 100%; max-width: ${W}px; height: auto; font-size: 11px; }
bbe-contraction text { fill: var(--muted); }
bbe-contraction .panel { fill: none; stroke: var(--rule-strong); }
bbe-contraction .code { fill: var(--panel); stroke: var(--rule-strong); }
bbe-contraction .src { font: 12.5px var(--mono); fill: var(--ink); text-anchor: middle; }
bbe-contraction .pt { font-weight: 600; font-size: 12px; fill: var(--text); }
bbe-contraction .ps { font-size: 10.5px; fill: var(--faint); }
bbe-contraction .box { fill: var(--card); stroke: var(--rule-strong); }
bbe-contraction .op { font: 11.5px var(--mono); fill: var(--ink); text-anchor: middle; }
bbe-contraction .wire { stroke: var(--faint); stroke-width: 1.2; fill: none; marker-end: url(#bbe-co-ah); }
bbe-contraction #bbe-co-ah path { fill: var(--faint); }
bbe-contraction .dot { fill: var(--faint); }
bbe-contraction .side { font-size: 10.5px; }
bbe-contraction .note { font-size: 10.5px; fill: var(--muted); }
bbe-contraction .no { fill: var(--orange); font-weight: 600; }
bbe-contraction .yes { fill: var(--purple); font-weight: 600; }
bbe-contraction .fuse { fill: var(--purple-soft); stroke: var(--purple); stroke-dasharray: 4 3; }
bbe-contraction .cmp { fill: var(--purple); }
bbe-contraction .ptx { fill: var(--panel); stroke: none; }
bbe-contraction .ins { font: 11px var(--mono); fill: var(--ink); }
bbe-contraction .ins.cmp { fill: var(--purple); }
bbe-contraction .rnd { font-size: 10.5px; fill: var(--muted); }
bbe-contraction .rnd2 { font-size: 10.5px; fill: var(--orange); }
bbe-contraction .rnd1 { font-size: 10.5px; fill: var(--purple); }
bbe-contraction .th { font-size: 10.5px; fill: var(--faint); }
bbe-contraction .num { font: 11px var(--mono); font-variant-numeric: tabular-nums; fill: var(--ink); text-anchor: end; }
bbe-contraction .rule { stroke: var(--rule); }
bbe-contraction .msg { font-weight: 600; font-size: 12.5px; fill: var(--text); }
`;

function panel(px, partial) {
  const o = [], cx = px + 12 + BW / 2, bx = px + 12, ax = px + 12 + BW + 14, y = (k) => PY + R[k];
  const box = (k, label) => `<rect class="box" x="${bx}" y="${y(k)}" width="${BW}" height="${BH}" rx="4"/><text class="op" x="${cx}" y="${y(k) + 18}">${label}</text>`;
  const wire = (y0, y1) => `<path class="wire" d="M${cx} ${y0} V${y1}"/>`;
  o.push(`<rect class="panel" x="${px + 0.5}" y="${PY + 0.5}" width="${PW - 1}" height="${PH}" rx="6"/>`);
  o.push(`<text class="pt" x="${px + 12}" y="${PY + 20}">${partial ? 'Partial block: with the mask' : 'Full block: no mask'}</text>`);
  o.push(`<text class="ps" x="${px + 12}" y="${PY + 35}">LLVM IR after InstCombine, then PTX</text>`);
  if (!partial) {
    // DAGCombiner's match: the fmul and the fsub become one fma
    o.push(`<rect class="fuse" x="${bx - 6}" y="${y('fmul') - 6}" width="${BW + 12}" height="${R.fsub - R.fmul + BH + 12}" rx="5"/>`);
  }
  o.push(box('fmul', 'fmul(qk, RCP_LN2)'));
  if (partial) {
    o.push(box('sel', 'select(valid, ·, −∞)'));
    o.push(wire(y('fmul') + BH, y('sel')), wire(y('sel') + BH, y('fsub')));
    o.push(`<text class="side no" x="${ax}" y="${y('fsub') + 18}">✗ no fma</text>`);
  } else {
    o.push(wire(y('fmul') + BH, y('fsub')));
    o.push(`<text class="side yes" x="${ax}" y="${y('fsub') + 18}">DAGCombiner → fma</text>`);
  }
  // the row max needs the (masked) product: a fork off the spine
  const fy = partial ? PY + R.fork : PY + R.fmul + BH + 17;
  o.push(`<path class="wire" d="M${cx} ${fy} H${ax - 4}"/><circle class="dot" cx="${cx}" cy="${fy}" r="2.5"/>` +
    `<text class="side" x="${ax}" y="${fy + 4}">row max</text>`);
  o.push(box('fsub', partial ? 'fsub(select, m)' : 'fsub(fmul, m)'), wire(y('fsub') + BH, y('ex2')), box('ex2', 'ex2(·)'));
  o.push(wire(y('ex2') + BH, y('p') - 10), `<text class="op" x="${cx}" y="${y('p') + 1}">p</text>`);
  // the PTX this path became
  const ty = y('ptx'), tx = px + 12;
  o.push(`<rect class="ptx" x="${tx}" y="${ty}" width="${PW - 24}" height="${PH - R.ptx - 12}" rx="4"/>`);
  if (partial) {
    o.push(`<text class="ins" x="${tx + 10}" y="${ty + 20}">mul.f32</text><text class="rnd" x="${tx + 96}" y="${ty + 20}">t = round(qk · RCP_LN2)</text>`);
    o.push(`<text class="ins" x="${tx + 10}" y="${ty + 40}">sub.f32</text><text class="rnd" x="${tx + 96}" y="${ty + 40}">round(t − m)</text>`);
    o.push(`<text class="rnd2" x="${tx + 10}" y="${ty + 80}">2 roundings</text>`);
  } else {
    o.push(`<text class="ins cmp" x="${tx + 10}" y="${ty + 20}">fma.rn.f32 %r, %r, 0f3FB8AA3B, %r</text>`);
    o.push(`<text class="rnd" x="${tx + 10}" y="${ty + 36}">round(qk · RCP_LN2 − m)</text>`);
    o.push(`<text class="ins" x="${tx + 10}" y="${ty + 56}">mul.f32</text><text class="rnd" x="${tx + 96}" y="${ty + 56}">round(qk · RCP_LN2), row max</text>`);
    o.push(`<text class="rnd1" x="${tx + 10}" y="${ty + 80}">1 rounding · 0x3FB8AA3B = fp32(log2 e)</text>`);
  }
  return o.join('');
}

function svg() {
  const o = [`<defs><marker id="bbe-co-ah" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker></defs>`];
  // the one source line, feeding both instantiations
  const cw = 300, lx = PW / 2, rx = PW + GAP + PW / 2;
  o.push(`<rect class="code" x="${W / 2 - cw / 2}" y="2" width="${cw}" height="26" rx="4"/><text class="src" x="${W / 2}" y="19">p = tl.math.exp2(qk * RCP_LN2 - m)</text>`);
  o.push(`<path class="wire" d="M${W / 2} 28 V38 H${lx} V${PY}"/><path class="wire" d="M${W / 2} 38 H${rx} V${PY}"/><circle class="dot" cx="${W / 2}" cy="38" r="2.5"/>`);
  o.push(panel(0, true), panel(PW + GAP, false));
  // census
  const cy = PY + PH + 24, c1 = 12, c2 = 150, c3 = 168;
  o.push(`<text class="th" x="${c1}" y="${cy}">PTX opcodes, whole flex kernel (4 copies of the loop body)</text>`);
  const rows = [['sub.f32', '132', '128 partial-block path + 4 scalar'], ['fma.rn.f32', '132', '128 full-block path + 4 other'], ['mul.f32', '771', '']];
  rows.forEach(([op, n, where], i) => {
    const ry = cy + 22 + i * 19;
    o.push(`<text class="ins${op.startsWith('fma') ? ' cmp' : ''}" x="${c1}" y="${ry}">${op}</text><text class="num" x="${c2}" y="${ry}">${n}</text><text class="note" x="${c3}" y="${ry}">${where}</text>`);
    o.push(`<path class="rule" d="M${c1} ${ry + 6} H${c3 + 230}"/>`);
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The Triton line p = exp2(qk * RCP_LN2 - m) compiles to mul plus sub on the masked path and to a single fma on the unmasked path.">${o.join('')}</svg>`;
}

class BbeContraction extends HTMLElement {
  connectedCallback() {
    if (!document.getElementById('bbe-contraction-css')) {
      const s = document.createElement('style'); s.id = 'bbe-contraction-css'; s.textContent = CSS; document.head.append(s);
    }
    this.innerHTML = svg();
  }
}
if (!customElements.get('bbe-contraction')) customElements.define('bbe-contraction', BbeContraction);
