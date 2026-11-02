// <bbe-fma>: a*b + c two ways.  Unfused round(round(a*b) + c) vs fused
// round(a*b + c), on a cancellation example where c = -round(a*b): the
// unfused result is exactly 0, the fused one is the product's rounding error.
// All values come from src/fp.js (exact BigInt fma), not from constants.
import { f32, mul32, add32, fma32, exact32, exactMul, exactSum, exactDecimal, hex32, fmt32 } from './fp.js';
const dec = (x) => exactDecimal(exact32(x), 20);

export function model() {
  const a = f32(1 + 2 ** -12), b = a;
  const p = mul32(a, b), c = -p;
  const prod = exactMul(exact32(a), exact32(b));           // exact a*b
  const unfused = add32(p, c), fused = fma32(a, b, c);
  // significand bits of the exact product, leading bit first
  const s = prod.m.toString(2), top = s.length - 1 + prod.e;   // top = exponent of leading bit
  const NB = 1 + 26, bits = [];
  for (let i = 0; i < NB; i++) bits.push(i < s.length ? +s[i] : 0);
  const exactSumAB = exactSum([prod, exact32(c)]);
  return { a, b, c, p, prod, top, bits, unfused, fused, exactSumAB };
}

const CSS = `
bbe-fma { display: block; height: 252px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.45 var(--sans); color: var(--text); }
bbe-fma .m { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 12.5px; }
bbe-fma .mut { color: var(--muted); }
bbe-fma .in { display: grid; grid-template-columns: max-content max-content max-content; gap: 1px 14px; align-items: baseline; }
bbe-fma .in .h { color: var(--faint); font-size: 12px; }
bbe-fma .strip { margin: 12px 0 4px; position: relative; }
bbe-fma .lab { font-size: 12px; color: var(--muted); margin-bottom: 3px; }
bbe-fma .cells { display: flex; align-items: flex-end; font: 12px var(--mono); }
bbe-fma .c { width: 18px; height: 22px; line-height: 22px; text-align: center; box-sizing: border-box;
  border-right: 1px solid var(--card); }
bbe-fma .c.k { background: var(--blue-soft); color: var(--ink); }
bbe-fma .c.d { background: var(--orange-soft); color: var(--ink); }
bbe-fma .c.d.one { background: var(--orange); color: var(--card); font-weight: 700; }
bbe-fma .c.k.one { font-weight: 700; }
bbe-fma .c.z { opacity: .55; }
bbe-fma .pt { width: 8px; text-align: center; }
bbe-fma .cut { width: 0; height: 34px; border-left: 2px solid var(--ink); margin: 0 1px -6px; }
bbe-fma .idx { display: flex; font: 10px var(--mono); color: var(--faint); height: 14px; }
bbe-fma .idx span { width: 18px; text-align: center; flex: none; white-space: nowrap; }
bbe-fma .idx span.pt { width: 8px; } bbe-fma .idx span.cut0 { width: 4px; }
bbe-fma .idx sup { font-size: 8px; }
bbe-fma .legend { display: flex; gap: 22px; font-size: 12px; color: var(--muted); margin-top: 4px; }
bbe-fma .legend i { display: inline-block; width: 11px; height: 11px; vertical-align: -1px; margin-right: 5px; background: var(--blue-soft); }
bbe-fma .legend i.o { background: var(--orange); }
bbe-fma .chip { background: var(--orange); color: var(--card); padding: 0 4px; border-radius: 2px; }
bbe-fma .two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }
bbe-fma .lane { border-top: 2px solid var(--rule); padding-top: 6px; }
bbe-fma .lane.fu { border-top-color: var(--blue); }
bbe-fma .lane.un { border-top-color: var(--orange); }
bbe-fma .lane h4 { margin: 0 0 4px; font: 600 13px var(--sans); color: var(--ink); }
bbe-fma .lane h4 code { font-size: 12px; }
bbe-fma .row { display: flex; justify-content: space-between; gap: 8px; }
bbe-fma .res { margin-top: 4px; font-weight: 600; color: var(--ink); }
bbe-fma .or { color: var(--orange); } bbe-fma .bl { color: var(--blue); }
`;

class BbeFma extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-fma-css')) {
      const st = document.createElement('style'); st.id = 'bbe-fma-css'; st.textContent = CSS; document.head.append(st);
    }
    const M = model(), KEEP = 23;
    const cells = M.bits.map((v, i) => {
      const kept = i <= KEEP, cls = `c ${kept ? 'k' : 'd'}${v ? ' one' : ''}${!kept && !v ? ' z' : ''}`;
      return (i === 1 ? '<span class="pt">.</span>' : '') + (i === KEEP + 1 ? '<span class="cut"></span>' : '') + `<span class="${cls}">${v}</span>`;
    }).join('');
    // fraction-position labels under the cells: 2^-1 … only where useful
    const sup = (n) => `2<sup>${n < 0 ? '−' + -n : n}</sup>`;
    const show = new Set([0, 11, 24]);
    const idx = M.bits.map((_, i) => (i === 1 ? '<span class="pt"></span>' : '') + (i === KEEP + 1 ? '<span class="cut0"></span>' : '') +
      `<span>${show.has(i) ? sup(M.top - i) : ''}</span>`).join('');
    this.innerHTML = `
      <div class="in m">
        <span>a = b = 1 + ${sup(-12)}</span><span>= ${dec(M.a)}</span><span class="mut">${hex32(M.a)}</span>
        <span>c = −round(a·b) = −(1 + ${sup(-11)})</span><span>= ${dec(M.c)}</span><span class="mut">${hex32(M.c)}</span>
      </div>
      <div class="strip">
        <div class="lab">exact a·b = 1 + ${sup(-11)} + ${sup(-24)}</div>
        <div class="cells">${cells}</div>
        <div class="idx">${idx}</div>
        <div class="legend"><span><i></i>fp32 significand</span><span><i class="o"></i>rounded away by round(a·b)</span></div>
      </div>
      <div class="two">
        <div class="lane un"><h4>separate: <code>round(round(a·b) + c)</code></h4>
          <div class="row m"><span>round(a·b)</span><span>1 + ${sup(-11)}</span></div>
          <div class="row m"><span>round(a·b) + c</span><span>0</span></div>
          <div class="row m res"><span>result</span><span>${fmt32(M.unfused)} <span class="mut">${hex32(M.unfused)}</span></span></div>
        </div>
        <div class="lane fu"><h4>fused: <code>fma(a, b, c) = round(a·b + c)</code></h4>
          <div class="row m"><span>a·b + c</span><span><span class="chip">${sup(-24)}</span></span></div>
          <div class="row m"><span>round(a·b + c)</span><span>${sup(-24)}</span></div>
          <div class="row m res"><span>result</span><span>${sup(-24)} ≈ ${fmt32(M.fused)} <span class="mut">${hex32(M.fused)}</span></span></div>
        </div>
      </div>`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-fma', BbeFma);
