// <bbe-transc-ex2>: how a GPU computes 2^x in software.  x = floor(x) + f;
// the integer part becomes the exponent field (exact), a polynomial supplies
// the significand of 2^f.  Every bit pattern comes from ex2Emu() in
// src/transc.js, a bit-exact emulation of FA4's ex2_emulation_2.
import { ex2Emu, ref2, ulpsApart } from './transc.js';
import { bits32, fmt32 } from './fp.js';

const CSS = `
bbe-transc-ex2 { display: block; height: 330px; max-width: 720px; margin: 18px 0 26px; font: 13px/1.4 var(--sans); color: var(--text); }
bbe-transc-ex2 .ctl { display: flex; align-items: center; gap: 14px; height: 30px; }
bbe-transc-ex2 .ctl input { width: 220px; accent-color: var(--blue); }
bbe-transc-ex2 .xv { font: 600 14px var(--mono); font-variant-numeric: tabular-nums; color: var(--ink); width: 92px; }
bbe-transc-ex2 .seg { display: inline-flex; margin-left: auto; }
bbe-transc-ex2 .seg button { font: 12px var(--sans); padding: 2px 9px; border: 1px solid var(--rule-strong);
  background: var(--card); color: var(--muted); cursor: pointer; }
bbe-transc-ex2 .seg button + button { border-left: none; }
bbe-transc-ex2 .seg button:first-child { border-radius: 4px 0 0 4px; }
bbe-transc-ex2 .seg button:last-child { border-radius: 0 4px 4px 0; }
bbe-transc-ex2 .seg button.on { background: var(--panel); color: var(--ink); font-weight: 600; cursor: default; }
bbe-transc-ex2 .grid { display: grid; grid-template-columns: 184px 426px 90px; column-gap: 8px; row-gap: 0; align-items: center; margin-top: 8px; }
bbe-transc-ex2 .lab { font-size: 12.5px; color: var(--ink); line-height: 1.2; padding: 5px 0; }
bbe-transc-ex2 .lab small { display: block; font: 11px var(--mono); color: var(--faint); margin-top: 1px; }
bbe-transc-ex2 .val { font: 12px var(--mono); font-variant-numeric: tabular-nums; color: var(--text); white-space: nowrap; }
bbe-transc-ex2 .bits { display: flex; font: 11px var(--mono); }
bbe-transc-ex2 .bits span { width: 13px; height: 20px; line-height: 20px; text-align: center; background: var(--panel);
  color: var(--muted); box-sizing: border-box; border-right: 1px solid var(--card); }
bbe-transc-ex2 .bits span.g { margin-left: 5px; }
bbe-transc-ex2 .bits span.z { color: var(--faint); opacity: .6; }
bbe-transc-ex2 .bits span.int { background: var(--blue-soft); color: var(--ink); font-weight: 600; }
bbe-transc-ex2 .bits span.poly { background: var(--amber-soft); color: var(--ink); }
bbe-transc-ex2 .bits span.diff { background: var(--orange); color: var(--card); font-weight: 700; }
bbe-transc-ex2 .hdr { display: flex; font: 10.5px var(--sans); color: var(--faint); }
bbe-transc-ex2 .hdr span { text-align: center; box-sizing: border-box; border-bottom: 1px solid var(--rule-strong); margin-bottom: 2px; white-space: nowrap; overflow: hidden; }
bbe-transc-ex2 .hdr span + span { margin-left: 5px; }
bbe-transc-ex2 .fv { display: flex; gap: 28px; font: 12px var(--mono); font-variant-numeric: tabular-nums; color: var(--text); padding-left: 2px; }
bbe-transc-ex2 .sep { grid-column: 1 / 4; border-top: 1px solid var(--rule-strong); margin: 4px 0 1px; }
bbe-transc-ex2 .lab small.o, bbe-transc-ex2 .o { color: var(--orange); font-weight: 600; }
bbe-transc-ex2 .lab small.k { color: var(--green); font-weight: 600; }
bbe-transc-ex2 .key { display: flex; gap: 18px; font-size: 12px; color: var(--muted); margin: 10px 0 0 192px; }
bbe-transc-ex2 .key i { display: inline-block; width: 12px; height: 12px; vertical-align: -2px; margin-right: 5px; }
bbe-transc-ex2 .key i.int { background: var(--blue-soft); } bbe-transc-ex2 .key i.poly { background: var(--amber-soft); }
bbe-transc-ex2 .key i.diff { background: var(--orange); }
`;

// 32 cells; cls(i) gives the class of bit i (0 = sign, 1..8 exponent, 9..31 significand)
function bitRow(u, cls) {
  const s = u.toString(2).padStart(32, '0');
  return `<div class="bits">${[...s].map((b, i) => `<span class="${cls(i, b)}${i === 1 || i === 9 ? ' g' : ''}">${b}</span>`).join('')}</div>`;
}
const sup = (s) => `<sup>${s}</sup>`;
const minus = (s) => String(s).replace('-', '−');

class BbeTranscEx2 extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    if (!document.getElementById('bbe-transc-ex2-css')) {
      const st = document.createElement('style'); st.id = 'bbe-transc-ex2-css'; st.textContent = CSS; document.head.append(st);
    }
    this.x = -2.3; this.deg = 3;
    this.innerHTML = `<div class="ctl">
        <label for="bbe-ex2-x">x</label><input id="bbe-ex2-x" type="range" min="-10" max="4" step="0.01" value="${this.x}">
        <span class="xv"></span>
        <span class="seg">${[3, 5].map((d) => `<button data-d="${d}">degree ${d}</button>`).join('')}</span>
      </div><div class="body"></div>`;
    this.querySelector('input').addEventListener('input', (e) => { this.x = +e.target.value; this.render(); });
    this.addEventListener('click', (e) => {
      const b = e.target.closest('[data-d]');
      if (b && +b.dataset.d !== this.deg) { this.deg = +b.dataset.d; this.render(); }
    });
    this.render();
  }
  render() {
    const S = ex2Emu(this.x, this.deg, true), cr = ref2(S.x), crBits = bits32(cr), u = ulpsApart(S.out, cr);
    const n = S.n, fExact = S.frac === S.xc - n;
    this.querySelector('.xv').textContent = 'x = ' + minus(fmt32(S.x));
    this.querySelectorAll('[data-d]').forEach((b) => b.classList.toggle('on', +b.dataset.d === this.deg));
    const outBits = S.outBits;
    const hdr = `<div class="hdr"><span style="width:13px">±</span><span style="width:104px">exponent</span><span style="width:299px">significand (23 bits)</span></div>`;
    this.querySelector('.body').innerHTML = `<div class="grid">
      <div></div>${hdr}<div></div>
      <div class="lab">① x + 1.5·2²³, rounded down<small>add.rm.f32</small></div>
      ${bitRow(bits32(S.xr), (i) => (i >= 23 ? 'int' : ''))}
      <div class="val">1.5·2²³ ${n < 0 ? '−' : '+'} ${Math.abs(n)}</div>
      <div class="lab">② ⌊x⌋ and f = x − ⌊x⌋<small>sub.f32, sub.f32</small></div>
      <div class="fv"><span>⌊x⌋ = ${minus(n)}</span><span>f ${fExact ? '=' : '<span class="o">≈</span>'} ${fmt32(S.frac)}</span></div>
      <div></div>
      <div class="lab">③ p = poly(f) ≈ 2${sup('f')}, in [1, 2)<small>degree ${this.deg} Horner, fma</small></div>
      ${bitRow(bits32(S.poly), (i) => (i >= 9 ? 'poly' : ''))}
      <div class="val">${fmt32(S.poly)}</div>
      <div class="lab">④ bits of ① shifted left 23<small>shl.b32</small></div>
      ${bitRow(S.shifted, (i, b) => (i < 9 ? 'int' : 'z'))}
      <div class="val">${minus(n)}·2²³</div>
      <div class="lab">⑤ ③ + ④ as integers = 2${sup('⌊x⌋')} · p<small>add.s32</small></div>
      ${bitRow(outBits, (i) => (i >= 1 && i <= 8 ? 'int' : i >= 9 ? 'poly' : ''))}
      <div class="val" style="font-weight:600;color:var(--ink)">${fmt32(S.out)}</div>
      <div class="sep"></div>
      <div class="lab">correctly rounded 2${sup('x')}<small class="${u ? 'o' : 'k'}">${u ? `${u > 0 ? '+' : '−'}${Math.abs(u)} ulp${Math.abs(u) === 1 ? '' : 's'}` : 'same bits'}</small></div>
      ${bitRow(crBits, (i) => (((outBits ^ crBits) >>> (31 - i)) & 1 ? 'diff' : ''))}
      <div class="val">${fmt32(cr)}</div>
    </div>
    <div class="key"><span><i class="int"></i>from ⌊x⌋</span><span><i class="poly"></i>from the polynomial</span><span><i class="diff"></i>differs from correctly rounded</span></div>`;
  }
}
if (typeof customElements !== 'undefined') customElements.define('bbe-transc-ex2', BbeTranscEx2);
