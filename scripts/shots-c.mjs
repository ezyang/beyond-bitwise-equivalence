// Group-C (fast math) widget check: loads dev/gallery-c.html in headless Chrome (light
// and dark), screenshots each widget in a few interaction states into
// /tmp/bbe-shots/c-*.png, reports console errors, fixed-height violations
// and geometric lint (overlapping text, text crossing box borders or the
// SVG edge, wires through text).  Self-contained: CDP over a pipe, after
// ../scaling-dsv3/scripts/cdp.mjs.
//
//   node scripts/shots-c.mjs [widget]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const OUT = '/tmp/bbe-shots';

async function chromePath() {
  if (process.env.CHROME) return process.env.CHROME;
  const cache = join(homedir(), 'Library/Caches/ms-playwright');
  const d = (await readdir(cache)).filter((x) => x.startsWith('chromium_headless_shell-')).sort().at(-1);
  for (const sub of await readdir(join(cache, d))) {
    const p = join(cache, d, sub, 'chrome-headless-shell');
    if (existsSync(p)) return p;
  }
  throw new Error('no chrome-headless-shell');
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const srv = createServer(async (req, res) => {
  try {
    const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const body = await readFile(join(root, p));
    res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' }); res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((r) => srv.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${srv.address().port}`;

const ch = spawn(await chromePath(), ['--headless', '--disable-gpu', '--hide-scrollbars', '--remote-debugging-pipe'],
  { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });
const pending = new Map(), listeners = [];
let id = 0, buf = Buffer.alloc(0);
ch.stdio[4].on('data', (d) => {
  buf = Buffer.concat([buf, d]);
  let i;
  while ((i = buf.indexOf(0)) >= 0) {
    const msg = JSON.parse(buf.subarray(0, i).toString()); buf = buf.subarray(i + 1);
    if (msg.id) { const p = pending.get(msg.id); pending.delete(msg.id); msg.error ? p.rej(new Error(msg.error.message)) : p.res(msg.result); }
    else listeners.forEach((f) => f(msg));
  }
});
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  pending.set(++id, { res, rej });
  ch.stdio[3].write(JSON.stringify({ id, method, params, sessionId }) + '\0');
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// geometric lint, evaluated in the page for one widget element
const LINT = (sel) => `(() => {
  const el = document.querySelector('${sel}'), svg = el.querySelector('svg'), out = [];
  if (!svg) return { h: el.getBoundingClientRect().height, issues: [] };
  const R = (e) => e.getBoundingClientRect();
  const ov = (a, b, m = 0.5) => a.left < b.right - m && b.left < a.right - m && a.top < b.bottom - m && b.top < a.bottom - m;
  const inside = (a, b) => a.left >= b.left - 0.5 && a.right <= b.right + 0.5 && a.top >= b.top - 0.5 && a.bottom <= b.bottom + 0.5;
  const vis = (e) => getComputedStyle(e).opacity !== '0' && getComputedStyle(e).visibility !== 'hidden' && !e.closest('[opacity="0"]');
  const texts = [...svg.querySelectorAll('text')].filter((t) => t.textContent.trim() && vis(t));
  const name = (t) => JSON.stringify(t.textContent.trim().slice(0, 28));
  const sr = R(svg);
  for (let i = 0; i < texts.length; i++) {
    const a = R(texts[i]);
    if (!inside(a, sr)) out.push('clipped ' + name(texts[i]));
    for (let j = i + 1; j < texts.length; j++) if (ov(a, R(texts[j]))) out.push('overlap ' + name(texts[i]) + ' / ' + name(texts[j]));
  }
  // boxes: rects that are drawn (not hit areas); text must be fully in or fully out
  const boxes = [...svg.querySelectorAll('rect')].filter((r) => vis(r) && !r.closest('defs') && !r.matches('.row,.hit,.nolint') && getComputedStyle(r).stroke !== 'none');
  for (const t of texts) { const a = R(t); for (const b of boxes) { const br = R(b); if (ov(a, br, 1) && !inside(a, br)) out.push('crosses box ' + name(t)); } }
  // wires: straight segments of paths with class wire/arr must not cross text
  const segs = [];
  for (const p of svg.querySelectorAll('path.wire, path.arr')) {
    if (!vis(p)) continue;
    const L = p.getTotalLength(), m = p.getScreenCTM();
    for (let s = 2; s < L - 2; s += 2) { const q = p.getPointAtLength(s); segs.push([m.a * q.x + m.c * q.y + m.e, m.b * q.x + m.d * q.y + m.f]); }
  }
  for (const t of texts) { const a = R(t); if (segs.some(([x, y]) => x > a.left + 1 && x < a.right - 1 && y > a.top + 1 && y < a.bottom - 1)) out.push('wire through ' + name(t)); }
  return { h: R(el).height, issues: [...new Set(out)] };
})()`;

const STATES = {
  'transc-ex2': [['default', ''], ['deg5', `q('bbe-transc-ex2 [data-d="5"]').click()`],
    ['x-0.2', `const i = q('bbe-transc-ex2 input'); i.value = -0.2; i.dispatchEvent(new Event('input'))`],
    ['x3.9', `const i = q('bbe-transc-ex2 input'); i.value = 3.99; i.dispatchEvent(new Event('input'))`],
    ['x-10', `const i = q('bbe-transc-ex2 input'); i.value = -10; i.dispatchEvent(new Event('input'))`]],
  'transc-err': [['deg5-fma', ''], ['deg5-mul', `q('bbe-transc-err [data-f="0"]').click()`],
    ['deg3-fma', `q('bbe-transc-err [data-d="3"]').click()`],
    ['deg3-hover', `q('bbe-transc-err [data-d="3"]').click(); const s = q('bbe-transc-err svg'), r = s.getBoundingClientRect(); s.querySelector('circle').dispatchEvent(new MouseEvent('mousemove', {bubbles: true, clientX: r.left + 600, clientY: r.top + 50}))`]],
  'transc-div': [['3-7', ''], ['1-3', `q('bbe-transc-div [data-i="0"]').click()`], ['7-10', `q('bbe-transc-div [data-i="2"]').click()`]],
};

await mkdir(OUT, { recursive: true });
let bad = 0;
for (const theme of ['light', 'dark']) {
  for (const [w, states] of Object.entries(STATES)) {
    if (process.argv[2] && w !== process.argv[2]) continue;
    for (const [sname, act] of states) {
      const { browserContextId } = await send('Target.createBrowserContext');
      const { targetId } = await send('Target.createTarget', { url: 'about:blank', browserContextId });
      const { sessionId: s } = await send('Target.attachToTarget', { targetId, flatten: true });
      const errs = [];
      const h = (m) => {
        if (m.sessionId !== s) return;
        if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) errs.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
        if (m.method === 'Runtime.exceptionThrown') errs.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
        if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errs.push(m.params.entry.text + ' ' + (m.params.entry.url ?? ''));
      };
      listeners.push(h);
      await send('Runtime.enable', {}, s); await send('Log.enable', {}, s); await send('Page.enable', {}, s);
      await send('Emulation.setDeviceMetricsOverride', { width: 900, height: 3200, deviceScaleFactor: 2, mobile: false }, s);
      await send('Page.navigate', { url: `${origin}/dev/gallery-c.html${theme === 'dark' ? '#dark' : ''}` }, s);
      await sleep(600);
      const ev = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true }, s)).result.value;
      const h0 = await ev(`document.querySelector('bbe-${w}').getBoundingClientRect().height`);
      if (act) { await ev(`(() => { const q = (x) => document.querySelector(x); ${act}; })()`); await sleep(700); }
      const lint = await ev(LINT(`bbe-${w}`));
      const r = await ev(`(() => { const b = document.querySelector('bbe-${w}').getBoundingClientRect(); return { x: b.left + scrollX, y: b.top + scrollY, w: b.width, h: b.height }; })()`);
      const svgH = await ev(`(() => { const e = document.querySelector('bbe-${w}'); return e.scrollHeight; })()`);
      const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x: r.x - 8, y: r.y - 8, width: Math.min(r.w, 760) + 16, height: r.h + 16, scale: 1 } }, s);
      const file = `${OUT}/c-${w}-${sname}-${theme}.png`;
      await writeFile(file, Buffer.from(shot.data, 'base64'));
      const issues = [...lint.issues];
      if (Math.abs(lint.h - h0) > 0.5) issues.push(`height changed ${h0} -> ${lint.h}`);
      if (svgH > lint.h + 0.5) issues.push(`content overflows element: ${svgH} > ${lint.h}`);
      issues.push(...errs.map((e) => 'console: ' + e));
      bad += issues.length;
      console.log(`${issues.length ? 'FAIL' : 'ok  '} ${w}/${sname}/${theme} h=${lint.h}${issues.map((i) => '\n     ' + i).join('')}`);
      listeners.splice(listeners.indexOf(h), 1);
      await send('Target.disposeBrowserContext', { browserContextId });
    }
  }
}
console.log(bad ? `${bad} issue(s)` : 'all clean');
ch.kill(); srv.close();
process.exit(bad ? 1 : 0);
