// see bottom for usage
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
// Whole-page check of index.html: console errors, widgets that never
// upgraded, and widget heights, in both themes; full-page screenshots.
//   node scripts/check-page.mjs
const OUTP = '/tmp/bbe-shots';
let bad = 0;
for (const theme of ['light', 'dark']) {
  const { browserContextId } = await send('Target.createBrowserContext');
  const { targetId } = await send('Target.createTarget', { url: 'about:blank', browserContextId });
  const { sessionId: s } = await send('Target.attachToTarget', { targetId, flatten: true });
  const errs = [];
  listeners.push((m) => {
    if (m.sessionId !== s) return;
    if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) errs.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
    if (m.method === 'Runtime.exceptionThrown') errs.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errs.push(m.params.entry.text + ' ' + (m.params.entry.url ?? ''));
  });
  await send('Runtime.enable', {}, s); await send('Log.enable', {}, s); await send('Page.enable', {}, s);
  await send('Emulation.setDeviceMetricsOverride', { width: 1366, height: 900, deviceScaleFactor: 1, mobile: false }, s);
  await send('Runtime.evaluate', { expression: '0' }, s);
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `localStorage.setItem('bbe-theme', '${theme}')` }, s);
  await send('Page.navigate', { url: `${origin}/${process.env.PAGE ?? 'index.html'}` }, s);
  await sleep(1500);
  const ev = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true }, s)).result.value;
  const ws = await ev(`[...document.querySelectorAll('[data-widget]')].map((e) => ({ tag: e.localName, def: !!customElements.get(e.localName), h: Math.round(e.getBoundingClientRect().height), w: Math.round(e.scrollWidth) }))`);
  for (const w of ws) { const ok = w.def && w.h > 0; if (!ok) bad++; console.log(`${ok ? 'ok  ' : 'BAD '} ${theme} ${w.tag} defined=${w.def} h=${w.h} w=${w.w}`); }
  for (const e of errs) { bad++; console.log(`BAD  ${theme} console: ${e}`); }
  const H = await ev('document.documentElement.scrollHeight');
  await send('Emulation.setDeviceMetricsOverride', { width: 1366, height: H, deviceScaleFactor: 1, mobile: false }, s);
  await sleep(300);
  const shot = await send('Page.captureScreenshot', { format: 'png' }, s);
  await writeFile(`${OUTP}/page-${theme}.png`, Buffer.from(shot.data, 'base64'));
  console.log(`page height ${H}px -> ${OUTP}/page-${theme}.png`);
}
console.log(bad ? `${bad} problem(s)` : 'all clean');
ch.kill(); srv.close(); process.exit(bad ? 1 : 0);
