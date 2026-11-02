// Bundle index.html into one self-contained HTML file for hosting as a single
// object: site.css is inlined, and each src/*.js module becomes a data: URL
// behind an import map (intra-module './x.js' imports are rewritten to 'bbe/x.js').
// Usage: node scripts/bundle.mjs [out.html]
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';

const root = new URL('..', import.meta.url);
const read = (p) => readFileSync(new URL(p, root), 'utf8');
const out = process.argv[2] ?? 'bbe.html';

const imports = {};
for (const f of readdirSync(new URL('src/', root)).filter((f) => f.endsWith('.js'))) {
  const js = read(`src/${f}`).replace(/from '\.\/([\w-]+\.js)'/g, "from 'bbe/$1'");
  imports[`bbe/${f}`] = 'data:text/javascript;base64,' + Buffer.from(js).toString('base64');
}

let html = read('index.html')
  .replace('<link rel="stylesheet" href="./site.css">', () => `<style>\n${read('site.css')}</style>`)
  .replace(/<script type="module" src="\.\/src\/([\w-]+\.js)"><\/script>/g, "<script type=\"module\">import 'bbe/$1';</script>");
html = html.replace('</head>', () => `<script type="importmap">${JSON.stringify({ imports })}</script>\n</head>`);

if (/src="\.\/|href="\.\//.test(html)) throw new Error('unbundled relative reference left in output');
writeFileSync(out, html);
console.log(`${out}: ${html.length} bytes`);
