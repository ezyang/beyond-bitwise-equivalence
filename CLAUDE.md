# Claude instructions for beyond-bitwise-equivalence

A single-page tutorial (`index.html`) on why implementations of the same
computation produce different bits, with interactive illustrations. Plain
HTML/JS ES modules, zero dependencies, no build step. Prefer less code over
generality. Sibling project with the same taste: `../scaling-dsv3`.

## Diagrams: no captions inside the diagram

- Diagrams carry normal labels, axis titles and legends, and nothing else.
  No sentences of explanation inside a widget: no "note" paragraphs, no
  "this shows that…", no takeaway lines, no annotations that narrate what
  the reader should notice.
- If something needs explaining, explain it in the prose below the figure.
- Before writing an explanation at all, ask whether the diagram's design
  could make the point by itself (layout, colour, alignment, a better
  encoding). Needing an explanation is a sign the design should change.
- Labels are short noun phrases. A legend entry is a swatch and a few words.

## Content: no internal-only kernels or data

- The page is meant to be public. Don't cite or describe Meta-internal
  kernels, forks, model configs, shapes, measurements, pastes or diffs. Talk
  about well-known optimization strategies and public code (upstream
  FlashAttention 4, flex attention / Triton / Inductor, PyTorch SDPA,
  cuBLAS behaviour, published papers) instead, and say "FA4-style" when
  describing a technique rather than a specific build.
- Any claim of a bitwise match must rest on public code.

## Prose

- Unmannered prose: plain, declarative, no flourishes.
- AI-drafted prose carries the draft disclosure at the top of the page and
  expects a human edit pass. Prose is author-owned: coordinate before
  rewriting passages the author has edited.
- Open questions in the prose are marked `<span class="flag-why">[[FLAG:
  reason]]</span>`; the author resolves and removes them.

## Widgets

- One custom element per file, `src/<name>.js`, `<bbe-name>`. Inject CSS
  once, scoped under the element name.
- Colours only via the CSS variables in `site.css`, which handles night mode.
  Blue = exact / kept, orange = rounded away / differs, green = agrees,
  amber = current / highlight, purple = compiler decision.
- Fixed height; interaction never changes layout. Width ≤ 940px (720 preferred).
- Floating-point shown in a widget is exact emulation (`src/fp.js`:
  Math.fround for single ops, BigInt fma, bit-level bf16), and
  `scripts/check-fp.mjs` / `scripts/check-transc.mjs` assert the numbers
  the widgets display.

## Testing

- `node scripts/check-page.mjs`: loads index.html in headless Chrome in both
  themes and reports console errors, widgets that never mount, and heights.
- `node scripts/shots-{a,b,c}.mjs [widget]`: per-widget screenshots,
  layout and console checks on the gallery pages in `dev/`.
- `node scripts/check-fp.mjs`, `node scripts/check-transc.mjs`: arithmetic.

## Git

- Commit only when asked.
