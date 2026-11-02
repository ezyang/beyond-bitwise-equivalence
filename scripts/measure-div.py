#!/usr/bin/env python3
"""Measure fp32 division, log2 and the final lse on an NVIDIA GPU for <bbe-lse-heatmap>.

Run on any machine with a CUDA GPU and torch + triton installed (the triton
that ships with torch is enough). From the repo root:

    python3 scripts/measure-div.py                    # writes data/div-measured.json
    python3 scripts/measure-div.py /tmp/out.json      # or anywhere else

No other files from this repo are needed. Each op runs as one PTX
instruction sequence through Triton's inline asm (tl.inline_asm_elementwise),
so neither the compiler nor ptxas can substitute another instruction. Values
pass through the asm as 32-bit integers and are moved into .f32 registers
inside it, which avoids any float-constraint differences between Triton
versions.

Output, JSON:
  device   torch.cuda.get_device_name()
  versions driver, CUDA, torch, triton, compute capability
  n        grid size; column i is O = fp32(2^((i+0.5)/n)), row j is
           l = fp32(2^((j+0.5)/n)), both in [1, 2)
  div      {name: [n*n ints]}, row-major [j*n + i]: signed ulps of the op's
           result from the correctly rounded fp32 O / l
  m        sweep size; l_k = fp32(2^(16 (k+0.5)/m)), l in (1, 2^16)
  log2     {name: [m ints]}: signed ulps from the correctly rounded fp32 log2(l)
  lse      {"n": n, "M": [n], "l": [n], kernel: [n*n ints]}: row j is the running max in log2
           units, M = fp32(-16 + 64 (j + frac(0.6180339887 (j+1))) / n) (jittered so
           every M has a full significand), column i is l = fp32(2^(15 (i+0.5)/n)),
           l in (1, 2^15); signed ulps of the fp32 natural-log lse each kernel
           writes, from the correctly rounded fp32 (M + log2(l)) * ln 2

lse, as each kernel computes it (bf16 inputs, so FA4's max_offset is 0):
  FA4  flash_attn/cute/flash_fwd_sm100.py, correction epilogue:
         lse = (row_max * softmax_scale_log2 + (log2(row_sum) - max_offset)) * LN2
       log2 = cute.math.log2(fastmath=True), taken here to be lg2.approx.ftz.f32;
       LN2 = fp32(ln 2). Modelled as RN(RN(M + lg2.approx(l)) * LN2), with M the
       fp32 product row_max * scale_log2 (if the compiler fuses that product
       into an fma with the add, M is not rounded first; not modelled).
  flex torch/_inductor/kernel/flex/templates/flex_attention.py.jinja:
         lse = m_i + tl.math.log2(l_i)          (log2 units, stored as fp32)
       then torch/nn/attention/flex_attention.py (_finalize_outputs,
       stats_are_log2) returns lse * math.log(2), an fp32 multiply.

The lse grid definition must match src/lse-heatmap.js. Signed ulps are the
difference of the two values' positions on the ordered fp32 line, so +1 means
the next fp32 value up.

References are computed on the host in float64 and rounded once to fp32. For
division that is exactly the correctly rounded quotient (53 >= 2*24 + 2). For
log2 it is correctly rounded unless log2(l) lies within ~2^-50 (relative) of
a midpoint between two fp32 values; the script counts such near-ties and
prints the count (expected 0).

FA4 (flash_attn/cute/softmax.py, Softmax.finalize) uses cute.arch.rcp_approx,
which CuTe DSL emits as nvvm.rcp.approx.ftz.f, then multiplies O by it; and
cute.math.log2(..., fastmath=True) for the LSE. Flex attention (Triton) lowers
fp32 `/` to div.full.f32. To add a variant, add an entry to DIV or LOG2 below:
an inline-PTX body, or a Python function of the input tensor.
"""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl

N, M = 128, 512

# PTX bodies: inputs in .f32 registers a, b (b unused for one-input ops); write q.
DIV = {
    "div.rn.f32": "div.rn.f32 q, a, b;",
    "div.full.f32": "div.full.f32 q, a, b;",
    "rcp.approx.ftz.f32 then mul.rn.f32": "rcp.approx.ftz.f32 r, b; mul.rn.f32 q, a, r;",
    "triton x / y": lambda x, y: _tl(x, y, 0),
}
LOG2 = {
    "lg2.approx.ftz.f32": "lg2.approx.ftz.f32 q, a;",
    "lg2.approx.f32": "lg2.approx.f32 q, a;",
    "log2f": lambda x, y: torch.log2(x.view(torch.float32)).view(torch.int32),  # CUDA libm log2f
    "triton tl.log2": lambda x, y: _tl(x, y, 1),
}
# a = M (log2 units), b = l; result: the fp32 natural-log lse
LSE = {
    "FA4-style": "lg2.approx.ftz.f32 r, b; add.rn.f32 q, a, r; mul.rn.f32 q, q, 0f3F317218;",
    "flex": lambda m, l: (_tl(m, l, 2).view(torch.float32) * math.log(2)).view(torch.int32),
}


@triton.jit
def _asm_kernel(x_ptr, y_ptr, o_ptr, n, ASM: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask)
    y = tl.load(y_ptr + i, mask=mask)
    o = tl.inline_asm_elementwise(ASM, "=r,r,r", [x, y], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(o_ptr + i, o, mask=mask)


@triton.jit
def _tl_kernel(x_ptr, y_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask).to(tl.float32, bitcast=True)
    y = tl.load(y_ptr + i, mask=mask).to(tl.float32, bitcast=True)
    if OP == 0:
        o = x / y
    elif OP == 1:
        o = tl.log2(x)
    else:
        o = x + tl.log2(y)
    tl.store(o_ptr + i, o.to(tl.int32, bitcast=True), mask=mask)


def _launch(kernel, x, y, arg):
    o = torch.empty_like(x)
    kernel[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), arg, BLOCK=1024)
    return o


def _tl(x, y, op):
    return _launch(_tl_kernel, x, y, op)


def run(op, x, y):
    """x, y: fp32 numpy arrays. Returns fp32 numpy array."""
    xt = torch.from_numpy(x.view(np.int32)).cuda()
    yt = torch.from_numpy(y.view(np.int32)).cuda()
    if callable(op):
        o = op(xt, yt)
    else:
        asm = "{ .reg .f32 a, b, r, q; mov.b32 a, $1; mov.b32 b, $2; " + op + " mov.b32 $0, q; }"
        o = _launch(_asm_kernel, xt, yt, asm)
    torch.cuda.synchronize()
    return o.cpu().numpy().view(np.float32)


def ordered(v):
    """fp32 -> int64 position on the ordered fp32 line (-0 and +0 both 0)."""
    b = v.view(np.int32).astype(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFF), b)


def ulps(got, ref):
    return (ordered(got) - ordered(ref)).tolist()


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        pathlib.Path(__file__).resolve().parent.parent / "data" / "div-measured.json"

    g = (2.0 ** ((np.arange(N) + 0.5) / N)).astype(np.float32)
    O = np.tile(g, N)            # column i
    L = np.repeat(g, N)          # row j
    ref = (O.astype(np.float64) / L.astype(np.float64)).astype(np.float32)
    div = {k: ulps(run(op, O, L), ref) for k, op in DIV.items()}

    l = (2.0 ** (16 * (np.arange(M) + 0.5) / M)).astype(np.float32)
    t = np.log2(l.astype(np.float64))
    lref = t.astype(np.float32)
    nb = np.where(t > lref, np.nextafter(lref, np.float32(np.inf)), np.nextafter(lref, np.float32(-np.inf)))
    near = int(np.sum(np.abs(t - (lref.astype(np.float64) + nb) / 2) <= 2.0 ** -50 * np.abs(t)))
    log2 = {k: ulps(run(op, l, l), lref) for k, op in LOG2.items()}

    Mg = (-16 + 64 * (np.arange(N) + (0.6180339887 * (np.arange(N) + 1)) % 1) / N).astype(np.float32)
    lg = (2.0 ** (15 * (np.arange(N) + 0.5) / N)).astype(np.float32)
    Mc, lc = np.repeat(Mg, N), np.tile(lg, N)
    e = (Mc.astype(np.float64) + np.log2(lc.astype(np.float64))) * math.log(2)
    eref = e.astype(np.float32)
    nb = np.where(e > eref, np.nextafter(eref, np.float32(np.inf)), np.nextafter(eref, np.float32(-np.inf)))
    near += int(np.sum(np.abs(e - (eref.astype(np.float64) + nb) / 2) <= 2.0 ** -48 * np.abs(e)))
    lse = {k: ulps(run(op, Mc, lc), eref) for k, op in LSE.items()}

    dev = torch.cuda.get_device_name()
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.split()[0]
    except (OSError, IndexError):
        driver = "unknown"
    versions = {"driver": driver, "cuda": torch.version.cuda, "torch": torch.__version__, "triton": triton.__version__,
                "sm": "%d%d" % torch.cuda.get_device_capability()}
    print(f"device: {dev}  {versions}")
    for name, d in [("div", div), ("log2", log2), ("lse", lse)]:
        for k, u in d.items():
            a = np.array(u)
            vals, cnt = np.unique(np.clip(a, -2, 2), return_counts=True)
            hist = "  ".join(f"{v:+d}:{c / a.size:.1%}" for v, c in zip(vals, cnt))
            print(f"  {name:4s} {k:36s} max|ulps| {np.abs(a).max():6d}   {hist}")
    rcp_only = run("rcp.approx.ftz.f32 q, b;", g, g)
    print(f"  rcp.approx.ftz(l) == RN(1/l): {np.mean(rcp_only == (1.0 / g.astype(np.float64)).astype(np.float32)):.1%} of l")
    print(f"  div.full == rcp.approx then mul: {np.mean(np.array(div['div.full.f32']) == np.array(div['rcp.approx.ftz.f32 then mul.rn.f32'])):.1%} of cells")
    print(f"  triton x / y == div.full: {div['triton x / y'] == div['div.full.f32']}")
    fa, fl = np.array(lse["FA4-style"]), np.array(lse["flex"])
    big = np.abs(eref) >= 1
    print(f"  lse bits differ, FA4-style vs flex: {np.mean(fa != fl):.1%} of cells; {np.mean((fa != fl)[big]):.1%} where |lse| >= 1")
    print(f"  max |ulps| where |lse| >= 1: FA4-style {np.abs(fa[big]).max()}, flex {np.abs(fl[big]).max()}")
    print(f"  reference near-ties (log2, lse): {near}")
    if any(div["div.rn.f32"]):
        print("  WARNING: div.rn.f32 differs from the host reference")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": dev, "versions": versions, "n": N, "div": div, "m": M, "log2": log2, "lse": {"n": N, "M": Mg.tolist(), "l": lg.tolist(), **lse}}, separators=(",", ":")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
