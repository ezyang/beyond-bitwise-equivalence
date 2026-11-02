#!/usr/bin/env python3
"""Lazy vs every-tile rescaling in online softmax, on a GPU, for the slide
"Rescale policy" in dev/slides.html.

Run on any machine with a CUDA GPU and torch + triton (the triton that ships
with torch is enough). From the repo root:

    python3 scripts/measure-rescale.py

Prints a table; writes nothing. One Triton program handles BM query rows;
each row runs the online-softmax loop over NT key tiles of BN keys with the
kernel recipe:
  scores s     fp32, already in log2 units (scale * log2 e folded in)
  reference m  per row; lazy: move to the tile max only when it exceeds m by
               more than THR = 8; every tile: m = max(m, tile max)
  alpha        ex2.approx.ftz.f32(m_old - m_new)    (first tile: ex2(-inf) = 0)
  p            ex2.approx.ftz.f32(sub.rn.f32(s, m)) (inline PTX, so the
               compiler cannot substitute or fuse anything)
  l            l * alpha + tl.sum(p)                 (fp32)
  O            tl.dot(bf16(p), v, O * alpha)         (bf16 MMA, fp32 accumulator)
  out          bf16(div.rn.f32(O, l))
Both policies run the identical code path; only m differs. Three variants:
  (a) m is any fp32 value (the tile max itself);
  (b) m is an integer in log2 units (ceil of the tile max), so alpha is
      always a power of two and every rescale is exact;
  (c) as (b), with p = ex2_neg(s - m) = 2^ceil(x) * ex2.approx(x - ceil(x)),
      an exp2 that only evaluates ex2.approx on (-1, 0]. Not a kernel recipe:
      a control for the ex2 finding below.
Score sets:
  qk       s = (q . k) * D^-0.5 * log2 e, q, k ~ N(0, 1) in bf16, head dim 128,
           fp32 product (max grows slowly; lazy almost never moves m)
  ramp     qk + 3 * tile index (lazy moves m every few tiles)
  *-grid   the same scores rounded to multiples of 2^-16, so that s - m is
           exact in fp32 for every reference used here
Also checks ex2.approx.ftz.f32 directly.

Result, NVIDIA B200 sm_100, driver 615.71.09, CUDA 13.2, torch 2.12.0+cu132,
triton 3.7.0 (65536 rows; % of rows whose bf16 out row differs anywhere):
              (a) any m   (b) integer m   (c) integer m, ex2_neg
  qk           87.06%       0.206%          0.198%
  ramp         99.93%       0.113%          0
  qk-grid      87.06%       0.0046%         0
  ramp-grid    99.93%       0.116%          0
ex2.approx.ftz.f32 is exact on integers and equals 2^trunc(x) * ex2(x - trunc(x))
for every x tested, so ex2(x - k) == ex2(x) * 2^-k exactly unless the sign of
the argument changes (then only ~6% agree). Lazy weights above 1 have positive
arguments where every-tile's are negative: that is the residual in (b) that
(c) removes. The rest of (b) is the rounding of s - m, which the grid removes.
"""
import math, subprocess, torch, triton, triton.language as tl

THR = 8.0
NT, BN, BM, D, HD = 8, 128, 64, 64, 128
ROWS = 1 << 16


@triton.jit
def ex2(x):
    return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def ex2_neg(x):
    # ex2 evaluated only on (-1, 0]: 2^ceil(x) * ex2(x - ceil(x)); both steps exact
    c = tl.math.ceil(x)
    return ex2(x - c) * ex2(c)


@triton.jit
def sub_rn(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def attn_row(S, V, Out, Moves, NT: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
             LAZY: tl.constexpr, INTREF: tl.constexpr, NEGEX2: tl.constexpr, THR: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.arange(0, D)
    m = tl.full([BM], float("-inf"), tl.float32)
    l = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    moves = tl.zeros([BM], tl.int32)
    for t in range(NT):
        cols = t * BN + tl.arange(0, BN)
        s = tl.load(S + rows[:, None] * (NT * BN) + cols[None, :])
        mt = tl.max(s, 1)
        if INTREF:
            mt = tl.math.ceil(mt)
        if LAZY:
            m_new = tl.where(mt - m > THR, mt, m)
        else:
            m_new = tl.maximum(m, mt)
        moves += ((m_new != m) & (t > 0)).to(tl.int32)
        alpha = ex2(m - m_new)
        if NEGEX2:
            p = ex2_neg(sub_rn(s, m_new[:, None]))
        else:
            p = ex2(sub_rn(s, m_new[:, None]))
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(V + cols[:, None] * D + dd[None, :])
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m = m_new
    out = tl.math.div_rn(acc, l[:, None])
    tl.store(Out + rows[:, None] * D + dd[None, :], out.to(tl.bfloat16))
    tl.store(Moves + rows, moves)


@triton.jit
def ex2_kernel(X, Y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, ex2(tl.load(X + i, mask=i < n)), mask=i < n)


def ex2_dev(x):
    y = torch.empty_like(x)
    ex2_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    return y


def run(S, V, lazy, intref, negex2):
    out = torch.empty(S.shape[0], D, device="cuda", dtype=torch.bfloat16)
    moves = torch.empty(S.shape[0], device="cuda", dtype=torch.int32)
    attn_row[(S.shape[0] // BM,)](S, V, out, moves, NT=NT, BN=BN, BM=BM, D=D, LAZY=lazy, INTREF=intref, NEGEX2=negex2, THR=THR)
    return out, moves


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(0)
    N = NT * BN
    q = torch.randn(ROWS, HD, device="cuda", generator=g).bfloat16().float()
    k = torch.randn(N, HD, device="cuda", generator=g).bfloat16().float()
    V = torch.randn(N, D, device="cuda", generator=g).bfloat16().contiguous()
    qk = (q @ k.T) * (HD ** -0.5 * math.log2(math.e))
    ramp = qk + 3.0 * (torch.arange(N, device="cuda") // BN).float()
    grid = lambda s: torch.round(s * 65536) / 65536
    sets = {"qk": qk, "ramp": ramp, "qk-grid": grid(qk), "ramp-grid": grid(ramp)}

    dev = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"{torch.cuda.get_device_name()} sm_{dev.major}{dev.minor}, driver {smi}, CUDA {torch.version.cuda}, torch {torch.__version__}, triton {triton.__version__}")
    print(f"{ROWS} rows, {NT} tiles x {BN} keys, head dim {D}, threshold {THR}")

    # ex2.approx.ftz.f32 on its own
    ints = torch.arange(-40, 41, device="cuda", dtype=torch.float32)
    print("ex2 of integers -40..40 exact:", bool((ex2_dev(ints) == torch.exp2(ints.double()).float()).all()))
    kk = torch.randint(1, 9, (1 << 20,), device="cuda", generator=g).float()
    for lo in (-8, 0):                                       # x on a 2^-16 grid, so x - k is exact
        x = torch.round((torch.rand(1 << 20, device="cuda", generator=g) * 8 + lo) * 65536) / 65536
        same = ex2_dev(x - kk) * torch.exp2(kk) == ex2_dev(x)
        flip = (x - kk < 0) != (x < 0)
        print(f"ex2(x - k) * 2^k == ex2(x), x in [{lo}, {lo + 8}), k in 1..8: {same.float().mean().item():.4f}"
              f" (sign of argument flips: {same[flip].float().mean().item() if flip.any() else float('nan'):.4f},"
              f" same sign: {same[~flip].float().mean().item():.4f})")
    x = torch.round((torch.rand(1 << 20, device="cuda", generator=g) * 32 - 16) * 65536) / 65536
    for nm, f in (("trunc", torch.trunc), ("floor", torch.floor)):
        print(f"ex2(x) == 2^{nm}(x) * ex2(x - {nm}(x)), x in [-16, 16):",
              f"{(ex2_dev(x - f(x)) * torch.exp2(f(x)) == ex2_dev(x)).float().mean().item():.4f}")

    print(f"{'scores':10} {'reference':14} {'rows differ':>12} {'elems differ':>13} {'lazy moves/row':>15}")
    for name, s in sets.items():
        s = s.contiguous()
        for intref, negex2, ref in ((False, False, "(a) any fp32"), (True, False, "(b) integer"), (True, True, "(c) b+ex2_neg")):
            (ol, ml), (oe, _) = run(s, V, True, intref, negex2), run(s, V, False, intref, negex2)
            diff = ol.view(torch.int16) != oe.view(torch.int16)
            print(f"{name:10} {ref:14} {diff.any(1).float().mean().item():12.4%} {diff.float().mean().item():13.4%} {ml.float().mean().item():15.3f}")


if __name__ == "__main__":
    main()
