"""Validate and benchmark AITER's CK INT8 GEMM (`aiter.gemm_a8w8`) on gfx90a.

Requires `configs/enable_gfx90a_asm_paths.py` to have been applied; without it
`module_gemm_a8w8` does not build on gfx90a at all. See
`docs/20-int8-gemm-gfx90a.md` for why.

Correctness is checked against an EXACT reference, not a tolerance guess.
INT8 x INT8 -> INT32 accumulation is exactly representable in float64 and MI210
has full-rate FP64, so `a.double() @ b.double().T` is the exact integer answer.
The kernel output is compared against the correctly-rounded cast of that value
into the output dtype, so the bar is "bit-exact, or within one output-dtype
ULP" -- and a kernel that is fast but wrong is reported WRONG, never timed away.

Two things worth knowing before reading the numbers:

  * On CDNA2, INT8 and BF16 share one peak: 181 TOPS and 181 TFLOPS. Unlike
    CDNA3, INT8 buys no arithmetic throughput on this hardware, only halved
    operand bytes. Expect a wash on compute-bound shapes and a large win on
    memory-bound ones.
  * There are no gfx90a rows in any AITER tuned CSV, so `not found tuned config
    ... will use default config!` is expected. These are UNTUNED numbers.
"""
import time

import torch
import aiter

torch.set_default_device("cuda:0")
torch.manual_seed(0)

# MI210 published peaks: 181 TFLOPS BF16/FP16, 181 TOPS INT8, 1638 GB/s HBM2e.
PEAK_OPS = 181e12
EPS = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11}

SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (2048, 4096, 4096),
    (1024, 8192, 8192),
    (16, 8192, 8192),      # decode-shaped: memory bound, where INT8 actually wins
]


def timed(fn, it=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def check(out, exact, odt):
    """Return (verdict, max_ulp) against the correctly-rounded exact answer."""
    want = exact.to(odt)
    if torch.equal(out, want):
        return "EXACT", 0.0
    diff = (out.double() - want.double()).abs()
    max_ulp = (diff / want.double().abs().clamp(min=1.0) / EPS[odt]).max().item()
    return ("PASS" if max_ulp <= 1.0 else "WRONG"), max_ulp


def main():
    print(f"{'M':>6} {'N':>6} {'K':>6} {'out':>8} {'int8 us':>10} {'TOP/s':>8} "
          f"{'%peak':>6} {'bf16 us':>10} {'TFLOP/s':>8} {'speedup':>8} "
          f"{'maxULP':>7}  verdict")

    for (M, N, K) in SHAPES:
        a = torch.randint(-127, 128, (M, K), dtype=torch.int8)
        b = torch.randint(-127, 128, (N, K), dtype=torch.int8)
        xs = torch.ones(M, 1, dtype=torch.float32)
        ws = torch.ones(1, N, dtype=torch.float32)
        exact = a.double() @ b.double().T
        flops = 2.0 * M * N * K

        ab, bb = a.to(torch.bfloat16), b.to(torch.bfloat16)
        t_bf16 = timed(lambda: ab @ bb.T)

        for odt in (torch.bfloat16, torch.float16):
            name = str(odt).split(".")[-1]
            try:
                def fn(odt=odt):
                    return aiter.gemm_a8w8(a, b, xs, ws, dtype=odt)
                out = fn()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{M:6} {N:6} {K:6} {name:>8}   FAILED: "
                      f"{str(e).splitlines()[0][:70]}")
                continue

            verdict, max_ulp = check(out, exact, odt)
            t_i8 = timed(fn)
            print(f"{M:6} {N:6} {K:6} {name:>8} {t_i8*1e6:10.1f} "
                  f"{flops/t_i8/1e12:8.1f} {flops/t_i8/PEAK_OPS*100:5.0f}% "
                  f"{t_bf16*1e6:10.1f} {flops/t_bf16/1e12:8.1f} "
                  f"{t_bf16/t_i8:7.2f}x {max_ulp:7.2f}  {verdict}")

    # With unit scales and K >= 4096, exact products reach ~1e8 and overflow
    # fp16's 65504 range -- inf is the correctly-rounded answer, but it makes
    # the ULP column meaningless. Re-check fp16 with scales that stay in range.
    print()
    for (M, N, K) in [(1024, 1024, 1024), (4096, 4096, 4096)]:
        a = torch.randint(-127, 128, (M, K), dtype=torch.int8)
        b = torch.randint(-127, 128, (N, K), dtype=torch.int8)
        s = 1.0 / 512.0
        xs = torch.full((M, 1), s, dtype=torch.float32)
        ws = torch.full((1, N), s, dtype=torch.float32)
        exact = (a.double() @ b.double().T) * (s * s)
        for odt in (torch.float16, torch.bfloat16):
            out = aiter.gemm_a8w8(a, b, xs, ws, dtype=odt)
            verdict, max_ulp = check(out, exact, odt)
            print(f"in-range {M}x{N}x{K} {str(odt).split('.')[-1]:>9} "
                  f"finite={bool(torch.isfinite(out).all())} "
                  f"maxULP={max_ulp:.2f}  {verdict}")

    # Non-trivial rowwise scales: confirm the scale path is actually applied.
    M = N = K = 1024
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    b = torch.randint(-127, 128, (N, K), dtype=torch.int8)
    xs = torch.rand(M, 1, dtype=torch.float32) * 0.01 + 0.001
    ws = torch.rand(1, N, dtype=torch.float32) * 0.01 + 0.001
    exact = (a.double() @ b.double().T) * xs.double() * ws.double()
    out = aiter.gemm_a8w8(a, b, xs, ws, dtype=torch.bfloat16)
    rel = (out.double() - exact).abs() / exact.abs().clamp(min=1e-9)
    ok = rel.max().item() < 2 * EPS[torch.bfloat16]
    print(f"\nrowwise scales {M}^3: max_rel={rel.max().item():.5f} "
          f"mean_rel={rel.mean().item():.6f}  {'PASS' if ok else 'WRONG'}")

    # gfx90a has no FP8 MFMA and the FP8 instances are excluded from the build,
    # so FP8 input must fail loudly rather than silently run an emulated path.
    # e4m3fn is what get_torch_fp8() resolves to here and so reaches the
    # compiled-out dispatch block; e4m3fnuz is rejected by the earlier generic
    # dtype check. Both must refuse.
    print()
    for fp8 in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        name = str(fp8).split(".")[-1]
        try:
            af = torch.randn(256, 256).to(fp8)
            aiter.gemm_a8w8(af, af, torch.ones(256, 1), torch.ones(1, 256),
                            dtype=torch.bfloat16)
            print(f"fp8 refusal {name}: NO -- accepted, which it must not be")
        except Exception as e:
            msg = str(e).replace("\n", " ")
            print(f"fp8 refusal {name}: YES -- ...{msg[-110:].strip()}")


if __name__ == "__main__":
    main()
