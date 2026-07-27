"""Compare block-scaled FP8 GEMM kernels on gfx90a, tuned and untuned.

Four things on identical Qwen3-14B shapes and identical operands:

  bf16    torch.matmul -- what FP8 has to match (it cannot beat it; on CDNA2
          FP8 and bf16 share one 181 TFLOP/s peak)
  vllm    vLLM's stock _w8a8_triton_block_scaled_mm
  aiter   aiter's _gemm_a8w8_blockscale_kernel, which is what
          AiterFp8BlockScaledMMKernel resolves to on this arch -- also Triton,
          because _hip_blockscale_supported() excludes gfx90a
  fast    the bit-trick decode from configs/enable_fast_fp8_dequant_gfx90a.py

Each is timed with its default config and with the best of a config sweep, so
the table separates "the kernel was untuned" from "the kernel was decoding FP8
in software". Every config is checked against a dequantized reference before it
is timed, and the winner is re-checked before its number is reported.

Requires configs/enable_fast_fp8_dequant_gfx90a.py to have been applied -- the
fast kernel is read back out of the patched vLLM module so that what is
measured is what actually ships.

    python bench_fp8_fast_dequant_gfx90a.py [--quick]
"""
import argparse
import itertools
import json
import sys
import time

import torch
import triton

import vllm.model_executor.layers.quantization.utils.fp8_utils as F

if not hasattr(F, "_gfx90a_fast_block_scaled_mm"):
    sys.exit(
        "vLLM is not patched. Run:\n"
        "    python configs/enable_fast_fp8_dequant_gfx90a.py"
    )

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (  # noqa: E402
    gemm_a8w8_blockscale,
)

BLOCK = [128, 128]
BN = BK = 128
DEV = "cuda"

SHAPES = [
    ("qkv_proj", 7168, 5120),
    ("o_proj", 5120, 5120),
    ("gate_up_proj", 34816, 5120),
    ("down_proj", 5120, 17408),
]
TOKENS = [1, 32, 4096]

VLLM_DEFAULT = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2,
}
AITER_DEFAULT = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16, "cache_modifier": None, "kpack": 2,
    "NUM_KSPLIT": 1,
}


def make_case(m, n, k):
    """Block-quantized operands plus the exact value the kernel should produce."""
    torch.manual_seed(1234)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16) / 4
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) / 4

    wv = w.float().view(n // BN, BN, k // BK, BK)
    ws = wv.abs().amax(dim=(1, 3)).clamp(min=1e-6) / 448.0
    wq = (wv / ws[:, None, :, None]).clamp(-448, 448).to(
        torch.float8_e4m3fn).view(n, k)

    xv = x.float().view(m, k // BK, BK)
    xs = xv.abs().amax(dim=2).clamp(min=1e-6) / 448.0
    xq = (xv / xs[:, :, None]).clamp(-448, 448).to(
        torch.float8_e4m3fn).view(m, k)

    xd = (xq.float().view(m, k // BK, BK) * xs[:, :, None]).view(m, k)
    wd = (wq.float().view(n // BN, BN, k // BK, BK)
          * ws[:, None, :, None]).view(n, k)
    ref = (xd @ wd.t()).to(torch.bfloat16)
    return x, w, xq, wq, xs.contiguous(), ws.contiguous(), ref


def rel_err(out, ref):
    return ((out.float() - ref.float()).abs().mean()
            / ref.float().abs().mean()).item()


def is_correct(out, ref, tol=0.02):
    """Written as `not (err < tol)` on purpose.

    `nan > tol` is False, so the natural spelling silently accepts a kernel that
    returns NaN -- which is exactly how a broken CK build got itself timed
    before this was caught.
    """
    return not (rel_err(out, ref) >= tol) and torch.isfinite(out).all().item()


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # microseconds


def _launch(kernel, A, B, As, Bs, config, out_dtype=torch.bfloat16):
    """Launch either block-scaled kernel; they share an argument order."""
    M = A.numel() // A.shape[-1]
    N, K = B.shape
    C = torch.empty(A.shape[:-1] + (N,), dtype=out_dtype, device=A.device)
    fast = kernel is F._gfx90a_fast_block_scaled_mm
    a, b = (A.view(torch.uint8), B.view(torch.uint8)) if fast else (A, B)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    kernel[grid](
        a, b, C, As, Bs, M, N, K, BLOCK[0], BLOCK[1],
        a.stride(-2), a.stride(-1), b.stride(1), b.stride(0),
        C.stride(-2), C.stride(-1), As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0), **config,
    )
    return C


def run_vllm(xq, wq, xs, ws, config):
    return _launch(F._w8a8_triton_block_scaled_mm, xq, wq, xs, ws, config)


def run_fast(xq, wq, xs, ws, config):
    return _launch(F._gfx90a_fast_block_scaled_mm, xq, wq, xs, ws, config)


def run_aiter(xq, wq, xs, ws, config):
    return gemm_a8w8_blockscale(xq, wq, xs, ws, dtype=torch.bfloat16,
                                config=dict(config))


def triton_space(quick):
    stages = [2] if quick else [2, 3, 4]
    bms = [16, 64] if quick else [16, 32, 64, 128, 256]
    return [
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
         "GROUP_SIZE_M": gm, "num_warps": nw, "num_stages": ns}
        for ns, bm, bk, bn, nw, gm in itertools.product(
            stages, bms, [64, 128], [32, 64, 128, 256], [4, 8], [1, 32])
    ]


def aiter_space():
    # GRID_MN is a constexpr, so each config recompiles per shape; this is the
    # neighbourhood of the winner found by a wider 128-config sweep.
    return [
        dict(AITER_DEFAULT, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=128,
             num_warps=4, NUM_KSPLIT=ks)
        for bm, ks in itertools.product([16, 32, 64], [1, 2, 4, 8])
    ]


def sweep(runner, space, xq, wq, xs, ws, ref, label):
    best, best_t, ok = None, float("inf"), 0
    t0 = time.time()
    for cfg in space:
        try:
            out = runner(xq, wq, xs, ws, cfg)
            torch.cuda.synchronize()
            if not is_correct(out, ref):
                continue
            t = timeit(lambda: runner(xq, wq, xs, ws, cfg), warmup=2, iters=5)
            ok += 1
            if t < best_t:
                best_t, best = t, cfg
        except Exception:
            continue
    print(f"      [{label}] {ok}/{len(space)} valid, best {best_t:.1f}us "
          f"({time.time() - t0:.0f}s)", flush=True)
    return best, best_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="smaller config sweep")
    ap.add_argument("--out", default="fp8-fast-dequant-results.json")
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name()}", flush=True)
    tspace, aspace = triton_space(args.quick), aiter_space()

    results = []
    for m in TOKENS:
        for name, n, k in SHAPES:
            print(f"\n=== M={m} {name} N={n} K={k}", flush=True)
            x, w, xq, wq, xs, ws, ref = make_case(m, n, k)
            row = {"M": m, "name": name, "N": n, "K": k}

            row["bf16_us"] = timeit(lambda: torch.matmul(x, w.t()))
            print(f"      bf16              {row['bf16_us']:9.1f} us", flush=True)

            for label, runner, default in (
                ("vllm", run_vllm, VLLM_DEFAULT),
                ("aiter", run_aiter, AITER_DEFAULT),
                ("fast", run_fast, VLLM_DEFAULT),
            ):
                try:
                    out = runner(xq, wq, xs, ws, default)
                    row[f"{label}_default_relerr"] = rel_err(out, ref)
                    row[f"{label}_default_us"] = timeit(
                        lambda: runner(xq, wq, xs, ws, default))
                    print(f"      {label} default     "
                          f"{row[f'{label}_default_us']:9.1f} us  "
                          f"relerr={row[f'{label}_default_relerr']:.6f}",
                          flush=True)
                except Exception as exc:
                    print(f"      {label} default FAILED: {exc}", flush=True)

            for label, runner, sp in (
                ("vllm", run_vllm, tspace),
                ("aiter", run_aiter, aspace),
                ("fast", run_fast, tspace),
            ):
                cfg, t = sweep(runner, sp, xq, wq, xs, ws, ref, label)
                if cfg is None:
                    continue
                out = runner(xq, wq, xs, ws, cfg)
                if not is_correct(out, ref):
                    print(f"      {label} tuned WINNER IS WRONG "
                          f"(relerr={rel_err(out, ref)}) -- not reported",
                          flush=True)
                    continue
                row[f"{label}_tuned_us"] = t
                row[f"{label}_tuned_relerr"] = rel_err(out, ref)
                row[f"{label}_tuned_config"] = cfg
                print(f"      {label} tuned       {t:9.1f} us  {cfg}",
                      flush=True)

            flops = 2 * m * n * k
            for key in [k2 for k2 in list(row) if k2.endswith("_us")]:
                row[key.replace("_us", "_tflops")] = (
                    flops / (row[key] * 1e-6) / 1e12)
            results.append(row)
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
