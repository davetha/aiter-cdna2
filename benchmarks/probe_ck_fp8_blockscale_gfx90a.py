"""Does CK's block-scaled FP8 GEMM work on gfx90a, and would it beat Triton?

Background: the INT8 work (docs/20, PR #9) built AITER's CK `module_gemm_a8w8`
for gfx90a, found that 32 of 36 FP8 instances compiled and 4 hung the register
allocator, and dropped the FP8 half on the reasoning that it is software
emulation that can never reach FP8 silicon. That was a statement about honesty
of description, not about speed -- the FP8 instances were never benchmarked.
This closes that gap for the kernel that actually matters.

`module_gemm_a8w8` is **rowwise**-scaled, so it cannot serve a block-quantized
128x128 checkpoint like Qwen3-14B-FP8 regardless of speed. The format-matching
module is `module_gemm_a8w8_blockscale`, which is what this probes.

What it establishes, in order:

  1. It builds. CK selects `mfma_f32_16x16x32f8f8` for `f8_t`, which gfx90a does
     not have, but `ck/utility/amd_xdlops.hpp` guards that with
     `#if defined(__gfx94__)` and falls back to converting each f8 to f32 and
     issuing 8x `intrin_mfma_f32_16x16x4f32`. So it compiles.
  2. What it compiled to -- disassembles the gfx90a code object out of the
     module's `.hip_fatbin` (the `.so` itself is host code; disassembling that
     shows no MFMA and is misleading).
  3. Whether it is correct. It is not: it returns exactly one quarter of the
     right answer, plus NaNs, because the fallback feeds one scalar per lane to
     a K=4 matrix instruction.

Correctness is checked with `not (err < tol)` rather than `err > tol`, because
`nan > tol` is False and a NaN-returning kernel would otherwise be timed.

    python probe_ck_fp8_blockscale_gfx90a.py
"""
import collections
import glob
import os
import subprocess
import sys
import tempfile
import time

import torch

BN = BK = 128
DEV = "cuda"
SENTINEL = -12345.0


def make_case(m, n, k, ones_scales=False):
    torch.manual_seed(1234)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16) / 4
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) / 4
    sn, sk = n // BN, k // BK

    wv = w.float().view(sn, BN, sk, BK)
    ws = (wv.abs().amax(dim=(1, 3)).clamp(min=1e-6) / 448.0).contiguous()
    wq = (wv / ws[:, None, :, None]).clamp(-448, 448).to(
        torch.float8_e4m3fn).view(n, k)
    xv = x.float().view(m, sk, BK)
    xs = (xv.abs().amax(dim=2).clamp(min=1e-6) / 448.0).contiguous()
    xq = (xv / xs[:, :, None]).clamp(-448, 448).to(
        torch.float8_e4m3fn).view(m, k)

    if ones_scales:
        xs = torch.ones_like(xs)
        ws = torch.ones_like(ws)

    xd = (xq.float().view(m, sk, BK) * xs[:, :, None]).view(m, k)
    wd = (wq.float().view(sn, BN, sk, BK) * ws[:, None, :, None]).view(n, k)
    return xq, wq, xs, ws, (xd @ wd.t())


def rel_err(out, ref):
    fin = torch.isfinite(out) & torch.isfinite(ref)
    if not fin.any():
        return float("nan")
    return ((out[fin] - ref[fin]).abs().mean() / ref[fin].abs().mean()).item()


def main() -> None:
    print(f"device: {torch.cuda.get_device_name()}")
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck

    print(f"arch  : {get_gfx()}\n")

    print("=" * 72)
    print("1. DOES IT BUILD AND RUN?")
    print("=" * 72)
    m, n, k = 64, 1024, 1024
    xq, wq, xs, ws, ref = make_case(m, n, k)
    Y = torch.full((m, n), SENTINEL, dtype=torch.bfloat16, device=DEV)
    t0 = time.time()
    try:
        out = gemm_a8w8_blockscale_ck(xq, wq, xs, ws, Y).float()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"  FAILED after {time.time() - t0:.0f}s: "
              f"{type(exc).__name__}: {str(exc)[:400]}")
        return
    print(f"  built and ran in {time.time() - t0:.0f}s")
    untouched = (out == SENTINEL).sum().item()
    print(f"  output cells left at sentinel: {untouched}/{out.numel()} "
          f"(0 means the kernel really did store)")

    print("\n" + "=" * 72)
    print("2. WHAT DID IT COMPILE TO?")
    print("=" * 72)
    so = ("/opt/python/lib/python3.14/site-packages/aiter/jit/"
          "module_gemm_a8w8_blockscale.so")
    llvm = "/opt/rocm/llvm/bin"
    tmp = tempfile.mkdtemp()
    try:
        subprocess.run([f"{llvm}/llvm-objcopy", "--dump-section",
                        f".hip_fatbin={tmp}/fat.bin", so], check=True)
        subprocess.run([f"{llvm}/clang-offload-bundler", "--type=o",
                        f"--input={tmp}/fat.bin",
                        "--targets=hipv4-amdgcn-amd-amdhsa--gfx90a",
                        f"--output={tmp}/dev.co", "--unbundle"], check=True)
        dis = subprocess.run([f"{llvm}/llvm-objdump", "-d", "--mcpu=gfx90a",
                              f"{tmp}/dev.co"], capture_output=True, text=True)
        c = collections.Counter()
        for line in dis.stdout.splitlines():
            if not line.startswith("\t"):
                continue
            op = line.strip().split(" ")[0]
            if op.startswith(("v_", "s_", "ds_", "buffer_", "global_", "flat_")):
                c[op] += 1
        print(f"  gfx90a code object: {sum(c.values())} instructions")
        print(f"    MFMA    : "
              f"{dict((kk, v) for kk, v in c.items() if 'mfma' in kk) or 'NONE'}")
        print(f"    FP8/BF8 : "
              f"{dict((kk, v) for kk, v in c.items() if 'fp8' in kk or 'bf8' in kk) or 'NONE'}")
    except Exception as exc:
        print(f"  could not extract device code: {exc}")

    print("\n" + "=" * 72)
    print("3. IS IT CORRECT?")
    print("=" * 72)
    for (mm, nn, kk) in [(64, 1024, 1024), (64, 5120, 5120), (128, 5120, 5120)]:
        xq, wq, xs, ws, ref = make_case(mm, nn, kk)
        Y = torch.zeros((mm, nn), dtype=torch.bfloat16, device=DEV)
        out = gemm_a8w8_blockscale_ck(xq, wq, xs, ws, Y).float()
        torch.cuda.synchronize()
        e = rel_err(out, ref)
        nans = torch.isnan(out).sum().item()
        ok = not (e >= 0.02) and nans == 0
        print(f"  M={mm:<4} N={nn:<5} K={kk:<5} relerr={e:<10.6f} "
              f"nan={nans:<6} {'CORRECT' if ok else 'WRONG'}")

    print("\n  ruling out scale layout -- all-ones scales, where it cannot matter:")
    xq, wq, xs, ws, ref = make_case(64, 1024, 1024, ones_scales=True)
    Y = torch.zeros((64, 1024), dtype=torch.bfloat16, device=DEV)
    out = gemm_a8w8_blockscale_ck(xq, wq, xs, ws, Y).float()
    torch.cuda.synchronize()
    print(f"    relerr={rel_err(out, ref):.6f}  (still wrong => not a layout bug)")

    print("\n  what fraction of the reduction did it do?")
    for div, label in ((1, "full"), (2, "full/2"), (4, "full/4")):
        print(f"    CK vs {label:<7}: {rel_err(out, ref / div):.6f}")
    print("\n  matching full/4 to bf16 rounding means CK computes exactly one")
    print("  quarter of the sum: the fallback feeds one scalar per lane to")
    print("  intrin_mfma_f32_16x16x4f32, a K=4 instruction, so 3 of 4 K lanes")
    print("  contribute nothing. No timings are quoted for a wrong kernel.")


if __name__ == "__main__":
    main()
