"""Correctness of AITER's ASM flash-attention forward (fmha_v3_fwd) on gfx90a.

`fmha_v3_fwd` is hand-written gfx9 ASM that AITER gates to gfx942/gfx950. 48 of
its 56 kernels -- every bf16 one -- are provably portable to gfx90a; only the 8
FP8 kernels are not, since CDNA2 has no FP8 ALU. After
`configs/repatch_gfx942_to_gfx90a.py` populates hsa/gfx90a/ and
`configs/enable_gfx90a_asm_paths.py` opens the arch gates, this test checks the
ASM kernels actually produce correct numbers.

Compares `aiter.flash_attn_func` against a PyTorch SDPA reference across head
dim, causal/non-causal, GQA ratio, sequence length and batch size.

    AITER_LOG_LEVEL=info python tests/test_fmha_v3_fwd_asm_gfx90a.py

With AITER_LOG_LEVEL=info the runtime prints a `LoadKernel: ... hsaco:
.../gfx90a/fmha_v3_fwd/MI300/....co` line the first time an ASM kernel loads.
That line is the only positive proof the ASM path ran rather than the CK
fallback -- correctness alone cannot distinguish them, since both should be
right. `--require-asm` turns its absence into a failure.

Why this test is not redundant with correctness-by-construction: the repatcher
proves each instruction re-encodes for gfx90a, but it cannot prove scheduling is
safe. V_MFMA_F32_16X16X16* is 4-pass on gfx942 and 8-pass on gfx90a, so ported
code can be short on wait states at an MFMA->consumer edge. That shows up as
wrong numbers under specific tile shapes, which is exactly what this grid hunts.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes


def capture_fds(fn):
    """Run fn() with OS-level fd 1/2 captured; return (result, captured_text).

    The `LoadKernel:` line is emitted by AITER's C++ runtime straight to fd 1, so
    contextlib.redirect_stdout cannot see it -- only a real dup2 can.
    """
    import tempfile
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            result = fn()
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        tmp.seek(0)
        return result, tmp.read()


def reference(q, k, v, causal, scale):
    """PyTorch SDPA reference. Inputs are (b, s, h, d); returns (b, s, h, d)."""
    hq = q.shape[2]
    hkv = k.shape[2]
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    if hq != hkv:
        rep = hq // hkv
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, scale=scale)
    return out.transpose(1, 2).to(q.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-asm", action="store_true",
                    help="fail if the ASM kernel never loaded")
    args = ap.parse_args()

    torch.set_default_device("cuda:0")
    gfx = aiter.jit.utils.chip_info.get_gfx()
    print(f"gfx: {gfx}")
    if os.environ.get("AITER_LOG_LEVEL", "").lower() != "info":
        print("NOTE: set AITER_LOG_LEVEL=info to see which kernel actually loads.")

    print(f"\n{'dtype':>5} {'hdq':>4} {'hdv':>4} {'causal':>7} {'b':>4} {'sq':>6} "
          f"{'sk':>6} {'hq':>4} {'hkv':>4} {'rel_rms':>10} {'%close':>8} "
          f"{'us':>9}  verdict")

    fails = 0
    ran = 0
    asm_loads = []
    for hdq, hdv in ((128, 128), (192, 128)):
        for causal in (False, True):
            for hq, hkv in ((8, 8), (16, 2), (32, 4)):
                for b, sq, sk in ((1, 256, 256), (2, 512, 512),
                                  (1, 1024, 1024), (4, 129, 129)):
                    torch.manual_seed(0)
                    scale = 1.0 / (hdq ** 0.5)
                    q = torch.empty(b, sq, hq, hdq, dtype=dtypes.bf16).uniform_(-1, 1)
                    k = torch.empty(b, sk, hkv, hdq, dtype=dtypes.bf16).uniform_(-1, 1)
                    v = torch.empty(b, sk, hkv, hdv, dtype=dtypes.bf16).uniform_(-1, 1)

                    ref = reference(q, k, v, causal, scale)
                    try:
                        got, log = capture_fds(
                            lambda: aiter.flash_attn_func(
                                q, k, v, softmax_scale=scale, causal=causal))
                        if "fmha_v3_fwd" in log:
                            asm_loads.append(
                                next(l.strip() for l in log.splitlines()
                                     if "fmha_v3_fwd" in l))
                    except Exception as exc:  # noqa: BLE001
                        print(f"{'bf16':>5} {hdq:>4} {hdv:>4} {str(causal):>7} "
                              f"{b:>4} {sq:>6} {sk:>6} {hq:>4} {hkv:>4} "
                              f"{'-':>10} {'-':>8} {'-':>9}  ERROR {exc}")
                        fails += 1
                        continue
                    if isinstance(got, tuple):
                        got = got[0]
                    torch.cuda.synchronize()

                    for _ in range(3):
                        aiter.flash_attn_func(q, k, v, softmax_scale=scale,
                                              causal=causal)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        aiter.flash_attn_func(q, k, v, softmax_scale=scale,
                                              causal=causal)
                    torch.cuda.synchronize()
                    us = (time.perf_counter() - t0) / 10 * 1e6

                    r, g = ref.float(), got.float()
                    rms = (((r - g) ** 2).mean().sqrt()
                           / r.abs().max().clamp(min=1e-6)).item()
                    pc = torch.isclose(r, g, atol=2e-2,
                                       rtol=2e-2).float().mean().item() * 100
                    ok = rms < 0.02 and pc > 99.0
                    fails += (not ok)
                    ran += 1
                    print(f"{'bf16':>5} {hdq:>4} {hdv:>4} {str(causal):>7} "
                          f"{b:>4} {sq:>6} {sk:>6} {hq:>4} {hkv:>4} "
                          f"{rms:>10.6f} {pc:>7.2f}% {us:>8.1f}  "
                          f"{'PASS' if ok else 'FAIL'}")

    # Varlen (packed THD) is the path a serving stack actually takes for
    # prefill, and it selects the *_group kernels rather than the batched ones,
    # so it needs its own coverage -- a batched pass says nothing about it.
    print(f"\n--- varlen (flash_attn_varlen_func, *_group kernels) ---")
    print(f"{'dtype':>5} {'hdq':>4} {'causal':>7} {'seqs':>5} {'lens':>18} "
          f"{'hq':>4} {'hkv':>4} {'rel_rms':>10} {'%close':>8} {'us':>9}  verdict")
    for hdq in (128, 192):
        for causal in (False, True):
            for hq, hkv in ((8, 8), (32, 4)):
                for lens in ([256], [128, 384], [64, 200, 500], [129, 129, 129, 129]):
                    torch.manual_seed(0)
                    hdv, scale = 128, 1.0 / (hdq ** 0.5)
                    total = sum(lens)
                    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)),
                                      dtype=torch.int32)
                    q = torch.empty(total, hq, hdq, dtype=dtypes.bf16).uniform_(-1, 1)
                    k = torch.empty(total, hkv, hdq, dtype=dtypes.bf16).uniform_(-1, 1)
                    v = torch.empty(total, hkv, hdv, dtype=dtypes.bf16).uniform_(-1, 1)

                    # Reference: attend within each sequence independently.
                    ref = torch.empty(total, hq, hdv, dtype=dtypes.bf16)
                    off = 0
                    for n in lens:
                        ref[off:off + n] = reference(
                            q[off:off + n].unsqueeze(0), k[off:off + n].unsqueeze(0),
                            v[off:off + n].unsqueeze(0), causal, scale).squeeze(0)
                        off += n

                    try:
                        got, log = capture_fds(
                            lambda: aiter.flash_attn_varlen_func(
                                q, k, v, cu, cu, max(lens), max(lens),
                                softmax_scale=scale, causal=causal))
                        if "fmha_v3_fwd" in log:
                            asm_loads.append(
                                next(l.strip() for l in log.splitlines()
                                     if "fmha_v3_fwd" in l))
                    except Exception as exc:  # noqa: BLE001
                        print(f"{'bf16':>5} {hdq:>4} {str(causal):>7} {len(lens):>5} "
                              f"{str(lens):>18} {hq:>4} {hkv:>4} {'-':>10} {'-':>8} "
                              f"{'-':>9}  ERROR {exc}")
                        fails += 1
                        continue
                    if isinstance(got, tuple):
                        got = got[0]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        aiter.flash_attn_varlen_func(q, k, v, cu, cu, max(lens),
                                                     max(lens), softmax_scale=scale,
                                                     causal=causal)
                    torch.cuda.synchronize()
                    us = (time.perf_counter() - t0) / 10 * 1e6

                    r, g = ref.float(), got.float()
                    rms = (((r - g) ** 2).mean().sqrt()
                           / r.abs().max().clamp(min=1e-6)).item()
                    pc = torch.isclose(r, g, atol=2e-2,
                                       rtol=2e-2).float().mean().item() * 100
                    ok = rms < 0.02 and pc > 99.0
                    fails += (not ok)
                    ran += 1
                    print(f"{'bf16':>5} {hdq:>4} {str(causal):>7} {len(lens):>5} "
                          f"{str(lens):>18} {hq:>4} {hkv:>4} {rms:>10.6f} "
                          f"{pc:>7.2f}% {us:>8.1f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n{ran} configs run, {fails} failures")

    if asm_loads:
        print(f"\nASM kernels loaded ({len(asm_loads)} distinct load events):")
        for line in sorted(set(asm_loads))[:8]:
            print(f"  {line}")
    else:
        msg = ("no fmha_v3_fwd ASM kernel was loaded -- these numbers came from "
               "the CK fallback, not ASM")
        if args.require_asm:
            print(f"\nFAIL: {msg}")
            sys.exit(1)
        print(f"\nNOTE: {msg}.\n"
              "      Run with AITER_LOG_LEVEL=info so the loader logs are visible.")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
