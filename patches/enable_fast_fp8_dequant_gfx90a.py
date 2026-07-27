"""Give gfx90a a block-scaled FP8 GEMM that is not bottlenecked on decoding FP8.

The problem
-----------
A block-quantized FP8 checkpoint runs correctly on an MI210 and keeps its
memory saving, but serves 10-15x slower than the bf16 checkpoint. The linear
layers are the whole of the difference: `TritonFp8BlockScaledMMKernel` is the
only FP8 GEMM reachable on CDNA2 and it peaks at 8.3 TFLOP/s against bf16's
~96 on identical shapes.

That was assumed to be a missing tuning config. It is not. Disassembling what
Triton generates for gfx90a shows the real cause:

    _w8a8_triton_block_scaled_mm     11,997 instructions
                                         64  v_mfma_f32_32x32x8f16
                                      7,106  v_cmp_ne_u16 / v_cndmask_b32

The matmul is 64 instructions. The other ~11,900 are almost entirely the
**e4m3 -> fp16 conversion**, emulated in software. gfx942 decodes FP8 with
`v_cvt_pk_f32_fp8`; gfx90a has no such instruction, so Triton open-codes the
IEEE-correct conversion -- a `v_cmp`/`v_cndmask` pair per inf, NaN and denormal
case, ~29 VALU ops per value. A standalone conversion kernel confirms the cost
in isolation: 117 VALU ops to decode 4 e4m3 values, against 11 for int8 or
e5m2, both of which gfx90a can decode with a shift.

So FP8 on CDNA2 is not slow because it lacks FP8 matrix hardware -- it never
needed any, the MFMA runs in fp16 either way. It is slow because it lacks an
FP8 *decoder*.

The fix
-------
The expensive conversion is expensive only because it is exact for inputs that
block-quantized weights never contain. Reinterpreting the bits is nearly free:

    e4m3fn:  s eeee mmm          (exponent bias 7)
    fp16:    s eeeee mmmmmmmmmm  (exponent bias 15)

    h = ((u & 0x80) << 8) | ((u & 0x7f) << 7)

places the 4 exponent bits at 13:10 and the 3 mantissa bits at 9:7, producing a
valid fp16 whose exponent carries bias 15 where the value assumed bias 7. The
result is therefore exactly 2^-8 times the true value -- and exactly, for
normals and denormals alike, because the denormal case shifts identically.
Folding 2^8 per operand back in (2^16 on the fp32 accumulator, one multiply per
output element) restores the true product.

Verified over the whole domain: all 254 non-NaN e4m3 byte patterns decode
bit-identically to `float8_e4m3fn.float()`, max_abs_diff 0.0. The two NaN
patterns (0x7f, 0xff) decode to +-480.0 instead of NaN; block-quantized weights
contain no NaN, and `_looks_like_e4m3_blockscale` below refuses anything that is
not a plain e4m3 block-scaled GEMM.

Measured on an MI210, Qwen3-14B shapes, same operands, output checked against a
dequantized reference every time (mean relative error 1e-6, identical to the
stock kernel's own error):

    kernel                       instructions   M=1      M=32     M=4096
    _w8a8_triton_block_scaled_mm       11,997   1123 us  1132 us  27796 us
    fast bit-trick dequant              3,954    204 us   184 us   4722 us

That is 5.5-6x, before any tuning, with the identical config and the identical
64 MFMA instructions doing the identical arithmetic.

What this does NOT do
---------------------
It does not give gfx90a FP8 arithmetic, and it does not touch the
`torch._scaled_mm` gate that `enable_vllm_aiter_gfx90a.py` narrowed. There is
still no FP8 MFMA and no `v_cvt_pk_fp8_f32` on CDNA2; this kernel loads FP8 and
multiplies in fp16, which is what the stock kernel already did. FP8 cannot beat
bf16 on CDNA2 -- they share one 181 TFLOP/s peak -- so the ceiling here is
matching bf16 speed while holding 1.75x less weight memory.

Gating
------
The hook replaces nothing. `w8a8_triton_block_scaled_mm` gains an early branch
that fires only when all of these hold:

  * ROCm, and `on_gfx9() and not on_mi3xx()` -- i.e. CDNA1/CDNA2 specifically.
    On MI300+ the stock path uses the hardware decoder and is faster.
  * both operands are `torch.float8_e4m3fn` (not fnuz, not e5m2, not int8)
  * scales are plain fp32 (E8M0 checkpoints fall through to the stock path)
  * block shape is the [128, 128] the trick's scale folding assumes

Anything else takes the original code path, unchanged.

    python enable_fast_fp8_dequant_gfx90a.py [--revert] [--check]
"""
import argparse
import sys

SITE = "/opt/python/lib/python3.14/site-packages"
FP8_UTILS = f"{SITE}/vllm/model_executor/layers/quantization/utils/fp8_utils.py"

# The kernel plus its dispatch predicate, inserted immediately before the
# stock kernel's @triton.jit decorator.
_KERNEL_ANCHOR = '''@triton.jit
def _w8a8_triton_block_scaled_mm(
'''

_KERNEL_PATCHED = '''# --- gfx90a fast e4m3 decode (enable_fast_fp8_dequant_gfx90a.py) ------------
# CDNA2 has no v_cvt_pk_f32_fp8, so Triton emulates e4m3 -> fp16 in ~29 VALU
# ops per value and the conversion, not the MFMA, dominates the kernel. The
# bit reinterpretation below is exact for every non-NaN e4m3 byte and costs 3.


@triton.jit
def _gfx90a_fast_fp8_to_f16(u):
    """Decode an e4m3fn byte to an fp16 holding exactly 2^-8 times its value."""
    u16 = u.to(tl.uint16)
    h = ((u16 & 0x80) << 8) | ((u16 & 0x7F) << 7)
    return h.to(tl.float16, bitcast=True)


@triton.jit
def _gfx90a_fast_block_scaled_mm(
    A, B, C, As, Bs,
    M, N, K,
    group_n, group_k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_As_m, stride_As_k,
    stride_Bs_k, stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """As _w8a8_triton_block_scaled_mm, but A and B arrive as uint8."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_u = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b_u = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)

        a = _gfx90a_fast_fp8_to_f16(a_u)
        b = _gfx90a_fast_fp8_to_f16(b_u)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # each operand was decoded 2^-8 too small
    c = (accumulator * 65536.0).to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_GFX90A_FAST_FP8 = None


def _gfx90a_fast_fp8_eligible(A, B, As, Bs, block_size):
    """Only plain e4m3 block-scaled GEMMs on CDNA1/CDNA2 take the fast path."""
    global _GFX90A_FAST_FP8
    if _GFX90A_FAST_FP8 is None:
        _GFX90A_FAST_FP8 = False
        if current_platform.is_rocm():
            try:
                from vllm.platforms.rocm import on_gfx9, on_mi3xx

                # MI300+ has a hardware FP8 decoder; the stock path wins there.
                _GFX90A_FAST_FP8 = bool(on_gfx9()) and not bool(on_mi3xx())
            except Exception:
                _GFX90A_FAST_FP8 = False
    if not _GFX90A_FAST_FP8:
        return False

    # The bit trick decodes e4m3fn specifically, and folds a fixed 2^16 that
    # assumes both operands went through it.
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        return False
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        return False
    if list(block_size) != [128, 128]:
        return False
    return True


# --- end gfx90a fast e4m3 decode -------------------------------------------


@triton.jit
def _w8a8_triton_block_scaled_mm(
'''

# The dispatch hook, inserted just before the stock kernel launch.
_LAUNCH_ANCHOR = '''    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
'''

_LAUNCH_PATCHED = '''    if _gfx90a_fast_fp8_eligible(A, B, As, Bs, block_size):
        _gfx90a_fast_block_scaled_mm[grid](
            A.view(torch.uint8),
            B.view(torch.uint8),
            C,
            As,
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            **config,
        )
        return C

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
'''

# (path, unpatched, patched, expected occurrences)
PATCHES = [
    (FP8_UTILS, _KERNEL_ANCHOR, _KERNEL_PATCHED, 1),
    (FP8_UTILS, _LAUNCH_ANCHOR, _LAUNCH_PATCHED, 1),
]


def apply(revert: bool = False, check: bool = False) -> int:
    failures = 0
    for path, old, new, want in PATCHES:
        if revert:
            old, new = new, old
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError as exc:
            print(f" ERROR    {path}: {exc}")
            failures += 1
            continue

        name = path.rsplit("/", 1)[-1]
        # Both rewrites wrap their anchor, so the patched form contains the
        # unpatched one. Count the patched form first and discount it.
        n_patched = text.count(new)
        n_unpatched = text.count(old) - (n_patched if old in new else 0)

        if check:
            state = "unpatched" if n_unpatched else "patched"
            print(f" {state:>9}  {name:<16} (want {want}, "
                  f"found {n_unpatched} unpatched / {n_patched} patched)")
            if n_unpatched + n_patched != want:
                failures += 1
            continue

        if n_patched == want and n_unpatched == 0:
            print(f" already    {name}")
            continue
        if n_unpatched != want:
            print(f" ERROR      {name}: expected {want} occurrence(s), found "
                  f"{n_unpatched}. Upstream code moved -- reread the source.")
            failures += 1
            continue

        with open(path, "w") as fh:
            fh.write(text.replace(old, new, want))
        print(f" patched    {name}")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true",
                    help="restore the stock kernel path")
    ap.add_argument("--check", action="store_true",
                    help="report state without writing")
    args = ap.parse_args()

    failures = apply(revert=args.revert, check=args.check)
    if failures:
        print(f"\n{failures} site(s) did not match as expected.")
        return 1
    if not args.check:
        print("\nOK. Verify with:  "
              "python enable_fast_fp8_dequant_gfx90a.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
