"""Let vLLM dispatch AITER *attention* to gfx90a, and keep FP8 away from it.

`enable_gfx90a_asm_paths.py` unlocks AITER's own ASM fast paths. This script
fixes the layer above it: vLLM refuses to route anything to AITER on an MI210,
so those kernels are unreachable from the serving stack no matter which
`VLLM_ROCM_USE_AITER_*` flags are set.

The block is not the documented one. `_aiter_ops.py` states in prose that its
check functions verify "(1) platform is ROCm, (2) device arch is gfx9, and
(3) aiter library is installed". gfx90a is gfx9, so by that description an
MI210 qualifies. The code implementing the check calls `on_mi3xx()`, which is

    _ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])

so it excludes CDNA2. Because `if_aiter_supported` returns `None` -- not
`False` -- when the check fails, every `is_*_enabled()` query answers falsy and
vLLM silently drops ROCM_AITER_FA and ROCM_AITER_UNIFIED_ATTN from the backend
candidate list. The failure is invisible: the engine logs a normal-looking
"Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN',
'TRITON_ATTN']" and serves happily from the Triton path, with AITER never
touched. Setting VLLM_ROCM_USE_AITER=1 changes nothing at all.

Why attention-only, and not the master gate
-------------------------------------------
An earlier version of this script widened `is_aiter_found_and_supported()`
itself. That works, but it admits gfx90a to *every* AITER op, including
`AiterFp8BlockScaledMMKernel` -- an FP8 GEMM on a chip with no FP8 ALU. It was
tempting to call that harmless because the measurements all pinned
`VLLM_ROCM_USE_AITER_LINEAR=0`, but "it is gated off elsewhere" is not a safety
property: the other gate can be removed by a later change, which is exactly the
defect class PR #8 closed. So the master gate is left alone and only the two
attention-specific checks are widened.

The cost is AITER's INT8 and bf16 linear paths, which stay unreachable on
gfx90a. That is an accepted trade -- there is no measurement showing they help,
and the 1.23x in benchmarks/vllm-aiter-asm-gfx90a.md came entirely from
attention.

Sites patched
-------------
1. `_aiter_ops.py` -- adds `is_aiter_attention_supported()` and the
   `if_aiter_attention_supported` decorator: the same check as the master one
   but keyed on `on_gfx9()`, the condition the surrounding docs already claim.

2. `_aiter_ops.py:is_mha_enabled` -- re-decorated with it. This is what puts
   ROCM_AITER_FA back in the backend candidate list.

3. `_aiter_ops.py:is_shuffle_kv_cache_enabled` -- re-decorated with it. The
   shuffled KV layout is what routes decode to `pa_fwd_asm`. It is read only by
   `rocm_aiter_fa.py`, so widening it cannot affect a non-attention path.

4. `rocm_aiter_fa.py:AiterFlashAttentionBackend.supports_compute_capability`
   -- a second, independent `on_mi3xx()` that rejects the backend during
   validation even once the gates above admit it.

5. `scaled_mm/pytorch.py:TorchFP8ScaledMMLinearKernel.is_supported` --
   *narrowed*, not widened. See below.

What is NOT touched
-------------------
`is_aiter_found_and_supported()` keeps `on_mi3xx()`, so AITER linear, MoE,
RMSNorm, quant and FP8 GEMM paths remain unreachable on gfx90a.

`rocm_aiter_ops.is_enabled()` therefore still answers falsy here. Inside the FA
backend that disables exactly one thing: `fused_rope_kvcache_supported()`, an
optional rope+KV-cache fusion. It is already unavailable whenever the shuffled
KV layout is on, so the ASM configuration is unaffected; the non-shuffled
variant loses the fusion. `flash_attn_varlen_func`, `triton_rope_and_cache` and
`paged_attention_common` are undecorated static methods and work regardless.

`is_linear_hipbmm_enabled` and the RowWise scaled-mm kernel keep their
`on_mi3xx()` checks. Both guard FP8 scaled-GEMM paths that CDNA2 cannot run.

The FP8 narrowing (site 5)
--------------------------
This one fixes a latent bug in the opposite direction. gfx90a reports compute
capability **9.0**, so the base `TorchFP8ScaledMMLinearKernel` admits it:

    if compute_capability is not None and compute_capability < 89:
        return False, "requires compute capability 89 and above."

Measured on an MI210, before this patch:

    PerTensorTorchFP8ScaledMMLinearKernel.is_supported(90)    -> (True, None)
    ChannelWiseTorchFP8ScaledMMLinearKernel.is_supported(90)  -> (True, None)
    RowWiseTorchFP8ScaledMMLinearKernel.is_supported(90)      -> (False, 'requires MI3xx.')

Only the RowWise variant refuses. A per-tensor or per-channel FP8 checkpoint
would therefore select one of the first two, reach `torch._scaled_mm`, and die
with a PyTorch error that says nothing about why:

    RuntimeError: torch._scaled_mm is only supported on CUDA devices with
    compute capability >= 9.0 or 8.9, or ROCm MI300+

The patch refuses at kernel-selection time instead, naming the actual cause.
This does not remove FP8 support on gfx90a: block-quantized checkpoints route
to the block-scaled kernels, a different code path that is left alone. See part
3 of benchmarks/vllm-aiter-asm-gfx90a.md, where a block-quantized FP8 model
runs (slowly) with no involvement from these kernels.

    python enable_vllm_aiter_gfx90a.py [--revert] [--check]
"""
import argparse
import sys

SITE = "/opt/python/lib/python3.14/site-packages"
AITER_OPS = f"{SITE}/vllm/_aiter_ops.py"
AITER_FA = f"{SITE}/vllm/v1/attention/backends/rocm_aiter_fa.py"
SCALED_MM = f"{SITE}/vllm/model_executor/kernels/linear/scaled_mm/pytorch.py"

# 1. Attention-specific support check + decorator, appended after the existing
#    if_aiter_supported definition.
_DECORATOR_ANCHOR = '''    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_aiter_found_and_supported():
            return func(*args, **kwargs)

        return None

    return wrapper
'''

_DECORATOR_PATCHED = _DECORATOR_ANCHOR + '''

def is_aiter_attention_supported() -> bool:
    """Can AITER's attention kernels run here?

    Deliberately broader than `is_aiter_found_and_supported`, and deliberately
    only broader for attention. AITER's fmha_v3_fwd and pa_fwd_asm kernels are
    hand-written gfx9 ASM and run on gfx90a once the code objects exist (see
    enable_gfx90a_asm_paths.py); they are validated there and benchmarked in
    benchmarks/vllm-aiter-asm-gfx90a.md.

    The master check stays on `on_mi3xx()` so that AITER's GEMM, MoE and FP8
    paths remain unreachable on CDNA2, which has no FP8 ALU and no gfx90a GEMM
    tuning configs.
    """
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_gfx9

        return on_gfx9()
    return False


def if_aiter_attention_supported(func: Callable) -> Callable:
    """As `if_aiter_supported`, but keyed on attention support."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_aiter_attention_supported():
            return func(*args, **kwargs)

        return None

    return wrapper
'''

# 2/3. Re-decorate the two attention checks.
_MHA_ANCHOR = '''    @classmethod
    @if_aiter_supported
    def is_mha_enabled(cls) -> bool:
'''
_MHA_PATCHED = '''    @classmethod
    @if_aiter_attention_supported
    def is_mha_enabled(cls) -> bool:
'''

_SHUFFLE_ANCHOR = '''    @classmethod
    @if_aiter_supported
    def is_shuffle_kv_cache_enabled(cls) -> bool:
'''
_SHUFFLE_PATCHED = '''    @classmethod
    @if_aiter_attention_supported
    def is_shuffle_kv_cache_enabled(cls) -> bool:
'''

# 4. The FA backend's own arch check.
_FA_IMPORT_ANCHOR = "        from vllm.platforms.rocm import on_mi3xx\n"
_FA_IMPORT_PATCHED = "        from vllm.platforms.rocm import on_gfx9\n"
_FA_RETURN_ANCHOR = "        return on_mi3xx()\n"
_FA_RETURN_PATCHED = "        return on_gfx9()\n"

# 5. Narrow the torch scaled_mm gate so CDNA2 is refused with a real reason.
_CC_ANCHOR = '''        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."

        return True, None
'''
_CC_PATCHED = '''        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."

        # gfx90a reports compute capability 9.0, so the check above admits it,
        # but CDNA2 has no FP8 ALU and torch._scaled_mm is gated to MI300+.
        # Without this, a per-tensor or per-channel FP8 checkpoint selects this
        # kernel and then dies inside ATen with an error that explains nothing.
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_mi3xx

            if not on_mi3xx():
                return False, (
                    "requires MI3xx on ROCm: torch._scaled_mm needs FP8 "
                    "hardware, which CDNA2 (gfx90a) does not have. Use a "
                    "block-quantized FP8 checkpoint, which routes to the "
                    "block-scaled kernels instead."
                )

        return True, None
'''

# (path, unpatched, patched, expected occurrences)
PATCHES = [
    (AITER_OPS, _DECORATOR_ANCHOR, _DECORATOR_PATCHED, 1),
    (AITER_OPS, _MHA_ANCHOR, _MHA_PATCHED, 1),
    (AITER_OPS, _SHUFFLE_ANCHOR, _SHUFFLE_PATCHED, 1),
    (AITER_FA, _FA_IMPORT_ANCHOR, _FA_IMPORT_PATCHED, 1),
    (AITER_FA, _FA_RETURN_ANCHOR, _FA_RETURN_PATCHED, 1),
    (SCALED_MM, _CC_ANCHOR, _CC_PATCHED, 1),
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
        # Some rewrites append to their anchor, so the patched form contains
        # the unpatched one. Count the patched form first and discount it.
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
                    help="restore the stock gates")
    ap.add_argument("--check", action="store_true",
                    help="report state without writing")
    args = ap.parse_args()

    failures = apply(revert=args.revert, check=args.check)
    if failures:
        print(f"\n{failures} site(s) did not match as expected.")
        return 1
    if not args.check:
        print("\nOK. Verify with:  python enable_vllm_aiter_gfx90a.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
