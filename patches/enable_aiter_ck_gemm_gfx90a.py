#!/usr/bin/env python3
"""Let the AITER CK int8 GEMM run on gfx90a, and let vLLM select it.

WHY. Round 38e's profile put `scaled_mm_kernel` at 44% of decode kernel time
-- by a wide margin the largest consumer, on a card whose decode is 99.9%
kernel-busy. That kernel is vLLM's GENERIC TRITON FALLBACK
(compressed_tensors/triton_scaled_mm.py). vLLM already ships a better one for
ROCm -- `AiterInt8ScaledMMLinearKernel` -> `aiter.gemm_a8w8_CK` -- and
`benchmarks/matrix/probe_a8w8_ck_gfx90a.py` measured it on this box at the
real Qwen3-30B-A3B shapes:

    decode  (M=1..16)   median 1.662x   (min 1.075x, max 1.810x)
    prefill (M=512..8k) median 1.210x   (min 1.051x, max 1.527x)

all within 3.5e-3 relative error of an fp32 reference. Two independent things
block it on CDNA2, and this script removes both.

BLOCKER 1 -- the build. `aiter.gemm_a8w8_CK` fails to JIT-build with

    fatal error: 'gemm_a8w8_manifest.h' file not found

which is a MISSING CODE-GENERATION STEP, not a compile error. Instance
codegen consults `GFX_CU_NUM_MAP` in `aiter/jit/utils/build_targets.py` to
resolve the current arch's CU count, that table holds only gfx942 / gfx950 /
gfx1250, and the lookup raises

    RuntimeError: Unknown gfx 'gfx90a' in GPU_ARCHS -- add it to
    GFX_CU_NUM_MAP in build_targets.py

so nothing is generated and the compile dies later on the absent header.
Note the shape of this bug: the table is consulted EVEN WHEN the tuned CSV
has no rows for your architecture (aiter/configs/a8w8_tuned_gemm.csv is 553
gfx950 rows + 26 gfx942 rows + zero gfx90a), so an arch that would otherwise
fall back to default kernels gets a hard error instead of the fallback.
Bypassing the CSV proves the templates are fine on CDNA2: generation then
emits 9 instances, a manifest and a lookup header, with this project's
AITER_A8W8_NO_FP8 guard correctly applied. The fix is one table entry --
MI210 reports 104 CUs (`torch.cuda.get_device_properties(0)`).

  Upstream: this is worth reporting. Related-but-different reports exist for
  the same CLASS of bug (ROCm/aiter#1415, #1552 -- missing arch in a mapping
  dict) and for gfx90a build failures generally (#179, fp8 kernels), but the
  CU-map path appears unreported. Its distinguishing detail is the one above:
  it defeats the no-tuned-config fallback.

BLOCKER 2 -- selection. `AiterInt8ScaledMMLinearKernel.is_supported()` gates
on `rocm_aiter_ops.is_linear_enabled()`, which carries `@if_aiter_supported`
-> `is_aiter_found_and_supported()` -> `on_mi3xx()` = gfx942|gfx950.
`enable_vllm_aiter_gfx90a.py` deliberately narrowed its carve-out to
ATTENTION ONLY, so linear stays blocked by our own patch as well as by
upstream's. This re-decorates `is_linear_enabled` the same way, against the
same found-and-gfx9 predicate.

ORDER. This script MUST run after `enable_vllm_aiter_gfx90a.py`, whose
inserted `is_aiter_attention_supported` this reuses. It asserts that, rather
than producing a subtly broken file.

RUNTIME. Both patches are necessary and neither is sufficient: serving still
needs VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_LINEAR=1. A run that
applies this and leaves the flags off is measuring the Triton kernel.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# --- 1. the CU-count table -------------------------------------------------

_CU_ANCHOR = 'GFX_CU_NUM_MAP = {\n'
_CU_PATCHED = (
    'GFX_CU_NUM_MAP = {\n'
    '    # MI210 = 104 CUs, read from the device on this box. gfx90a is not\n'
    '    # one part: MI250 is 104 CUs per GCD and matches, but MI250X is 110\n'
    '    # per GCD and would need the CU_NUM override this table documents.\n'
    '    # Only ever a no-live-GPU default -- get_cu_num() wins when a card is\n'
    '    # visible. See configs/enable_aiter_ck_gemm_gfx90a.py\n'
    '    "gfx90a": 104,\n'
)

# --- 2. the selection gate -------------------------------------------------

_LINEAR_ANCHOR = '''    @classmethod
    @if_aiter_supported
    def is_linear_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._LINEAR_ENABLED
'''
_LINEAR_PATCHED = '''    @classmethod
    def is_linear_enabled(cls) -> bool:
        # gfx90a carve-out (configs/enable_aiter_ck_gemm_gfx90a.py). The
        # stock decorator resolves to on_mi3xx() and excludes CDNA2, but the
        # CK a8w8 GEMM measures 1.08-1.81x over the Triton fallback here.
        # is_aiter_attention_supported() is "aiter importable AND gfx9",
        # which is exactly the predicate this needs; it is reused rather than
        # duplicated so the two carve-outs cannot drift apart.
        if not is_aiter_attention_supported():
            return False
        return cls._AITER_ENABLED and cls._LINEAR_ENABLED
'''

# --- 3. op REGISTRATION -----------------------------------------------------
#
# Found the hard way: with 1 and 2 applied, serving still died at the first
# qkv_proj with
#
#   AttributeError: '_OpNamespace' 'vllm' object has no attribute
#                   'rocm_aiter_w8a8_gemm'
#
# because register_ops_once() carries @if_aiter_supported too. On CDNA2 it
# returns None and NOT ONE aiter custom op is ever registered. The attention
# carve-out never tripped over this: the ASM flash-attention backend calls
# into aiter directly rather than through torch.ops.vllm.
#
# Registration only DECLARES ops. Every impl imports its aiter symbol lazily
# inside the call body, so declaring MI300-only ops on gfx90a costs nothing
# and cannot execute them -- their own is_*_enabled() gates remain
# on_mi3xx().
_REGISTER_ANCHOR = '''    @staticmethod
    @if_aiter_supported
    def register_ops_once() -> None:
        global _OPS_REGISTERED
'''
# vLLM >= 0.26.1 already decorates register_ops_once with
# @if_aiter_attention_supported. That is semantically identical to the
# hand-rolled body guard below, so when this form is present the site needs no
# patch. Without this case the anchor matches 0 times, the script aborts, and
# -- because the write is atomic -- the is_linear_enabled carve-out that DID
# match is never written to disk either. Symptom: "patched ... is_linear_enabled
# carve-out" printed, followed by FATAL, followed by --check reporting it as
# "not patched".
_REGISTER_EQUIV = '''    @if_aiter_attention_supported
    def register_ops_once() -> None:
'''

_REGISTER_PATCHED = '''    @staticmethod
    def register_ops_once() -> None:
        # gfx90a carve-out -- see configs/enable_aiter_ck_gemm_gfx90a.py.
        # Without this, torch.ops.vllm.rocm_aiter_w8a8_gemm does not exist on
        # CDNA2 and the CK GEMM cannot be called no matter what the selection
        # gate says.
        if not is_aiter_attention_supported():
            return
        global _OPS_REGISTERED
'''

# The prerequisite from enable_vllm_aiter_gfx90a.py.
_PREREQ = "def is_aiter_attention_supported() -> bool:"


def _paths(site: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    return (
        site / "aiter" / "jit" / "utils" / "build_targets.py",
        site / "vllm" / "_aiter_ops.py",
    )


def _read(p: pathlib.Path) -> str:
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p.read_text()


def check(site: pathlib.Path) -> int:
    bt, ops = _paths(site)
    bt_s, ops_s = _read(bt), _read(ops)
    cu_done = '"gfx90a": 104' in bt_s
    lin_done = _LINEAR_PATCHED in ops_s and _LINEAR_ANCHOR not in ops_s
    # Satisfied either by our carve-out or by upstream's own
    # @if_aiter_attention_supported decorator (vLLM >= 0.26.1). Both mean the
    # ops get registered on gfx90a, which is all this site exists to ensure.
    reg_ours = _REGISTER_PATCHED in ops_s and _REGISTER_ANCHOR not in ops_s
    reg_upstream = _REGISTER_EQUIV in ops_s
    reg_done = reg_ours or reg_upstream
    reg_how = (
        "PATCHED" if reg_ours
        else "OK (upstream-gated)" if reg_upstream
        else "not patched"
    )
    print(f"  CU map (gfx90a: 104)       : {'PATCHED' if cu_done else 'not patched'}")
    print(f"  is_linear_enabled carve-out: {'PATCHED' if lin_done else 'not patched'}")
    print(f"  register_ops_once carve-out: {reg_how}")
    return 0 if (cu_done and lin_done and reg_done) else 1


def apply(site: pathlib.Path) -> int:
    bt, ops = _paths(site)

    # --- CU map
    bt_s = _read(bt)
    # Test the PRECISE marker, not a bare '"gfx90a"'. The module already
    # mentions the arch at build_targets.py:11 (`1: "gfx90a"` in the id->name
    # table), so the loose test reports "already patched" and silently skips
    # -- which is exactly what it did on the first build attempt. Only
    # GFX_CU_NUM_MAP is missing the entry.
    if '"gfx90a": 104' in bt_s:
        print(f"  CU map already has the gfx90a entry; leaving {bt} alone")
    else:
        n = bt_s.count(_CU_ANCHOR)
        if n != 1:
            sys.exit(
                f"FATAL: GFX_CU_NUM_MAP anchor matched {n} times in {bt}, "
                "expected exactly 1. Upstream changed the table; re-derive "
                "this patch rather than forcing it."
            )
        bt.write_text(bt_s.replace(_CU_ANCHOR, _CU_PATCHED))
        print(f"  patched {bt}: gfx90a -> 104 CUs")

    # --- selection gate
    ops_s = _read(ops)
    if _PREREQ not in ops_s:
        sys.exit(
            f"FATAL: {ops} has no is_aiter_attention_supported(). Run "
            "configs/enable_vllm_aiter_gfx90a.py FIRST -- this patch reuses "
            "the predicate that script inserts."
        )
    for name, anchor, patched, equivalent in (
        ("is_linear_enabled", _LINEAR_ANCHOR, _LINEAR_PATCHED, None),
        ("register_ops_once", _REGISTER_ANCHOR, _REGISTER_PATCHED, _REGISTER_EQUIV),
    ):
        if patched in ops_s:
            print(f"  {name} already carved out; leaving it alone")
            continue
        if equivalent is not None and equivalent in ops_s:
            # Upstream (>=0.26.1) decorates this with @if_aiter_attention_supported
            # directly, which is exactly what our carve-out achieves by hand.
            # Nothing to do -- and crucially, not a reason to abort the whole
            # patch and leave is_linear_enabled unwritten.
            print(f"  {name} already gated by if_aiter_attention_supported upstream")
            continue
        n = ops_s.count(anchor)
        if n != 1:
            sys.exit(
                f"FATAL: {name} anchor matched {n} times in {ops}, expected "
                "exactly 1. Upstream changed the method; re-derive this patch "
                "rather than forcing it."
            )
        ops_s = ops_s.replace(anchor, patched)
        print(f"  patched {ops}: {name} carve-out")
    ops.write_text(ops_s)
    return 0


def revert(site: pathlib.Path) -> int:
    """Undo both patches, or say plainly that it could not.

    Each branch VERIFIES the replacement instead of assuming it. An earlier
    version keyed the CU-map branch on the marker `"gfx90a": 104` but replaced
    the full _CU_PATCHED block; against a file patched by a different revision
    of this script (the comment above the entry has changed once already) the
    replace silently matched nothing and it printed "reverted" anyway. A
    revert that reports success without reverting is worse than one that
    fails.
    """
    bt, ops = _paths(site)
    rc = 0

    bt_s = _read(bt)
    if '"gfx90a": 104' not in bt_s:
        print(f"  CU map has no gfx90a entry; nothing to revert in {bt}")
    else:
        new = bt_s.replace(_CU_PATCHED, _CU_ANCHOR)
        if new == bt_s:
            print(f"  COULD NOT revert {bt}: it carries a gfx90a CU entry that "
                  "does not match this script's block, so it was written by a "
                  "different revision. Remove the entry by hand.")
            rc = 1
        else:
            bt.write_text(new)
            print(f"  reverted {bt}")

    ops_s = _read(ops)
    touched = False
    for name, anchor, patched in (
        ("is_linear_enabled", _LINEAR_ANCHOR, _LINEAR_PATCHED),
        ("register_ops_once", _REGISTER_ANCHOR, _REGISTER_PATCHED),
    ):
        if patched not in ops_s:
            print(f"  {name} carve-out not present; nothing to revert")
            continue
        ops_s = ops_s.replace(patched, anchor)
        touched = True
        print(f"  reverted {name}")
    if touched:
        ops.write_text(ops_s)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="/opt/python/lib/python3.14/site-packages")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--assert-patched", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    site = pathlib.Path(args.site)

    if args.check:
        return check(site)
    if args.assert_patched:
        rc = check(site)
        if rc:
            print("ASSERTION FAILED: CK GEMM patches are not fully applied.")
        return rc
    if args.revert:
        return revert(site)
    return apply(site)


if __name__ == "__main__":
    sys.exit(main())
