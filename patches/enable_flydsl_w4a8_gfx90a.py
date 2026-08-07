"""Enable FlyDSL W4A8 (int4 weight + int8 activation + INT8 MFMA) on gfx90a.

FlyDSL ships a W4A8 MoE kernel (``moe_gemm_2stage.py``, ``in_dtype="int4"``)
that unpacks packed int4 weights to int8 in-kernel and runs them through the
INT8 MFMA pipeline.  176 kernel configs for ``aint8_wint4`` are registered at
import time.  Two gates prevent this from being reachable on gfx90a:

1. ``flydsl/utils/smem_allocator.py`` — ``SMEM_CAPACITY_MAP`` omits gfx90a,
   so ``is_flydsl_available()`` returns False and the entire FlyDSL path
   is skipped.

2. ``aiter/ops/flydsl/moe_kernels.py`` — ``compile_flydsl_moe_stage1`` has
   no ``elif`` for ``a_dtype="int8", b_dtype="int4"``, falling through to
   "Unsupported stage1 dtype combination".

This script rewrites both sites.  It is idempotent and asserts on expected
match counts.

    python enable_flydsl_w4a8_gfx90a.py [--revert] [--check]

Sites patched
-------------

Site 1 — SMEM_CAPACITY_MAP (flydl/utils/smem_allocator.py, ~line 239)
    Insert ``"gfx90a": 65536,`` before the ``"gfx942"`` entry.
    gfx90a has the same 64 KB LDS per CU as gfx942 (CDNA3).

    **IMPORTANT**: after patching, delete ``__pycache__/smem_allocator*.pyc``
    — Python serves stale bytecode otherwise.

Site 2 — compile_flydsl_moe_stage1 dispatch (aiter/ops/flydsl/moe_kernels.py, ~line 556)
    Insert an ``elif a_dtype == "int8" and b_dtype == "int4"`` case before
    the ``else: raise ValueError(...)``.  The case mirrors the existing
    ``int4_bf16`` (W4A16) path but passes ``in_dtype="int4"`` (the W4A8
    path from moe_gemm_2stage.py).

Correctness caveat
------------------

The kernel **compiles and executes** on gfx90a (verified 2026-08-06, output
shape correct).  However, the FlyDSL int8 path uses
``mfma_i32_16x16x32_i8`` (K=32 INT8 MFMA), which is a gfx942 instruction.
gfx90a only has ``mfma_i32_16x16x16_i8`` (K=16) — see
[docs/int8-gemm.md](../docs/int8-gemm.md) and the port-matrix.  The K=32
spelling is **rejected by the gfx90a assembler**, so the FlyDSL-generated
kernel may silently produce incorrect output (observed alternating zeros
in test output).  A correctness comparison against a bf16 reference is the
outstanding work before trusting this for real inference.
"""

import argparse
import glob
import os
import sys

AITER_ROOT = os.environ.get(
    "AITER_ROOT",
    "/opt/python/lib/python3.14/site-packages",
)


def _site1_smem_map(action: str) -> bool:
    """Patch or revert SMEM_CAPACITY_MAP."""
    path = os.path.join(AITER_ROOT, "flydsl/utils/smem_allocator.py")
    with open(path) as f:
        src = f.read()

    has_gfx90a = '"gfx90a": 65536' in src

    if action == "check":
        return has_gfx90a

    if action == "apply" and not has_gfx90a:
        assert '"gfx942": 65536,' in src, "Site 1: gfx942 entry not found"
        src = src.replace(
            '"gfx942": 65536,',
            '"gfx90a": 65536,  # CDNA2 MI210 — same LDS as CDNA3\n    "gfx942": 65536,',
            1,
        )
        with open(path, "w") as f:
            f.write(src)
        # Clear .pyc cache
        pyc_dir = os.path.join(AITER_ROOT, "flydsl/utils/__pycache__")
        for pyc in glob.glob(os.path.join(pyc_dir, "smem_allocator*.pyc")):
            os.remove(pyc)
        print(f"Site 1: added gfx90a to SMEM_CAPACITY_MAP + cleared .pyc")
        return True

    if action == "revert" and has_gfx90a:
        src = src.replace(
            '    "gfx90a": 65536,  # CDNA2 MI210 — same LDS as CDNA3\n', "", 1
        )
        with open(path, "w") as f:
            f.write(src)
        print("Site 1: reverted gfx90a from SMEM_CAPACITY_MAP")
        return True

    return False


def _site2_dispatch(action: str) -> bool:
    """Patch or revert the W4A8 dispatch case in compile_flydsl_moe_stage1."""
    path = os.path.join(AITER_ROOT, "aiter/ops/flydsl/moe_kernels.py")
    with open(path) as f:
        src = f.read()

    marker = 'a_dtype == "int8" and b_dtype == "int4"'
    has_case = marker in src

    if action == "check":
        return has_case

    if action == "apply" and not has_case:
        old_else = '    else:\n        raise ValueError(\n            f"Unsupported stage1 dtype combination'
        assert old_else in src, "Site 2: dispatch else clause not found"

        w4a8_case = (
            '    elif a_dtype == "int8" and b_dtype == "int4":\n'
            "        # W4A8: int8 activations + int4 weights + int8 MFMA\n"
            "        from .kernels.moe_gemm_2stage import compile_moe_gemm1\n"
            "        _use_cshuffle = None if k_batch > 1 else False\n"
            "        return compile_moe_gemm1(\n"
            "            model_dim=model_dim,\n"
            "            inter_dim=inter_dim,\n"
            "            experts=experts,\n"
            "            topk=topk,\n"
            "            tile_m=tile_m,\n"
            "            tile_n=tile_n,\n"
            "            tile_k=tile_k,\n"
            "            doweight_stage1=doweight_stage1,\n"
            '            in_dtype="int4",\n'
            "            group_size=32,\n"
            "            out_dtype=out_dtype,\n"
            "            use_cshuffle_epilog=_use_cshuffle,\n"
            "            scale_is_bf16=True,\n"
            "            k_batch=k_batch,\n"
            "        )\n"
        )
        src = src.replace(old_else, w4a8_case + old_else, 1)
        with open(path, "w") as f:
            f.write(src)
        # Clear .pyc
        pyc_dir = os.path.join(AITER_ROOT, "aiter/ops/flydsl/__pycache__")
        for pyc in glob.glob(os.path.join(pyc_dir, "moe_kernels*.pyc")):
            os.remove(pyc)
        print(f"Site 2: added W4A8 dispatch case + cleared .pyc")
        return True

    if action == "revert" and has_case:
        # Remove the inserted elif block (from marker to the next '    else:')
        lines = src.split("\n")
        out = []
        skip = False
        for line in lines:
            if marker in line:
                skip = True
                continue
            if skip and line.strip().startswith("else:") and line.startswith("    else:"):
                skip = False
            if not skip:
                out.append(line)
        with open(path, "w") as f:
            f.write("\n".join(out))
        print("Site 2: reverted W4A8 dispatch case")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revert", action="store_true", help="Undo the patches")
    group.add_argument("--check", action="store_true", help="Check if patches are applied")
    args = parser.parse_args()

    action = "revert" if args.revert else ("check" if args.check else "apply")

    s1 = _site1_smem_map(action)
    s2 = _site2_dispatch(action)
    print(f"\nSite 1 (SMEM_CAPACITY_MAP): {'applied' if s1 else 'not applied'}")
    print(f"Site 2 (W4A8 dispatch):     {'applied' if s2 else 'not applied'}")

    if action == "apply" and not (s1 and s2):
        print("\nWARNING: not all sites patched successfully", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
