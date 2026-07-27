"""Enable AITER's bf16 ASM fast paths on gfx90a (MI210 / CDNA2).

AITER ships hand-written ASM kernels for `fmha_v3_fwd` (flash-attention forward)
and MLA, but gates them to gfx942/gfx950. The gate is not a hardware statement --
`aiter/ops/mha.py` even calls the kernels "hand-written gfx9 ASM", and gfx90a is
gfx9. AMD simply never built or validated gfx90a binaries.

Of the 56 `fmha_v3_fwd` kernels, 48 (every bf16 one) are provably portable to
gfx90a; only the 8 FP8 kernels are not, because CDNA2 has no FP8 ALU. Once
`repatch_gfx942_to_gfx90a.py` has populated `hsa/gfx90a/`, the remaining blockers
are five literal architecture-string comparisons.

This script rewrites those five sites. It is idempotent, and every rewrite
asserts on its expected match count, so an upstream change that moves the code
fails loudly here instead of silently leaving the fast path disabled.

    python enable_gfx90a_asm_paths.py [--revert] [--check]

Sites patched
-------------
Python, `aiter/ops/mha.py` -- two copies of `can_impl_fmha_v3_fwd` (batched and
varlen). Each is widened to admit gfx90a and then explicitly re-narrowed to
bf16, so an FP8 tensor can never reach a kernel CDNA2 cannot execute.

C++, `aiter_meta/csrc/cpp_itfs/mha_fwd.cu` -- three sites:
  * `get_kernel_name_key`  picks a kernel config only for gfx942/gfx950
  * `get_kernel_co_name`   inserts the `MI300/` or `MI308/` product subdirectory
  * `init_fmha_fwd_v3_args` applies an hdim 192x128 tuning override

Python + C++, `csrc/ck_gemm_a8w8/` -- four sites that together stop the CK
INT8/FP8 GEMM module from generating FP8 kernel instances on gfx90a. CDNA2 has
native INT8 MFMA but no FP8 at all, and four of the FP8 instances hang the
compiler outright, taking the whole module (including all 36 working INT8
instances) down with them. See the block comment on those patches.

gfx90a is treated exactly like gfx942 in all three fmha sites, which is correct because the
gfx90a `.co` files are byte-derived from the gfx942 ones: same kernel names, same
config table, same kernarg layout. The MI300 branch is the right one -- MI210's
PCI chip id (0x740f) is not in `MI308_CHIP_IDS`, so `is_mi308_device()` is false.

After running this, the affected JIT modules must be rebuilt so they pick up the
new C++ (delete `aiter/jit/module_fmha_v3_fwd.so` and its build dir, then call
the op). A stale module whose kernarg layout predates the installed `.co` files
is exactly how `pa_fwd_asm` appeared broken for weeks -- see
docs/18-pa-fwd-asm-resolved.md.

Not covered here: the MFMA wait-state hazard. V_MFMA_F32_16X16X16* is 4-pass on
gfx942 but 8-pass on gfx90a, so ported code can be short on wait states at an
MFMA->consumer edge. That is a per-kernel correctness question and must be
settled by numerical validation, not by this patch.
"""
import argparse
import os
import sys

SITE = "/opt/python/lib/python3.14/site-packages"
MHA_PY = f"{SITE}/aiter/ops/mha.py"
MHA_CU = f"{SITE}/aiter_meta/csrc/cpp_itfs/mha_fwd.cu"
A4W4_PY = f"{SITE}/aiter/ops/gemm_op_a4w4.py"
QUANT_PY = f"{SITE}/aiter/ops/quant.py"
CORE_PY = f"{SITE}/aiter/jit/core.py"
GEMM_GEN_PY = f"{SITE}/aiter_meta/csrc/ck_gemm_a8w8/gen_instances.py"
GEMM_CU = f"{SITE}/aiter_meta/csrc/ck_gemm_a8w8/gemm_a8w8.cu"

MARK = "gfx90a-asm-enable"

# (path, original, replacement, expected occurrences)
PATCHES = [
    # NOTE: the bare `ret = get_gfx() in ("gfx942", "gfx950")` line occurs FOUR
    # times in mha.py -- twice in `can_impl_fmha_v3_fwd` (batched and varlen) and
    # twice in `is_fmha_v3_fp8`. Only the first pair may be widened: admitting
    # gfx90a into `is_fmha_v3_fp8` would route FP8 tensors at a CDNA2 GPU that
    # has no FP8 ALU. Anchor on the preceding comment, which is unique to each.
    (
        MHA_PY,
        '        # fmha v3 is hand-written gfx9 ASM; non-gfx9 must fall back to ck-tile.\n'
        '        ret = get_gfx() in ("gfx942", "gfx950")\n',
        '        # fmha v3 is hand-written gfx9 ASM; non-gfx9 must fall back to ck-tile.\n'
        '        # {MARK}: gfx90a is gfx9. Its bf16 kernels are byte-portable from\n'
        '        # gfx942; its FP8 ones are not, so re-narrow to bf16 below.\n'
        '        ret = get_gfx() in ("gfx942", "gfx950", "gfx90a")\n'
        '        ret = ret and not (get_gfx() == "gfx90a" and q.dtype != dtypes.bf16)\n',
        1,
    ),
    (
        MHA_PY,
        '        # ck-tile (mha_varlen_fwd, the else branch below).\n'
        '        ret = get_gfx() in ("gfx942", "gfx950")\n',
        '        # ck-tile (mha_varlen_fwd, the else branch below).\n'
        '        # {MARK}: gfx90a is gfx9. Its bf16 kernels are byte-portable from\n'
        '        # gfx942; its FP8 ones are not, so re-narrow to bf16 below.\n'
        '        ret = get_gfx() in ("gfx942", "gfx950", "gfx90a")\n'
        '        ret = ret and not (get_gfx() == "gfx90a" and q.dtype != dtypes.bf16)\n',
        1,
    ),
    (
        MHA_CU,
        '            else if(arch_id == "gfx942" && cfg.bf16_cvt == bf16_cvt)\n',
        '            // {MARK}: gfx90a uses the gfx942 config table verbatim.\n'
        '            else if((arch_id == "gfx942" || arch_id == "gfx90a") &&\n'
        '                    cfg.bf16_cvt == bf16_cvt)\n',
        1,
    ),
    (
        MHA_CU,
        '    if(arch_id == "gfx942")\n'
        '    {\n'
        "        auto pos = cfg_co_name.rfind('/');\n",
        '    // {MARK}: MI210 chip id 0x740f is not in MI308_CHIP_IDS, so this\n'
        '    // resolves to the MI300/ subdirectory -- the one repatched for gfx90a.\n'
        '    if(arch_id == "gfx942" || arch_id == "gfx90a")\n'
        '    {\n'
        "        auto pos = cfg_co_name.rfind('/');\n",
        1,
    ),
    (
        MHA_CU,
        '    if(a.hdim_q == 192 && a.hdim_v == 128 && arch_id == "gfx942")\n',
        '    // {MARK}: same tuning override applies to the gfx90a port.\n'
        '    if(a.hdim_q == 192 && a.hdim_v == 128 &&\n'
        '       (arch_id == "gfx942" || arch_id == "gfx90a"))\n',
        1,
    ),
    # The load-bearing one. This early-out is written in the NEGATED form, so a
    # grep for `arch_id == "gfx942"` misses it -- and it returns -1 before the
    # kernel-config lookup ever runs, which surfaces as the unhelpful
    # `RuntimeError: invalid argument for fmha_fwd`.
    (
        MHA_CU,
        '       ((arch_id != "gfx942") && (arch_id != "gfx950")))\n',
        '       // {MARK}\n'
        '       ((arch_id != "gfx942") && (arch_id != "gfx950") &&\n'
        '        (arch_id != "gfx90a")))\n',
        1,
    ),
    # Grid-dimension selection for the hdim 192x128 kernels. These must follow
    # the gfx942 shape because the gfx90a .co files ARE the gfx942 kernels, so
    # they expect the same workgroup decomposition.
    (
        MHA_CU,
        '    if(arch_id == "gfx942" && a.is_group_mode && a.hdim_q == 192 && a.hdim_v == 128)\n',
        '    // {MARK}: gfx90a runs the gfx942 kernels, so it needs their grid too.\n'
        '    if((arch_id == "gfx942" || arch_id == "gfx90a") && a.is_group_mode &&\n'
        '       a.hdim_q == 192 && a.hdim_v == 128)\n',
        1,
    ),
    (
        MHA_CU,
        '    if(arch_id == "gfx942" && a.hdim_q == 192 && a.hdim_v == 128)\n',
        '    // {MARK}: gfx90a runs the gfx942 kernels, so it needs their grid too.\n'
        '    if((arch_id == "gfx942" || arch_id == "gfx90a") &&\n'
        '       a.hdim_q == 192 && a.hdim_v == 128)\n',
        1,
    ),

    # ---------------------------------------------------------------------
    # Escape hatches: the INVERSE bug. These guards name only gfx942, so
    # gfx90a *passes* them and walks into FP4 code paths CDNA2 has no hardware
    # for. They are silent -- no warning, no exception at the gate -- so the
    # failure surfaces far from its cause. Widening them is not an
    # optimisation, it is the fallback policy working correctly: gfx90a should
    # be routed away from FP4 exactly as gfx942 is.
    # ---------------------------------------------------------------------
    (
        A4W4_PY,
        '    if gfx_arch in ["gfx942"]:\n',
        '    # {MARK}: CDNA2 has no FP4 hardware either -- refuse, do not proceed.\n'
        '    if gfx_arch in ["gfx942", "gfx90a"]:\n',
        1,
    ),
    (
        QUANT_PY,
        '        and get_gfx() != "gfx942"\n',
        '        # {MARK}: gfx90a has no FP4 ALU; keep it out of the f4 quant path.\n'
        '        and get_gfx() not in ("gfx942", "gfx90a")\n',
        1,
    ),
    (
        CORE_PY,
        '        if get_gfx() != "gfx942" and int(os.getenv("AITER_FP4x2", "1")) > 0:\n',
        '        # {MARK}: do not advertise FP4x2 to the compiler on gfx90a.\n'
        '        if (get_gfx() not in ("gfx942", "gfx90a")\n'
        '                and int(os.getenv("AITER_FP4x2", "1")) > 0):\n',
        1,
    ),

    # ---------------------------------------------------------------------
    # CK a8w8 GEMM: stop building FP8 instances on gfx90a.
    #
    # CDNA2 has native INT8 MFMA (v_mfma_i32_16x16x16i8 / _32x32x8i8, 181
    # TOPS) but no FP8 anything -- v_cvt_pk_fp8_f32 does not even assemble.
    # Only the gfx942 INT8 *encoding* (v_mfma_i32_16x16x32_i8, K=32) is
    # unavailable, not the INT8 capability.
    #
    # All 36 INT8 instances compile cleanly in ~30 s each. Four of the 36 FP8
    # instances do not compile at all: tile 256x224x256x128 ... intrawave_v3,
    # across abF8 x {dB16,dF16,dF32} x {eB16,eF16}. With no FP8 MFMA to lower
    # to, the backend scalarises the matrix op and the register allocator
    # drowns -- observed at >17 minutes and ~5 GB RSS per translation unit
    # before being killed, with no sign of converging. Ninja stops at the
    # first failure, so those four take the whole module down, which is the
    # only reason `aiter.gemm_a8w8` reports unavailable on gfx90a.
    #
    # The other 32 FP8 instances do compile, but only by emulating FP8 in
    # software; they are dead weight that can never reach FP8 hardware. So
    # rather than excise one tile, drop the FP8 half of the cross product
    # entirely: it halves the build (36 instances, not 72) and it is the
    # honest description of the hardware. `gemm_a8w8` then refuses FP8 input
    # with a clear message instead of silently running an emulated path.
    #
    # This is worth reporting upstream as an LLVM issue: a target with no FP8
    # MFMA should diagnose or fall back, not hang the register allocator.
    #
    # Three coupled sites, all keyed off get_gfx() at codegen time:
    #   1. gen_instances.py imports  -- derive AB_DTYPES and FP8_GUARD_DEFINE
    #   2. gen_instance()            -- emit only the AB_DTYPES instances
    #   3. gen_manifest_head()       -- emit AITER_A8W8_NO_FP8 into the
    #                                   generated manifest, which
    #   4. gemm_a8w8.cu              -- reads to compile out the FP8 dispatch
    #                                   (otherwise it would reference template
    #                                   instantiations that no longer exist).
    # On a non-gfx90a target all four are no-ops.
    # ---------------------------------------------------------------------
    # NOTE: this anchor deliberately runs through `class gemm_a8w8_fwd_codegen:`.
    # Anchoring on the import block alone would make the pre-patch text a strict
    # PREFIX of the post-patch text, so `apply()` would count it as both present
    # and already-applied and insert the block again on every run.
    (
        GEMM_GEN_PY,
        'from gemm_a8w8_common import (  # noqa: E402\n'
        '    default_kernels_dict,\n'
        '    kernelInstance,\n'
        '    kernels_list,\n'
        ')\n'
        '\n'
        '\n'
        'class gemm_a8w8_fwd_codegen:\n',
        'from gemm_a8w8_common import (  # noqa: E402\n'
        '    default_kernels_dict,\n'
        '    kernelInstance,\n'
        '    kernels_list,\n'
        ')\n'
        '\n'
        '# {MARK}: see the comment in configs/enable_gfx90a_asm_paths.py.\n'
        '# AITER_CORE_DIR is already on sys.path from the chip_info import above.\n'
        '#\n'
        '# NOTE: instance generation is driven by the dict gen_instances() is\n'
        '# called with -- default_kernels_dict via get_tune_dict() in the normal\n'
        '# build, kernels_list only under --tune. Filtering kernels_list alone has\n'
        '# no effect on a normal build.\n'
        'from chip_info import get_gfx  # noqa: E402\n'
        '\n'
        '_NO_FP8 = get_gfx() == "gfx90a"\n'
        'AB_DTYPES = ["I8"] if _NO_FP8 else ["I8", "F8"]\n'
        'FP8_GUARD_DEFINE = "#define AITER_A8W8_NO_FP8 1\\n" if _NO_FP8 else ""\n'
        '\n'
        '\n'
        'class gemm_a8w8_fwd_codegen:\n',
        1,
    ),
    (
        GEMM_GEN_PY,
        '                for ABDtype in ["I8", "F8"]:\n',
        '                for ABDtype in AB_DTYPES:  # {MARK}\n',
        1,
    ),
    (
        GEMM_GEN_PY,
        '#include <torch/extension.h>\n'
        '"""\n'
        '        MAINFEST_template = """\n',
        '#include <torch/extension.h>\n'
        '""" + FP8_GUARD_DEFINE  # {MARK}\n'
        '        MAINFEST_template = """\n',
        1,
    ),
    (
        GEMM_CU,
        '  else\n'
        '  {\n'
        '    if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)\n',
        '  else\n'
        '  {\n'
        '#ifdef AITER_A8W8_NO_FP8\n'
        '    // {MARK}: CDNA2 has no FP8 MFMA, so codegen did not emit the FP8\n'
        '    // instances. Refuse loudly rather than dispatch into kernels that\n'
        '    // are not in this build.\n'
        '    TORCH_CHECK(false,\n'
        '                "gemm_a8w8: fp8 input is not supported on this build -- gfx90a "\n'
        '                "has no fp8 MFMA, so the fp8 kernel instances are excluded. "\n'
        '                "Use int8 (torch.int8) inputs instead.");\n'
        '#else\n'
        '    if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)\n',
        1,
    ),
    (
        GEMM_CU,
        '      TORCH_CHECK(false, "Unsupported scales/output dtype!");\n'
        '    }\n'
        '  }\n'
        '  return Y;\n',
        '      TORCH_CHECK(false, "Unsupported scales/output dtype!");\n'
        '    }\n'
        '#endif  // {MARK}\n'
        '  }\n'
        '  return Y;\n',
        1,
    ),
]


def expand(text):
    return text.replace("{MARK}", MARK)


def apply(revert=False, check=False):
    ok = True
    for path, old, new, count in PATCHES:
        new = expand(new)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            return False
        with open(path) as fh:
            body = fh.read()

        frm, to = (new, old) if revert else (old, new)
        have = body.count(frm)
        already = body.count(to)

        if check:
            state = "patched" if already else ("clean" if have else "UNKNOWN")
            print(f"{state:>8}  {os.path.basename(path)}  "
                  f"(want {count}, found {have} unpatched / {already} patched)")
            ok = ok and (already == count or have == count)
            continue

        if have == 0 and already >= count:
            print(f"  skip (already applied): {os.path.basename(path)}")
            continue
        if have != count:
            print(f"  FAIL {os.path.basename(path)}: expected {count} occurrence(s) "
                  f"of the target text, found {have}. Upstream source changed -- "
                  f"re-derive this patch rather than forcing it.")
            return False

        with open(path, "w") as fh:
            fh.write(body.replace(frm, to))
        print(f"  {'reverted' if revert else 'patched'} {count}x "
              f"{os.path.basename(path)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ok = apply(revert=args.revert, check=args.check)
    if not args.check and ok and not args.revert:
        print("\nNow rebuild the JIT module so it picks up the new C++:\n"
              "  rm -f  {0}/aiter/jit/module_fmha_v3_fwd.so\n"
              "  rm -rf {0}/aiter/jit/build/module_fmha_v3_fwd\n"
              "then call the op once to trigger the build.".format(SITE))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
