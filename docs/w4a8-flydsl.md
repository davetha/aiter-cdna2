# W4A8 FlyDSL kernel on gfx90a: hard ISA wall (K=32 int8 MFMA not selectable)

**Date**: 2026-08-06 (initial) · **Corrected**: 2026-08-07 (GCN ISA dump + instrumentation)
**Hardware**: 2× AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each
**Software**: ROCm 7.14.0, amd-aiter 0.1.19 (FlyDSL 0.2.4), Python 3.14

> **⚠️ This document supersedes its earlier version.** The prior "alternating
> zeros / K=32 silently runs as K=16" diagnosis was an artifact of a **stale
> `~/.flydsl` compile cache**. Dumping the actual generated GPU assembly and
> instrumenting the kernel generator showed the real blocker is a **hard ISA
> selection failure**, detailed below. The earlier byte-reordering experiments
> (`_pack_i32_pair_to_i64` swaps) were operating on a cached binary and are not
> meaningful — they are preserved in git history only.

## TL;DR

W4A8 (int8 activation × int4 weight) via AITER's FlyDSL `moe_gemm_2stage.py`
**cannot run on gfx90a**. The int8 compute path emits the **K=32 int8 MFMA**
(`v_mfma_i32_16x16x32_i8`), an **MI300-only (gfx940/942) instruction that does
not exist on gfx90a**. LLVM refuses to select it:

```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.mfma.i32.16x16x32.i8
```

gfx90a only has the **K=16** int8 MFMA (`v_mfma_i32_16x16x16_i8`). The K=16
primitive *exists* in FlyDSL's `rocdl` and assembles on gfx90a — so a K=16 int8
rewrite is *theoretically* possible — but the entire W4A8 data path (operand
packing, preshuffle layout, K-micro-step tiling) is built around K=32, so this
is a kernel rewrite, not a patch.

## How this was proven (2026-08-07)

FlyDSL exposes an IR/ISA dumper (`FLYDSL_DUMP_IR=1`, `FLYDSL_DUMP_DIR=…`) that
writes each lowering stage including `21_final_isa.s` (real GCN assembly via
`gpu-module-to-binary{format=isa}`). The dump fires at **kernel invocation**
time (lazy JitFunction), so the kernel must actually be run, and the on-disk
cache (`~/.flydsl`) must be cleared to force regeneration. Scripts:
[`build/dump_w4a8_isa.py`](../build/dump_w4a8_isa.py),
[`build/instrument_w4a8.py`](../build/instrument_w4a8.py),
[`build/instr_loops.py`](../build/instr_loops.py),
[`build/instr_mfma.py`](../build/instr_mfma.py),
[`build/test_w4a8_t128.py`](../build/test_w4a8_t128.py).

### Finding 1 — the registered tile (`tile_n=32`) silently produces ZERO compute

`num_acc_n = tile_n // 64` (`moe_gemm_2stage.py:706`, since `n_per_wave =
tile_n//4` and `num_acc_n = n_per_wave//16`). With the tile size used in all
prior tests (`tile_n=32`): `num_acc_n = 0`. The innermost N compute loop
(`for ni in range_constexpr(num_acc_n)`) runs **zero times**, so `mfma_k64` is
never called and the generated kernel contains **no MFMA of any kind** — it is
pure data staging (buffer loads + barriers + address math). Output: all zeros.
The `21_final_isa.s` had zero `v_mfma`; `00_origin.mlir` had zero `rocdl.mfma`
and zero `i32x4` (the MFMA result type).

Instrumentation (`instr_loops.py`) confirmed it directly:
```
[DBGA] int8 else k_unroll= 4 m_repeat= 2 num_acc_n= 0
```

### Finding 2 — with a valid tile (`tile_n=128`), the path crashes before MFMA

Using the int8-valid tile `tile_n=128` (`num_acc_n = 2`, matching the kernel
registry `tile_ns=[128]` for non-fp4) the compute loop finally tries to run —
but hits an `UnboundLocalError: 'a0'`. Cause: the int8 A-load `if/else`
(`moe_gemm_2stage.py:1150`) is missing the `const_expr(...)` wrapper that the
W4A16 branch (`:1043`) has — a copy-paste omission. The int8 path was never
exercised end-to-end. One-line fix (`test_w4a8_t128.py` applies it):

```python
# int8 branch (broken)            vs   W4A16 branch (works)
if (                              if const_expr(
    (a0_prefetch is not None)         (a0_prefetch is not None)
    and (ku == 0) and (mi == 0)       and (ku == 0) and (mi == 0)
):                                ):
```

### Finding 3 — past that fix, LLVM hard-errors on the K=32 instruction

With the `const_expr` fix applied, the int8 path emits its MFMA and LLVM fails
instruction selection on gfx90a:

```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.mfma.i32.16x16x32.i8
```

`moe_gemm_2stage.py:210` hardcodes the K=32 op with no K=16 fallback:
```python
mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) \
               or getattr(rocdl, "mfma_i32_16x16x32_i8", None)
```
and `mfma_k64`'s int8 fallthrough (`:1014`) calls it with i64 operands (8 int8
elements/lane = K=32). gfx90a has no such instruction.

### Finding 4 — the K=16 primitive exists, but the data path does not

`from flydsl.expr import rocdl` exposes `mfma_i32_16x16x16i8` (K=16, 4 int8
elements/lane) — and that instruction **assembles and is selectable on gfx90a**.
So K=16 int8 MFMA is available in principle. But the W4A8 path is built around
K=32 end-to-end: `load_b_pack_k32`, `lds_load_packs_k64` (K=64 = 2×K=32),
`kpack_bytes`, and the preshuffle B layout all assume the K=32 micro-step.
Switching the op alone (tried earlier) changes nothing because the surrounding
data pipeline still feeds K=32-shaped operands.

## What "fixing" W4A8 on gfx90a would require

A real K=16 int8 MoE GEMM in FlyDSL — a kernel-authoring task, not a patch:

1. Add a K=16 int8 compute branch: emit `rocdl.mfma_i32_16x16x16i8` (i32
   operands, 4 int8/lane) instead of the K=32 i64 path.
2. Add a `load_b_pack_k16` (or K-parametrized loader) feeding 4 int8/lane, and
   rework `lds_load_packs_k64` so each MFMA micro-step consumes a K=16 pack.
3. Rework the B **preshuffle layout** for the K=16 lane-to-element mapping.
4. Fix the `const_expr` copy-paste bug and ensure tiles satisfy
   `num_acc_n = tile_n//64 ≥ 1` (i.e. `tile_n ≥ 64`; registry uses 128).
5. Re-run the dequantized-reference correctness gate, then benchmark.

llama.cpp's `MMQ_MFMA` already does K=16 int8 MFMA with int4 weights on gfx90a
(verified) — a working reference for the K=16 data-path pattern. That is the
practical path to int4+int8 compute on this hardware today; see
[`mi210-llamacpp-glm52-6x`](../../mi210-vllm/docs/LLAMACPP-GFX90A.md).

## Relationship to other work

- [`int8-gemm.md`](int8-gemm.md) — INT8 W8A8 GEMM on gfx90a (CK-based, working)
  — also uses the K=16 int8 MFMA, confirming the instruction is usable here.
- [`port-matrix.md`](port-matrix.md) — 242/1422 gfx942 kernels portable.
- The working AITER int4 path on gfx90a is **W4A16** (bf16 activations), via a
  different module (`mixed_moe_gemm_2stage.py`), not the W4A8 path patched here.
