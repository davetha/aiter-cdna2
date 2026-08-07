# W4A8 FlyDSL kernel on gfx90a: compiles and runs, K=32 correctness fail diagnosed

**Date**: 2026-08-06
**Hardware**: 2× AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each
**Software**: ROCm 7.14.0, amd-aiter 0.1.19 (FlyDSL 0.2.4), Python 3.14

## Summary

AITER's FlyDSL kernel generator (`moe_gemm_2stage.py`) supports a W4A8 path
(`in_dtype="int4"`) that unpacks packed int4 weights to int8 in-kernel and runs
them through the INT8 MFMA pipeline — 176 kernel configs for `aint8_wint4` are
registered at import time. Two gates prevented this from being reachable on
gfx90a. Both are one-line patches; see
[`patches/enable_flydsl_w4a8_gfx90a.py`](../patches/enable_flydsl_w4a8_gfx90a.py).

After patching, `flydsl_moe_stage1(a_dtype="int8", b_dtype="int4")` **compiled
and executed** on the MI210, producing correct-shaped output `[M, topk, N]` in
bf16. See [`tests/test_flydsl_w4a8_gfx90a.py`](../tests/test_flydsl_w4a8_gfx90a.py).

## Correctness: FAILS — K=32 MFMA confirmed (2026-08-07)

A dequantized fp32 reference comparison definitively confirmed the K=32 issue:

- **Even-indexed output elements: exactly 0.0000** (all zero).
- **Odd-indexed output elements: non-zero** (0.01–0.08 range).
- **even/odd ratio: 0.000** — perfect alternating zeros.
- **max_abs_diff: 0.414**, **mean_rel_err: 15307%** — garbage, not quantization noise.

The K=32 instruction (`v_mfma_i32_16x16x32_i8`) silently executes on gfx90a as
a K=16 operation, zeroing half the output lanes. This matches the
[`int8-gemm.md`](int8-gemm.md) finding that gfx90a rejects the K=32 assembler
spelling but the FlyDSL-generated code object bypasses the assembler check.

## K=16 adaptation attempted: instruction swap insufficient (2026-08-07)

Swapped all `mfma_i32_16x16x32i8` → `mfma_i32_16x16x16i8` (the K=16 variant
that EXISTS in rocdl and assembles on gfx90a). 22 string references replaced.
**Result: output unchanged** — same alternating zeros, identical values.

Root cause: the K=32 tiling is embedded in **three coordinated sites** in the
FlyDSL kernel template (`moe_gemm_2stage.py`), not just the MFMA instruction:

1. **MFMA instruction** (line ~211): `mfma_i32_16x16x32i8` — the math op.
   Swappable to K=16, but insufficient alone.
2. **Data loading** (line ~744): `load_b_pack_k32(...)` — loads K=32 weight
   elements per pack from the preshuffled layout. The K=16 MFMA only consumes
   the first 16; the rest are silently dropped → alternating zeros.
3. **Data layout** (line ~420): `kpack_bytes = 8` for int4 — tied to the K=32
   packing density.

A proper K=16 adaptation requires **coordinated changes to all three**: a
`load_b_pack_k16` variant (or modified `load_b_pack_k32` that loads K=16), the
K-tiling loop step (32 → 16), and the packing layout. This is FlyDSL kernel
template engineering, not a string replacement.

## What is needed

1. **K=16 data path**: create a `load_b_pack_k16` (or parameterize
   `load_b_pack_k32` with a K-dimension argument). Adjust the K-tiling loop to
   iterate in steps of 16. Adjust `kpack_bytes` for K=16 consumption.
2. **Correctness re-test**: once the data path feeds K=16, re-run the
   dequantized-reference comparison. The alternating zeros should disappear.
3. **Performance**: once correct, benchmark W4A8 prefill vs W4A16.
4. **Reference**: llama.cpp's `MMQ_MFMA` already does K=16 int8 MFMA with int4
   weights on gfx90a (verified from source) — a working reference for the
   K=16 data path pattern.

## Relationship to other work

- [`int8-gemm.md`](int8-gemm.md) — INT8 W8A8 GEMM on gfx90a (CK-based, working).
- [`port-matrix.md`](port-matrix.md) — 242/1422 gfx942 kernels portable.
- llama.cpp's Q4 MMQ already uses INT8 MFMA (`MMQ_MFMA=ON`) with K=16 shapes
  on gfx90a — a working reference for int4+int8 compute on this hardware.
