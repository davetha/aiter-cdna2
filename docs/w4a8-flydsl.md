# W4A8 FlyDSL kernel on gfx90a: compiles and runs, correctness unverified

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

## The K=32 / K=16 problem

The FlyDSL int8 path emits `mfma_i32_16x16x32_i8` (K=32 INT8 MFMA). gfx90a only
has `mfma_i32_16x16x16_i8` (K=16). This is documented in
[`int8-gemm.md`](int8-gemm.md) and the [port matrix](port-matrix.md):

| Instruction | gfx90a | gfx942 |
|---|---|---|
| `v_mfma_i32_16x16x16_i8` | assembles | — |
| `v_mfma_i32_16x16x32_i8` (K=32) | **rejected** | assembles |

The test output showed alternating zeros (`[0.0000, 0.0139, 0.0000, ...]`),
which is consistent with the K=32 instruction silently failing on gfx90a and
zeroing half the lanes. **Correctness is unverified.**

## What is needed before trusting this

1. **Correctness comparison**: run identical data through a bf16 reference MoE
   and the W4A8 kernel; compare outputs. If the K=32 instruction is the
   problem, the error will be large (not just quantization noise).
2. **K=16 adaptation**: if K=32 is confirmed as the blocker, the FlyDSL kernel
   needs to be adapted to use `16x16x16_i8` (K=16) instead of `16x16x32_i8`
   (K=32). This is a FlyDSL kernel-template change, not a source patch.
3. **Performance measurement**: once correct, benchmark W4A8 prefill vs W4A16
   to measure the actual INT8 MFMA speedup on gfx90a.

## Relationship to other work

- [`int8-gemm.md`](int8-gemm.md) — INT8 W8A8 GEMM on gfx90a (CK-based, working).
- [`port-matrix.md`](port-matrix.md) — 242/1422 gfx942 kernels portable.
- llama.cpp's Q4 MMQ already uses INT8 MFMA (`MMQ_MFMA=ON`) with K=16 shapes
  on gfx90a — a working reference for int4+int8 compute on this hardware.
