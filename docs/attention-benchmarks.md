# ASM attention on gfx90a — measured

**Date**: 2026-07-27 · **Hardware**: AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e)
**Software**: ROCm 7.14.0, PyTorch 2.11, amd-aiter 0.1.17
**Reproduce**: `python benchmarks/bench_attention_gfx90a.py`

Every number below was produced by a backend that **passed a correctness check
against a reference immediately before being timed**. This is deliberate: docs 16
and 17 published ASM throughput for gfx90a that was actually the CK or Triton
fallback, because `mha.py` gated ASM to gfx942/gfx950. The harness reports a
fast-but-wrong backend as `WRONG` rather than as a result, and `--require-asm`
in the correctness tests fails if the ASM code object never loads.

MI210 theoretical peaks for reference: **181 TFLOP/s** bf16 matrix, **1638 GB/s**
HBM2e.

---

## Prefill — varlen packed THD, bf16, head_dim 128, 32 Q heads / 4 KV heads

ASM is `fmha_v3_fwd` via `aiter.flash_attn_varlen_func`; torch is
`F.scaled_dot_product_attention` on the same data viewed as a dense batch.

| seqs × len | causal | ASM µs | ASM TFLOP/s | torch µs | torch TFLOP/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 1 × 2048 | no | 800 | **85.9** | 1175 | 58.5 | **1.47×** |
| 1 × 2048 | yes | 590 | 58.3 | 748 | 45.9 | 1.27× |
| 4 × 1024 | no | 812 | **84.6** | 1513 | 45.4 | **1.86×** |
| 4 × 1024 | yes | 565 | 60.8 | 956 | 35.9 | 1.69× |
| 8 × 2048 | no | 7028 | 78.2 | 9581 | 57.4 | 1.36× |
| 8 × 2048 | yes | 4006 | 68.6 | 5679 | 48.4 | 1.42× |
| 16 × 1024 | no | 3282 | 83.8 | 5787 | 47.5 | **1.76×** |
| 16 × 1024 | yes | 2116 | 64.9 | 3639 | 37.8 | 1.72× |
| 1 × 8192 | no | 13311 | 82.6 | 15026 | 73.2 | 1.13× |
| 1 × 8192 | yes | 6116 | **89.9** | 8332 | 66.0 | 1.36× |

**ASM prefill is 1.13–1.86× faster than PyTorch SDPA**, peaking at **89.9
TFLOP/s** — about 50% of the card's bf16 matrix peak. `rel_rms` is identical to
torch's own error against the fp32 reference (4.8e-5 … 6.0e-4), so the speedup
costs no accuracy.

### The Triton prefill path does not work on gfx90a at all

`aiter.ops.triton.attention.mha.flash_attn_varlen_func` fails with
`KeyError: 'default'` for every shape. The cause is not a missing tuning entry
but a missing *architecture*: `aiter/ops/triton/configs/` contains only
`gfx942-*.json` and `gfx950-*.json` files — **there are no gfx90a configs
anywhere**, so the config lookup cannot even fall back to a default.

This is worth stating plainly because it inverts the usual framing: for these
shapes ASM is not merely the faster prefill path on an MI210, it is the *only
working* one among the two aiter offers. (ATOM's `unified_attention` is a
different Triton kernel and does run — see the ATOM section of doc 19.)

---

## Decode — paged KV cache, bf16, head_dim 128, block_size 16

ASM is `aiter.pa_fwd_asm`; HIP is `aiter.paged_attention_rocm`, which takes the
same vLLM cache layout (only the ASM path wants V shuffled).

| seqs × ctx | GQA | ASM µs | ASM GB/s | HIP µs | HIP GB/s | speedup |
|---|---|---:|---:|---:|---:|---:|
| 1 × 1024 | 8 | 88 | 24 | 89 | 24 | 1.00× |
| 32 × 1024 | 8 | 93 | 726 | 100 | 669 | 1.09× |
| 128 × 1024 | 8 | 267 | **1006** | 453 | 593 | **1.70×** |
| 32 × 4096 | 8 | 255 | **1052** | 439 | 611 | **1.72×** |
| 32 × 1024 | 16 | 88 | 190 | 91 | 184 | 1.03× |
| 128 × 1024 | 16 | 94 | 714 | 93 | 718 | 0.99× |
| 32 × 4096 | 16 | 91 | 734 | 93 | 722 | 1.02× |

**The ASM advantage is concentrated exactly where serving needs it**: large
batch or long context, where the kernel becomes bandwidth-bound. At 128 requests
× 1024 tokens and at 32 × 4096 it is **~1.7× faster**, sustaining **>1 TB/s** —
about 64% of HBM2e peak, against 36% for the HIP kernel.

Below roughly 700 GB/s the two are equal because both are launch-latency bound
(~88 µs floor); at batch 1 neither saturates anything and the comparison is
meaningless. GQA 16 shows little benefit at these shapes — the extra Q heads per
KV head already give the HIP kernel enough work to hide latency.

---

## What this does not measure

- **End-to-end serving throughput.** These are kernel microbenchmarks. Whether
  ATOM or vLLM actually reaches these kernels is a separate question — see the
  dispatch analysis in doc 19.
- **fp8 / int8 paths.** CDNA2 has no FP8 ALU, so they do not exist here.
- **Multi-GPU.** Single MI210 only; the pair has no xGMI bridge.
- **GEMM.** The missing-gfx90a-config problem that breaks Triton prefill also
  affects every tuned GEMM CSV under `aiter/configs/`, which have no gfx90a rows
  either. That is likely a larger end-to-end effect than attention and is not
  benchmarked here.
