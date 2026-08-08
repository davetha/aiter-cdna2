# INT8 GEMM on gfx90a: enabling it, and what it is actually worth

**Date**: 2026-07-27
**Hardware**: 2× AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each
**Software**: ROCm 7.14.0, PyTorch 2.11.0, amd-aiter 0.1.17, Python 3.14

`aiter.gemm_a8w8` — AITER's CK-based rowwise-quantized INT8/FP8 GEMM — reported
as unavailable on gfx90a. This document explains why, fixes it, and then
measures the result carefully enough to answer the question that actually
matters: **is INT8 worth using on an MI210?**

The short answer: **yes for decode, no for prefill**, and the reason is a CDNA2
architecture detail that is easy to get wrong.

---

## 1. gfx90a has INT8 matrix hardware. The ASM port matrix does not say otherwise.

[`19-aiter-operator-port-matrix.md`](port-matrix.md) reports
that every INT8 ASM kernel is blocked on gfx90a, and calls the split "physics,
not effort". That is correct **about the pre-compiled ASM blobs**, and it is
easy to misread as a statement about the hardware. It is not one.

Verified against the LLVM assembler:

| Instruction | gfx90a | gfx942 |
|---|---|---|
| `v_mfma_i32_16x16x16i8` | assembles | — |
| `v_mfma_i32_32x32x8i8` | assembles | — |
| `v_mfma_i32_16x16x32_i8` (K=32) | **rejected** | assembles |
| `v_cvt_pk_fp8_f32` | **rejected** | assembles |

So CDNA2 has native INT8 MFMA at its full published rate of **181 TOPS**. What
it lacks is the gfx942 *encoding*: gfx942's INT8 MFMA takes K=32 per
instruction, gfx90a's takes K=16. The two ISAs are disjoint at that opcode,
which is exactly why AITER's prebuilt gfx942 INT8 blobs cannot be binary-patched
the way the bf16 attention kernels were. Recompiling INT8 kernels *from source*
for gfx90a has never been blocked by anything.

FP8 is the genuinely absent capability. `v_cvt_pk_fp8_f32` does not assemble and
there is no FP8 MFMA at all.

## 2. What actually stopped the module from building

Building `module_gemm_a8w8` for gfx90a produces 72 translation units: 9 kernel
tiles × 2 output dtypes (B16, F16) × 2 D dtypes (F32, E) × 2 AB dtypes (I8, F8).

- All **36 INT8** instances compile cleanly, ~30 s each.
- **32 of the 36 FP8** instances compile (slowly, and only by emulating FP8 in
  software — they could never reach FP8 hardware that does not exist).
- **4 FP8 instances never finish.** All four are the same tile,
  `256x224x256x128 ... intrawave_v3`, across `dB16_eB16 / dF16_eF16 /
  dF32_eB16 / dF32_eF16`. Observed at **>17 minutes and ~5 GB RSS** per
  translation unit before being killed, with no sign of converging, against
  ~30 s for every other instance. With no FP8 MFMA to lower to, the backend
  scalarises the matrix operation and the register allocator drowns in the
  resulting live ranges.

Ninja stops at the first failure. Those four hung compiles took down the entire
module — including all 36 working INT8 instances. That is the *only* reason
`aiter.gemm_a8w8` reported unavailable on gfx90a. Nothing about INT8 was ever
the problem.

This looks worth reporting upstream as an LLVM issue: a target with no FP8 MFMA
should diagnose or fall back, not hang the register allocator indefinitely.

## 3. What drives instance generation (and the fix that does not work)

The obvious fix — remove the offending tile from `kernels_list` in
`csrc/ck_gemm_a8w8/gemm_a8w8_common.py` — **has no effect**. Codegen still emits
all 72 instances.

The reason is in the `__main__` block of `csrc/ck_gemm_a8w8/gen_instances.py`:

```python
if args.tune:
    codegen.gen_instances(kernels_list)          # --tune only
else:
    codegen.gen_instances(get_tune_dict(args.tune_file))
```

`kernels_list` (72 entries, keyed by `kernelId`) drives instance generation
**only under `--tune`**, which builds the separate `module_gemm_a8w8_tune`. The
normal build goes through `get_tune_dict()`, which returns
`default_kernels_dict` — 9 entries, negative integer keys — optionally
overlaid with rows from `aiter/configs/a8w8_tuned_gemm.csv` that match the
current target. There are no gfx90a rows in any AITER tuned CSV, so on this
hardware `get_tune_dict()` returns `default_kernels_dict` unchanged, and
`kernels_list` is never consulted at all.

Two further details matter for any fix here:

- **The negative keys are compile lists, not dispatch entries.**
  `write_lookup_header()` skips negative-int keys in non-tune mode, so
  `default_kernels_dict` decides only *which kernels get compiled*. Untuned
  shapes are routed by `rowwise_heuristic_dispatch()` in `gemm_a8w8.cu`, which
  names its nine kernels as string literals.
- **Therefore you cannot simply drop a tile from `default_kernels_dict`.** The
  heuristic's fallback branch names the `256x224x256x128` kernel directly, so
  removing it from the dict deletes the manifest declaration the heuristic
  needs, and the module fails to compile for a new reason.

## 4. The fix

Rather than excise one tile, drop the FP8 half of the cross product entirely on
gfx90a. This halves the build (36 instances, not 72), removes all four hangs,
and is the honest description of the hardware — the 32 FP8 instances that *do*
compile are software emulation that can never reach FP8 silicon, so shipping
them would mean silently running an emulated path on a GPU with no FP8 unit.

Four coupled sites, all keyed off `get_gfx()` at codegen time, added to
[`../patches/enable_gfx90a_asm_paths.py`](../patches/enable_gfx90a_asm_paths.py):

1. `gen_instances.py` imports — derive `AB_DTYPES` and `FP8_GUARD_DEFINE`.
2. `gen_instance()` — iterate `AB_DTYPES` instead of the literal `["I8", "F8"]`.
3. `gen_manifest_head()` — emit `#define AITER_A8W8_NO_FP8 1` into the generated
   `gemm_a8w8_manifest.h`.
4. `gemm_a8w8.cu` — read that define to compile out the FP8 dispatch block,
   which would otherwise reference template instantiations that no longer
   exist. It becomes a `TORCH_CHECK(false, ...)` naming the real reason.

On a non-gfx90a target all four are no-ops, so the patch is safe to leave
applied. As with every patch in that script, each rewrite asserts on its
expected match count.

Result: **`module_gemm_a8w8` builds in ~170–190 s** (48-core EPYC 74F3) and
`aiter.gemm_a8w8` imports and runs. FP8 input now fails loudly with a message
naming the cause instead of dispatching into kernels that are not in the build.

## 5. Correctness

INT8 × INT8 → INT32 accumulation is exactly representable in float64, and MI210
has full-rate FP64, so `a.double() @ b.double().T` is the **exact** integer
answer, not an approximation. The kernel output is compared against the
correctly-rounded cast of that exact value into the output dtype.

Every shape tested is **bit-exact** (`torch.equal`), for both bf16 and fp16
output, with unit scales and with non-trivial rowwise scales:

| Shape (M×N×K) | bf16 out | fp16 out |
|---|---|---|
| 4096³ | bit-exact | bit-exact |
| 8192³ | bit-exact | bit-exact |
| 2048×4096×4096 | bit-exact | bit-exact |
| 1024×8192×8192 | bit-exact | bit-exact |
| 16×8192×8192 | bit-exact | bit-exact |

Rowwise scale path, 1024³ with random per-row/per-column scales:
`max_rel = 0.0039`, `mean_rel = 0.0014` — consistent with bf16 output rounding
(bf16 eps = 2⁻⁸ = 0.0039) and nothing more.

One caveat worth stating: with unit scales and K ≥ 4096, exact products reach
~10⁸ and **overflow fp16's 65504 range**. The kernel returns `inf`, which is the
correctly-rounded answer, but any real use of the fp16 output path needs scales
that keep results in range. Re-tested at scale 2⁻⁹ the fp16 path is finite and
bit-exact.

## 6. Performance — and the CDNA2 detail that decides everything

**On CDNA2, INT8 and BF16 have the same peak: 181 TOPS and 181 TFLOPS.** This is
not true of CDNA3, where INT8 is 2× FP16. So on an MI210, INT8 buys no
arithmetic throughput at all. Its only advantage is that the operands are half
the bytes.

That single fact explains every number below. Measured with the default
(untuned) kernel selection — there are no gfx90a rows in any AITER tuned CSV, so
the log line `not found tuned config ... will use default config!` is expected.

| M | N | K | INT8 | vs 181 peak | BF16 (torch/hipBLASLt) | speedup |
|---:|---:|---:|---|---|---|---|
| 4096 | 4096 | 4096 | 102.4 TOP/s | 57% | 95.5 TFLOP/s | 1.07× |
| 8192 | 8192 | 8192 | 95.8 TOP/s | 53% | 94.9 TFLOP/s | 1.01× |
| 2048 | 4096 | 4096 | 87.9 TOP/s | 49% | 93.5 TFLOP/s | 0.94× |
| 1024 | 8192 | 8192 | 91.2 TOP/s | 50% | 88.2 TFLOP/s | 1.03× |
| 16 | 8192 | 8192 | 38.8 TOP/s | 21% | 8.9 TFLOP/s | **4.36×** |

**Compute-bound shapes: INT8 is a wash.** It lands at 49–57% of the 181 TOPS
ceiling — but bf16 via hipBLASLt lands at 49–53% of the *same* 181 TFLOPS
ceiling. The untuned CK INT8 kernel is already at parity with, or slightly ahead
of, the vendor-tuned bf16 library in efficiency terms. There is no headroom to
win here because both paths share one ceiling.

**Memory-bound shapes: INT8 is a large win.** At M=16, N=K=8192 the weight
matrix dominates: 134 MB in bf16, 67 MB in INT8. Against MI210's 1638 GB/s HBM
bandwidth the floors are 82 µs and 41 µs. Measured 242 µs (34% of bandwidth) and
55 µs (**73% of bandwidth**). So INT8 wins 4.36×, not 2× — half the bytes, moved
by a much better kernel.

### Is a tuning run worth doing?

**Low priority.** On compute-bound shapes the ceiling is shared with bf16, which
already nearly saturates it, so even a perfect tune buys perhaps 20–25% on GEMMs
that are not the bottleneck anyway. On the memory-bound shapes where INT8
actually matters, the kernel is already at 73% of peak HBM bandwidth, leaving
little to recover. The effort is better spent elsewhere.

### What this means for the LLM stack

- **Decode (memory-bound, small M):** INT8 weights are worth real throughput on
  MI210 — 4.36× on the shape measured, from halved weight traffic plus a
  bandwidth-efficient kernel.
- **Prefill (compute-bound, large M):** INT8 is roughly neutral. Do not expect
  the 2× that CDNA3 INT8 delivers; CDNA2 does not have it to give.
- Any quantization plan for this hardware should be justified by **memory
  traffic and capacity**, not by arithmetic throughput.

## 7. Reproducing

```bash
python patches/enable_gfx90a_asm_paths.py            # applies all sites
python patches/enable_gfx90a_asm_paths.py --check    # verify
rm -rf  <site-packages>/aiter/jit/build/module_gemm_a8w8
rm -f   <site-packages>/aiter/jit/module_gemm_a8w8.so
python -c "import torch, aiter; a=torch.randint(-127,128,(4096,4096),dtype=torch.int8,device='cuda'); \
  print(aiter.gemm_a8w8(a,a,torch.ones(4096,1,device='cuda'),torch.ones(1,4096,device='cuda'),dtype=torch.bfloat16).shape)"
```

Benchmark and validation harness: [`../benchmarks/bench_int8_gemm_gfx90a.py`](../benchmarks/bench_int8_gemm_gfx90a.py).

If a build is interrupted, remove **both** `aiter/jit/build/lock_module_gemm_a8w8`
(otherwise the next run deadlocks waiting for baton release) and
`aiter/jit/build/module_gemm_a8w8` (otherwise codegen does not re-run).

## 8. Getting vLLM to actually call it

Everything above is the kernel in isolation. vLLM will not select it on gfx90a
without [`../patches/enable_aiter_ck_gemm_gfx90a.py`](../patches/enable_aiter_ck_gemm_gfx90a.py),
and the failure is silent — the engine serves correct output from
`TritonInt8ScaledMMLinearKernel` and nothing in the log says the fast kernel was
skipped. Two independent blockers:

**1. Codegen.** `GFX_CU_NUM_MAP` in `aiter/jit/utils/build_targets.py` has
gfx942/gfx950/gfx1250 only. The lookup raises before instance generation runs,
so the build dies later on a missing `gemm_a8w8_manifest.h`. Note the shape of
this bug: the table is consulted **even when the tuned CSV has no rows for your
architecture**, so an arch that would otherwise fall back to default kernels
gets a hard error instead of the fallback. MI210 reports 104 CUs.

**2. Selection.** `AiterInt8ScaledMMLinearKernel.is_supported()` gates on
`rocm_aiter_ops.is_linear_enabled()`, which carries `@if_aiter_supported` →
`is_aiter_found_and_supported()` → `on_mi3xx()` = gfx942|gfx950.
`enable_vllm_aiter_gfx90a.py` deliberately narrows its carve-out to **attention
only**, so linear stays blocked by this project's own patch as well as by
upstream's.

Both are necessary; neither is sufficient. Serving also needs
`VLLM_ROCM_USE_AITER=1` **and** `VLLM_ROCM_USE_AITER_LINEAR=1` — a run with the
patches and without the flags is measuring Triton.

The one log line that distinguishes them:

```
INFO [__init__.py:670] Selected AiterInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

Grep for `Selected .*ScaledMMLinearKernel` before trusting any W8A8 number.

### Measured, end to end

Qwen3.6-27B W8A8 (int8 channel weights, int8 per-token activations), 1× MI210,
bf16, `--attention-backend ROCM_AITER_FA`. Same server and flags, only the
kernel differs:

| depth | tg256 Triton | tg256 AITER | speedup |
|---|---:|---:|---:|
| 0 | 10.12 | 35.62 | 3.5× |
| 8192 | 10.04 | 33.42 | 3.3× |
| 16384 | 9.92 | 31.35 | 3.2× |
| 32768 | 9.59 | 28.34 | 3.0× |

Prefill gains too: 907.7 → 1,055.5 t/s at d32768 (+17%), since the same linear
layers carry prefill. This is larger than §4's isolated kernel ratio because the
Triton fallback also costs scheduler overhead the microbenchmark never sees.

### Tuning it is a null result

§6 notes there are no gfx90a rows in any AITER tuned CSV. Tuning them does not
help at decode shapes. `gemm_a8w8_tune.py --compare --update_improved` over the
27B's four linear shapes at M ∈ {1, 2, 8} — 12 shapes, 1,032 candidates:

```
Total shapes: 12 | Updated: 0 (improved: 0, new: 0) | Skipped: 12
```

Every shape between **−1.57% and +0.07%**. The reason is visible in the timings:
M=1 and M=8 on the same shape take 30.60 vs 33.64 µs — 8× the arithmetic for 10%
more time. These GEMMs sit at the bandwidth roofline, and tile-config tuning
schedules *compute*. Consistent with §6's conclusion that CDNA2 INT8 wins on
memory traffic, not arithmetic.

Cheap pre-check before committing a tuning round: time the kernel at M=1 and
M=16. If they are within ~10%, the shape is latency-bound and tuning will not
move it.
