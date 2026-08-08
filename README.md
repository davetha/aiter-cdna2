# aiter-cdna2

Running AMD's [AITER](https://github.com/ROCm/aiter) hand-written assembly
kernels on **CDNA2 / gfx90a** — MI210, MI250, MI250X.

AITER ships ASM code objects for `gfx942`, `gfx950` and `gfx1250` only. There is
no `gfx90a` directory, and every dispatch path checks the architecture by name,
so on an MI210 the ASM kernels are simply unreachable — including from vLLM,
where `VLLM_ROCM_USE_AITER=1` has no effect whatsoever and the engine silently
serves from a generic fallback.

This repository translates the kernels that can be translated, opens the
dispatch paths that gate them, and provides the tests and benchmarks to prove
which of it actually runs.

Verified on 2× MI210 (gfx90a, 64 GB HBM2e), ROCm 7.14, PyTorch 2.11,
amd-aiter 0.1.17 and 0.1.19.

---

## What actually works, and what does not

**242 of 1,422 gfx942 kernels translate to gfx90a.** That number is a ceiling,
not a starting point, and it was reached by three independent methods that agree
on the identical file set.

The blockers are hardware, not packaging:

| Blocker | Kernels | Why |
|---|---:|---|
| FP8 / BF8 | many | CDNA2 has no FP8 ALU at all. `v_cvt_pk_fp8_f32` does not assemble. |
| `v_mfma_i32_16x16x32_i8` | many | gfx90a has INT8 MFMA, but at K=16 (`16x16x16i8`). The two spellings are disjoint — neither chip accepts the other's. |
| `global_atomic_pk_add_bf16` | 539 | Absent on CDNA2. |
| SMFMAC (sparse), XF32 | some | CDNA3+ only. |
| gfx950-shaped MFMA | some | `16x16x32_bf16` etc. |

**No kernel is blocked by cosmetic differences alone.** That was tested
explicitly, because it is the hypothesis everyone forms first. An early guess
here was that the 51 `bf16gemm` kernels were merely spelled wrong; 49 of them
need `global_atomic_pk_add_bf16`, which the hardware does not have.

CDNA2 also has a **shared arithmetic ceiling** that CDNA3 does not: bf16 and
INT8 both peak at 181 TFLOP/s / 181 TOPS. On CDNA3, FP8 is 2× bf16. So the
quantization strategies that pay off on an MI300 do not necessarily pay off
here — INT8 buys you memory bandwidth, not arithmetic throughput.

## Results

Measured on one MI210, correctness-checked against a reference before timing.

**Flash attention (prefill), bf16, head_dim 128, 32Q/4KV:**

| shape | causal | ASM | PyTorch SDPA | speedup |
|---|---|---:|---:|---:|
| 4 × 1024 | no | 84.6 TFLOP/s | 45.4 | **1.86×** |
| 16 × 1024 | no | 83.8 | 47.5 | 1.76× |
| 1 × 8192 | yes | **89.9** | 66.0 | 1.36× |

Peak 89.9 TFLOP/s is about 50% of the card's bf16 matrix peak.

**Paged attention (decode), bf16, block_size 16:**

| shape | GQA | ASM | HIP | speedup |
|---|---|---:|---:|---:|
| 128 × 1024 | 8 | **1006 GB/s** | 593 | **1.70×** |
| 32 × 4096 | 8 | **1052 GB/s** | 611 | **1.72×** |

The ASM advantage lands exactly where serving needs it — large batch or long
context, where decode becomes bandwidth-bound. Above 1 TB/s is ~64% of HBM2e
peak, against 36% for the HIP kernel.

**Also here:** INT8 GEMM (bit-exact, 4.36× at M=16) and a fix for FP8
block-scaled GEMM that took it from 2.7 to 29.2 tok/s — the cause was a missing
decode instruction, not a tuning problem.

### The Triton path does not work on gfx90a at all

`aiter/ops/triton/configs/` contains only `gfx942-*.json` and `gfx950-*.json`.
There are **no gfx90a configs anywhere**, so the lookup cannot even fall back to
a default and fails with `KeyError: 'default'`. For prefill, ASM is not merely
the faster of AITER's two paths on an MI210 — it is the only working one.

---

## Layout

```
tools/
  repatch_gfx942_to_gfx90a.py   translate ASM code objects gfx942 -> gfx90a
  classify_gfx942_kernels.py    classify every kernel by what blocks it
patches/
  enable_gfx90a_asm_paths.py    open AITER's own gfx90a dispatch (~16 sites)
  enable_vllm_aiter_gfx90a.py   let vLLM route ATTENTION to AITER on gfx90a
  enable_aiter_ck_gemm_gfx90a.py  let vLLM route LINEAR to the CK int8 GEMM
  enable_fast_fp8_dequant_gfx90a.py   FP8 e4m3 decode fix
  prefer_aiter_fa_gfx90a.py     make ROCM_AITER_FA selectable, not just admitted
  skip_fp8_tune_instances_gfx90a.py   drop FP8 instances before a tuning run
build/
  build_vllm_aiter_gfx90a.sh    end-to-end patched vLLM image
tests/         correctness, with --require-asm to fail if ASM never loads
benchmarks/    every number above is reproducible from these
docs/          the full derivation for each result
```

## Quick start

```bash
# 1. translate what can be translated (prints a TALLY; expect OK=242)
python3 tools/repatch_gfx942_to_gfx90a.py \
    /path/to/site-packages/aiter_meta/hsa/gfx942 ./gfx90a
cp -r ./gfx90a /path/to/site-packages/aiter_meta/hsa/gfx90a

# 2. open AITER's dispatch paths
python3 patches/enable_gfx90a_asm_paths.py

# 3. if you are serving with vLLM, open its gate too
python3 patches/enable_vllm_aiter_gfx90a.py

# 3b. ATTENTION ONLY is what step 3 opens. For an int8 (W8A8) checkpoint the
#     linear layers stay on vLLM's generic Triton kernel until this runs too --
#     silently, with no log line to distinguish the two. Worth 2.9-3.5x decode.
python3 patches/enable_aiter_ck_gemm_gfx90a.py
python3 patches/enable_aiter_ck_gemm_gfx90a.py --check

# 4. prove it
python3 tests/test_fmha_v3_fwd_asm_gfx90a.py --require-asm
python3 benchmarks/bench_attention_gfx90a.py
```

Or build a patched vLLM image in one step:

```bash
./build/build_vllm_aiter_gfx90a.sh rocm-vllm-aiter-gfx90a:latest v0.1.19
```

---

## Things that will cost you a day if you do not know them

**Translation is proven by the assembler, not assumed.** `repatch` disassembles
every instruction, substitutes the CDNA2 mnemonic, and *re-assembles for
gfx90a*. Anything that fails to assemble, or changes encoding length, is
reported NOT PORTABLE and no file is written. Hand-guessed byte patches are how
you get a kernel that runs at full speed and produces silent garbage — an
earlier attempt patched `v_mfma_f32_16x16x16_bf16` to the `f16` opcode (`D3CD`)
instead of `bf16_1k` (`D3E7`), which is a perfectly valid instruction that
computes the wrong thing.

**Manifests must be pruned with the kernels.** The loader hard-fails on a
missing code object, and kernel selection picks by *shape* from the CSV without
checking the file exists. A manifest that outlives its kernels turns an
unsupported shape into a crash instead of a fallback. `repatch` does this in a
second pass.

**A fast number with no proof of ASM is probably the fallback.** Two documents
in the parent project published gfx90a "ASM" throughput that was actually the CK
or Triton path, because nothing verified which kernel loaded. Every benchmark
here checks correctness against a reference before timing, and reports a
fast-but-wrong backend as `WRONG` rather than as a result. The `LoadKernel:`
line from AITER's C++ runtime — visible with `AITER_LOG_LEVEL=info` — is the
only direct evidence. Note it is written straight to fd 1, so
`contextlib.redirect_stdout` cannot capture it; you need a real `dup2`.

**vLLM's block is not the documented one.** `_aiter_ops.py` says in prose that
it checks for "gfx9", and gfx90a *is* gfx9 — but the code calls `on_mi3xx()`,
which is `gfx942 | gfx950`. Because the decorator returns `None` rather than
`False`, every check answers falsy and vLLM drops the AITER backends from its
candidate list while logging a completely normal-looking
`Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN','TRITON_ATTN']`.

**Widen attention only.** An earlier version of the vLLM patch widened the
master gate, which admits gfx90a to *every* AITER op including an FP8 GEMM on a
chip with no FP8 ALU. "It is gated off elsewhere" is not a safety property.

**Installing AITER can downgrade triton and segfault.** amd-aiter 0.1.19 pulls
triton 3.7.0 over 3.7.1 as a transitive dependency — it does not declare triton
at all — and 3.7.0 dies with SIGSEGV inside `triton/knobs.py` on import. `import
triton` alone still succeeds; only the path AITER takes through
`runtime/autotuner` crashes. Reinstall `triton==3.7.1` afterwards.

**LLVM tool paths vary.** A distro ROCm puts them in `/opt/rocm/llvm/bin`; the
`rocm/vllm` images ship ROCm as a Python wheel with no `/opt/rocm` at all.
`repatch` auto-detects and honours `ROCM_LLVM_BIN`.

**Kill a build, orphan a lock.** AITER's JIT uses an `mp_lock` baton. Killing a
build mid-flight leaves it held and every subsequent run hangs on "waiting for
baton release", which reads exactly like a compiler hang. Clear
`aiter/jit/build/lock_module_*` after any kill.

## Related

Derived from work in
[davetha/mi210-llm-stack](https://github.com/davetha/mi210-llm-stack), which has
the full investigation history including the dead ends.

## License

MIT. The AITER kernels themselves are AMD's, under AITER's own license — this
repository contains no AITER source, only tools that transform an existing
installation.
