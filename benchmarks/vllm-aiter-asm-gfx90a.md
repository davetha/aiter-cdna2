# AITER ASM attention under vLLM on gfx90a — measured

**Date**: 2026-07-27 · **Hardware**: AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e, 104 CU)
**Software**: ROCm 7.14.0, PyTorch 2.11, amd-aiter 0.1.17, vLLM 0.23.1.dev1+g9ddef7117
**Models**: Qwen/Qwen3-14B and Qwen/Qwen3-14B-FP8 — 40 layers, 40 Q heads / 8 KV
heads, head_dim 128

**Reproduce**: apply `configs/enable_vllm_aiter_gfx90a.py`, then
`benchmarks/run-vllm-aiter-ab.sh` (drives `bench_vllm_serving.py`).
Supporting probes: `benchmarks/probe_weight_dtypes.py` for what is actually
resident in VRAM, `benchmarks/bench_fp8_block_gemm_gfx90a.py` for the FP8 GEMM.

`asm-attention-gfx90a.md` measured AITER's hand-written ASM attention kernels in
isolation and found them 1.86x faster than PyTorch SDPA at prefill and 1.72x
faster than the HIP kernel at decode. It closed by naming the obvious open
question: *whether vLLM actually reaches these kernels is a separate question.*

This is the answer to that question, in three parts: whether vLLM can reach the
kernels at all (part 1), what they are worth in bf16 serving (part 2), and
whether an FP8 checkpoint can keep them while halving weight memory (part 3).

---

**The short answer: yes, but only on long prompts under concurrency — 1.23x
output throughput at 4096-token prompts, and nothing at all on short ones.**
And the credit does not go where the microbenchmarks suggested: the ASM *decode*
kernel — the one measured at 1.72x in isolation — is worth about 1% here, which
is inside this harness's run-to-run noise. Essentially all of the gain comes
from the prefill side.

| | 128-token prompts | 4096-token prompts |
|---|---|---|
| concurrency 1 | 1.02x | 1.01x |
| concurrency 8 | 1.00x | **1.23x** |
| concurrency 32 | 1.02x | **1.23x** |

Part 3 adds a separate result: an FP8 checkpoint **does** keep the bf16 ASM
attention kernels and **does** keep its memory saving (1.75x on weights, 1.40x
more KV cache), but serves 10–15x slower, because the only FP8 GEMM available on
CDNA2 is an untuned Triton kernel running at 4.6% of the card's peak.

> ⚠️ **Part 3's slowness has since been fixed, and the diagnosis below is
> wrong.** It was not a missing tuning config. gfx90a has no FP8 *decode*
> instruction, so the kernel emulated `e4m3 → fp16` in software — 7,106 of
> 11,997 instructions, against 64 MFMA. A 3-op bit-reinterpret decode takes FP8
> from 2.7 to **29.2 tok/s**, i.e. 0.67–0.85x of bf16 rather than 0.07x. Parts 1
> and 2 (the bf16 ASM A/B) are unaffected and stand as written. See
> [`../docs/21-fp8-block-gemm-gfx90a.md`](../docs/fp8-block-gemm.md).

---

## Part 1 — vLLM could not reach AITER on gfx90a at all

Before any measurement was possible, a gate had to be found and removed. It is
worth describing precisely, because it fails silently and every log line it
produces looks healthy.

`vllm/_aiter_ops.py` documents its own dispatch rule in prose:

> All check functions (`is_*_enabled`) are decorated with `@if_aiter_supported`,
> which verifies: (1) platform is ROCm, (2) device arch is gfx9, and (3) aiter
> library is installed.

gfx90a **is** gfx9, so by that description an MI210 qualifies. The code
implementing the check does something else:

```python
def is_aiter_found_and_supported() -> bool:
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_mi3xx
        return on_mi3xx()          # _ON_MI3XX = gfx942 or gfx950
    return False
```

`on_mi3xx()` is MI300-series only. On an MI210 it is False, so
`if_aiter_supported` short-circuits — and because it returns `None` rather than
`False`, every `is_*_enabled()` query answers falsy. vLLM then drops
`ROCM_AITER_FA` and `ROCM_AITER_UNIFIED_ATTN` from the backend candidate list
and logs, entirely calmly:

```
Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].
```

Nothing in that line says AITER was considered and rejected. Setting
`VLLM_ROCM_USE_AITER=1` changes it not at all. A second, independent
`on_mi3xx()` in `AiterFlashAttentionBackend.supports_compute_capability()`
rejects the backend again even if the first gate is widened.

`configs/enable_vllm_aiter_gfx90a.py` opens this up for **attention only**. It
adds an attention-specific `is_aiter_attention_supported()`, keyed on
`on_gfx9()` — the condition the surrounding documentation already claims — and
re-decorates exactly two checks with it: `is_mha_enabled` (which puts
`ROCM_AITER_FA` back in the candidate list) and `is_shuffle_kv_cache_enabled`
(which routes decode to `pa_fwd_asm`, and is read only by `rocm_aiter_fa.py`).
The `supports_compute_capability` site above becomes `on_gfx9()` too.

**The master `is_aiter_found_and_supported()` is deliberately left on
`on_mi3xx()`.** Widening it instead — which an earlier draft of this work did —
also admits gfx90a to every other AITER op, including
`AiterFp8BlockScaledMMKernel`, an FP8 GEMM on a chip with no FP8 ALU. Pinning
`VLLM_ROCM_USE_AITER_LINEAR=0` hides that, but "it is gated off elsewhere" is
not a safety property: the other gate can be removed later. That is the defect
class PR #8 closed, so it is not reintroduced here.

Measured after the narrow patch, with `VLLM_ROCM_USE_AITER_LINEAR=1` set
explicitly to prove the point:

```
master  is_aiter_found_and_supported : False
attn    is_aiter_attention_supported : True
is_mha_enabled                       : True
is_shuffle_kv_cache_enabled          : True
is_enabled / is_linear_enabled / is_linear_fp8_enabled : None
AiterFp8BlockScaledMMKernel.is_supported(90)  -> (None, 'Only supported on ROCm ... with aiter')
TritonFp8BlockScaledMMKernel.is_supported(90) -> (True, None)
```

The price is AITER's INT8 and bf16 linear paths, which stay unreachable on
gfx90a. No measurement here shows they help, and the 1.23x below came entirely
from attention. Serving throughput is byte-for-byte unaffected by the
narrowing — the full sweep was re-run and matched to within the noise floor
(80.0 vs 79.9 output tok/s at 4096 tokens / concurrency 32).

One consequence worth knowing: with the master gate closed,
`rocm_aiter_ops.is_enabled()` stays falsy on gfx90a, which disables
`fused_rope_kvcache_supported()` — an optional rope + KV-cache fusion inside the
FA backend. That fusion is already unavailable whenever the shuffled KV layout
is on, so the ASM configuration is unaffected. `flash_attn_varlen_func`,
`triton_rope_and_cache` and `paged_attention_common` are undecorated static
methods and work regardless.

**This corrects a premise this work started from.** The expectation was that
vLLM's AITER gates were already `is_on_gfx9()` and no vLLM patching would be
needed. That string does appear at `_aiter_ops.py:1315` — but inside the
docstring quoted above, not in executable code.

### Selecting the backend is a second, separate step

Opening the gate makes `ROCM_AITER_FA` *selectable*, not *selected*. vLLM's
ROCm priority list puts `ROCM_ATTN` first, so the default still avoids AITER:

```
Overriding with ROCM_ATTN out of potential backends:
    ['ROCM_ATTN', 'ROCM_AITER_FA', 'TRITON_ATTN']
```

(`ROCM_AITER_UNIFIED_ATTN` is *not* in that list, because its gate is the
untouched master check rather than `is_mha_enabled`. That backend is a Triton
kernel, not the ASM path, and is not wanted here — but it is a good marker that
the narrowing did what it claims.)

`ROCM_ATTN` reaches no AITER code at all — its prefill is a Triton kernel and
its decode is vLLM's own paged attention. Two further points cost real time
here and are worth recording:

- **`VLLM_ATTENTION_BACKEND` is not a recognised variable in vLLM 0.23.x.**
  Setting it produces `WARNING: Unknown vLLM environment variable detected` and
  is otherwise ignored. Backend choice goes through
  `--attention-config '{"backend":"ROCM_AITER_FA"}'`.
- **The ATOM plugin registers a vLLM *platform* plugin**, which replaces
  `RocmPlatform` wholesale and therefore replaces backend selection. Every run
  here sets `VLLM_PLUGINS=` so that dispatch is vLLM's own.

### Which kernel each path actually reaches

Only `ROCM_AITER_FA` touches the ASM kernels, and only its prefill does so
unconditionally:

| path | prefill | decode |
|---|---|---|
| `ROCM_ATTN` (vLLM default) | Triton | vLLM paged attention |
| `ROCM_AITER_FA` | **`fmha_v3_fwd` ASM** | `torch.ops.aiter.paged_attention_v1` (HIP) |
| `ROCM_AITER_FA` + `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | **`fmha_v3_fwd` ASM** | **`pa_fwd_asm` ASM** |

The ASM decode kernel needs the shuffled KV layout, which is a separate env var
from any `VLLM_ROCM_USE_AITER_*` flag. `VLLM_ROCM_USE_AITER_PAGED_ATTN` does
not select it.

### Proof the ASM kernels load

With `AITER_LOG_LEVEL=info`, from the running server's log (the EngineCore
subprocess inherits the server's stdout, so a plain redirect captures it):

```
[aiter] LoadKernel: _ZN5aiter37fmha_fwd_hd128_bf16_causal_rtna_groupE
    hsaco: .../aiter_meta/hsa//gfx90a/fmha_v3_fwd/MI300/fwd_hd128_bf16_causal_rtna_group.co
[aiter] LoadKernel: _ZN5aiter27pa_bf16_noquant_gqa8_1tg_4wE
    hsaco: .../aiter_meta/hsa//gfx90a/pa/pa_bf16_noquant_gqa8_1tg_4w.co
```

Both paths are `gfx90a/`, i.e. the ported code objects from
`repatch_gfx942_to_gfx90a.py`, not a gfx942 binary loaded by accident.

Note what this does and does not prove. It proves the kernels load and are
reached. It does **not** prove every decode step uses `pa_fwd_asm` — see the
batch-size heuristic below, which is the single most important fact for
interpreting the numbers.

### The ASM decode kernel is gated on batch size, inside AITER

`aiter/ops/attention.py:_should_use_asm_kernel` refuses the ASM path unless

```python
total_heads = num_seqs * num_heads > 2 * cu_num
```

On a 104-CU MI210 serving Qwen3-14B (40 Q heads), that is
`num_seqs > 5.2`, so **the ASM decode kernel is not used below 6 concurrent
requests** no matter how it is configured. This is not a gfx90a quirk; it is
AITER's own heuristic, and it agrees with the microbenchmarks, which found the
HIP and ASM decode kernels equal until the kernel becomes bandwidth-bound.

Any concurrency-1 decode number below is therefore a measurement of the HIP
kernel in both configurations, and should show no difference by construction.

---

## Part 2 — what the kernels are worth end to end

Qwen3-14B bf16 on one MI210, 128 output tokens per request, `ignore_eos`, no
prefix caching. Every request in every cell below succeeded; there are no
dropped requests hiding in these averages.

| prompt | conc | stock ttft | aiter-fa ttft | asm ttft | stock tpot | aiter-fa tpot | asm tpot | stock tok/s | aiter-fa tok/s | **asm tok/s** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 66.3 | 66.1 | 65.9 | 25.29 | 25.01 | 24.89 | 39.1 | 39.5 | **39.7** |
| 128 | 8 | 299.5 | 333.9 | 323.0 | 38.40 | 38.22 | 38.14 | 197.6 | 197.2 | **198.1** |
| 128 | 32 | 959.7 | 1001.5 | 1039.1 | 45.36 | 44.80 | 43.80 | 608.1 | 610.8 | **619.3** |
| 4096 | 1 | 1407.3 | 1362.1 | 1374.8 | 26.23 | 26.02 | 25.95 | 27.0 | 27.4 | **27.4** |
| 4096 | 8 | 5109.4 | 4392.2 | 4469.0 | 117.97 | 95.89 | 93.31 | 50.7 | 61.5 | **62.4** |
| 4096 | 32 | 12501.2 | 10556.3 | 10716.2 | 394.16 | 321.50 | 315.27 | 64.9 | 79.0 | **79.9** |

TTFT and TPOT in ms, mean. Throughput is output tokens/s across the whole case.

### Long prompts under load: 1.23x

The only cells that move are 4096-token prompts at concurrency 8 and 32, and
they move together: **1.23x** output throughput, **1.25x** better TPOT
(394 → 315 ms at concurrency 32), and **14%** lower TTFT (12.5 → 10.7 s).

### Short prompts: nothing

At 128-token prompts the three configurations are within 2% of each other at
every concurrency — which is within run-to-run noise for this harness. A
128-token prefill is too small to occupy an MI210, so a faster attention kernel
has nothing to speed up, and decode at these batch sizes is bound by streaming
28 GB of weights rather than by the attention kernel.

### Concurrency 1: nothing, at either prompt length

Single-stream serving sees no benefit at all (1.01–1.02x). This is expected and
was predictable from the gating described above:

- Decode cannot use `pa_fwd_asm` at all below 6 concurrent requests, because
  `num_seqs * 40 heads` fails AITER's `> 2 * 104 CU` test.
- Prefill *does* use the ASM kernel, but at concurrency 1 it buys only 2.3%
  (TTFT 1407 → 1375 ms at 4096 tokens). The ASM prefill kernel's advantage
  comes from packing several variable-length sequences into one launch; with
  one sequence there is little to pack.

**The consistent story across all six cells is that these kernels pay off only
when the GPU has enough parallel work to expose their advantage.** That is the
same conclusion the microbenchmarks reached — they found the decode kernels
identical below ~700 GB/s and 1.7x apart once bandwidth-bound — now confirmed
at the serving layer.

### Where the gain actually comes from — not the decode kernel

`aiter-fa` and `aiter-fa-asm` differ in exactly one kernel, so the difference
between those two columns is the ASM decode kernel's entire end-to-end
contribution:

| prompt | conc | ASM decode kernel worth |
|---|---:|---:|
| 128 | 32 | +1.4% |
| 4096 | 8 | +1.5% |
| 4096 | 32 | +1.1% |

**About 1% — and a repeat of the same configuration varies by up to 1.0%
(see below), so this is at the edge of measurement noise and cannot be
distinguished from zero.**

The 1.72x decode speedup measured in isolation almost entirely disappears into
everything else a decode step does — the QKV and MLP GEMMs, RMSNorm, sampling,
and the scheduler — none of which the kernel touches. This is the clearest
result here, and it is a caution about reading microbenchmarks as forecasts: a
1.7x kernel was worth ~1% of the workload, if anything.

The direction is at least consistent — `aiter-fa-asm` is ahead in all three
cells where the ASM decode kernel can engage — but proving a 1% effect would
need repeated trials this run does not have.

The remaining ~22% comes from the switch to `ROCM_AITER_FA` itself, which
changes prefill from Triton to the ASM `fmha_v3_fwd` **and** changes decode
from vLLM's paged attention to AITER's HIP `paged_attention_v1`. This
experiment does not separate those two, because isolating the ASM prefill would
mean reverting the aiter ASM enablement and re-running — see the limitations
below.

### Correctness

Every configuration was checked before its throughput was quoted. All three
answer a short factual prompt identically, and all three recover a fact planted
at the start of a ~2,600-token prompt (`RECALL_OK: True`), which a broken
long-context attention kernel would lose.

Outputs are **not** bit-identical across backends — different attention kernels
give slightly different bf16 rounding, and under greedy decoding that
eventually diverges the continuation. On the long probe all three state the
planted code correctly and then continue differently. That is expected, not a
defect; what would be a defect is a wrong or degenerate answer, and there is
none.

### llama.cpp, as a loose reference only

`llama-coder-80b` on the second MI210 — **a different model, so this is context,
not a comparison**: Qwen3-Coder-Next-80B MoE at Q4_K_M with q8_0 KV cache and 8
parallel slots, versus a 14B dense model at bf16 with bf16 KV.

| prompt | conc | llama.cpp 80B-Q4 tok/s | vLLM 14B-bf16 (asm) tok/s |
|---|---:|---:|---:|
| 128 | 1 | 68.3 | 39.7 |
| 128 | 32 | 150.6 | 619.3 |
| 4096 | 1 | 31.4 | 27.4 |
| 4096 | 32 | 44.9 | 79.9 |

The single-stream advantage is llama.cpp's 4-bit weights (an 80B MoE with ~3B
active parameters at Q4 moves far less memory per token than 28 GB of bf16).
The throughput advantage under load is vLLM's continuous batching against
llama.cpp's 8 fixed slots — at concurrency 32 llama.cpp queues, and its TTFT
degrades to 17.9 s (128-token prompts) and 61.9 s (4096-token), against 1.0 s
and 10.7 s for vLLM. None of this says anything about the ASM kernels.

---

## Part 3 — FP8 weights with bf16 ASM attention

The hypothesis worth testing: attention operates on Q/K/V *activations*, not on
weights, so an FP8 checkpoint whose linear layers dequantize to bf16 should
still reach the bf16 ASM attention kernels. That would combine FP8's halved
weight memory with ASM attention at no arithmetic cost, since on CDNA2 FP8 and
bf16 share one 181 TFLOP/s ceiling rather than FP8 being 2x as on CDNA3.

Tested with `Qwen/Qwen3-14B-FP8` — the same architecture as everything above, so
it compares directly. It is **block-quantized** FP8 (`weight_block_size
[128,128]`, dynamic activation scaling, e4m3).

**The attention half of the hypothesis is exactly right. The arithmetic half is
catastrophically wrong: FP8 keeps its memory saving and runs 10–15x slower.**

### Does it load, and is the memory saving real?

It loads and generates coherent text. The memory saving survives in full —
weights are **not** dequantized at load time.

Determined two ways rather than inferred. vLLM's own accounting:

| | weights | KV cache | KV tokens |
|---|---:|---:|---:|
| bf16 | 27.52 GiB | 26.25 GiB | 172,000 |
| FP8 | **15.71 GiB** | **36.77 GiB** | **240,992** |

And by reading the tensors actually resident in VRAM after load
(`benchmarks/` probe, `named_parameters()` walk):

```
torch.float8_e4m3fn        160 tensors     12.30 GiB
torch.bfloat16             164 tensors      2.91 GiB
model.layers.0.self_attn.qkv_proj.weight    torch.float8_e4m3fn  (7168, 5120)
model.layers.0.mlp.gate_up_proj.weight      torch.float8_e4m3fn  (34816, 5120)
```

The projections are genuinely stored as `float8_e4m3fn`; the 2.91 GiB of bf16 is
embeddings, norms and the LM head. Dequantization happens per tile inside the
GEMM, so **1.75x less weight memory and 1.40x more KV cache** — 240,992 tokens
against 172,000.

### Do the ASM attention kernels still engage? Yes

Same proof standard, from the FP8 server's log — identical kernels to the bf16
runs:

```
[aiter] LoadKernel: _ZN5aiter37fmha_fwd_hd128_bf16_causal_rtna_groupE
    hsaco: .../hsa//gfx90a/fmha_v3_fwd/MI300/fwd_hd128_bf16_causal_rtna_group.co
[aiter] LoadKernel: _ZN5aiter27pa_bf16_noquant_gqa8_1tg_4wE
    hsaco: .../hsa//gfx90a/pa/pa_bf16_noquant_gqa8_1tg_4w.co
```

Attention sees bf16 activations regardless of how the weights are stored, so the
ASM path is untouched by quantization. **The mechanism Dave proposed is real.**

### But the FP8 GEMM makes it unusable

| prompt | conc | bf16 asm tok/s | FP8 asm tok/s | FP8 stock tok/s | FP8 vs bf16 |
|---|---:|---:|---:|---:|---:|
| 128 | 1 | 39.7 | 2.7 | 2.7 | **0.07x** |
| 128 | 8 | 198.1 | 19.8 | 19.6 | **0.10x** |
| 4096 | 1 | 27.4 | 2.1 | 2.1 | **0.08x** |

TPOT sits at 373–383 ms in *every* FP8 cell — independent of prompt length and
concurrency — against 25–26 ms for bf16. A per-step cost that does not move with
batch size is the signature of a kernel that is overhead-bound rather than doing
useful work.

Note the FP8 `stock` column: with attention as the only difference, it matches
the ASM column to within 1%. **The 1.23x from Part 2 vanishes entirely under
FP8**, because attention is no longer anywhere near the bottleneck. Getting ASM
attention "for free" alongside FP8 is worth nothing when the GEMM is 12x slower.

### Why: the only available FP8 kernel is an untuned Triton one

vLLM logs its choice:

```
Selected TritonFp8BlockScaledMMKernel for Fp8LinearMethod
WARNING Using default W8A8 Block FP8 kernel config. Performance might be
    sub-optimal! Config file not found at ...device_name=AMD_Instinct_MI210...
```

Every other block-scaled FP8 kernel is unavailable on CDNA2 — DeepGEMM wants
Hopper, CUTLASS and Marlin are CUDA-only, and the AITER one needs AITER's linear
ops (deliberately off here, see below). Triton's is the only candidate, and it
has no tuned configuration for this device.

Timing it directly against the bf16 matmul it replaces, at Qwen3-14B's own
projection shapes (`bench_fp8_block_gemm_gfx90a.py`):

| layer | M | bf16 µs | FP8 µs | FP8 TFLOP/s | slowdown |
|---|---:|---:|---:|---:|---:|
| qkv_proj | 1 | 87.6 | 1134.1 | 0.1 | 12.9x |
| down_proj | 1 | 218.5 | 3663.7 | 0.0 | 16.8x |
| qkv_proj | 32 | 92.9 | 1143.1 | 2.1 | 12.3x |
| down_proj | 32 | 244.3 | 3696.6 | 1.5 | 15.1x |
| qkv_proj | 4096 | 3240.2 | 39030.7 | 7.7 | 12.0x |
| gate_up_proj | 4096 | 15229.1 | 187410.9 | 7.8 | 12.3x |
| down_proj | 4096 | 7719.6 | 88326.7 | 8.3 | 11.4x |

**The FP8 GEMM peaks at 8.3 TFLOP/s — 4.6% of the card's 181 TFLOP/s — while the
bf16 matmul on the same shapes reaches ~96 TFLOP/s.** The 7–17x kernel gap fully
accounts for the 10–15x end-to-end gap; nothing else needs explaining.

The M=1 and M=32 rows are nearly identical (1134 vs 1143 µs), confirming the
kernel is latency-bound at decode shapes and doing essentially no useful work —
which is precisely why serving TPOT was pinned at ~375 ms.

### Verdict

> ⚠️ **Superseded (2026-07-27).** The measurements below stand; the *cause* is
> wrong. It was not the missing tuning config (that was real, and worth ~3x). It
> was that gfx90a has no FP8 **decoder** instruction, so Triton emulates
> `e4m3 -> fp16` at ~29 VALU ops per value — 7,106 of the kernel's 11,997
> instructions are the conversion, against 64 MFMA. A 3-instruction bit
> reinterpretation, exact on every non-NaN e4m3 byte, takes this run from
> **2.7 to 28.9 tok/s (10.7x)** and TPOT from 373 ms to 34 ms, making FP8
> ~0.73x of bf16 rather than 0.07x. See
> [`docs/21-fp8-block-gemm-gfx90a.md`](../docs/fp8-block-gemm.md).

**FP8 weight-only on gfx90a works, keeps its memory saving, and preserves ASM
attention — but is not usable for serving.** Trading 12 GiB of VRAM for a 12x
throughput loss is not a trade anyone wants. The blocker is not the hypothesis,
which held; it is that CDNA2 has no tuned FP8 GEMM in any framework, which is
the same missing-gfx90a-tuning-config problem that breaks Triton prefill
attention and every tuned GEMM CSV under `aiter/configs/`.

This is worth writing down because it closes the line of enquiry cleanly: the
route to FP8 on CDNA2 is **not** blocked by the hardware lacking an FP8 ALU (the
dequant-to-bf16 design sidesteps that exactly as predicted). It is blocked by
kernel tuning. If someone tuned the Triton block-FP8 config for gfx90a — the
warning names the exact missing file — this could plausibly become viable, and
the memory-saving prize is real: 1.75x on weights and 1.40x more KV cache.

### A latent bug found on the way, and fixed

`torch._scaled_mm` is hard-gated below MI300, verified directly on this card:

```
RuntimeError: torch._scaled_mm is only supported on CUDA devices with
compute capability >= 9.0 or 8.9, or ROCm MI300+
```

That did not affect the runs above, because block-quantized checkpoints route to
the block-scaled kernels instead. But **gfx90a reports compute capability 9.0**,
which *passes* the `>= 89` check in the base `TorchFP8ScaledMMLinearKernel`.
Measured on the MI210 before any patch:

```
PerTensorTorchFP8ScaledMMLinearKernel.is_supported(90)    -> (True, None)
ChannelWiseTorchFP8ScaledMMLinearKernel.is_supported(90)  -> (True, None)
RowWiseTorchFP8ScaledMMLinearKernel.is_supported(90)      -> (False, 'requires MI3xx.')
```

Only the RowWise variant refuses. So a **per-tensor or per-channel** FP8
checkpoint selects one of the first two, reaches `torch._scaled_mm`, and dies
inside ATen with an error that explains nothing about the real cause.

This is the rare gate that wants **narrowing**, not widening, so
`enable_vllm_aiter_gfx90a.py` narrows it: on ROCm below MI3xx the kernel now
refuses at selection time, naming CDNA2's missing FP8 ALU and pointing at the
block-quantized alternative. After the patch all four report `False`, while
`TritonFp8BlockScaledMMKernel` still reports `True` — so block-quantized FP8
still runs exactly as measured above, re-verified end to end (same 15.71 GiB,
same Triton kernel selected, coherent output).

No per-tensor FP8 checkpoint was ever run, so the *failure* this prevents
remains a prediction from code reading plus a direct `_scaled_mm` test. What is
measured is the gate's before-and-after answer, shown above.

---

## Method

Three configurations, differing only in attention:

| label | flags |
|---|---|
| `stock` | `VLLM_ROCM_USE_AITER=0` → `ROCM_ATTN` |
| `aiter-fa` | AITER on, `ROCM_AITER_FA`, shuffle layout off → ASM prefill, HIP decode |
| `aiter-fa-asm` | AITER on, `ROCM_AITER_FA`, shuffle layout on → ASM prefill, ASM decode |

The AITER master switch also enables AITER's linear, MoE, RMSNorm and FP8-BMM
ops. All of those are pinned **off** in both `aiter-*` configs, so the only
difference between `stock` and `aiter-fa-asm` is the attention implementation.
Left on, the comparison would have measured "AITER vs not" instead of "ASM
attention vs not" — and AITER's RMSNorm in particular is unvalidated on gfx90a.

`aiter-fa` vs `aiter-fa-asm` is the cleaner of the two contrasts: those two
differ in exactly one kernel, the decode one.

Every server ran with `--no-enable-prefix-caching`, `--max-model-len 8192`,
`--gpu-memory-utilization 0.85`, `--max-num-seqs 64`, on one MI210 with the
second card left running its usual workload. Prompts are built to an exact
token count and are unique per request, so prefill is never a cache hit. Output
length is pinned with `ignore_eos`, so decode work is identical across
configurations. TTFT is the first streamed token; TPOT is the mean inter-token
gap after it.

Exact invocation:

```bash
# ASM prefill + ASM decode
VLLM_PLUGINS= HIP_VISIBLE_DEVICES=0 \
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 \
VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0 \
VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0 \
VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 AITER_LOG_LEVEL=info \
vllm serve Qwen/Qwen3-14B --port 8000 --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --max-num-seqs 64 \
    --attention-config '{"backend":"ROCM_AITER_FA"}'

python benchmarks/bench_vllm_serving.py --label aiter-fa-asm \
    --out results-aiter-fa-asm.json --model Qwen/Qwen3-14B
```

### Two operational notes that cost time

**`vllm serve` leaves an orphan.** Killing the parent leaves a `VLLM::EngineCore`
process whose cmdline does not match `vllm serve`; it holds ~58 GB of VRAM
indefinitely. The next server then dies with `Free memory on device cuda:0
(8.19/63.98 GiB) ... less than desired GPU memory utilization`, which reads like
a configuration problem and is not. Tear down by matching `VLLM::` as well, and
wait on `rocm-smi` actually reporting the memory back rather than on a fixed
sleep.

**HIP device order is the reverse of `rocm-smi`'s.** `HIP_VISIBLE_DEVICES=0` is
the card `rocm-smi` lists as `GPU[1]`. Picking the wrong one lands on the busy
card and fails with the same misleading free-memory error.

---

## What this does not measure

- **ASM prefill in isolation.** The 22-point remainder is the `ROCM_AITER_FA`
  backend as a whole: ASM prefill *and* AITER's HIP paged decode, versus
  Triton prefill and vLLM's paged decode. Separating them would need a run with
  the aiter ASM enablement reverted so `fmha_v3_fwd` falls back to CK, which
  was out of scope here.
- **Prompts beyond 4096 tokens.** `--max-model-len` was 8192. The trend from
  128 to 4096 is strongly increasing, so the 1.23x is likely a floor rather
  than a ceiling for long-context serving.
- **INT8 and MoE.** Untouched here. FP8 dense is covered in part 3; INT8 GEMM
  is `docs/20-int8-gemm-gfx90a.md`.
- **Per-tensor FP8 checkpoints.** Only a block-quantized one was run. The
  `torch._scaled_mm` prediction in part 3 is code reading plus a direct kernel
  test, not an end-to-end measurement.
- **A tuned FP8 config.** Part 3's conclusion is about the *untuned* Triton
  block-FP8 kernel, which is the only one available on CDNA2 today. Whether
  tuning it would close the 12x gap is untested.
- **Multi-GPU.** Single MI210. The pair has no xGMI bridge.
- **Variance.** Each cell is one run of 8–96 requests. `aiter-fa-asm` was
  re-run end to end as a reproducibility check and agreed to within **1.0%**
  at the noisiest cell:

  | prompt | conc | first run | repeat |
  |---|---:|---:|---:|
  | 128 | 1 | 39.7 | 39.7 |
  | 128 | 8 | 198.1 | 199.5 |
  | 128 | 32 | 619.3 | 625.4 |
  | 4096 | 1 | 27.4 | 27.3 |
  | 4096 | 8 | 62.4 | 62.5 |
  | 4096 | 32 | 79.9 | 80.1 |

  So the 1.23x at long context is far outside noise and is real; every
  sub-2% difference in this document is not, and should be read as "no
  measurable difference".
