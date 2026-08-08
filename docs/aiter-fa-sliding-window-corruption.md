# ROCM_AITER_FA corrupts sliding-window attention on gfx90a

**Measured 2026-08-08**, 2× MI210 (gfx90a), vLLM `0.26.1rc0+mi210.1`,
image `local/vllm-mi210:dsa7-aiterint8`, amd-aiter 0.1.19.

`ROCM_AITER_FA` produces **silently wrong output** for models whose layers use
sliding-window attention. Same weights, same prompt, same server, greedy
decoding — only `--attention-backend` differs:

| | `ROCM_AITER_FA` | `TRITON_ATTN` |
|---|---|---|
| docstring | `"Reverss a linked list"` | `"Reverse a singly linked list"` |
| indentation | `while` body not indented | correct |
| algorithm | `reversed_head.next = reversed_head` (self-referential) | correct `prev`/`current`/`next_node` walk |
| termination | same block repeats until `max_tokens` | terminates cleanly |

It is not a crash, an assert, or a log line. The server reports healthy, the
benchmark reports excellent throughput, and the tokens are grammatical enough
to skim past. **Only reading the output catches it.**

## Reproduction

Model: `poolside/Laguna-S-2.1-INT4` (symmetric INT4, compressed-tensors).
48 layers, `layer_types` = 36 `sliding_attention` + 12 `full_attention`,
`sliding_window: 512`, `LagunaForCausalLM`.

```bash
# BAD -- corrupted output
vllm serve /models/laguna --attention-backend ROCM_AITER_FA \
  --dtype bfloat16 -tp 2 ...

# GOOD -- correct output
VLLM_ROCM_USE_AITER=0 vllm serve /models/laguna --attention-backend TRITON_ATTN \
  --dtype bfloat16 -tp 2 ...
```

Probe with a raw completion, which bypasses chat template, reasoning parser and
tool parser — all of which were ruled out as causes this way:

```bash
curl -s localhost:8035/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"laguna","prompt":"def reverse_linked_list(head):","max_tokens":150,"temperature":0}'
```

Under AITER FA this returns misspelled tokens, broken indentation and an
infinite repetition loop. Under `TRITON_ATTN` it returns a correct reversal
plus a `Node` class.

## Scope: sliding window only

Three hybrid-attention models were run on the same box and image. Only the
sliding-window one is affected:

| model | attention | AITER FA output |
|---|---|---|
| Laguna-S-2.1 | 36 sliding + 12 full | ❌ corrupt |
| Qwen3.6-27B / KAT-Coder-V2.5 | linear (GDN) + full | ✅ correct |
| Nemotron-3-Super-120B | Mamba + ~9 full attention | ✅ correct |

So this is not "AITER FA is broken on gfx90a" — it is specifically the
sliding-window path. `vllm/v1/attention/backends/rocm_aiter_fa.py` does carry
sliding-window support (`AiterChunkSlidingWindowMetadata`, and
`sliding_window_configs` collected from each layer at
`build`), and the backend advertises `supports_sliding_window()`. The support
exists and is selected; it computes incorrectly on CDNA2.

Not yet isolated: whether the fault is in the chunked-SWA metadata, the ASM
kernel's handling of a 512-token window, or the gfx90a repatch of the underlying
`fmha_v3_fwd` code objects. A next step is running the same model with
`sliding_window` forced to `None` — if output cleans up under AITER FA, the
window handling is the culprit rather than the kernel port.

## Cost of the workaround

`TRITON_ATTN` is correct but needs more workspace. On this model, 128k context
OOM'd at `--gpu-memory-utilization 0.9x` where AITER FA had fit; it served at
64k with `--kv-cache-memory=6000000000`. Expect roughly half the usable context
for the same memory budget, plus the loss of AITER's prefill advantage.

## Why this matters beyond one model

**A throughput benchmark cannot detect this.** Laguna was benchmarked at
3,820 t/s prefill and 51.8 t/s decode under AITER FA — numbers that would have
looked like a win and gone into a comparison table. They measured a kernel
emitting garbage. The same discipline `probe/probe_image_patches.sh` applies to
kernel *selection* is needed for kernel *correctness*: assert on the output, not
just on the flag or the rate.

Any sliding-window model served through AITER FA on gfx90a should be treated as
suspect until its raw completions are read. Candidates in this class include
Gemma (interleaved SWA), Mistral SWA variants, and any config whose
`layer_types` contains `sliding_attention`.

## Relationship to other work

- [`attention-benchmarks.md`](attention-benchmarks.md) — AITER FA throughput on
  gfx90a. Those numbers stand for full-attention models; they say nothing about
  sliding-window correctness.
- [`patches/enable_vllm_aiter_gfx90a.py`](../patches/enable_vllm_aiter_gfx90a.py)
  and [`prefer_aiter_fa_gfx90a.py`](../patches/prefer_aiter_fa_gfx90a.py) — these
  open and prefer the AITER FA path. Neither is wrong, but a sliding-window
  model is a reason to override the preference.
