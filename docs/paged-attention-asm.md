# pa_fwd_asm on gfx90a: RESOLVED — It Was Never an ISA Problem

**Date**: 2026-07-27
**Status**: ✅ ASM paged-attention decode works on MI210. 48/48 configs match a PyTorch reference.
**Supersedes**: [`17-pa-fwd-investigation-transcript.md`](https://github.com/davetha/mi210-llm-stack/blob/main/docs/17-pa-fwd-investigation-transcript.md)

---

## Summary

Doc 17 concluded that gfx942 `pa_fwd` binaries **cannot** run on gfx90a, and that the
incompatibility is at the instruction-encoding level. That conclusion is **wrong** for
the noquant bf16/fp16 kernels.

The actual blockers were two ordinary bugs. Neither is an architectural incompatibility:

1. A **stale JIT module** whose kernarg struct predated the installed `.co` files. This
   made every `buffer_store` in the kernel get silently discarded.
2. The **wrong MFMA opcode** on 3 of the 8 portable pa kernels.

After fixing both, `pa_fwd_asm` is numerically exact on gfx90a.

Doc 17's conclusion *is* correct for the fp8/int8 variants — but for a concrete reason
(instructions that genuinely do not exist on CDNA2), not a general binary-compat argument.

---

## Root Cause 1: Stale JIT Module

`aiter/jit/module_attention_asm.so` was compiled at **20:29**. The aiter package —
`csrc/py_itfs_cu/asm_pa.cu` **and** every `hsa/**/*.co` — was replaced at **22:05**.
An `mtp` field was added to `KernelArgs` in between:

```c
unsigned int KVs;   p3 _p17;
unsigned int mtp;   p3 _p18;   // <-- added at 22:05
unsigned int GQA;   p3 _p19;
```

The cached `.so` therefore wrote `GQA` at kernarg `+0xe0`, while the newly installed
`.co` reads it from `+0xf0`. The kernel got `GQA = 0`.

That is fatal, because the kernel derives its **output buffer descriptor** from it:

```asm
s_load_dword s77, s[0:1], 0xf0      ; GQA         -> reads 0
s_mul_i32    s76, 0x100, s77        ; num_records -> 0
s_mov_b32    s10, s76               ; V# word2
...
buffer_store_dword v128, v8, s[8:11], 0 offen
```

Per both the CDNA2 and CDNA3 ISA guides, **`num_records == 0` puts every buffer access
out of range**, and out-of-range writes are *dropped*. So the kernel:

- launched with no error,
- ran at full speed (258 µs at ctx=1024 × 128 seqs — ~1 TB/s of KV reads, near HBM peak),
- reached `s_endpgm`,
- and **wrote nothing at all**.

The "incoherent output" in doc 17 was never the kernel's output. It was whatever
happened to be in the freshly allocated `torch.empty_like(Q)` tensor. Confirmed by
pre-filling the output with a sentinel: **100% of elements untouched**.

### Fix

```bash
rm /opt/python/lib/python3.14/site-packages/aiter/jit/module_attention_asm.so
# aiter rebuilds it on next use (~13s)
```

---

## Root Cause 2: Wrong MFMA Opcode on 3 Kernels

Only `pa_bf16_noquant_gqa8_1tg_4w.co` had been re-patched with the correct
`D3E1 → D3E7`. Three others still carried the older, wrong `D3E1 → D3CD` patch:

| Kernel | Was | Should be |
|---|---|---|
| `pa_bf16_noquant_gqa16_1tg_4w` | `v_mfma_f32_16x16x16f16` | `v_mfma_f32_16x16x16bf16_1k` |
| `pa_bf16_noquant_gqa8_1tg_4w_mtp_msk0` | `v_mfma_f32_16x16x16f16` | `v_mfma_f32_16x16x16bf16_1k` |
| `pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1` | `v_mfma_f32_16x16x16f16` | `v_mfma_f32_16x16x16bf16_1k` |

Before: bf16 gqa16 had `rel_rms` up to **272** at 0.2% element match.
After: `5.8e-4` at 100%.

fp16 kernels need **no** MFMA patch — `v_mfma_f32_16x16x16_f16` (D3CD) is already valid
gfx90a. They only need the `e_flags` change.

---

## Where Doc 17's ISA Analysis Went Wrong

Doc 17 inferred incompatibility from partial binary inspection. The decisive test —
never run — is to **disassemble and re-assemble for the target**:

```bash
llvm-objdump -d --mcpu=gfx942 pa_bf16_noquant_gqa8_1tg_4w.co > k.s
# substitute the BF16 MFMA mnemonic, then:
llvm-mc -arch=amdgcn -mcpu=gfx90a k.s
```

**Result: zero errors across all 3,303 instructions.** Comparing encodings byte-for-byte,
the *only* difference in the entire kernel is the MFMA opcode.

| Doc 17 claim | Reality |
|---|---|
| "0 `v_accvgpr_read/write` ⇒ gfx942 register model, can't run" | gfx90a has a **unified** VGPR/AGPR file. MFMA writing ArchVGPRs directly is normal on CDNA2, not a gfx942 artifact. |
| "FLAT instructions may use gfx942-specific addressing modes" | No. All FLAT/global encodings are byte-identical and assemble clean for gfx90a. |
| "VOP3P modifier bits have different meanings" | No. Byte-identical. |
| "GPU HANG at `--block-size 16`" | The ASM kernel **requires** block_size 16 — declared in `pa_asm.csv` (`blkSz=16`). No hang occurs once the stale module is rebuilt. |
| vgpr patches (`vgpr=256/512`) | Unnecessary and wrong. The kernel descriptor needs **no** change: `accum_offset=256`, 512 VGPRs, 64 KB LDS are all valid gfx90a. |

Also verified against both ISA guides: `packed-tid` and the buffer-descriptor
range-check rules are **identical** on gfx90a and gfx942. Neither is a porting hazard.

---

## What Genuinely Cannot Be Ported

Of 56 kernels in `hsa/gfx942/pa/`, only **8** are portable. The other 48 use
instructions with no gfx90a equivalent:

- `v_cvt_pk_fp8_f32` — all fp8 KV-cache variants
- `v_mfma_i32_16x16x32_i8` — all int8 variants (gfx90a's int8 MFMA is K=16, not K=32)

**fp8/int8 paged attention really is gfx942+ only.**

> **Update:** this taxonomy is incomplete. A full sweep of all 1,422 gfx942
> kernels found two further blockers not visible in the `pa` family alone:
> **packed-bf16 atomics** (`global_atomic_pk_add_bf16`, 539 kernels — the second
> largest blocker overall) and 64-bit VALU ops (3 kernels). See
> [`19-aiter-operator-port-matrix.md`](port-matrix.md).

Portable set (all verified):

```
pa_bf16_noquant_gqa8_1tg_4w            pa_fp16_noquant_gqa8_1tg_4w
pa_bf16_noquant_gqa16_1tg_4w           pa_fp16_noquant_gqa16_1tg_4w
pa_bf16_noquant_gqa8_1tg_4w_mtp_msk0   pa_fp16_noquant_gqa8_1tg_4w_mtp_msk0
pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1   pa_fp16_noquant_gqa8_1tg_4w_mtp_msk1
```

---

## Verification

`tests/test_pa_fwd_asm_gfx90a.py` compares `aiter.pa_fwd_asm` against a pure-PyTorch
reference. **48/48 PASS, 0 failures.**

| dtype | gqa | ctx range | seqs | rel_rms | element match |
|---|---|---|---|---|---|
| bf16 | 8 | 16 – 4097 | 1 – 128 | 5.2e-4 – 7.7e-4 | 100.00% |
| bf16 | 16 | 16 – 4097 | 1 – 128 | 5.2e-4 – 7.8e-4 | 100.00% |
| fp16 | 8 | 16 – 4097 | 1 – 128 | 1.6e-4 – 2.5e-4 | 100.00% |
| fp16 | 16 | 16 – 4097 | 1 – 128 | 1.6e-4 – 2.4e-4 | 100.00% |

End-to-end `atom.examples.simple_inference` with Qwen3-0.6B is coherent in English and
Chinese (TPOT 0.028 s):

```
Prompt:     'introduce yourself'
Completion: "<think>\nOkay, the user wants me to introduce myself. Let me start by
             recalling what I know. I'm an"
```

**Caveat**: Qwen3-0.6B has `gqa_ratio = 2`, and ASM PA kernels exist only for gqa 8/10/16
— so it never routes through `pa_fwd_asm`. The op-level test above is the actual proof;
the e2e run only shows the stack is healthy.

---

## Full-Tree Audit of the 1,251 Patched Code Objects

Running the assembler-verified repatcher over **all** of `hsa/gfx942` (1,422 kernels, 59 s):

```
TALLY (all): {'OK': 242, 'NOTPORT': 1180}
```

Cross-referenced against what is actually installed in `hsa/gfx90a`:

| | Count |
|---|---|
| installed `.co` in `gfx90a/` | 1,251 |
| provably portable (whole tree) | 242 |
| **installed but NOT portable** | **1,147** |
| installed and portable | 104 |
| — of those, **mis-patched** | **74** |

### 1,147 installed kernels contain gfx942-only instructions

Mostly `fmoe` (830), `fmoe_2stages` (186), `pa` (48), `bf16gemm` (22), `mla` (13).
They use `v_cvt_pk_fp8_f32`, `v_mfma_*_fp8_fp8`, `v_mfma_i32_16x16x32_i8`. The original
bulk patch script copied them into `gfx90a/` after rewriting only the 16×16×16 BF16 MFMA,
without checking whether the rest of the kernel was even valid gfx90a.

### 74 of the 104 portable ones are mis-patched

By family: **48 `fmha_v3_fwd`, 11 `mla`, 8 `fmoe`, 7 top-level**
(`all_reduce.co`, `allreduce_rmsnorm_N8192.co`, …).

A second missed opcode. The original script only handled `v_mfma_f32_16x16x16_bf16`;
these kernels use **`v_mfma_f32_32x32x8_bf16`**, which needs `v_mfma_f32_32x32x8bf16_1k`.
Left unpatched, those bytes are not decodable as MFMA on gfx90a at all:

```
fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co
  gfx942 original : 176 x v_mfma_f32_32x32x8_bf16
  installed       : (llvm-objdump --mcpu=gfx90a decodes no MFMA — invalid)
  correct         : 176 x v_mfma_f32_32x32x8bf16_1k
```

### …but they are inert, which invalidates doc 17's "works" rows

`aiter/ops/mha.py` and `aiter/mla.py` gate the v3 ASM paths on
`get_gfx() in ("gfx942", "gfx950")`. On gfx90a those branches are never taken, so the
mis-patched `fmha_v3_fwd` and `mla` code objects are **never dispatched** — dead files,
not a live hazard.

> **Update — this is no longer true, and the prediction two sections below came
> true immediately.** `patches/enable_gfx90a_asm_paths.py` deliberately removes
> that gate, so `fmha_v3_fwd` **is** now dispatched on gfx90a. It is safe only
> because the kernel tree was regenerated first (change 29 §2), leaving no
> mis-patched files to dispatch. Had the gate been opened against the old tree,
> this reasoning would have failed exactly as warned. Do not rely on "gated off"
> as a safety property.

That also means doc 17's performance table is misattributed:

| Doc 17 row | Reality |
|---|---|
| ASM flash attention ✅ 4,791,074 tok/s | ASM path is gated off on gfx90a; this measured the CK/Triton fallback |
| ASM MLA prefill ✅ 3,013,378 tok/s | same — `mla.py` gates to gfx942/gfx950 |
| ASM MLA decode ✅ 0.090 ms/step | same |

Those were throughput numbers with no correctness check, on kernels that were not the
ASM ones. `pa_fwd` was different precisely *because* it is **not** gated in Python —
`asm_pa.cu` selects by `get_gpu_arch()` directory lookup and therefore genuinely picked
up `hsa/gfx90a/pa/`. It was the one op where the patch actually had to be correct.

---

## Diagnostic Method (Reusable)

The technique that cracked it, after the symptom "runs at full speed, writes nothing":

1. **Sentinel the output.** Pre-fill `out_` with an *exactly bf16-representable* value
   (`-768.0`, not `-777.0` — bf16 rounds that to `-776.0`, and a naive equality check
   then reports everything as "written").
2. **Truncate the kernel.** Binary-patch a `buffer_store` + `s_endpgm` in right after the
   output descriptor is built. This separates "stores work" from "math works". It showed
   the store failing *before any attention math ran*.
3. **Dump the kernargs the kernel actually sees.** Build a V# over the kernarg segment
   itself, have each lane load one dword, store it out. This showed `GQA=8` sitting at
   `+0xe0` while the code read `+0xf0`.

Steps 2–3 generalise to any ported ASM kernel that silently misbehaves.

---

## Remaining Risks

1. **MFMA hazard shortfall.** `V_MFMA_F32_16X16X16*` is **4-pass on gfx942 but 8-pass on
   gfx90a**. Required software wait states: gfx942 needs 7, gfx90a needs **11** (LLVM
   `SMFMA16x16WriteVgprVALUMemExpReadWaitStates = 11`; CDNA2 Table 26 has no 4-pass row
   at all). Every gfx942 binary is therefore ~4 wait states short at each MFMA→consumer
   edge. `pa_fwd` survives because its scheduling leaves ~19 states of natural spacing —
   but **other ported kernels may be silently wrong**. Check `s_nop` padding first on any
   ported kernel that is subtly inaccurate.

2. **`sc1`.** MUBUF bit 15 is `SC1` (scope) on gfx942 but *reserved, must be zero* on
   gfx90a. Any kernel using `sc0/sc1` scope bits must be rejected, not patched. The pa
   kernels use only `nt` (bit 17 = `slc` on gfx90a), which is benign.

3. **`mha_varlen_fwd_bf16_*` cannot be rebuilt** in the current container. CK's
   `ck_tile/host/device_prop.hpp` fails with 6 redefinition errors (`get_device_name`,
   `is_gfx11_supported`, `get_num_cus`, …). The existing pre-upgrade `.so` still works,
   but if it is ever evicted, flash attention cannot be rebuilt until that header
   conflict is fixed.

---

## Tooling

[`tools/repatch_gfx942_to_gfx90a.py`](../tools/repatch_gfx942_to_gfx90a.py) replaces
hand-guessed byte patching. For each kernel it disassembles, applies the mnemonic
substitution table, and **re-encodes every instruction through the assembler**, refusing
any kernel that does not assemble for gfx90a or whose encoding changes length.
Portability is *proven*, not assumed.

```bash
python tools/repatch_gfx942_to_gfx90a.py \
    /path/to/aiter_meta/hsa/gfx942  ./out            # whole tree
python tools/repatch_gfx942_to_gfx90a.py \
    /path/to/aiter_meta/hsa/gfx942  ./out  pa/       # one family
```

Treat `hsa/gfx90a/` as **generated**. Rebuild it with this tool, never by hand.

### Recommended follow-ups — ✅ both done

1. ~~Delete the 1,147 non-portable `.co` files from `hsa/gfx90a/` so kernel selection can
   never pick an unrunnable kernel.~~
2. ~~Reinstall the 74 mis-patched ones from the repatcher — harmless today because they are
   gated off, but wrong on disk and a trap if a future aiter drops the gate.~~

Both were completed in change 29 §2: `hsa/gfx90a/` was regenerated wholesale
from the repatcher, leaving exactly the 242 portable kernels. The trap named in
item 2 was real — the gate was dropped days later by
`patches/enable_gfx90a_asm_paths.py`, and only the prior regeneration made that
safe.

---

## Upstream

The stale-JIT failure mode is architecture-independent and worth reporting to
[ROCm/aiter](https://github.com/ROCm/aiter): cached `aiter/jit/*.so` modules survive an
in-place package upgrade with no version or ABI check. Two cheap guards would have turned
a multi-day silent failure into an immediate error:

1. Stamp the built `.so` with the aiter version / a hash of its source set, and rebuild on
   mismatch.
2. Validate `sizeof(KernelArgs)` against the `.co`'s `.kernarg_segment_size` before
   dispatch — the metadata already carries it (`kernarg_segment_size: 272`).
