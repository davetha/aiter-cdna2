# The AITER ASM Operator Port Matrix for gfx90a

**Date**: 2026-07-27
**Hardware**: 2× AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each
**Software**: ROCm 7.14.0, PyTorch 2.11.0, amd-aiter 0.1.17, atom 0.1.5, Python 3.14

This document answers, exhaustively and with evidence, the question the earlier
docs kept guessing at: **which AITER ASM operators can run on an MI210, which
cannot, and why.**

It supersedes the portability claims in
[`17-pa-fwd-investigation-transcript.md`](https://github.com/davetha/mi210-llm-stack/blob/main/docs/17-pa-fwd-investigation-transcript.md)
and extends [`18-pa-fwd-asm-resolved.md`](paged-attention-asm.md) from one
operator to the whole ASM surface.

---

## Headline results

1. **242 of AITER's 1,422 gfx942 ASM kernels are portable to gfx90a.** That is a
   hard ceiling, not a work-in-progress number — see "Why 242 is the ceiling".
2. **The split is exactly by data type.** Every bf16/fp16 ASM kernel in the
   attention families is portable. Every FP8 and INT8 kernel is not. CDNA2 has
   no FP8 ALU and no gfx942-shaped INT8 MFMA, so this is physics, not effort.

   > **Scope note (added 2026-07-27).** The INT8 half of that sentence is about
   > these **prebuilt ASM blobs**, and must not be read as "MI210 cannot do
   > INT8". gfx90a has native INT8 MFMA at the full 181 TOPS —
   > `v_mfma_i32_16x16x16i8` and `v_mfma_i32_32x32x8i8` both assemble for it.
   > What is missing is gfx942's *encoding*: its INT8 MFMA takes K=32,
   > gfx90a's takes K=16, so the two ISAs are disjoint at that opcode and the
   > blobs cannot be patched across. INT8 kernels compiled **from source** for
   > gfx90a work and are bit-exact — see
   > [`20-int8-gemm-gfx90a.md`](int8-gemm.md). FP8 is the one
   > genuinely absent capability; `v_cvt_pk_fp8_f32` does not assemble at all.
3. **ASM flash-attention forward (`fmha_v3_fwd`) now works on gfx90a** — 48/48
   correctness configs pass. It was never a hardware limitation; it was six
   architecture-string comparisons in AITER's dispatch code.
4. **1,147 of the 1,251 `.co` files previously installed under `hsa/gfx90a/`
   could not run.** They have been removed. The directory is now generated,
   not hand-edited.

---

## The portability ceiling

`configs/classify_gfx942_kernels.py` disassembles every gfx942 kernel, collects
the distinct instruction texts across the whole tree, and asks the assembler
whether each one re-encodes for gfx90a after the known rewrites. Judging per
*instruction text* rather than per *mnemonic* matters: `flat_load_dword v13,
v[12:13]` is valid gfx90a while `flat_load_dword v13, v[12:13] sc0 sc1` is not,
and a per-mnemonic verdict would wrongly condemn the first because of the second.

| Family | Total | Portable | Blocked | Blocking capability |
|---|---:|---:|---:|---|
| `fmha_v3_bwd` | 156 | 138 | 18 | bf16_atomic |
| `fmha_v3_fwd` | 56 | 48 | 8 | fp8 |
| `topksoftmax` | 22 | 22 | 0 | — |
| `mla` | 24 | 11 | 13 | fp8 |
| `pa` | 56 | 8 | 48 | fp8, int8, valu64 |
| `fmoe` | 838 | 8 | 830 | fp8, int8, bf16_atomic |
| (top-level) | 45 | 7 | 38 | fp8, int8, bf16_atomic |
| `bf16gemm` | 22 | 0 | 22 | bf16_atomic |
| `fmoe_2stages` | 186 | 0 | 186 | fp8, int8 |
| `i8gemm` | 9 | 0 | 9 | int8 |
| `fp8gemm_blockscale` | 6 | 0 | 6 | fp8 |
| `topk_per_row_*` | 2 | 0 | 2 | valu64 |
| **Total** | **1422** | **242** | **1180** | |

Blocking capabilities, by number of kernels affected (a kernel may need several):

| Capability | Kernels | Why CDNA2 cannot do it |
|---|---:|---|
| `fp8` | 647 | No FP8 ALU. `v_cvt_pk_fp8_f32`, `v_mfma_*_fp8_fp8` do not exist. `torch._scaled_mm` is hardware-gated to MI300+ for the same reason. |
| `bf16_atomic` | 539 | No `global_atomic_pk_add_bf16`. gfx90a has `global_atomic_pk_add_f16` and `global_atomic_add_f32`, but not the bf16 packed form. |
| `int8` | 482 | No `v_mfma_i32_*` in the gfx942 shapes. |
| `valu64` | 3 | No `v_lshl_add_u64`, no `v_mov_b64`. |

### Why 242 is the ceiling

The interesting question is not how many kernels are blocked but whether any are
blocked *only* by something cosmetic — a renamed mnemonic or a renamed cache
modifier — because those would be recoverable by patching. Two such rewrites do
exist and are size-compatible:

| gfx942 | gfx90a | Encoding |
|---|---|---|
| `v_fmamk_f32 v1, v2, 0x4f800000, v1` | `v_madmk_f32 …` | 8 bytes → 8 bytes |
| `flat_load_dword v13, v[12:13] sc0 sc1` | `… glc slc` | 8 bytes → 8 bytes |

**No kernel is blocked by these alone.** The classifier checks for exactly this
and reports it. 51 kernels initially looked recoverable (including all 22
`bf16gemm`); on inspection, 49 of them need `global_atomic_pk_add_bf16` and the
other 2 need 64-bit VALU ops that expand to multiple instructions and therefore
cannot be patched in place. So 242 is not "242 so far" — it is the complete set
reachable by binary patching, and three independent methods agree on the
identical file set:

```
classifier portable : 242
repatcher produced  : 242
identical sets      : True
```

### What this means practically

The blocked kernels are precisely the ones whose *arithmetic* CDNA2 cannot
perform. An MI210 cannot run an FP8 model regardless of which kernel you hand
it. So the 1,180 blocked kernels do not represent lost capability — they
represent data types this hardware does not implement. For the data types an
MI210 does run (bf16, fp16, and Q4-class GGUF via llama.cpp), the ASM kernels
are available.

---

## ASM flash attention on gfx90a

`fmha_v3_fwd` is AITER's hand-written ASM flash-attention forward. AITER gates it
to gfx942/gfx950. The gate is not a hardware statement — `aiter/ops/mha.py:1674`
describes the kernels as *"hand-written gfx9 ASM"*, and gfx90a **is** gfx9. AMD
simply never built or validated gfx90a binaries.

All 48 bf16 kernels are portable. The 8 FP8 ones are not. Once `hsa/gfx90a/` is
populated, six architecture-string comparisons stand in the way:

| File | Site | Form |
|---|---|---|
| `aiter/ops/mha.py` | `can_impl_fmha_v3_fwd` (batched) | `get_gfx() in ("gfx942","gfx950")` |
| `aiter/ops/mha.py` | `can_impl_fmha_v3_fwd` (varlen) | `get_gfx() in ("gfx942","gfx950")` |
| `cpp_itfs/mha_fwd.cu` | `fmha_fwd_v3` early-out | `(arch_id != "gfx942") && (arch_id != "gfx950")` |
| `cpp_itfs/mha_fwd.cu` | `get_kernel_name_key` | `arch_id == "gfx942"` |
| `cpp_itfs/mha_fwd.cu` | `get_kernel_co_name` | `arch_id == "gfx942"` |
| `cpp_itfs/mha_fwd.cu` | `get_grid_dim` ×2, `init_fmha_fwd_v3_args` | `arch_id == "gfx942"` |

`patches/enable_gfx90a_asm_paths.py` rewrites all of them. Three details matter:

- **The early-out is written in the negated form**, so a grep for
  `arch_id == "gfx942"` misses it. It returns `-1` before the kernel lookup
  runs, surfacing as the unhelpful `RuntimeError: invalid argument for
  fmha_fwd`. This one site is the actual blocker; the others are follow-on.
- **`is_fmha_v3_fp8()` must NOT be widened.** The identical line
  `ret = get_gfx() in ("gfx942","gfx950")` appears four times in `mha.py` — twice
  in `can_impl_fmha_v3_fwd` and twice in `is_fmha_v3_fp8`. Widening the latter
  would route FP8 tensors at a GPU with no FP8 ALU. The patch anchors on the
  preceding comment so it can only hit the correct pair, and re-narrows to bf16
  explicitly.
- **The `MI300/` subdirectory is correct for MI210.** `get_kernel_co_name`
  inserts `MI300/` or `MI308/` by PCI chip id. MI210 is `0x740f`, not in
  `MI308_CHIP_IDS = {0x74A2, 0x74A8, 0x74B6, 0x74BC}`, so it resolves to
  `MI300/` — the directory the repatcher populates.

No change to `hsa/codegen.py` is needed: `archs_supported` is a glob of
`hsa/*/`, so gfx90a is picked up automatically once the directory exists, and
`AITER_GPU_ARCHS` resolves to `gfx90a` by native detection. The generated config
table contains 28 gfx90a rows.

### Validation

`tests/test_fmha_v3_fwd_asm_gfx90a.py` — **80 configurations, 0 failures**,
rel_rms 1.1e-4 … 7.6e-4, 100.00% element match at atol/rtol 2e-2:

- 48 batched (`flash_attn_func`): head dim 128 and 192×128, causal/non-causal,
  GQA ratio 1/8/8, seqlen 129…1024, batch 1/2/4.
- 32 varlen (`flash_attn_varlen_func`): the packed-THD path a serving stack
  actually takes for prefill. It selects the `*_group` kernels rather than the
  batched ones, so a batched pass says nothing about it — it needs its own
  coverage. Ragged batches up to 4 sequences of differing length.

Seven distinct ASM kernels are exercised: `{hd128, hd192x128}` ×
`{plain, causal}` × `{batched, _group}`.

Proof the ASM path actually ran, rather than a silent fall back to CK:

```
LoadKernel: _ZN5aiter24fmha_fwd_hd128_bf16_rtnaE
  hsaco: .../hsa/gfx90a/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtna.co
LoadKernel: _ZN5aiter31fmha_fwd_hd128_bf16_causal_rtnaE       …/fwd_hd128_bf16_causal_rtna.co
LoadKernel: _ZN5aiter28fmha_fwd_hd192x128_bf16_rtnaE          …/fwd_hd192x128_bf16_rtna.co
LoadKernel: _ZN5aiter35fmha_fwd_hd192x128_bf16_causal_rtnaE   …/fwd_hd192x128_bf16_causal_rtna.co
```

The test captures these at the file-descriptor level (the C++ runtime writes
straight to fd 1, so `contextlib.redirect_stdout` cannot see them) and `--require-asm`
turns their absence into a failure. This matters because correctness alone cannot
distinguish ASM from the CK fallback — both should be right.

---

## Manifests must be pruned with the kernels

Removing a non-portable `.co` is only half the job. Each kernel family carries a
`.csv` manifest that `hsa/codegen.py` turns into the config table the dispatcher
selects from **by shape** — and that selection never checks whether the file
exists. The loader then hard-fails:

```c
AITER_CHECK(file.is_open(), "failed to open ", full_path.c_str());
```
— `aiter_meta/csrc/include/aiter_hip_common.h:394`

So a manifest that outlives its kernels converts an unsupported shape from a
graceful fallback into a crash. After the first pass of pruning, **754 manifest
rows pointed at kernels that no longer existed** — 47 of the 55 `pa` rows among
them.

Nothing reachable actually broke, because every dangling `pa` row is a quantised
variant that a bf16/fp16 workload cannot select, and `fmoe`'s ASM path is
unreachable on gfx90a anyway. But that is precisely the "it is gated off, so it
is harmless" reasoning that [doc 18](paged-attention-asm.md) used about
`fmha_v3_fwd` — and which this work then invalidated by opening the gate. Do not
lean on it twice.

`repatch_gfx942_to_gfx90a.py` therefore prunes manifests in a second pass, after
it knows which kernels it produced, so the generated tree is self-consistent **by
construction**:

```
TALLY: {'OK': 242, 'NOTPORT': 1180}
manifests: 207 rows kept, 754 dangling rows dropped
```

Note `co_name` is relative to the manifest's own directory, except for fmha,
where the dispatcher inserts an `MI300/` or `MI308/` product subdirectory at load
time — a row is only dangling if none of those resolve.

Because the config table is generated from these manifests at JIT build time,
changing them requires rebuilding the affected module.

## The MFMA wait-state hazard

`V_MFMA_F32_16X16X16*` is a 4-pass instruction on gfx942 but an **8-pass**
instruction on gfx90a. The required wait states between an MFMA and a consumer
of its result are correspondingly higher — 11 on gfx90a versus 7 on gfx942. Code
scheduled by AMD's assembler for gfx942 can therefore be up to 4 wait states
short at every MFMA→consumer edge on gfx90a.

This is a real hazard and the repatcher cannot detect it: every instruction
re-encodes perfectly, so static verification passes while the *schedule* is
unsound. It has to be settled empirically, per kernel.

Status so far: `pa_fwd_asm` (48/48) and `fmha_v3_fwd` (48/48) both pass with no
sign of it — `pa_fwd` survives on roughly 19 wait states of natural instruction
spacing. **Do not generalise this to untested kernels.** Any newly enabled ASM
family needs its own numerical validation before it is trusted.

---

## Reproducing

Inside the ROCm container, with `hsa/gfx942` intact:

```bash
# 1. Classify (optional; ~8 min) -- proves the ceiling
python configs/classify_gfx942_kernels.py \
    $SITE/aiter_meta/hsa/gfx942 \
    --gfx90a-dir $SITE/aiter_meta/hsa/gfx90a \
    --json kernel_classification.json

# 2. Generate the gfx90a kernel set (~60 s) -- 242 files
python tools/repatch_gfx942_to_gfx90a.py $SITE/aiter_meta/hsa/gfx942 ./port_v2

# 3. Install. hsa/gfx90a is GENERATED -- never hand-edit it.
cp -a $SITE/aiter_meta/hsa/gfx90a $SITE/aiter_meta/hsa/gfx90a.bak
rm -rf $SITE/aiter_meta/hsa/gfx90a && cp -a ./port_v2 $SITE/aiter_meta/hsa/gfx90a

# 4. Open the ASM arch gates
python patches/enable_gfx90a_asm_paths.py

# 5. Force the affected JIT modules to rebuild against the patched C++.
#    A stale module is how pa_fwd_asm appeared broken for weeks -- see doc 18.
rm -f  $SITE/aiter/jit/module_fmha_v3_fwd.so
rm -rf $SITE/aiter/jit/build/module_fmha_v3_fwd

# 6. Validate
AITER_LOG_LEVEL=info python tests/test_pa_fwd_asm_gfx90a.py
AITER_LOG_LEVEL=info python tests/test_fmha_v3_fwd_asm_gfx90a.py --require-asm
```

`enable_gfx90a_asm_paths.py --revert` undoes step 4; `--check` reports state.
Every rewrite asserts on its expected match count, so an upstream change that
moves the code fails loudly rather than silently leaving the fast path disabled.

---

## Open items

- **The CK flash-attention fallback does not build** on this container:
  `mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale` fails with redefinition errors
  in `ck_tile/host/device_prop.hpp` (`get_device_name`, `is_gfx11_supported`,
  `is_gfx12_supported`, `is_gfx95_supported`, `get_num_cus`). This is
  pre-existing and independent of the gfx90a port, but it means that before this
  work `flash_attn_func` had **no** working path on this box — neither ASM nor
  CK. Any earlier benchmark row claiming ASM flash-attention throughput on
  gfx90a should be treated as unverified.
- `fmha_v3_bwd` (138 portable kernels) is not yet enabled or validated.
- `mla` (11 portable kernels) is not yet enabled or validated.
- The 539 kernels blocked by `global_atomic_pk_add_bf16` could in principle be
  recovered by recompiling from source with a CAS-loop emulation, but AITER does
  not publish ASM sources, so this would require reimplementation rather than
  patching.
