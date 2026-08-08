#!/usr/bin/env python3
"""Instrument the int8 compute loop levels to find where execution stops."""
import os, shutil
P = "/opt/python/lib/python3.14/site-packages"
for cd in [os.path.expanduser("~/.flydsl"), "/root/.flydsl"]:
    shutil.rmtree(cd, ignore_errors=True)
sm = f"{P}/flydsl/utils/smem_allocator.py"; s = open(sm).read()
if "gfx90a" not in s:
    open(sm, "w").write(s.replace('"gfx942": 65536,', '"gfx90a": 65536,\n    "gfx942": 65536,', 1))
mk = f"{P}/aiter/ops/flydsl/moe_kernels.py"; s = open(mk).read()
if 'a_dtype == "int8" and b_dtype == "int4"' not in s:
    w4 = '''    elif a_dtype == "int8" and b_dtype == "int4":
        from .kernels.moe_gemm_2stage import compile_moe_gemm1
        _cs = None if k_batch > 1 else False
        return compile_moe_gemm1(model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage1=doweight_stage1, in_dtype="int4", group_size=32,
            out_dtype=out_dtype, use_cshuffle_epilog=_cs, scale_is_bf16=True, k_batch=k_batch)
'''
    old = '    else:\n        raise ValueError(\n            f"Unsupported stage1 dtype combination'
    open(mk, "w").write(s.replace(old, w4 + old, 1))

mg = f"{P}/aiter/ops/flydsl/kernels/moe_gemm_2stage.py"; src = open(mg).read()

# A: print loop bounds at int8 else-entry (anchor unique to int8 branch)
a_else = "                        for ku in range_constexpr(k_unroll):\n                            b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]"
if "[DBGA]" not in src:
    src = src.replace(a_else,
        "                        print('[DBGA] int8 else k_unroll=', k_unroll, 'm_repeat=', m_repeat, 'num_acc_n=', num_acc_n)\n"
        "                        for ku in range_constexpr(k_unroll):\n"
        "                            print('[DBGB] ku=', ku)\n"
        "                            b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]", 1)

# C: innermost loop body marker (anchor unique: b_gate_packs0[ni],)
a_ni = "                                    gate_list[acc_idx] = mfma_k64(\n                                        gate_list[acc_idx],\n                                        a0,\n                                        a1,\n                                        b_gate_packs0[ni],"
if "[DBGC]" not in src:
    src = src.replace(a_ni,
        "                                    print('[DBGC] ni loop body reached, calling mfma_k64')\n"
        "                                    gate_list[acc_idx] = mfma_k64(\n                                        gate_list[acc_idx],\n                                        a0,\n                                        a1,\n                                        b_gate_packs0[ni],", 1)

open(mg, "w").write(src)
print("[PATCH] loop instrumentation added")

for root, dirs, files in os.walk(P):
    for fn in files:
        if fn.endswith(".pyc"):
            try: os.remove(os.path.join(root, fn))
            except OSError: pass

import torch
torch.set_default_device("cuda"); torch.manual_seed(42)
M, K, N, E, topk, bm = 32, 256, 256, 4, 2, 32
w1q = torch.randint(-8, 8, (E, N * 2, K // 2), dtype=torch.int8)
a1q = torch.randint(-128, 127, (M, K), dtype=torch.int8)
w1s = torch.ones(E, N * 2, K // 32, dtype=torch.bfloat16)
a1s = torch.ones(M, K // 32, dtype=torch.bfloat16)
sids = torch.arange(M * topk, dtype=torch.int32); seids = sids % E
swts = torch.full((M * topk,), 0.5, dtype=torch.bfloat16)
nv = torch.tensor(M * topk, dtype=torch.int32)
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
out = flydsl_moe_stage1(a=a1q, w1=w1q, sorted_token_ids=sids, sorted_expert_ids=seids,
    num_valid_ids=nv, topk=topk, tile_m=bm, tile_n=32, tile_k=256, a_dtype='int8', b_dtype='int4',
    out_dtype='bf16', w1_scale=w1s, a1_scale=a1s, sorted_weights=swts)
torch.cuda.synchronize()
print(f"[DONE] out[:4]={out.reshape(-1)[:4].tolist()}")
