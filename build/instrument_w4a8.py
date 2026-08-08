#!/usr/bin/env python3
"""Instrument compute_tile to determine whether the int8 (W4A8) compute branch
is actually executed during MLIR generation. Clears cache -> forces regen."""
import os, shutil

P = "/opt/python/lib/python3.14/site-packages"
os.environ["FLYDSL_DUMP_IR"] = "0"  # don't need dumps here, just the prints

for cd in [os.path.expanduser("~/.flydsl"), "/root/.flydsl"]:
    shutil.rmtree(cd, ignore_errors=True)

# Base patches
sm = f"{P}/flydsl/utils/smem_allocator.py"
with open(sm) as f: s = f.read()
if "gfx90a" not in s:
    s = s.replace('"gfx942": 65536,', '"gfx90a": 65536,\n    "gfx942": 65536,', 1)
    open(sm, "w").write(s)

mk = f"{P}/aiter/ops/flydsl/moe_kernels.py"
with open(mk) as f: s = f.read()
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
    s = s.replace(old, w4 + old, 1)
    open(mk, "w").write(s)

# INSTRUMENT moe_gemm_2stage.py compute_tile
mg = f"{P}/aiter/ops/flydsl/kernels/moe_gemm_2stage.py"
with open(mg) as f: src = f.read()

# Print 1: compute_tile entry (line 928 anchor)
anchor1 = "                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4"
if "[DBG1]" not in src:
    src = src.replace(anchor1,
        "                    print(f'[DBG1] compute_tile ENTERED is_int8={is_int8} is_int4_bf16={is_int4_bf16} is_int4={is_int4}')\n" + anchor1, 1)

# Print 2: W4A16 branch taken
anchor2 = "                    if const_expr(is_int4_bf16 or is_int4_bf16_groupwise):"
if "[DBG2]" not in src:
    src = src.replace(anchor2,
        "                    print('[DBG2] branch check: is_int4_bf16 or groupwise')\n" + anchor2, 1)

# Print 3: int8 else-branch entered
anchor3 = "                        for ku in range_constexpr(k_unroll):\n                            b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]"
if "[DBG3]" not in src:
    src = src.replace(anchor3,
        "                        print('[DBG3] INT8 ELSE-BRANCH ENTERED -> will emit mfma')\n" + anchor3, 1)

open(mg, "w").write(src)
print("[PATCH] instrumentation added to compute_tile")

# Clear .pyc
for root, dirs, files in os.walk(P):
    for fn in files:
        if fn.endswith(".pyc"):
            try: os.remove(os.path.join(root, fn))
            except OSError: pass

import torch
torch.set_default_device("cuda")
torch.manual_seed(42)
M, K, N, E, topk, bm = 32, 256, 256, 4, 2, 32
w1q = torch.randint(-8, 8, (E, N * 2, K // 2), dtype=torch.int8)
a1q = torch.randint(-128, 127, (M, K), dtype=torch.int8)
w1s = torch.ones(E, N * 2, K // 32, dtype=torch.bfloat16)
a1s = torch.ones(M, K // 32, dtype=torch.bfloat16)
sids = torch.arange(M * topk, dtype=torch.int32)
seids = torch.arange(M * topk, dtype=torch.int32) % E
swts = torch.full((M * topk,), 0.5, dtype=torch.bfloat16)
nv = torch.tensor(M * topk, dtype=torch.int32)

print("\n[INVOKE] running flydsl_moe_stage1...")
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
out = flydsl_moe_stage1(a=a1q, w1=w1q, sorted_token_ids=sids,
    sorted_expert_ids=seids, num_valid_ids=nv, topk=topk,
    tile_m=bm, tile_n=32, tile_k=256, a_dtype='int8', b_dtype='int4',
    out_dtype='bf16', w1_scale=w1s, a1_scale=a1s, sorted_weights=swts)
torch.cuda.synchronize()
print(f"[DONE] out[:4]={out.reshape(-1)[:4].tolist()}")
