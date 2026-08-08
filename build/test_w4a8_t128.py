#!/usr/bin/env python3
"""W4A8 correctness test with VALID tile_n=128 (num_acc_n = 128//64 = 2).
Prior tests used tile_n=32 -> num_acc_n=0 -> no compute -> all zeros."""
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
for root, dirs, files in os.walk(P):
    for fn in files:
        if fn.endswith(".pyc"):
            try: os.remove(os.path.join(root, fn))
            except OSError: pass

# Patch: int8 A-load if/else missing const_expr() wrapper (copy-paste bug vs W4A16)
mg = f"{P}/aiter/ops/flydsl/kernels/moe_gemm_2stage.py"; src = open(mg).read()
bare = ("                                if (\n"
        "                                    (a0_prefetch is not None)\n"
        "                                    and (ku == 0)\n"
        "                                    and (mi == 0)\n"
        "                                ):")
wrapped = ("                                if const_expr(\n"
           "                                    (a0_prefetch is not None)\n"
           "                                    and (ku == 0)\n"
           "                                    and (mi == 0)\n"
           "                                ):")
if bare in src:
    src = src.replace(bare, wrapped)  # fix all bare occurrences (int8 stage1 + stage2)
    open(mg, "w").write(src)
    print("[PATCH] int8 A-load if/else: added const_expr() wrapper")
else:
    print("[PATCH] bare-if not found (already wrapped?)")

import torch, torch.nn.functional as F
torch.set_default_device("cuda"); torch.manual_seed(42)

# Config: VALID int8 tiles. tile_n=128 -> num_acc_n=2.
M, K, Ninter, E, topk = 64, 256, 256, 4, 2
tile_m, tile_n, tile_k = 64, 128, 256

def quant_int4(w, gs=32):
    e, nd, kk = w.shape; ng = kk // gs
    wg = w.float().reshape(e, nd, ng, gs)
    sc = (wg.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-8)
    wq = (wg / sc).round().clamp(-8, 7).to(torch.int8).reshape(e, nd, kk)
    wp = torch.zeros(e, nd, kk // 2, dtype=torch.int8)
    for i in range(0, kk, 2):
        wp[:, :, i // 2] = ((wq[:, :, i] & 0xF) << 4) | (wq[:, :, i + 1] & 0xF)
    return wp, sc.squeeze(-1).reshape(e, nd, ng).to(torch.bfloat16)

def quant_int8(x, gs=32):
    m, kk = x.shape; ng = kk // gs
    xg = x.float().reshape(m, ng, gs)
    sc = (xg.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-8)
    return (xg / sc).round().clamp(-128, 127).to(torch.int8).reshape(m, kk), \
           sc.squeeze(-1).reshape(m, ng).to(torch.bfloat16)

w1_bf = torch.randn(E, 2 * Ninter, K, dtype=torch.bfloat16) / 10
inp_bf = torch.randn(M, K, dtype=torch.bfloat16) / 10
w1q, w1s = quant_int4(w1_bf)
a1q, a1s = quant_int8(inp_bf)

# MoE routing: token t -> experts (t+k)%E
sids = torch.tensor([t for t in range(M) for _ in range(topk)], dtype=torch.int32)
seids = torch.tensor([(t + k) % E for t in range(M) for k in range(topk)], dtype=torch.int32)
swts = torch.full((M * topk,), 1.0 / topk, dtype=torch.bfloat16)
nv = torch.tensor(M * topk, dtype=torch.int32)

print(f"[CFG] M={M} K={K} Ninter={Ninter} E={E} topk={topk} tile={tile_m}x{tile_n}x{tile_k}")
print(f"[CFG] num_acc_n = {tile_n // 64} (tile_n//64)")
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
out = flydsl_moe_stage1(a=a1q, w1=w1q, sorted_token_ids=sids, sorted_expert_ids=seids,
    num_valid_ids=nv, topk=topk, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    a_dtype='int8', b_dtype='int4', out_dtype='bf16',
    w1_scale=w1s, a1_scale=a1s, sorted_weights=swts)
torch.cuda.synchronize()
print(f"[KERNEL] out[:6]={out.reshape(-1)[:6].tolist()}")

# Reference
def unpack_int4(packed):
    e, nd, k2 = packed.shape; kk = k2 * 2
    o = torch.zeros(e, nd, kk, dtype=torch.float32)
    for i in range(k2):
        hi = (packed[:, :, i].int() >> 4) & 0xF
        lo = packed[:, :, i].int() & 0xF
        o[:, :, 2 * i] = torch.where(hi >= 8, hi - 16, hi).float()
        o[:, :, 2 * i + 1] = torch.where(lo >= 8, lo - 16, lo).float()
    return o
ng = K // 32
w1i = unpack_int4(w1q)
w1d = (w1i.reshape(E, 2 * Ninter, ng, 32) * w1s.unsqueeze(-1).float()).reshape(E, 2 * Ninter, K)
a1d = (a1q.float().reshape(M, ng, 32) * a1s.unsqueeze(-1).float()).reshape(M, K)
ref = torch.zeros(M, topk, Ninter, dtype=torch.float32)
for t in range(M):
    for k in range(topk):
        e = (t + k) % E
        g = a1d[t] @ w1d[e, :Ninter, :].T
        u = a1d[t] @ w1d[e, Ninter:, :].T
        ref[t, k, :] = F.silu(g) * u * (1.0 / topk)

ko = out.float()
mask = ref.abs() > 1e-4
d = (ref - ko).abs()
print(f"[REF]    ref[:6]={ref.reshape(-1)[:6].tolist()}")
print(f"[CHECK] nonzero_kernel_outs={(ko.abs()>1e-6).sum().item()}/{ko.numel()}")
print(f"[CHECK] max_diff={d.max().item():.4f}  mean_rel_diff={(d[mask]/(ref[mask].abs()+1e-6)).mean().item()*100:.2f}%")
ea = ko.reshape(-1)[0::2].abs().mean().item()
oa = ko.reshape(-1)[1::2].abs().mean().item()
r = min(ea, oa) / (max(ea, oa) + 1e-8)
print(f"[CHECK] even_abs={ea:.4f} odd_abs={oa:.4f} evenodd_ratio={r:.3f}")
if r > 0.3 and (ko.abs() > 1e-6).any():
    if d.max().item() < 0.05: print("[VERDICT] CORRECT (no alt-zeros, low diff)")
    else: print("[VERDICT] COMPUTES but wrong values (no alt-zeros) -> layout/scale issue")
elif not (ko.abs() > 1e-6).any():
    print("[VERDICT] ALL ZEROS (still no compute)")
else:
    print("[VERDICT] ALTERNATING ZEROS persist")
