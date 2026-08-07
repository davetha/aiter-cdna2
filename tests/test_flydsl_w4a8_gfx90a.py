#!/usr/bin/env python3
"""Test: FlyDSL W4A8 (int4 weight + int8 act + INT8 MFMA) MoE kernel on gfx90a.

Prerequisites:
  python ../patches/enable_flydsl_w4a8_gfx90a.py

Run (inside the ROCm container with GPU access):
  python test_flydsl_w4a8_gfx90a.py

The kernel COMPILES and EXECUTES on gfx90a (verified 2026-08-06). Output shape
is correct ([M, topk, N] bf16). However, correctness is UNVERIFIED — see the
K=32/K=16 MFMA shape caveat in docs/w4a8-flydsl.md and docs/int8-gemm.md.
"""
import torch

torch.set_default_device("cuda")

M, K, N, E, topk, bm = 32, 256, 256, 4, 2, 32
torch.manual_seed(42)


def quant_int4(w, gs=32):
    E, N, K = w.shape
    ng = K // gs
    wg = w.float().reshape(E, N, ng, gs)
    sc = (wg.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-8)
    wq = (wg / sc).round().clamp(-8, 7).to(torch.int8).reshape(E, N, K)
    wp = torch.zeros(E, N, K // 2, dtype=torch.int8)
    for i in range(0, K, 2):
        wp[:, :, i // 2] = ((wq[:, :, i] & 0xF) << 4) | (wq[:, :, i + 1] & 0xF)
    return wp, sc.squeeze(-1).reshape(E, N, ng).to(torch.bfloat16)


def quant_int8(x, gs=32):
    M, K = x.shape
    ng = K // gs
    xg = x.float().reshape(M, ng, gs)
    sc = (xg.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-8)
    xq = (xg / sc).round().clamp(-128, 127).to(torch.int8).reshape(M, K)
    return xq, sc.squeeze(-1).reshape(M, ng).to(torch.bfloat16)


w1 = torch.randn(E, N * 2, K, dtype=torch.bfloat16) / 10
w1q, w1s = quant_int4(w1)
inp = torch.randn(M, K, dtype=torch.bfloat16) / 10
a1q, a1s = quant_int8(inp)

sids, seids, swts = [], [], []
for t in range(M):
    for k in range(topk):
        sids.append(t)
        seids.append((t + k) % E)
        swts.append(1.0 / topk)
sorted_ids = torch.tensor(sids, dtype=torch.int32)
sorted_expert_ids = torch.tensor(seids, dtype=torch.int32)
sorted_weights = torch.tensor(swts, dtype=torch.bfloat16)
num_valid = torch.tensor(M * topk, dtype=torch.int32)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Data: M={M} K={K} N={N} E={E} topk={topk}")

from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

out = flydsl_moe_stage1(
    a=a1q, w1=w1q,
    sorted_token_ids=sorted_ids,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid,
    topk=topk, tile_m=bm, tile_n=32, tile_k=256,
    a_dtype="int8", b_dtype="int4", out_dtype="bf16",
    w1_scale=w1s, a1_scale=a1s,
    sorted_weights=sorted_weights,
)
torch.cuda.synchronize()
print(f"Output: {out.shape} {out.dtype}")
print(f"Sample: {out.reshape(-1)[:8]}")
print("\nW4A8 kernel compiled + executed on gfx90a.")
print("WARNING: correctness unverified (K=32/K=16 MFMA shape risk).")
print("Next: compare against bf16 reference on identical data.")
