"""Correctness + perf validation of AITER's ASM paged-attention decode on gfx90a.

Compares `aiter.pa_fwd_asm` against a pure-PyTorch reference across dtype, GQA
ratio, context length and batch size. Deliberately standalone: it does not use
aiter's own op_tests harness, which pulls in the Composable Kernel reference path
and fails to build in some containers.

Expected on a correctly patched MI210: 48/48 PASS, rel_rms ~1e-4..8e-4, 100%
element match. See docs/18-pa-fwd-asm-resolved.md.

    python tests/test_pa_fwd_asm_gfx90a.py

NOTE: bf16 gqa16 was the case that exposed the bad D3E1->D3CD MFMA patch
(rel_rms ~272 at 0.2% match), so keep gqa 16 in the grid.
"""
import time, torch, aiter
from aiter import dtypes

torch.set_default_device("cuda:0")
D, bs, x = 128, 16, 8


def shuf(V):
    nb, nkv, hs, b = V.shape
    return V.view(nb, nkv, hs, b // x, x).permute(0, 1, 3, 2, 4).contiguous()


def reference(query, k_cache, v_cache, bt, ctx, seqs, qh, kvh, dt):
    kf = k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, kvh, D)
    vf = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, kvh, D)
    out = torch.zeros_like(query)
    btl = bt.cpu().tolist()
    r = qh // kvh
    scale = 1.0 / (D ** 0.5)
    for i in range(seqs):
        idx = [int(btl[i][j // bs]) * bs + (j % bs) for j in range(ctx)]
        keys = torch.repeat_interleave(kf[idx], r, dim=1).float()
        vals = torch.repeat_interleave(vf[idx], r, dim=1).float()
        lg = torch.einsum("hd,chd->hc", query[i].float(), keys) * scale
        out[i] = torch.einsum("hc,chd->hd", torch.softmax(lg, -1), vals).to(dt)
    return out


print(f"{'dtype':>5} {'qh':>4} {'kvh':>4} {'gqa':>4} {'seqs':>5} {'ctx':>6} "
      f"{'rel_rms':>10} {'%close':>8} {'us':>9}  verdict")
fails = 0
for dt in (dtypes.bf16, dtypes.fp16):
    for qh, kvh in ((32, 4), (8, 1), (32, 2), (16, 1)):
        for seqs, ctx in ((1, 16), (4, 57), (128, 128), (32, 257), (128, 1024), (8, 4097)):
            torch.manual_seed(0)
            nbps = (ctx + bs - 1) // bs
            NB = seqs * nbps + 8
            bt = torch.randperm(NB)[: seqs * nbps].view(seqs, nbps).to(torch.int32).contiguous()
            sl = torch.full((seqs,), ctx, dtype=torch.int32)
            q = torch.empty(seqs, qh, D, dtype=dt).uniform_(-1, 1).contiguous()
            kc = torch.empty(NB, kvh, D // x, bs, x, dtype=dt).uniform_(-1, 1)
            vc = torch.empty(NB, kvh, D, bs, dtype=dt).uniform_(-1, 1)
            vs = shuf(vc)
            ref = reference(q, kc, vc, bt, ctx, seqs, qh, kvh, dt)
            got = aiter.pa_fwd_asm(q, kc, vs, bt, sl, bt.stride(0))
            torch.cuda.synchronize()
            for _ in range(3):
                aiter.pa_fwd_asm(q, kc, vs, bt, sl, bt.stride(0))
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20):
                aiter.pa_fwd_asm(q, kc, vs, bt, sl, bt.stride(0))
            torch.cuda.synchronize()
            us = (time.perf_counter() - t) / 20 * 1e6
            r_, g_ = ref.float(), got.float()
            rms = (((r_ - g_) ** 2).mean().sqrt() / r_.abs().max().clamp(min=1e-6)).item()
            pc = torch.isclose(r_, g_, atol=2e-2, rtol=2e-2).float().mean().item() * 100
            ok = rms < 0.02 and pc > 99.0
            fails += (not ok)
            name = "bf16" if dt == dtypes.bf16 else "fp16"
            print(f"{name:>5} {qh:>4} {kvh:>4} {qh//kvh:>4} {seqs:>5} {ctx:>6} "
                  f"{rms:>10.6f} {pc:>7.2f}% {us:>8.1f}  {'PASS' if ok else 'FAIL'}")
print(f"\n{fails} failures")
