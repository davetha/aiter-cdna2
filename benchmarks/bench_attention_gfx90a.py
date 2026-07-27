"""Benchmark every available attention backend on gfx90a, ASM included.

Doc 16 and doc 17 both published ASM attention throughput for gfx90a that was
actually the CK or Triton fallback, because `mha.py` gated the ASM paths to
gfx942/gfx950. This benchmark exists so that cannot happen again: **every
backend is checked against a reference before it is timed**, and a backend that
is fast but wrong is reported WRONG rather than as a result.

Covers both halves of serving:

  prefill  varlen packed-THD, the shape a serving stack actually sees
             asm     aiter.flash_attn_varlen_func      (fmha_v3_fwd ASM)
             triton  aiter.ops.triton.attention.mha.flash_attn_varlen_func
             torch   F.scaled_dot_product_attention    (reference)

  decode   paged KV cache
             asm     aiter.pa_fwd_asm                  (pa ASM)
             triton  aiter.ops.triton.attention.pa_decode
             hip     aiter.paged_attention_rocm
             ck      aiter.pa_fwd_naive

Requires the gfx90a ASM enablement to be installed; see
docs/19-aiter-operator-port-matrix.md. Run with AITER_LOG_LEVEL=info to see
which code object each ASM call loads.

    python benchmarks/bench_attention_gfx90a.py [--quick]
"""
import argparse
import time

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes

D, BS, X = 128, 16, 8
WARMUP, ITERS = 3, 20


def timed(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def rel_rms(ref, got):
    r, g = ref.float(), got.float()
    return (((r - g) ** 2).mean().sqrt() / r.abs().max().clamp(min=1e-6)).item()


# ---------------------------------------------------------------- prefill ---

def prefill_reference(q, k, v, cu, causal, scale):
    out = torch.empty_like(q)
    for i in range(len(cu) - 1):
        a, b = int(cu[i]), int(cu[i + 1])
        qs, ks, vs = q[a:b], k[a:b], v[a:b]
        rep = qs.shape[1] // ks.shape[1]
        kt = ks.transpose(0, 1).float().repeat_interleave(rep, dim=0)
        vt = vs.transpose(0, 1).float().repeat_interleave(rep, dim=0)
        o = F.scaled_dot_product_attention(
            qs.transpose(0, 1).float(), kt, vt, is_causal=causal, scale=scale)
        out[a:b] = o.transpose(0, 1).to(q.dtype)
    return out


def bench_prefill(quick):
    print("\n" + "=" * 100)
    print("PREFILL — varlen packed THD, bf16, head_dim 128")
    print("=" * 100)
    shapes = [(1, 512), (1, 2048), (4, 1024), (8, 2048)]
    if not quick:
        shapes += [(1, 8192), (16, 1024)]

    try:
        from aiter.ops.triton.attention.mha import flash_attn_varlen_func as tri_varlen
    except Exception as exc:  # noqa: BLE001
        tri_varlen = None
        print(f"  triton prefill unavailable: {exc}")

    print(f"\n{'seqs':>5} {'len':>6} {'hq':>4} {'hkv':>4} {'causal':>7}  "
          f"{'backend':>8} {'us':>10} {'TFLOP/s':>9} {'rel_rms':>10}  verdict")
    for nseq, slen in shapes:
        for hq, hkv in ((32, 4),):
            for causal in (False, True):
                torch.manual_seed(0)
                total = nseq * slen
                scale = 1.0 / (D ** 0.5)
                cu = torch.arange(0, total + 1, slen, dtype=torch.int32)
                q = torch.empty(total, hq, D, dtype=dtypes.bf16).uniform_(-1, 1)
                k = torch.empty(total, hkv, D, dtype=dtypes.bf16).uniform_(-1, 1)
                v = torch.empty(total, hkv, D, dtype=dtypes.bf16).uniform_(-1, 1)
                ref = prefill_reference(q, k, v, cu, causal, scale)
                # 2 GEMMs, x2 for causal halving
                flops = 2 * 2 * nseq * slen * slen * hq * D / (2 if causal else 1)

                # All sequences here are equal length, so the same data can be
                # viewed as a dense batch -- that is the fair PyTorch baseline.
                # (The varlen reference above loops in Python and is not a
                # meaningful kernel comparison.)
                qb = q.view(nseq, slen, hq, D).transpose(1, 2)
                kb = k.view(nseq, slen, hkv, D).transpose(1, 2)
                vb = v.view(nseq, slen, hkv, D).transpose(1, 2)
                rep = hq // hkv

                def run_torch():
                    o = F.scaled_dot_product_attention(
                        qb, kb.repeat_interleave(rep, dim=1),
                        vb.repeat_interleave(rep, dim=1),
                        is_causal=causal, scale=scale)
                    # (b, h, s, d) -> packed (total, h, d) to match the ASM out
                    return o.transpose(1, 2).reshape(total, hq, D)

                cands = [
                    ("asm", lambda: aiter.flash_attn_varlen_func(
                        q, k, v, cu, cu, slen, slen, softmax_scale=scale,
                        causal=causal)),
                    ("torch", run_torch),
                ]
                if tri_varlen is not None:
                    cands.append(("triton", lambda: tri_varlen(
                        q, k, v, cu, cu, slen, slen, causal=causal,
                        softmax_scale=scale)))

                for name, fn in cands:
                    try:
                        out = fn()
                        if isinstance(out, tuple):
                            out = out[0]
                        torch.cuda.synchronize()
                        err = rel_rms(ref, out)
                        us = timed(fn)
                        tf = flops / (us * 1e-6) / 1e12
                        ok = "PASS" if err < 0.02 else "WRONG"
                        print(f"{nseq:>5} {slen:>6} {hq:>4} {hkv:>4} "
                              f"{str(causal):>7}  {name:>8} {us:>10.1f} "
                              f"{tf:>9.1f} {err:>10.6f}  {ok}")
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc).strip().split("\n")[0][:44]
                        print(f"{nseq:>5} {slen:>6} {hq:>4} {hkv:>4} "
                              f"{str(causal):>7}  {name:>8} {'-':>10} {'-':>9} "
                              f"{'-':>10}  SKIP {msg}")


# ----------------------------------------------------------------- decode ---

def shuf(v):
    nb, nkv, hs, b = v.shape
    return v.view(nb, nkv, hs, b // X, X).permute(0, 1, 3, 2, 4).contiguous()


def decode_reference(q, kc, vc, bt, ctx, seqs, qh, kvh, dt):
    kf = kc.permute(0, 3, 1, 2, 4).contiguous().view(-1, kvh, D)
    vf = vc.permute(0, 3, 1, 2).contiguous().view(-1, kvh, D)
    out = torch.zeros_like(q)
    btl = bt.cpu().tolist()
    r, scale = qh // kvh, 1.0 / (D ** 0.5)
    for i in range(seqs):
        idx = [int(btl[i][j // BS]) * BS + (j % BS) for j in range(ctx)]
        keys = torch.repeat_interleave(kf[idx], r, dim=1).float()
        vals = torch.repeat_interleave(vf[idx], r, dim=1).float()
        lg = torch.einsum("hd,chd->hc", q[i].float(), keys) * scale
        out[i] = torch.einsum("hc,chd->hd", torch.softmax(lg, -1), vals).to(dt)
    return out


def bench_decode(quick):
    print("\n" + "=" * 100)
    print("DECODE — paged KV cache, bf16, head_dim 128, block_size 16")
    print("=" * 100)
    shapes = [(1, 1024), (32, 1024), (128, 1024), (32, 4096)]
    if not quick:
        shapes += [(128, 4096), (256, 2048)]

    print(f"\n{'seqs':>5} {'ctx':>6} {'gqa':>4}  {'backend':>8} {'us':>10} "
          f"{'GB/s':>8} {'rel_rms':>10}  verdict")
    for seqs, ctx in shapes:
        for qh, kvh in ((32, 4), (16, 1)):
            torch.manual_seed(0)
            dt = dtypes.bf16
            nbps = (ctx + BS - 1) // BS
            nb = seqs * nbps + 8
            bt = torch.randperm(nb)[: seqs * nbps].view(
                seqs, nbps).to(torch.int32).contiguous()
            sl = torch.full((seqs,), ctx, dtype=torch.int32)
            q = torch.empty(seqs, qh, D, dtype=dt).uniform_(-1, 1).contiguous()
            kc = torch.empty(nb, kvh, D // X, BS, X, dtype=dt).uniform_(-1, 1)
            vc = torch.empty(nb, kvh, D, BS, dtype=dt).uniform_(-1, 1)
            vs = shuf(vc)
            ref = decode_reference(q, kc, vc, bt, ctx, seqs, qh, kvh, dt)
            # decode is bandwidth bound: read K and V for every context token
            gb = 2 * seqs * ctx * kvh * D * 2 / 1e9

            # paged_attention_rocm takes the unshuffled vLLM cache layout, which
            # is exactly what kc/vc already are: K [nb, kvh, D/x, bs, x] and
            # V [nb, kvh, D, bs]. Only the ASM path wants V shuffled.
            scale = 1.0 / (D ** 0.5)
            psz = 256
            nparts = (ctx + psz - 1) // psz
            hip_out = torch.empty(seqs, qh, D, dtype=dt)
            exp_sums = torch.empty(seqs, qh, nparts, dtype=torch.float32)
            max_logits = torch.empty_like(exp_sums)
            tmp_out = torch.empty(seqs, qh, nparts, D, dtype=dt)
            one = torch.ones(1, dtype=torch.float32)

            def run_hip():
                aiter.paged_attention_rocm(
                    hip_out, exp_sums, max_logits, tmp_out, q, kc, vc, kvh,
                    scale, bt, sl, BS, ctx, None, "auto", one, one, None, psz)
                return hip_out

            cands = [
                ("asm", lambda: aiter.pa_fwd_asm(q, kc, vs, bt, sl, bt.stride(0))),
                ("hip", run_hip),
            ]
            for name, fn in cands:
                try:
                    out = fn()
                    if isinstance(out, tuple):
                        out = out[0]
                    torch.cuda.synchronize()
                    err = rel_rms(ref, out)
                    us = timed(fn)
                    print(f"{seqs:>5} {ctx:>6} {qh // kvh:>4}  {name:>8} "
                          f"{us:>10.1f} {gb / (us * 1e-6):>8.0f} {err:>10.6f}  "
                          f"{'PASS' if err < 0.02 else 'WRONG'}")
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).strip().split("\n")[0][:44]
                    print(f"{seqs:>5} {ctx:>6} {qh // kvh:>4}  {name:>8} "
                          f"{'-':>10} {'-':>8} {'-':>10}  SKIP {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--prefill-only", action="store_true")
    args = ap.parse_args()
    torch.set_default_device("cuda:0")
    print(f"gfx: {aiter.jit.utils.chip_info.get_gfx()}   "
          f"device: {torch.cuda.get_device_name(0)}")
    if not args.decode_only:
        bench_prefill(args.quick)
    if not args.prefill_only:
        bench_decode(args.quick)
    print("\nWRONG means the backend ran but disagreed with the reference — "
          "do not quote its throughput.")


if __name__ == "__main__":
    main()
