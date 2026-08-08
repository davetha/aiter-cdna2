#!/usr/bin/env python3
"""Dump the W4A8 FlyDSL kernel GCN assembly on gfx90a.

The dump fires at kernel INVOCATION time (lazy JitFunction), so we must
actually run the kernel with FLYDSL_DUMP_IR=1 set. Clear the on-disk cache
first so lowering is forced.
"""
import os, glob, time, shutil

P = "/opt/python/lib/python3.14/site-packages"
DUMP_DIR = "/hosttmp/w4a8_dump"
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = DUMP_DIR

# Clear any FlyDSL on-disk cache so lowering (and the dump) is forced
for cd in [os.path.expanduser("~/.flydsl"), "/root/.flydsl"]:
    shutil.rmtree(cd, ignore_errors=True)
shutil.rmtree(DUMP_DIR, ignore_errors=True)
os.makedirs(DUMP_DIR, exist_ok=True)

# --- Patch 1: SMEM_CAPACITY_MAP gfx90a ---
sm = f"{P}/flydsl/utils/smem_allocator.py"
with open(sm) as f: s = f.read()
if "gfx90a" not in s:
    s = s.replace('"gfx942": 65536,', '"gfx90a": 65536,\n    "gfx942": 65536,', 1)
    with open(sm, "w") as f: f.write(s)
    print("[PATCH] SMEM_CAPACITY_MAP += gfx90a")

# --- Patch 2: W4A8 dispatch case ---
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
    with open(mk, "w") as f: f.write(s)
    print("[PATCH] moe_kernels += W4A8 dispatch")

# Clear .pyc
for root, dirs, files in os.walk(P):
    for fn in files:
        if fn.endswith(".pyc"):
            try: os.remove(os.path.join(root, fn))
            except OSError: pass

from flydsl.utils import env
print(f"[ENV] dump_ir={env.debug.dump_ir!r} dump_dir={env.debug.dump_dir!r}")

import torch  # noqa
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

print("[INVOKE] running flydsl_moe_stage1 (forces lowering + dump)...")
t0 = time.time()
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
out = flydsl_moe_stage1(a=a1q, w1=w1q, sorted_token_ids=sids,
    sorted_expert_ids=seids, num_valid_ids=nv, topk=topk,
    tile_m=bm, tile_n=32, tile_k=256, a_dtype='int8', b_dtype='int4',
    out_dtype='bf16', w1_scale=w1s, a1_scale=a1s, sorted_weights=swts)
torch.cuda.synchronize()
print(f"[INVOKE] done in {time.time()-t0:.1f}s  out={out.reshape(-1)[:4].tolist()}")

# Scan dump artifacts (broad)
print("\n===== DUMP ARTIFACTS =====")
files = sorted(glob.glob(f"{DUMP_DIR}/**/*", recursive=True))
files = [f for f in files if os.path.isfile(f)]
for p in files:
    print(f"  {os.path.relpath(p, DUMP_DIR)}  ({os.path.getsize(p)} bytes)")
print(f"TOTAL: {len(files)} files")

# Grep MFMA across all dump files
import re
from collections import Counter
allmfs = Counter()
isa_file = None
for p in files:
    try:
        txt = open(p, errors="ignore").read()
    except Exception:
        continue
    mfs = re.findall(r"(?:v_mfma|rocdl\.mfma|mfma_i32)\S*", txt)
    for m in mfs: allmfs[m] += 1
    if "v_mfma" in txt and p.endswith(".s"): isa_file = p

print("\n===== MFMA occurrences across all dump stages =====")
for instr, cnt in allmfs.most_common():
    print(f"  {instr}: {cnt}")

if not allmfs:
    print("[!] NO mfma found in any dump file.")
    # Show what stages exist
    for p in files[:5]:
        print(f"  --- head of {os.path.basename(p)} ---")
        print("\n".join(open(p, errors='ignore').read().splitlines()[:8]))
