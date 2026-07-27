"""Measure what it costs gfx90a to decode one FP8 value, per format.

Block-scaled FP8 GEMM is ~12x slower than bf16 on an MI210, and the obvious
suspect -- vLLM warns about it by name -- is the missing tuning config. It is
not. Disassembling the kernel shows 64 MFMA instructions inside ~12,000, with
~7,100 of them `v_cmp_ne_u16`/`v_cndmask_b32` pairs. That is a software
emulation of the e4m3 -> fp16 conversion: gfx942 decodes FP8 with
`v_cvt_pk_fp8_f32`, and gfx90a has no such instruction.

This isolates the conversion from the GEMM. Each kernel does nothing but load
a byte, widen it to fp16 and scale it, so the instruction count is the decode
cost and nothing else.

Expect e4m3 (both the OCP `fn` and AMD `fnuz` spellings) to cost ~10x what
int8 and e5m2 cost. e5m2's exponent field is already fp16-shaped, so it decodes
with a shift; e4m3's 4-bit exponent needs re-biasing and its denormal boundary
moves, and Triton open-codes every case.

    python probe_fp8_convert_cost_gfx90a.py
"""
import collections
import glob
import os
import shutil
import subprocess
import tempfile

CACHE = os.path.join(tempfile.gettempdir(), "fp8_convert_probe")
shutil.rmtree(CACHE, ignore_errors=True)
os.makedirs(CACHE, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = CACHE

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

N = 8192


def build(name):
    @triton.jit
    def _k(src, dst, BLOCK: tl.constexpr, DT: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(src.to(tl.pointer_type(DT)) + offs)
        tl.store(dst + offs, v.to(tl.float16) * 2.0)

    _k.__name__ = name
    return _k


CASES = [
    ("e4m3fn_ocp", torch.float8_e4m3fn, tl.float8e4nv),
    ("e4m3fnuz_amd", torch.float8_e4m3fnuz, tl.float8e4b8),
    ("e5m2_ocp", torch.float8_e5m2, tl.float8e5),
    ("int8_baseline", torch.int8, tl.int8),
]


def objdump_bin():
    for cand in ("/opt/rocm/llvm/bin/llvm-objdump", "llvm-objdump"):
        if cand == "llvm-objdump" or os.path.exists(cand):
            return cand
    return "llvm-objdump"


def main() -> None:
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name()}\n")
    objdump = objdump_bin()

    results = []
    for name, torch_dt, tl_dt in CASES:
        subdir = os.path.join(CACHE, name)
        shutil.rmtree(subdir, ignore_errors=True)
        os.makedirs(subdir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = subdir

        src = torch.randint(0, 120, (N,), device=dev, dtype=torch.uint8).view(torch_dt)
        dst = torch.empty(N, device=dev, dtype=torch.float16)

        try:
            build(f"conv_{name}")[(N // 1024,)](
                src, dst, BLOCK=1024, DT=tl_dt, num_warps=4
            )
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"{name:<16} FAILED: {type(exc).__name__}: {exc}")
            results.append((name, None))
            continue

        counts = collections.Counter()
        for hsaco in glob.glob(os.path.join(subdir, "**", "*.hsaco"), recursive=True):
            dis = subprocess.run(
                [objdump, "-d", "--mcpu=gfx90a", hsaco],
                capture_output=True, text=True,
            )
            for line in dis.stdout.splitlines():
                if not line.startswith("\t"):
                    continue
                op = line.strip().split(" ")[0]
                if op.startswith(("v_", "s_", "ds_", "buffer_", "global_", "flat_")):
                    counts[op] += 1

        valu = sum(v for k, v in counts.items() if k.startswith("v_"))
        top = ", ".join(f"{k}={v}" for k, v in counts.most_common(6)
                        if k.startswith("v_"))
        print(f"{name:<16} total={sum(counts.values()):<6} VALU={valu}")
        print(f"{'':<16} {top}")
        results.append((name, valu))

    ref = dict(results).get("int8_baseline")
    print("\n--- VALU ops to decode 4 values per lane to fp16 ---")
    for name, valu in results:
        if valu is None:
            continue
        rel = f"  {valu / ref:.1f}x int8" if ref else ""
        print(f"  {name:<16} {valu:>5}{rel}")


if __name__ == "__main__":
    main()
