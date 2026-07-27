"""Time vLLM's Triton block-scaled FP8 GEMM against bf16 on gfx90a.

Serving Qwen3-14B-FP8 on an MI210 is ~15x slower per token than the bf16
checkpoint, even though the ASM attention kernels are identical in both runs
(same LoadKernel lines). That leaves the linear layers as the only suspect.
This isolates them: same shapes, same device, FP8 block-scaled GEMM versus the
plain bf16 matmul it replaces.

vLLM selects TritonFp8BlockScaledMMKernel here because every other FP8 path is
unavailable on CDNA2 -- torch._scaled_mm is hardware-gated to MI300+, and
CUTLASS and Marlin are CUDA-only. It also warns that it has no tuned config for
this device:

    Using default W8A8 Block FP8 kernel config. Performance might be
    sub-optimal! Config file not found at ...device_name=AMD_Instinct_MI210...

    python bench_fp8_block_gemm_gfx90a.py
"""
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)

BLOCK = [128, 128]

# Qwen3-14B projection shapes (hidden 5120), as (name, N, K).
SHAPES = [
    ("qkv_proj", 7168, 5120),
    ("o_proj", 5120, 5120),
    ("gate_up_proj", 34816, 5120),
    ("down_proj", 5120, 17408),
]
# Token counts: batch-1 decode, batch-32 decode, prefill chunk.
TOKENS = [1, 32, 4096]


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


def main() -> None:
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"{'layer':<14} {'M':>5} {'N':>6} {'K':>6} "
          f"{'bf16 us':>9} {'fp8 us':>9} {'fp8 TFLOP/s':>12} {'slowdown':>9}")

    for m in TOKENS:
        for name, n, k in SHAPES:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            w_bf16 = torch.randn(n, k, device=dev, dtype=torch.bfloat16)

            # Block-quantized weight and its scales, laid out as the kernel wants.
            w_fp8 = w_bf16.to(torch.float8_e4m3fn)
            ws = torch.ones(
                (n + BLOCK[0] - 1) // BLOCK[0],
                (k + BLOCK[1] - 1) // BLOCK[1],
                device=dev, dtype=torch.float32,
            )
            qx, xs = per_token_group_quant_fp8(x, BLOCK[1])

            t_bf16 = timeit(lambda: torch.matmul(x, w_bf16.t()))
            t_fp8 = timeit(
                lambda: w8a8_triton_block_scaled_mm(
                    qx, w_fp8, xs, ws, BLOCK, torch.bfloat16
                )
            )
            tflops = 2 * m * n * k / (t_fp8 * 1e-6) / 1e12
            print(f"{name:<14} {m:>5} {n:>6} {k:>6} "
                  f"{t_bf16:>9.1f} {t_fp8:>9.1f} {tflops:>12.1f} "
                  f"{t_fp8 / t_bf16:>8.1f}x")


if __name__ == "__main__":
    main()
