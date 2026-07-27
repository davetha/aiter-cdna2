#!/usr/bin/env bash
# Build a vLLM image that can actually reach AITER ASM kernels on gfx90a.
#
#   ./build_vllm_aiter_gfx90a.sh [output-tag] [aiter-tag]
#
# Stock rocm/vllm cannot use AITER on an MI210, and it says nothing about it.
# The engine logs a perfectly normal-looking
#
#   Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN','TRITON_ATTN']
#
# and serves from the generic ROCm path. AITER never enters the candidate list,
# so VLLM_ROCM_USE_AITER=1 has no effect at all. Measured cost on
# Qwen3-30B-A3B AWQ: 15.1k prefill at ~2,660 tok/s, roughly 17 TFLOP/s
# effective against the card's 181 TFLOP/s bf16 peak, while this repo's own ASM
# flash attention benchmarked at 89.9 TFLOP/s on the same silicon.
#
# Four things are broken, at four layers. All must be fixed or the stack
# quietly stays slow:
#
#   0. The image ships amd-aiter 0.1.13. Upstream is v0.1.19 (2026-07-27). The
#      repo's patches were derived against 0.1.17+, and 0.1.13's gates are
#      spelled differently (`get_gfx() == "gfx942"` rather than the tuple
#      form), so they refuse to apply. We install v0.1.19 from source rather
#      than pinning backwards.
#   1. AITER ships NO gfx90a code objects -- only gfx942, gfx950, gfx1250.
#      repatch_gfx942_to_gfx90a.py disassembles each gfx942 kernel, substitutes
#      CDNA2 mnemonics, RE-ASSEMBLES for gfx90a, and emits only those that
#      genuinely translate. Portability is proven by the assembler, never
#      assumed; kernels that cannot translate are reported, not silently
#      skipped.
#   2. AITER's own dispatch refuses gfx90a in ~11 places, including a negated
#      arch test in mha_fwd.cu that returns -1 before kernel lookup.
#      enable_gfx90a_asm_paths.py opens exactly those.
#   3. vLLM gates AITER behind on_mi3xx() = gfx942|gfx950.
#      enable_vllm_aiter_gfx90a.py widens ONLY the two attention checks, so
#      AITER's GEMM/MoE/FP8 paths stay unreachable on a chip with no FP8 ALU.
#
# Nothing here is committed to an image until every layer verifies, because a
# half-patched image fails by being SLOW rather than by erroring -- which is
# exactly the failure this repo has already published bad numbers from twice.
set -euo pipefail

TAG="${1:-rocm-vllm-aiter-gfx90a:latest}"
AITER_TAG="${2:-v0.1.19}"
BASE="rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0"
CFG="${CFG_DIR:-/mnt/llm-storage/bench-matrix/configs}"
BUILDER="vllm-aiter-build"
SITE=/opt/python/lib/python3.14/site-packages

echo "=== base  : $BASE"
echo "=== aiter : $AITER_TAG"
echo "=== out   : $TAG"

docker rm -f "$BUILDER" >/dev/null 2>&1 || true
docker run -d --name "$BUILDER" \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
  -v "$CFG":/cfg \
  -v /var/cache/mi210-ccache:/ccache \
  -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=100G -e CCACHE_DEPEND=1 \
  -e CCACHE_SLOPPINESS=locale,time_macros,include_file_ctime,include_file_mtime \
  --entrypoint sleep "$BASE" infinity

echo ""
echo "=== 1. install AITER $AITER_TAG from source"
# --recursive is required: aiter vendors Composable Kernel as a submodule and a
# non-recursive clone builds into confusing missing-header errors much later.
# PREBUILD_KERNELS is left off so the huge AOT kernel build is skipped and
# AITER's JIT compiles what is actually used -- with the shared ccache behind
# it, that is the difference between minutes and hours.
docker exec "$BUILDER" bash -lc "
  set -e
  ccache --zero-stats >/dev/null 2>&1 || true
  git clone --recursive --depth 1 --branch $AITER_TAG \
      https://github.com/ROCm/aiter.git /src/aiter
  cd /src/aiter
  python3 -m pip uninstall -y amd-aiter aiter 2>/dev/null || true
  python3 -m pip install --no-build-isolation . 2>&1 | tail -15

  # Installing aiter drags triton DOWN from the 3.7.1 the image ships to
  # 3.7.0, and 3.7.0 SEGFAULTS on import against this image's torch --
  # 'import aiter' dies with SIGSEGV inside triton/knobs.py loading a native
  # module. amd-aiter does not even declare triton as a dependency
  # (einops, flydsl, ninja, packaging, pandas, psutil, pybind11), so this is
  # a transitive downgrade, not an intentional pin. Restore it explicitly.
  # Confirmed: with 3.7.1 back, 'import aiter' succeeds and JIT-builds
  # module_aiter_core normally.
  python3 -m pip install --quiet 'triton==3.7.1'
  python3 -c 'import triton; assert triton.__version__ == \"3.7.1\", triton.__version__'
"

# Import aiter for real before going further. A broken aiter does not fail the
# later steps -- they are file edits and would all 'succeed' -- it fails at
# serve time, hours later, looking like a model problem.
docker exec "$BUILDER" python3 -c "import aiter; print('aiter imports OK')" 2>&1 | tail -2
docker exec "$BUILDER" python3 -c "
import aiter, os
print('aiter at', os.path.dirname(aiter.__file__))
" 2>&1 | tail -2

echo ""
echo "=== 2. translate gfx942 ASM -> gfx90a"
docker exec "$BUILDER" bash -lc "
  set -e
  HSA=\$(ls -d $SITE/aiter_meta/hsa 2>/dev/null || ls -d $SITE/aiter/hsa 2>/dev/null || ls -d /src/aiter/hsa)
  echo \"hsa root: \$HSA\"
  ls \"\$HSA\"
  python3 /cfg/repatch_gfx942_to_gfx90a.py \"\$HSA/gfx942\" /tmp/gfx90a 2>&1 | tail -6
  rm -rf \"\$HSA/gfx90a\"
  cp -r /tmp/gfx90a \"\$HSA/gfx90a\"
  echo \"installed \$(find \"\$HSA/gfx90a\" -name '*.co' | wc -l) gfx90a code objects\"
"

echo ""
echo "=== 3. open AITER's gfx90a dispatch paths"
docker exec "$BUILDER" python3 /cfg/enable_gfx90a_asm_paths.py

echo ""
echo "=== 4. let vLLM route attention to AITER"
docker exec "$BUILDER" python3 /cfg/enable_vllm_aiter_gfx90a.py

echo ""
echo "=== 5. verify every layer before committing"
docker exec "$BUILDER" python3 /cfg/enable_vllm_aiter_gfx90a.py --check
docker exec "$BUILDER" bash -lc "
  HSA=\$(ls -d $SITE/aiter_meta/hsa 2>/dev/null || ls -d $SITE/aiter/hsa)
  n=\$(find \"\$HSA/gfx90a\" -name '*.co' 2>/dev/null | wc -l)
  echo \"gfx90a code objects: \$n\"
  [ \"\$n\" -gt 0 ] || { echo 'FATAL: no gfx90a code objects' >&2; exit 1; }
"
docker exec "$BUILDER" sh -c "ccache -s 2>/dev/null | head -4" || true

echo ""
echo "=== 6. commit"
docker commit "$BUILDER" "$TAG" >/dev/null
docker rm -f "$BUILDER" >/dev/null 2>&1 || true
echo "committed $TAG"
echo ""
echo "Serve with VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1."
echo "PROOF OF USE, in order of strength:"
echo "  1. the log must NOT contain 'Overriding with ROCM_ATTN'"
echo "  2. with AITER_LOG_LEVEL=info the runtime prints the .co it loads per"
echo "     ASM call -- that line is the only direct evidence the ASM kernel"
echo "     ran rather than a silent fallback to CK or Triton."
