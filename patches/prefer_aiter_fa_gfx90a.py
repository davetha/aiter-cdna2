"""Make vLLM's AITER flash-attention backend selectable ahead of ROCM_ATTN.

`enable_vllm_aiter_gfx90a.py` gets ROCM_AITER_FA into the candidate list on
gfx90a. It does not get it *chosen*. Backend selection in
`vllm/platforms/rocm.py:_get_backend_priorities()` is a plain ordered list, and
the first valid entry wins:

    backends = []
    if not use_kv_connector:
        backends.append(AttentionBackendEnum.ROCM_ATTN)      # priority 0
    if rocm_aiter_ops.is_mha_enabled():
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)  # priority 1
    ...

ROCM_ATTN is appended first unconditionally, so ROCM_AITER_FA is only ever
reached when ROCM_ATTN is *invalid*. The observable result on an MI210, with
every AITER gate already opened, is a log line that looks like success:

    Overriding with ROCM_ATTN out of potential backends:
        ['ROCM_ATTN', 'ROCM_AITER_FA', 'TRITON_ATTN']

AITER is present, admitted, and unused.

This is a hardcoded preference, not a capability test -- the ordering is the
same on MI300, so upstream evidently prefers ROCM_ATTN in general. That may
well be right on CDNA3. It is worth questioning on CDNA2, where ROCM_ATTN's
custom paged-decode kernel is refused above `max_seq_len > 128*1024`
(`use_rocm_custom_paged_attention()`) and decode falls back to a Triton
implementation for which aiter ships no gfx90a configs at all.

So this patch does NOT reorder unconditionally. It makes the order switchable:

    VLLM_PREFER_AITER_FA=1   -> ROCM_AITER_FA first
    unset / 0                -> stock order, ROCM_ATTN first

An env switch rather than a static reorder, because the whole point is to
measure which is actually faster on this hardware, and a benchmark whose two
arms need different images is not a benchmark. Both arms run the same binary
and differ by one variable.

    python prefer_aiter_fa_gfx90a.py [--revert] [--check]
"""
import argparse
import sys

SITE = "/opt/python/lib/python3.14/site-packages"
ROCM_PY = f"{SITE}/vllm/platforms/rocm.py"

_ANCHOR = (
    "    backends = []\n"
    "    # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is\n"
    "    # incompatible with KV connectors that require blocks-first layout.\n"
    "    if not use_kv_connector:\n"
    "        backends.append(AttentionBackendEnum.ROCM_ATTN)\n"
    "    if rocm_aiter_ops.is_mha_enabled():\n"
    "        backends.append(AttentionBackendEnum.ROCM_AITER_FA)\n"
)

_PATCHED = (
    "    backends = []\n"
    "    # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is\n"
    "    # incompatible with KV connectors that require blocks-first layout.\n"
    "    #\n"
    "    # gfx90a: VLLM_PREFER_AITER_FA=1 puts ROCM_AITER_FA ahead of\n"
    "    # ROCM_ATTN. Upstream appends ROCM_ATTN first unconditionally and the\n"
    "    # first valid backend wins, so AITER flash attention is otherwise\n"
    "    # admitted to the candidate list and never selected. Off by default:\n"
    "    # this exists to A/B the two on CDNA2, not to assert a winner.\n"
    "    _prefer_aiter_fa = os.environ.get('VLLM_PREFER_AITER_FA', '0') == '1'\n"
    "    if _prefer_aiter_fa and rocm_aiter_ops.is_mha_enabled():\n"
    "        backends.append(AttentionBackendEnum.ROCM_AITER_FA)\n"
    "    if not use_kv_connector:\n"
    "        backends.append(AttentionBackendEnum.ROCM_ATTN)\n"
    "    if not _prefer_aiter_fa and rocm_aiter_ops.is_mha_enabled():\n"
    "        backends.append(AttentionBackendEnum.ROCM_AITER_FA)\n"
)

PATCHES = [(ROCM_PY, _ANCHOR, _PATCHED, 1)]


def apply(revert: bool = False, check: bool = False) -> int:
    failures = 0
    for path, old, new, want in PATCHES:
        if revert:
            old, new = new, old
        try:
            text = open(path).read()
        except OSError as exc:
            print(f" ERROR    {path}: {exc}")
            failures += 1
            continue
        name = path.rsplit("/", 1)[-1]
        n_new, n_old = text.count(new), text.count(old)

        if check:
            print(f" {'patched' if n_new else 'unpatched':>9}  {name} "
                  f"(unpatched {n_old} / patched {n_new})")
            if n_old + n_new != want:
                failures += 1
            continue
        if n_new == want and n_old == 0:
            print(f" already    {name}")
            continue
        if n_old != want:
            print(f" ERROR      {name}: expected {want}, found {n_old}. "
                  f"Upstream moved -- reread the source, do not force it.")
            failures += 1
            continue
        open(path, "w").write(text.replace(old, new, want))
        print(f" patched    {name}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    failures = apply(revert=args.revert, check=args.check)
    if failures:
        print(f"\n{failures} site(s) did not match.")
        return 1
    if not args.check:
        print("\nOK. Serve with VLLM_PREFER_AITER_FA=1 and confirm the log says "
              "'Using ROCM_AITER_FA' rather than 'Overriding with ROCM_ATTN'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
