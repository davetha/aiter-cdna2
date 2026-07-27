"""Classify every gfx942 ASM kernel by why it can or cannot run on gfx90a.

Companion to `repatch_gfx942_to_gfx90a.py`. That script answers "can this kernel
be ported?"; this one answers "and if not, *what* is blocking it, and is the
blocker a hardware capability CDNA2 lacks or merely a syntax difference?"

The distinction matters. A mnemonic that gfx90a spells differently, or a cache
modifier that was renamed between CDNA2 and CDNA3, is recoverable by in-place
binary patching. A packed-bf16 atomic is not: no such instruction exists on
CDNA2, and emulating it needs a CAS loop, which changes code size and so cannot
be patched in place.

Method: disassemble every .co for gfx942 and collect the set of DISTINCT full
instruction TEXTS across the whole tree, then decide gfx90a-validity once per
distinct text with a batched llvm-mc call. Judging per *text* rather than per
*mnemonic* matters -- `flat_load_dword v13, v[12:13]` is valid gfx90a while
`flat_load_dword v13, v[12:13] sc0 sc1` is not, and a per-mnemonic verdict would
wrongly condemn the first because of the second.

Blockers are bucketed by the hardware capability they need:

    fp8         v_cvt_*_fp8/bf8, *_fp8_*/*_bf8_* MFMA -- no FP8 ALU on CDNA2
    int8        v_mfma_i32_* with gfx942 shapes       -- no such MFMA on CDNA2
    xf32        v_mfma_*_xf32_*                       -- no XF32 on CDNA2
    smfmac      v_smfmac_*                            -- no sparse MFMA on CDNA2
    gfx950      CDNA4-only MFMA shapes
    bf16_atomic global_atomic_pk_add_bf16             -- no packed-bf16 atomic
    valu64      v_lshl_add_u64, v_mov_b64             -- no 64-bit VALU forms
    cosmetic    renamed mnemonic or cache modifier    -- RECOVERABLE, see SUBST
    other       anything else                         -- investigate these

Usage (inside the ROCm container):

    python classify_gfx942_kernels.py <hsa/gfx942> [--gfx90a-dir <hsa/gfx90a>] \
        [--json out.json]

Prints a per-family table and a blocker-bucket tally. With --gfx90a-dir it also
reports which non-portable kernels are nevertheless installed under gfx90a/ --
those are the landmines: a kernel the loader can find but the GPU cannot run.

Result as of 2026-07-27 (aiter 0.1.17): 242 of 1422 portable. Every one of the
1180 blocked kernels needs at least one capability CDNA2 does not have; none is
blocked by a cosmetic difference alone, so 242 is the hard ceiling for binary
patching. See docs/19-aiter-operator-port-matrix.md.
"""
import argparse
import json
import os
import re
import struct
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field

MC = "/opt/rocm/llvm/bin/llvm-mc"
OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"

EFLAGS_OFF = 0x30
GFX942 = 0x4C

# gfx942 -> gfx90a rewrites that are semantically identical AND assemble to the
# same number of bytes, so they can be applied by in-place binary patching.
# Mirrors repatch_gfx942_to_gfx90a.py, plus the cosmetic cases this tool found.
SUBST = {
    # MFMA: gfx90a spells the bf16 and f16 forms without the underscore, and
    # uses the _1k suffix for the bf16 variants that take a full K=16 tile.
    "v_mfma_f32_16x16x16_bf16": "v_mfma_f32_16x16x16bf16_1k",
    "v_mfma_f32_32x32x8_bf16": "v_mfma_f32_32x32x8bf16_1k",
    "v_mfma_f32_4x4x4_bf16": "v_mfma_f32_4x4x4bf16_1k",
    "v_mfma_f32_16x16x16_f16": "v_mfma_f32_16x16x16f16",
    "v_mfma_f32_32x32x8_f16": "v_mfma_f32_32x32x8f16",
    "v_mfma_f32_4x4x4_f16": "v_mfma_f32_4x4x4f16",
    # VOP2 fused-multiply-add with literal K: gfx90a keeps the legacy name.
    "v_fmamk_f32": "v_madmk_f32",
    "v_fmaak_f32": "v_madak_f32",
}

# CDNA3 renamed the memory cache-control bits. SC0/SC1 express scope and NT
# expresses non-temporal; CDNA2 spells the equivalent conservative behaviour
# GLC/SLC. Applied to the operand tail, not the mnemonic.
MODIFIER_SUBST = [(" sc0", " glc"), (" sc1", " slc"), (" nt", " slc")]

BUCKETS = [
    ("fp8", re.compile(r"fp8|bf8")),
    ("smfmac", re.compile(r"smfmac")),
    ("xf32", re.compile(r"xf32")),
    ("int8", re.compile(r"mfma_i32|_i8\b|_iu8\b")),
    ("gfx950", re.compile(r"16x16x32_bf16|32x32x16_bf16|16x16x32_f16|32x32x16_f16")),
    ("bf16_atomic", re.compile(r"atomic_pk_add_bf16")),
    # v_mov_b64_e32 etc. carry an encoding suffix, so anchor on the type token
    # rather than a word boundary.
    ("valu64", re.compile(r"_u64|_b64")),
]


@dataclass
class Family:
    total: int = 0
    portable: int = 0
    blocked: int = 0
    installed_unrunnable: int = 0
    buckets: "defaultdict[str, int]" = field(
        default_factory=lambda: defaultdict(int))


def bucket_for(text):
    mnemonic = text.split()[0] if text.split() else text
    if mnemonic in SUBST or _modifier_only(text):
        return "cosmetic"
    for name, pat in BUCKETS:
        if pat.search(mnemonic):
            return name
    return "other"


def _modifier_only(text):
    """True if the text differs from valid gfx90a only by cache modifiers."""
    return any(old in text for old, _ in MODIFIER_SUBST)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def rewrite(text):
    """Apply the known gfx942 -> gfx90a rewrites to one instruction's text."""
    parts = text.split(None, 1)
    mnemonic = parts[0]
    tail = parts[1] if len(parts) > 1 else ""
    mnemonic = SUBST.get(mnemonic, mnemonic)
    for old, new in MODIFIER_SUBST:
        tail = tail.replace(old, new)
    return f"{mnemonic} {tail}".strip()


def gfx90a_valid(texts):
    """Return the subset of `texts` whose rewritten form assembles for gfx90a.

    One batched call resolves the common all-valid case. On any error, bisect so
    a single bad line cannot condemn the whole batch.
    """
    texts = list(texts)
    if not texts:
        return set()
    body = "\n".join(rewrite(t) for t in texts)
    p = run([MC, "-arch=amdgcn", "-mcpu=gfx90a", "-show-encoding"], input=body)
    if "error" not in p.stderr:
        return set(texts)
    if len(texts) == 1:
        return set()
    mid = len(texts) // 2
    return gfx90a_valid(texts[:mid]) | gfx90a_valid(texts[mid:])


def disassemble(path):
    """Return the list of instruction texts in one .co, in program order."""
    dis = run([OBJDUMP, "-d", "--mcpu=gfx942", path]).stdout
    out = []
    for line in dis.splitlines():
        m = re.match(r"^\t(.*?)\s*//\s*[0-9A-Fa-f]+:\s*[0-9A-Fa-f ]+$", line)
        if not m:
            continue
        text = m.group(1).strip()
        if text:
            out.append(text)
    return out


def is_gfx942(path):
    with open(path, "rb") as fh:
        head = fh.read(EFLAGS_OFF + 4)
    if len(head) < EFLAGS_OFF + 4:
        return False
    return struct.unpack_from("<I", head, EFLAGS_OFF)[0] & 0xFF == GFX942


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gfx942_dir")
    ap.add_argument("--gfx90a-dir", default=None,
                    help="installed hsa/gfx90a tree, to flag unrunnable installs")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    kernels = {}
    all_texts = set()
    for dirpath, _, files in os.walk(args.gfx942_dir):
        for fn in sorted(files):
            if not fn.endswith(".co"):
                continue
            src = os.path.join(dirpath, fn)
            if not is_gfx942(src):
                continue
            rel = os.path.relpath(src, args.gfx942_dir)
            texts = set(disassemble(src))
            kernels[rel] = texts
            all_texts |= texts

    valid = gfx90a_valid(sorted(all_texts))
    invalid = all_texts - valid

    installed = set()
    if args.gfx90a_dir:
        for dirpath, _, files in os.walk(args.gfx90a_dir):
            for fn in files:
                if fn.endswith(".co"):
                    installed.add(os.path.relpath(
                        os.path.join(dirpath, fn), args.gfx90a_dir))

    families: "defaultdict[str, Family]" = defaultdict(Family)
    records = []
    for rel, texts in sorted(kernels.items()):
        fam = rel.split(os.sep)[0] if os.sep in rel else "(top-level)"
        blocking = sorted(texts & invalid)
        buckets = sorted({bucket_for(t) for t in blocking})
        portable = not blocking
        f = families[fam]
        f.total += 1
        if portable:
            f.portable += 1
        else:
            f.blocked += 1
            for b in buckets:
                f.buckets[b] += 1
        unrunnable = (not portable) and rel in installed
        if unrunnable:
            f.installed_unrunnable += 1
        records.append({
            "kernel": rel, "family": fam, "portable": portable,
            "blocking_instructions": blocking[:20],
            "blocking_mnemonics": sorted({t.split()[0] for t in blocking}),
            "buckets": buckets,
            "installed_on_gfx90a": rel in installed,
            "installed_but_unrunnable": unrunnable,
        })

    print(f"{'family':<24} {'total':>6} {'port':>6} {'blocked':>8} "
          f"{'bad-install':>12}  blockers")
    for fam in sorted(families):
        f = families[fam]
        bl = ",".join(f"{k}:{v}" for k, v in sorted(f.buckets.items()))
        print(f"{fam:<24} {f.total:>6} {f.portable:>6} {f.blocked:>8} "
              f"{f.installed_unrunnable:>12}  {bl}")

    tot = len(records)
    port = sum(r["portable"] for r in records)
    bad = sum(r["installed_but_unrunnable"] for r in records)
    print(f"\n{tot} kernels: {port} portable, {tot - port} blocked")
    print(f"{bad} blocked kernels are installed under gfx90a/ and cannot run")

    agg: "defaultdict[str, int]" = defaultdict(int)
    for r in records:
        for b in r["buckets"]:
            agg[b] += 1
    print("\nblocker buckets (kernels may appear in several):")
    for b, n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<12} {n}")

    # A kernel blocked ONLY by cosmetic differences is a bug in the patcher, not
    # a hardware limit -- surface it loudly rather than silently writing it off.
    cosmetic_only = [r["kernel"] for r in records if r["buckets"] == ["cosmetic"]]
    if cosmetic_only:
        print(f"\nRECOVERABLE -- blocked only by cosmetic differences "
              f"({len(cosmetic_only)}); the patcher should handle these:")
        for k in cosmetic_only[:20]:
            print(f"  {k}")
    else:
        print("\nNo kernel is blocked by cosmetic differences alone: every "
              "blocked kernel needs a capability CDNA2 lacks.")

    if "other" in agg:
        other = sorted({t.split()[0] for r in records
                        for t in r["blocking_instructions"]
                        if bucket_for(t) == "other"})
        print(f"\nunbucketed blocking mnemonics ({len(other)}) -- investigate:")
        for m in other:
            print(f"  {m}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "kernels": records,
                "families": {k: {"total": v.total, "portable": v.portable,
                                 "blocked": v.blocked,
                                 "installed_unrunnable": v.installed_unrunnable,
                                 "buckets": dict(v.buckets)}
                             for k, v in families.items()},
                "invalid_instructions": sorted(invalid),
            }, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
