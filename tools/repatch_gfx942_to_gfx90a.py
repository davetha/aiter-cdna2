"""Principled gfx942 -> gfx90a .co repatcher.

For each gfx942 code object it disassembles every instruction, applies the known
gfx942->gfx90a mnemonic substitutions, and RE-ASSEMBLES each instruction for
gfx90a. An instruction is only patched if the gfx90a encoding is the same length.
If anything fails to assemble for gfx90a (fp8/int8/xf32 MFMA, etc.) the kernel is
reported NOT PORTABLE and no file is written.

This replaces hand-guessed byte patches: portability is proven by the assembler,
not assumed.

Usage (run inside the ROCm container, where the LLVM tools below exist):

    python repatch_gfx942_to_gfx90a.py <hsa/gfx942> <outdir> [name-filter]

    # whole tree
    python repatch_gfx942_to_gfx90a.py .../aiter_meta/hsa/gfx942 ./out
    # just the paged-attention family
    python repatch_gfx942_to_gfx90a.py .../aiter_meta/hsa/gfx942 ./out pa/

Prints one line per non-portable kernel and a final TALLY. Only files reported OK
are written to <outdir>; copy those over hsa/gfx90a/. Treat hsa/gfx90a/ as
generated -- rebuild it with this tool, never by hand.

Result as of 2026-07-27 (aiter 0.1.17): {'OK': 242, 'NOTPORT': 1180}.
See docs/18-pa-fwd-asm-resolved.md.
"""
import csv, os, re, shutil, sys, struct, subprocess

# LLVM tool locations differ by how ROCm was installed. A distro//opt/rocm
# install puts them in /opt/rocm/llvm/bin, but the rocm/vllm images ship ROCm as
# a Python wheel and there is no /opt/rocm at all -- the tools live under
# site-packages/_rocm_sdk_devel/lib/llvm/bin. Hardcoding the first path makes
# this script die with a bare FileNotFoundError on those images, after it has
# already reported success on the directory scan. Auto-detect, and allow an
# explicit override via ROCM_LLVM_BIN.
def _find_llvm_bin():
    override = os.environ.get("ROCM_LLVM_BIN")
    if override:
        return override
    candidates = ["/opt/rocm/llvm/bin"]
    for base in sys.path:
        if base.endswith("site-packages"):
            candidates.append(os.path.join(base, "_rocm_sdk_devel/lib/llvm/bin"))
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "llvm-mc")):
            return cand
    raise SystemExit(
        "cannot locate llvm-mc/llvm-objdump/llvm-readelf. Tried:\n  "
        + "\n  ".join(candidates)
        + "\nSet ROCM_LLVM_BIN to the directory containing them."
    )


_LLVM_BIN = _find_llvm_bin()
MC = os.path.join(_LLVM_BIN, "llvm-mc")
OBJDUMP = os.path.join(_LLVM_BIN, "llvm-objdump")
READELF = os.path.join(_LLVM_BIN, "llvm-readelf")

# gfx942 mnemonic -> gfx90a mnemonic. Only semantically identical ops.
SUBST = {
    "v_mfma_f32_16x16x16_bf16": "v_mfma_f32_16x16x16bf16_1k",
    "v_mfma_f32_32x32x8_bf16":  "v_mfma_f32_32x32x8bf16_1k",
    "v_mfma_f32_4x4x4_bf16":    "v_mfma_f32_4x4x4bf16_1k",
    "v_mfma_f32_16x16x32_bf16": None,   # gfx950 only
    "v_mfma_f32_16x16x16_f16":  "v_mfma_f32_16x16x16f16",
    "v_mfma_f32_32x32x8_f16":   "v_mfma_f32_32x32x8f16",
    "v_mfma_f32_4x4x4_f16":     "v_mfma_f32_4x4x4f16",
}
# Anything matching these is gfx942+ only with no gfx90a equivalent.
NOT_PORTABLE = re.compile(r"_(fp8|bf8)_|_xf32|smfmac|_i8$|_i8\b")

EFLAGS_OFF = 0x30
GFX942, GFX90A = 0x4c, 0x3f


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def assemble(text):
    p = run([MC, "-arch=amdgcn", "-mcpu=gfx90a", "-show-encoding"], input=text)
    if "error" in p.stderr:
        return None
    out = []
    for m in re.findall(r"encoding: \[(.*?)\]", p.stdout):
        out.append(bytes(int(b, 16) for b in m.split(",")))
    return out


def text_range(path):
    p = run([READELF, "-S", path])
    for line in p.stdout.splitlines():
        m = re.search(r"\.text\s+PROGBITS\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m:
            return int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)
    return None


def convert(src, dst):
    """Returns (status, detail)."""
    data = bytearray(open(src, "rb").read())
    flags = struct.unpack_from("<I", data, EFLAGS_OFF)[0]
    if flags & 0xFF != GFX942:
        return "SKIP", f"not gfx942 (e_flags={flags:#x})"

    dis = run([OBJDUMP, "-d", "--mcpu=gfx942", src]).stdout
    tr = text_range(src)
    if not tr:
        return "ERROR", "no .text"
    vaddr, foff, _ = tr

    insns = []
    for L in dis.splitlines():
        m = re.match(r"^\t(.*?)\s*//\s*([0-9A-Fa-f]+):\s*([0-9A-Fa-f ]+)$", L)
        if m:
            insns.append((m.group(1).strip(), int(m.group(2), 16), m.group(3).split()))
    if not insns:
        return "ERROR", "no instructions decoded"

    # find instructions needing substitution
    todo = []
    for txt, addr, words in insns:
        mn = txt.split()[0]
        if NOT_PORTABLE.search(mn):
            return "NOTPORT", f"gfx942-only op: {mn}"
        if mn in SUBST:
            if SUBST[mn] is None:
                return "NOTPORT", f"no gfx90a equivalent: {mn}"
            todo.append((txt, addr, words, SUBST[mn]))

    # verify EVERY instruction is valid gfx90a (catches unknown gfx942-only ops)
    probe = "\n".join(SUBST.get(t.split()[0], t.split()[0]) + " " + t.split(" ", 1)[1]
                      if " " in t and t.split()[0] in SUBST else t
                      for t, _, _ in insns)
    if assemble(probe) is None:
        p = run([MC, "-arch=amdgcn", "-mcpu=gfx90a"], input=probe)
        bad = [l for l in p.stderr.splitlines() if "error" in l][:2]
        return "NOTPORT", f"does not assemble for gfx90a: {bad}"

    # Batch-assemble every substituted instruction in ONE llvm-mc call.
    # (Per-instruction calls made this ~1500x slower on MFMA-heavy kernels.)
    newtxts = [newmn + txt[len(txt.split()[0]):] for txt, _, _, newmn in todo]
    encs = assemble("\n".join(newtxts)) if newtxts else []
    if encs is None:
        return "NOTPORT", "substituted instructions do not assemble for gfx90a"
    if len(encs) != len(todo):
        return "ERROR", f"batch assemble returned {len(encs)} of {len(todo)}"

    npatch = 0
    for (txt, addr, words, newmn), enc in zip(todo, encs):
        old = b"".join(bytes.fromhex(w.zfill(8))[::-1] for w in words)
        if len(enc) != len(old):
            return "NOTPORT", f"size change on {newmn} @ {addr:#x}"
        off = foff + (addr - vaddr)
        if bytes(data[off:off + len(old)]) != old:
            return "ERROR", f"byte mismatch at {addr:#x}"
        data[off:off + len(enc)] = enc
        npatch += 1

    struct.pack_into("<I", data, EFLAGS_OFF, (flags & ~0xFF) | GFX90A)
    open(dst, "wb").write(bytes(data))
    return "OK", f"{npatch} MFMA patched, e_flags {flags:#x}->{(flags & ~0xFF) | GFX90A:#x}"


def prune_csv(src, dst, produced, dst_root):
    """Copy a kernel manifest, dropping rows whose .co was not produced.

    The loader hard-fails on a missing code object --
    `AITER_CHECK(file.is_open(), "failed to open ", ...)` in
    aiter_hip_common.h -- and kernel selection picks by shape from the
    CSV-derived config table WITHOUT checking the file exists. So a manifest
    that outlives its kernels turns an unsupported shape into a crash instead
    of a fallback. Keeping the manifest in step with the blobs is what makes
    the generated tree self-consistent; do not rely on some other arch gate
    happening to keep the dangling row unreachable.

    `co_name` is relative to the manifest's directory, except for fmha, where
    the dispatcher inserts an `MI300/` or `MI308/` product subdirectory at
    load time -- so check those too before calling a row dangling.
    """
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        shutil.copyfile(src, dst)
        return 0, 0
    col = next((c for c in rows[0] if c and "co_name" in c), None)
    if col is None:
        shutil.copyfile(src, dst)
        return 0, 0

    reldir = os.path.dirname(os.path.relpath(dst, dst_root))
    keep = []
    for row in rows:
        co = (row.get(col) or "").strip()
        cands = [os.path.join(reldir, co)]
        cands += [os.path.join(reldir, p, co) for p in ("MI300", "MI308")]
        if any(os.path.normpath(c) in produced for c in cands):
            keep.append(row)

    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    return len(rows) - len(keep), len(rows)


if __name__ == "__main__":
    src_root, dst_root = sys.argv[1], sys.argv[2]
    only = sys.argv[3] if len(sys.argv) > 3 else ""
    tally = {}
    produced = set()
    deferred = []

    # Pass 1: convert code objects, recording which ones were actually written.
    for dirpath, _, files in os.walk(src_root):
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(dirpath, fn), src_root)
            if only and only not in rel:
                continue
            s = os.path.join(dirpath, fn)
            d = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if not fn.endswith(".co"):
                if os.path.isfile(s):
                    deferred.append((s, d))
                continue
            st, detail = convert(s, d)
            tally[st] = tally.get(st, 0) + 1
            if st == "OK":
                produced.add(os.path.normpath(rel))
            else:
                print(f"  {st:8s} {rel}: {detail}")

    # Pass 2: manifests, now that we know which kernels exist.
    dropped = kept_total = 0
    for s, d in deferred:
        if s.endswith(".csv"):
            n_drop, n_rows = prune_csv(s, d, produced, dst_root)
            dropped += n_drop
            kept_total += n_rows - n_drop
            if n_drop:
                print(f"  PRUNED   {os.path.relpath(d, dst_root)}: "
                      f"dropped {n_drop} of {n_rows} rows (kernel not portable)")
        else:
            shutil.copyfile(s, d)

    print("\nTALLY:", tally)
    print(f"manifests: {kept_total} rows kept, {dropped} dangling rows dropped")
