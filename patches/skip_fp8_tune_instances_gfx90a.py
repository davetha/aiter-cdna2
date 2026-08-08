#!/usr/bin/env python3
"""Stop the a8w8 tuner generating FP8 instances that CDNA2 cannot execute.

WHY. `csrc/ck_gemm_a8w8/gen_instances.py` emits, for every kernel in the tuning
list, BOTH an `abI8` (int8) and an `abF8` (fp8) instance -- its own comment says
"Generate both I8 and F8 instances for tuning". On gfx90a that is exactly half
the build wasted:

    83  abI8   int8   usable
    83  abF8   fp8    unusable -- gfx90a has no FP8 MFMA at all
   166  total

`docs/49` established the FP8 gap at the assembler, not by inference:

    v_mfma_f32_16x16x32_fp8_fp8   gfx90a  REJECTS -- not supported on this GPU
    v_mfma_f32_16x16x32_fp8_fp8   gfx942  ACCEPTS

So no `abF8` instance can ever win a tuning round here; it cannot even run.

AND THE COST IS NOT MERELY 2x. Round 52 measured the tail: a single FP8
instance,

    a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2
        _intrawave_v3_abF8_dF32_eB16.cpp

consumed **2073 s of CPU (34.5 min)** on one translation unit while the other
165 instances had already finished under `ninja -j 38`. The whole build sat on
one straggler that compiles a kernel the hardware rejects. Without FP8 in the
list, round 52's build would have completed in roughly the time it took to
compile the int8 half.

That tail is also what made the run LOOK hung -- load average fell from 38 to
1.78 with both GPUs at 0%, which is indistinguishable from a deadlock unless
you check for compiler processes. See round 52's header for the discriminator.

SCOPE. Only the `istune` branch is touched. The non-tuning branch below it
still generates the full matrix, because that path builds the production module
whose manifest is keyed off the tuned CSV -- narrowing it could desynchronise a
prebuilt module from its manifest, which is a different and worse failure than
a slow build.

NOT A CORRECTNESS CHANGE. Removing candidates that cannot execute cannot change
which kernel wins; it only removes entries that would fail or be skipped. If a
future card in this stack has FP8, revert this.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ANCHOR = """            # F8 instances
            for EDtype in ["B16"]:
                INSTANCE_abF8 = INSTANCE_template.format(
                    name=k.name, dtypes=f"F8, F32, {EDtype}"
                )
                Path(
                    os.path.join(
                        self.instances_path, f"{k.name}_abF8_dF32_e{EDtype}.cpp"
                    )
                ).write_text(INSTANCE_abF8)
"""

_PATCHED = """            # F8 instances -- SKIPPED on gfx90a.
            # configs/skip_fp8_tune_instances_gfx90a.py. CDNA2 has no FP8 MFMA
            # (v_mfma_f32_16x16x32_fp8_fp8 is rejected by the assembler), so
            # these instances cannot execute and cannot win a tuning round.
            # They are half of all generated instances, and one of them burned
            # 2073 s of CPU on a single translation unit in round 52.
            if False:  # gfx90a carve-out
                for EDtype in ["B16"]:
                    INSTANCE_abF8 = INSTANCE_template.format(
                        name=k.name, dtypes=f"F8, F32, {EDtype}"
                    )
                    Path(
                        os.path.join(
                            self.instances_path, f"{k.name}_abF8_dF32_e{EDtype}.cpp"
                        )
                    ).write_text(INSTANCE_abF8)
"""

_REL = "csrc/ck_gemm_a8w8/gen_instances.py"


def _target(root: pathlib.Path) -> pathlib.Path:
    p = root / _REL
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p


def check(root: pathlib.Path) -> int:
    s = _target(root).read_text()
    done = _PATCHED in s
    print(f"  skip_fp8_tune_instances: {'PATCHED' if done else 'not patched'}")
    if not done and _ANCHOR not in s:
        print("  WARNING: anchor not found either -- upstream changed this block.")
    return 0 if done else 1


def apply(root: pathlib.Path) -> int:
    p = _target(root)
    s = p.read_text()
    if _PATCHED in s:
        print("  already patched")
        return 0
    n = s.count(_ANCHOR)
    if n != 1:
        sys.exit(f"FATAL: anchor matched {n} times, expected 1. "
                 "Upstream changed gen_instances.py; re-derive this patch.")
    p.write_text(s.replace(_ANCHOR, _PATCHED))
    print("  patched: FP8 tuning instances will no longer be generated")
    return 0


def revert(root: pathlib.Path) -> int:
    p = _target(root)
    s = p.read_text()
    if _PATCHED not in s:
        print("  not patched; nothing to revert")
        return 0
    p.write_text(s.replace(_PATCHED, _ANCHOR))
    print("  reverted")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/src/aiter",
                    help="aiter source root (default: /src/aiter)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--assert-patched", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    if a.check:
        return check(root)
    if a.assert_patched:
        rc = check(root)
        if rc:
            print("ASSERTION FAILED: FP8 tune-instance skip not applied.")
        return rc
    if a.revert:
        return revert(root)
    return apply(root)


if __name__ == "__main__":
    sys.exit(main())
