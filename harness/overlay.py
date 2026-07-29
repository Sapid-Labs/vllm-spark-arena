"""Kernel overlay: let the arena own engine source files without touching the wheel.

THE PROBLEM

Some wins on this hardware are not runtime Python. The DeepSeek-V4 block-size
deadlock, for instance, is one instantiation list in a CUDA file. Editing that
file inside `site-packages` works exactly once: a `pip install --force` or a
version bump silently reverts it, nothing is versioned, and two such changes
cannot be composed or reviewed.

THE MECHANISM

flashinfer compiles its kernels on the machine, from source files it locates
through `jit_env.FLASHINFER_CSRC_DIR`. Its own `env.py` says:

    # NOTE(lequn): Do not "from .jit.env import xxx".
    # Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
    # This helps AOT script to override envs.

That indirection is deliberate, and it is the whole opening. Reassigning that
attribute redirects the build to a directory we control. Verified 2026-07-28:
with `site-packages` left entirely stock, an overlay produced a `.so` carrying
30 DSv4 decode kernels instead of 15.

The overlay is a SYMLINK FARM. Every stock file is linked; only the files the
arena owns are real copies. So we carry the diff, not a fork of 181 files that
would drift silently.

COMPOUNDING

This is what makes kernel wins stack. The incumbent is the accumulated set of
owned files, exactly as the llama.cpp arena's incumbent is its accumulated diff
against a pinned tree. A new submission owns more files, or a newer version of
one. Promotion appends.

The safety property is MANIFEST.json: it records the sha256 of the *upstream
original* each owned file was derived from. If the wheel moves under us, the
hash mismatches and the build stops. Without that check, an overlay silently
reverts whatever upstream fixed in that file — which is how patch stacks rot.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stock_csrc_dir() -> Path:
    """Where the installed flashinfer keeps its kernel sources."""
    from flashinfer.jit import env as jit_env
    return Path(jit_env.FLASHINFER_CSRC_DIR)


def build_overlay(owned_dir: Path, out_dir: Path, *, verify: bool = True) -> Path:
    """Link every stock source into out_dir, then replace the files we own.

    `owned_dir` is a submission's kernels/ directory: kernels/csrc/*.cu plus a
    MANIFEST.json pinning the upstream hashes.
    """
    stock = stock_csrc_dir()
    owned_csrc = owned_dir / "csrc"
    manifest_path = owned_dir / "MANIFEST.json"

    if verify and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for rel, meta in manifest.get("files", {}).items():
            upstream = stock / Path(rel).name
            if not upstream.exists():
                raise SystemExit(
                    f"overlay: {rel} does not exist upstream any more.\n"
                    f"  The file this submission owns was removed or renamed in "
                    f"flashinfer {manifest.get('version')}. Re-derive it; do not "
                    f"carry the old copy forward."
                )
            got = _sha256(upstream)
            want = meta["upstreamSha256"]
            if got != want:
                raise SystemExit(
                    f"overlay: upstream {rel} has changed.\n"
                    f"  manifest: {want[:16]}\n"
                    f"  installed: {got[:16]}\n"
                    f"  The arena's copy was derived from the manifest version. Using it "
                    f"now would silently revert whatever upstream changed in that file. "
                    f"Re-derive the owned file against the new upstream and update "
                    f"MANIFEST.json."
                )

    csrc_out = out_dir / "csrc"
    csrc_out.mkdir(parents=True, exist_ok=True)
    # Rebuild the farm each time: a stale link is indistinguishable from a real
    # file at compile time, and that failure mode is very hard to see.
    for existing in csrc_out.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()

    for f in sorted(stock.iterdir()):
        if f.is_file() and not f.name.endswith(".orig"):
            (csrc_out / f.name).symlink_to(f)

    owned = 0
    if owned_csrc.is_dir():
        for f in sorted(owned_csrc.iterdir()):
            if not f.is_file():
                continue
            dst = csrc_out / f.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(f, dst)
            owned += 1
    return csrc_out


def activate(owned_dir: Path, out_dir: Path) -> Path:
    """Build the overlay and point flashinfer's JIT at it.

    Called from a submission's sitecustomize.py, so it runs at interpreter
    startup and therefore applies in the engine and worker processes vLLM
    spawns — the same reason the patch mechanism uses PYTHONPATH at all.
    """
    csrc = build_overlay(Path(owned_dir), Path(out_dir))
    from flashinfer.jit import env as jit_env
    jit_env.FLASHINFER_CSRC_DIR = csrc
    return csrc


if __name__ == "__main__":
    # Diagnostic: build an overlay and report what it owns.
    owned = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("patches/flashinfer-dsv4-pbs256/kernels")
    out = Path(os.environ.get("ARENA_OVERLAY_DIR", "/tmp/arena-overlay"))
    csrc = build_overlay(owned, out)
    links = sum(1 for p in csrc.iterdir() if p.is_symlink())
    files = sum(1 for p in csrc.iterdir() if p.is_file() and not p.is_symlink())
    print(f"overlay at {csrc}\n  {links} linked from upstream\n  {files} owned by the arena")
