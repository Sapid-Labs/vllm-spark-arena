"""Point flashinfer's JIT at this submission's kernel sources.

Runs at interpreter startup because the harness puts this directory on
PYTHONPATH — which is also why it reaches the engine and worker processes vLLM
spawns. A later monkeypatch would miss them.

Nothing in site-packages is modified. The overlay is a symlink farm over the
installed wheel with this submission's files copied over the top, so a
`pip install --force` cannot revert the change and two submissions can own
different files without colliding.
"""

import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ARENA = _HERE.parent.parent


class _PageBlockSizes(int):
    """Compares equal to every page size the overlay kernel instantiates.

    flashinfer gates dispatch with `page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE`
    in two places, both reading the module global at call time. Rebinding that
    global to this int subclass widens both checks at once, without editing the
    installed file or copying function bodies that upstream may change.

    Python resolves `int == subclass-of-int` by trying the SUBCLASS's reflected
    __eq__ first, so our comparison wins even though we are on the right-hand
    side. It still behaves as 64 everywhere else (arithmetic, hashing), which is
    what the non-comparison uses of the constant expect.
    """

    _ACCEPTED = (64, 256)

    def __eq__(self, other):
        return other in self._ACCEPTED

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(int(self))


def _widen_python_guard() -> None:
    from flashinfer.mla import _sparse_mla_sm120 as m
    if m._DECODE_DSV4_PAGE_BLOCK_SIZE == 256:
        return
    m._DECODE_DSV4_PAGE_BLOCK_SIZE = _PageBlockSizes(64)


def _apply() -> None:
    try:
        import flashinfer  # noqa: F401
    except Exception:
        # Not a flashinfer interpreter (the launcher, a helper subprocess).
        # Never raise from here — this file is imported by every python3 that
        # sees it on PYTHONPATH.
        return
    _widen_python_guard()
    sys.path.insert(0, str(_ARENA / "harness"))
    try:
        from overlay import activate
        out = pathlib.Path(os.environ.get("ARENA_OVERLAY_DIR", "/tmp/arena-overlay"))
        activate(_HERE / "kernels", out)
    except SystemExit:
        raise  # manifest mismatch — must be loud, not swallowed
    except Exception as e:
        print(f"[arena] kernel overlay NOT applied: {e}", file=sys.stderr)


_apply()
