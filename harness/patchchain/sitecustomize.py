"""Run SEVERAL patches at interpreter startup, in order.

Python imports `sitecustomize` exactly once -- the first one it finds on
sys.path. Measured: with two directories on PYTHONPATH, each holding a
sitecustomize.py, only the first prints. So putting two patch directories on
PYTHONPATH does not stack them; it SILENTLY DISCARDS all but one.

That matters because a target can require a patch to run at all. On
deepseek-v4-flash the required overlay instantiates the DSv4 decode kernel at
page-block-size 256, without which the model does not serve. A candidate patch
prepended to PYTHONPATH would have taken its place, and the failure would look
like the candidate's fault.

This directory is the only thing on PYTHONPATH. It reads the ordered list of
real patch directories from ARENA_PATCH_CHAIN and executes each one's
sitecustomize.py itself. The order is the order the harness put them in:
required patches first, the candidate last, so the candidate can override.
"""

import os
import runpy
import sys

_chain = os.environ.get("ARENA_PATCH_CHAIN", "")
for _d in [p for p in _chain.split(os.pathsep) if p]:
    _f = os.path.join(_d, "sitecustomize.py")
    if not os.path.isfile(_f):
        print(f"arena patchchain: no sitecustomize.py in {_d}", file=sys.stderr)
        continue
    # The real directory must be importable too -- a patch may ship helper
    # modules beside its sitecustomize.py.
    if _d not in sys.path:
        sys.path.insert(0, _d)
    try:
        runpy.run_path(_f, run_name="sitecustomize")
    except Exception as _e:  # noqa: BLE001 - never take the interpreter down
        print(f"arena patchchain: {_d} failed: {type(_e).__name__}: {_e}", file=sys.stderr)
        raise
