"""Let the NvFP4 LINEAR kernel oracle consider FlashInfer B12X on GB10.

WHAT THIS CHANGES

Dispatch only. Same serve config, same weights, same arithmetic inside every
kernel. It puts one already-shipped kernel class back into the auto-selection
list it was removed from. Kernel selection is named in the contract as in scope.

WHY

vLLM 0.26.0 removes exactly one entry from the NvFP4 linear kernel order, by
hand, and says why:

    _POSSIBLE_NVFP4_KERNELS = {
        PlatformEnum.CUDA: [
            FlashInferCuteDslNvFp4LinearKernel,
            # FlashInferB12xNvFp4LinearKernel excluded from auto-selection until
            # upstream CUTLASS SM121 MMA op guard is resolved; use
            # --linear-backend flashinfer_b12x to opt in explicitly.
            FlashInferCutlassNvFp4LinearKernel,
            ...

SM121 is this part, and it is the reason the entry is missing. On this target
every kernel ahead of Marlin fails its support check, so the oracle walks down
to Marlin and the server says:

    WARNING [marlin.py:35] Your GPU does not have native support for FP4
    computation but FP4 quantization is being used. Weight-only FP4 compression
    will be used ...

That is a weight-only path: the weights are FP4 but the maths is not. B12X is
the Blackwell-12x kernel, and the vendor's own DGX Spark image turns the B12X
path on (`VLLM_USE_B12X_MOE=1`) for a different NvFP4 model on this same chip,
which is the evidence that B12X runs here at all.

HOW

`_POSSIBLE_NVFP4_KERNELS` is a module-level dict, so this inserts the class at
the position the comment removed it from -- ahead of CUTLASS and therefore ahead
of Marlin -- and changes nothing else. The oracle's own `is_supported()` still
decides: if B12X says no on this shape, the list simply continues as before and
the result is identical to the baseline.

Deliberately NOT done: `--linear-backend flashinfer_b12x`, the opt-in the comment
suggests. That is a serve flag, and tuning serve flags is a recipe under this
contract, not a submission. The submission is the dispatch order.

Applied at interpreter startup through PYTHONPATH so it reaches the engine and
worker processes vLLM spawns. The edit itself runs from a post-import hook,
because this file is imported by every python3 that sees it on PYTHONPATH --
including the launcher -- and importing torch and vLLM from all of them would be
slow and fragile.
"""

import importlib.abc
import importlib.util
import sys

_TARGET = "vllm.model_executor.kernels.linear"
_TAG = "[arena nvfp4-b12x-linear]"


def _install(mod) -> None:
    if getattr(mod, "_arena_b12x_linear_installed", False):
        return
    try:
        from vllm.platforms import PlatformEnum

        b12x = mod.FlashInferB12xNvFp4LinearKernel
        table = mod._POSSIBLE_NVFP4_KERNELS
        order = table.get(PlatformEnum.CUDA)
        if order is None:
            print(f"{_TAG} no CUDA entry in the NvFP4 kernel table", file=sys.stderr)
            return
        if b12x in order:
            print(f"{_TAG} B12X already in the order; nothing to do", file=sys.stderr)
            mod._arena_b12x_linear_installed = True
            return
        # Exactly where the comment removed it: after CuteDSL, before CUTLASS.
        anchor = mod.FlashInferCutlassNvFp4LinearKernel
        at = order.index(anchor) if anchor in order else 0
        order.insert(at, b12x)
        mod._arena_b12x_linear_installed = True
        print(f"{_TAG} inserted {b12x.__name__} at position {at}; order is now "
              f"{[k.__name__ for k in order]}", file=sys.stderr)
    except Exception as e:                                        # noqa: BLE001
        # A dispatch preference must never be the reason a server fails to boot.
        print(f"{_TAG} not applied ({type(e).__name__}: {e})", file=sys.stderr)


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _PatchAfterImport) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchAfterImport())
if _TARGET in sys.modules:
    _install(sys.modules[_TARGET])
