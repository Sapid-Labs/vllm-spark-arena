"""Let the NvFP4 MoE oracle consider FLASHINFER_B12X on GB10.

WHAT THIS CHANGES

Nothing about the serve config, the weights, or any kernel's arithmetic. It
changes DISPATCH: which expert kernel the oracle picks for an NvFP4 MoE layer,
with `moe_backend='auto'` exactly as the target already sets it. Kernel
selection is named in the contract as in scope.

WHY

vLLM 0.26.0 excludes one backend from auto-selection by hand, and says so:

    # NOTE: the kernels are selected in the following order.
    # FLASHINFER_B12X is intentionally excluded from auto-selection until
    # the upstream CUTLASS SM121 MMA op guard is resolved; use
    # moe_backend="flashinfer_b12x" to opt in explicitly.

sm_121 IS this part. Everything ahead of it in the list fails its support check
here, so the oracle walks the whole list and lands on MARLIN -- the weight-only
fallback, which announces itself:

    WARNING [marlin.py:34] Your GPU does not have native support for FP4
    computation but FP4 quantization is being used. Weight-only FP4 compression
    will be used ...

So this target currently runs NvFP4 experts with no FP4 compute path at all.
B12X is the Blackwell-12x path, and the vendor's own DGX Spark image enables it
(`VLLM_USE_B12X_MOE=1`) for a different NvFP4 model on this same chip, which is
the evidence that it runs here.

HOW, AND WHAT IT DELIBERATELY DOES NOT DO

The exclusion lives in a list local to `select_nvfp4_moe_backend`, so there is
no global to rebind. This wraps that function, asks the module's OWN
`backend_to_kernel_cls` and `is_supported_config` whether B12X supports this
deployment, and only then returns it. If the answer is no, the original function
runs untouched. Two guards preserve upstream's semantics:

  * `moe_backend != 'auto'` is left alone. An explicit choice by the operator is
    not ours to override.
  * `swiglu_limit is not None` is left alone. B12X is not in
    NVFP4_BACKENDS_WITH_CLAMP, so preferring it there would silently drop the
    SwiGLU clamp -- a numerical change, which would be a recipe, not this.

Applied at interpreter startup through PYTHONPATH, so it reaches the engine and
worker processes vLLM spawns. The patch itself is installed by a post-import
hook rather than by importing vLLM here: this file is imported by every python3
that sees it on PYTHONPATH, including the launcher, and pulling in torch and
vLLM from all of them would be both slow and fragile.
"""

import importlib.abc
import importlib.util
import sys

_TARGET = "vllm.model_executor.layers.fused_moe.oracle.nvfp4"
_TAG = "[arena nvfp4-moe-b12x]"


def _install(nv) -> None:
    if getattr(nv, "_arena_b12x_installed", False):
        return
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    original = nv.select_nvfp4_moe_backend
    b12x = nv.NvFp4MoeBackend.FLASHINFER_B12X

    def select(config, weight_key, activation_key):
        try:
            # Upstream's own guards, not ours to relax.
            if getattr(config, "moe_backend", "auto") != "auto":
                return original(config, weight_key, activation_key)
            if getattr(config, "swiglu_limit", None) is not None:
                return original(config, weight_key, activation_key)

            batched = config.moe_parallel_config.use_batched_activation_format
            fmt = (mk.FusedMoEActivationFormat.BatchedExperts if batched
                   else mk.FusedMoEActivationFormat.Standard)

            for k_cls in nv.backend_to_kernel_cls(b12x):
                # Same odd calling convention as the oracle: the class is passed
                # as the first argument.
                supported, reason = k_cls.is_supported_config(
                    k_cls, config, weight_key, activation_key, fmt)
                if supported:
                    print(f"{_TAG} selected FLASHINFER_B12X ({k_cls.__name__}) "
                          f"instead of the auto-selection order", file=sys.stderr)
                    return b12x, k_cls
                print(f"{_TAG} B12X unsupported here: {reason}", file=sys.stderr)
        except Exception as e:                                   # noqa: BLE001
            # A dispatch preference must never be the reason a server dies.
            print(f"{_TAG} not applied ({type(e).__name__}: {e})", file=sys.stderr)
        return original(config, weight_key, activation_key)

    nv.select_nvfp4_moe_backend = select
    nv._arena_b12x_installed = True
    print(f"{_TAG} dispatch hook installed", file=sys.stderr)


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    """Run _install as soon as the oracle module finishes importing."""

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)          # avoid recursing into ourselves
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
# Already imported (an interpreter that loaded vLLM before us): patch in place.
if _TARGET in sys.modules:
    _install(sys.modules[_TARGET])
