"""A no-op submission, kept as the shape reference.

Loaded at interpreter startup when this directory is on PYTHONPATH, which is how
the arena applies a candidate patch. Startup injection is not incidental: vLLM
spawns engine and worker processes, and only something imported this early
reaches all of them. A post-import monkeypatch in your own script would patch
the API server and nothing that actually runs the model.

A real submission changes dispatch, kernel selection, memory layout or
scheduling — and must leave the tokens byte-identical. If it changes what the
model emits, it is a RECIPE, not a submission, and belongs on howtospark.com
with eval evidence.

Precedent worth reading: howtospark's bench/patches/laguna-int4 forces the
Triton MoE path over Marlin in four lines, because vLLM's Marlin WNA16 MoE
asserts on grouped 8-bit experts.
"""


def _apply() -> None:
    try:
        import vllm  # noqa: F401
    except Exception:
        # Not a vLLM interpreter (the launcher, a subprocess helper). Nothing to
        # do — and never raise from here, or you break every python3 on PATH.
        return

    # ---- your change goes here ----
    #
    # from vllm.model_executor.layers.fused_moe import fused_moe
    # fused_moe.some_dispatch_hook = _faster_thing
    #
    # Two rules that will save you a rejected submission:
    #   1. Do not touch sampling. Gate 2 compares tokens byte-for-byte.
    #   2. Do not memoize completions. Gate 3 runs freshly generated prompts AND
    #      times them, so a cache shows up as a win that does not generalize.


_apply()
