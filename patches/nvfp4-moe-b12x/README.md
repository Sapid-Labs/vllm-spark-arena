# nvfp4-moe-b12x

**Target:** `qwen3-6-27b-nvfp4` · **Scope:** engine-general for NvFP4 MoE on sm_121

## Claim

Let the NvFP4 MoE oracle consider `FLASHINFER_B12X` under `moe_backend='auto'`.
No config, weight or arithmetic change — only which expert kernel is dispatched.

## Why there is headroom

vLLM 0.26.0 excludes B12X from auto-selection by hand, and says so in a comment:
*"intentionally excluded from auto-selection until the upstream CUTLASS SM121 MMA
op guard is resolved"*. sm_121 is this part. Everything ahead of it in the list
fails its support check here, so the oracle walks the whole list to `MARLIN` and
the server says:

```
WARNING [marlin.py:34] Your GPU does not have native support for FP4 computation
but FP4 quantization is being used. Weight-only FP4 compression will be used ...
```

So this target runs NvFP4 experts with no FP4 compute path at all. B12X is the
Blackwell-12x path, and the vendor's own DGX Spark image sets
`VLLM_USE_B12X_MOE=1` for a different NvFP4 model on this same chip — which is
the evidence that it runs here.

## What it does not do

- `moe_backend != 'auto'` is left alone; an explicit operator choice is not ours.
- `swiglu_limit is not None` is left alone. B12X is not in
  `NVFP4_BACKENDS_WITH_CLAMP`, so preferring it there would drop the SwiGLU
  clamp — a numerical change, which is a recipe, not a submission.
- If the module's own `is_supported_config` says no, the original selector runs.

## Risk

A different expert kernel is a different summation order, so gate 2 (token
identity) is the real test and may reject this. That outcome is still worth
recording: it would mean the 13% documented for the CUTLASS path is only
reachable as a recipe on this hardware, not as an arena submission.
