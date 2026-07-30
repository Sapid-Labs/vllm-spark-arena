# nvfp4-b12x-linear

**Target:** `qwen3-6-27b-nvfp4` · **Scope:** engine-general for NvFP4 linear layers on sm_121

## Claim

Put `FlashInferB12xNvFp4LinearKernel` back into the NvFP4 linear auto-selection
order. Dispatch only — same serve config, same weights, no kernel's arithmetic
touched.

## Why there is headroom

vLLM 0.26.0 removes exactly one entry from `_POSSIBLE_NVFP4_KERNELS`, by hand:

```python
FlashInferCuteDslNvFp4LinearKernel,
# FlashInferB12xNvFp4LinearKernel excluded from auto-selection until
# upstream CUTLASS SM121 MMA op guard is resolved; use
# --linear-backend flashinfer_b12x to opt in explicitly.
FlashInferCutlassNvFp4LinearKernel,
```

SM121 is this part. Every kernel ahead of Marlin fails its support check here, so
the oracle walks down to Marlin and the server reports weight-only FP4 — the
weights are FP4 but the maths is not.

Measured on the node: `FlashInferB12xNvFp4LinearKernel.is_supported()` returns
`(True, None)`. It is excluded by a note, not by a capability check.

## Why not the flag the comment suggests

`--linear-backend flashinfer_b12x` is a serve flag, and tuning serve flags is a
recipe under this contract, not a submission. The submission is the dispatch
order, which the contract puts in scope.

## Risk

A different linear kernel is a different summation order, so gate 2 (token
identity) is the real test and may reject this. If it does, that is worth
recording: the native FP4 path on this chip would then be reachable only as a
recipe, with eval evidence, and not as an arena submission.
