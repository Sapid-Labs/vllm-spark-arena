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

---

## OUTCOME: rejected — no valid target, 2026-07-30

Measured, twice, and it is a no-op:

```
pair 1: decode x0.9992  prefill x1.0090     (all four prompt hashes identical)
```

The patch does apply — the candidate log shows B12X inserted at position 1, ahead
of Marlin — and both of the oracle's checks pass (`is_supported() -> (True, None)`,
`can_implement() -> (True, None)`). Marlin is still used, because **this
checkpoint never reaches that oracle at all.**

`nvidia/Qwen3.6-27B-NVFP4` declares `quant_algo=W4A16_NVFP4`, and vLLM routes
that to a dedicated linear method:

```python
# W4A16_NVFP4   -> W4A16: FP4 Marlin GEMM with bf16/fp16 activations
elif quant_method == "W4A16_NVFP4":
    self.LinearMethodCls = ModelOptNvFp4W4A16LinearMethod
```

W4A16 means 4-bit weights and **BF16 activations**. There is no FP4 arithmetic to
dispatch, so FP4-compute kernels are irrelevant and Marlin is the correct choice,
not a fallback. The warning that started this hunt —

```
WARNING [marlin.py:35] Your GPU does not have native support for FP4 computation
```

— is misleading in this context: it fires because the checkpoint's activations
are BF16 by design, not because sm_121 lacks a capability.

## What a valid target would need

The documented sm_121 NvFP4 gap ("FLASHINFER_CUTLASS removed on sm_121 and it
costs 13%") is real, but it needs a **W4A4 NVFP4** checkpoint — FP4 activations
as well as weights — ideally MoE, since that is where the CUTLASS note points.
Neither arena has one today. That is a target-acquisition task, not a kernel one.

Kept in the repo as a recorded negative. It cannot be promoted: `promote` requires
a passing bench record, and this one does not beat the incumbent.
