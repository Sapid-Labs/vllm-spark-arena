# flashinfer: instantiate the DSv4 decode kernel at page block size 256

This is **enablement, not an optimization**. It does not make anything faster.
It makes DeepSeek-V4-Flash *dispatchable* on GB10 / sm_121, where today it is
not, so the model can later become a normal arena target.

It is also not a `sitecustomize.py` submission — it changes a CUDA file that
flashinfer compiles on the machine. It is kept here because this is where the
sm_121 work lives.

## The problem

Two hard-coded integers in released code disagree, and no serve flag reconciles
them:

| Component | Requires |
|---|---|
| vLLM `DeepseekV4IndexerBackend.get_supported_kernel_block_sizes()` | exactly `[256]` |
| flashinfer `sparse_mla_sm120_decode_dsv4.cu` launcher | exactly `64` |

At `--block-size 256` the decode kernel is not dispatchable, the call falls
through to the prefill orchestrator, and that asserts `num_tokens > 64` — which
single-user decode never satisfies. Below 128 is arithmetically dead anyway
(`MLAAttentionSpec.storage_block_size` is `block_size // compress_ratio`, and
this checkpoint's `compress_ratios` contains 128, so `64 // 128 == 0` gives a
zero-byte page). On sm_121 there is no second sparse-MLA backend to fall back
to.

## Why the fix is small

The kernel has **no page-size limitation**. `PAGE_BLOCK_SIZE` is a template
parameter, and inside the kernel it only feeds index decomposition:

```cuda
const int block_idx_g = idx / section_pbs;
const int local_idx_g = idx - block_idx_g * section_pbs;
```

The extra-KV section already passes a **runtime** page size of 2, which proves
the generality. And shared memory is sized from `DSV4_BI` (64, the candidate
tile), not from the page size — so a wider page costs no extra smem, which was
the obvious thing to fear.

Only the launcher was specific: one guard and a dispatch macro that hard-coded
`64`. The patch parameterizes the macro and instantiates both sizes. The `.so`
goes from 15 kernels to 30.

## Verification

`pbs_equiv3.py` compares the stage-1 decode partials at page size 64 and 256,
built from **one** logical per-token source.

The layout matters and is easy to get wrong: a page is structure-of-arrays,
`[pbs * 576 bytes KV][pbs * 8 bytes scales]`, so the scale section starts at
`pbs * IO_STRIDE` and the split point *moves with the page size*. Reshaping an
array-of-structures buffer feeds the two kernels different data and reports a
false mismatch.

Comparison is on **bit patterns**, not values: synthetic bytes are not always
valid FP8, and `NaN != NaN` reports a false mismatch by value.

`pbs_control3.py` is the negative control, and it is not optional here — two
earlier versions of this test passed vacuously (all-zero outputs, then NaN
inequality). Calibration matters too: perturbing a *single byte* changes
nothing, because one FP8 element among 1024 candidates × 512 dims is
softmax-weighted down below bf16 resolution. Perturb a whole token.

Measured on Spark-1, flashinfer 0.6.14, vLLM 0.26.0:

| check | differing elements | expected |
|---|---|---|
| 64 vs 256, same data | 0 | 0 |
| one whole token changed, @64 | 128 | > 0 |
| one whole token changed, @256 | 128 | > 0 |
| perturbed 64 vs perturbed 256 | 0 | 0 |

Four head/topk shapes × two seeds, all bit-identical.

## What is NOT verified

**The kernel has not served a real model.** `DeepSeek-V4-Flash` is 222 GB and
lives only on node 2; node 1 has 141 GB free, and TP2 needs every rank to read
all shards. So the end-to-end serve at `--block-size 256` is blocked on disk,
not on this patch.

Speed at 256 is also unmeasured. A wider page changes the access pattern; it
could be slower. Nothing here claims otherwise.

## Applying it

The patch edits files inside `site-packages`, so a reinstall of flashinfer
removes it:

```bash
cd ~/venvs/vllm-026/lib/python3.12/site-packages
patch -p1 < .../dsv4-pbs256.patch
# flashinfer rebuilds the changed translation unit on next use (~7 s incremental)
```

Upstream is the right home for this. It is a flashinfer change, it is small, and
every GB10 owner needs it.
