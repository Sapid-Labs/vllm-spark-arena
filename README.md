# vLLM Spark arena

Crowd-optimized vLLM for the NVIDIA DGX Spark (GB10, **sm_121**).

Make the engine faster on this part without changing what it outputs. A
submission is a Python patch applied over a pinned vLLM wheel, measured as a
paired ratio on a real Spark, and gated on producing byte-identical tokens.

Sibling repo: [**llamacpp-spark-arena**](https://github.com/Sapid-Labs/llamacpp-spark-arena),
same contract, different substrate.

## Why this exists

Nothing is built for sm_121. On vLLM specifically the consequences are
documented and expensive: `FLASHINFER_CUTLASS` for NVFP4 MoE was removed on
sm_121 *"and it costs 13%"*, `HUMMING` cannot initialize, and `VLLM_CUTLASS` /
`FLASHINFER_TRTLLM` fail outright with *"kernel does not support current
device"*. Every one of those is unclaimed performance on hardware people already
own.

## What a submission is

**Not** a source edit. vLLM here is a pip wheel with precompiled CUDA extensions
(`_C_stable_libtorch.abi3.so`, `_flashmla_C.abi3.so`, …), and nobody has built it
from source on aarch64/GB10 — so there is no vendored tree to patch. A submission
is a directory:

```
patches/<your-name>/sitecustomize.py
```

put on `PYTHONPATH` for the candidate arm only. **Startup injection is the
point**: vLLM spawns engine and worker processes, and only something imported
this early reaches all of them. A monkeypatch in your own script would patch the
API server and nothing that runs the model.

That means kernels in `csrc/` are **out of scope for now**. Reaching them needs
a from-source vLLM build for sm_121, which does not exist yet on this hardware —
it is tracked as its own milestone, and it is also what would unlock recovering
that 13%.

## The loop

```bash
git clone https://github.com/Sapid-Labs/vllm-spark-arena && cd $_
python3 harness/arena.py baseline --target qwen3-6-27b-nvfp4     # your node's baseline
cp -r patches/example-noop patches/my-idea                        # then edit it
python3 harness/arena.py bench --target qwen3-6-27b-nvfp4 --patch my-idea
```

No build step, so iteration is fast. Model load dominates instead — budget a few
minutes per arm rather than the 15-minute rebuild the llama.cpp arena pays.

## Scoring

```
score = decode_speedup^0.75 * prefill_speedup^0.25      # both floor at 0.95, hard
```

Paired ratios, arms alternating in one session on one node. A baseline on this
fleet has drifted **24.05 → 20.09 tok/s overnight**; the ratio cancels the room,
an absolute number does not.

| # | Gate | Runs where |
|---|------|-----------|
| 1 | **Config identity** — same wheel, same model, same serve args; only `PYTHONPATH` differs | your node |
| 2 | **Token identity** — byte-identical output under the pinned config | your node |
| 3 | **Held-out identity *and speedup*** — on prompts you have never seen | referee's node |
| 4 | **Speedup floors** — ≥ 0.95 on both axes | your node |
| 5 | **Beat the incumbent** | referee's node |

### Gate 3 is stricter here than in the llama.cpp arena, on purpose

There, the editable surface is CUDA kernels, so a submission *physically cannot*
memoize a response. Here the submission is arbitrary Python. A patch could cache
completions keyed on the prompt, return byte-identical tokens at absurd speed,
and sail through gates 1, 2 and 4.

So gate 3 does two things. Prompts are **generated from a random seed at
verification time** — never stored, so a cache cannot have seen them, and the
seed is recorded so anyone can regenerate them afterwards to audit. And the
held-out arm is **timed**, not just compared: the speedup has to reproduce to
within 50% of the claimed gain. A real engine-level win generalizes to unseen
inputs; a lookup table does not.

Identity alone would let the cheat through. That is why the two halves are one
gate.

## The pinned serve config is part of the contract

This is the one structural difference from the llama.cpp arena, and it is not
bureaucracy.

Gates 2 and 3 compare a **baseline server** against a **candidate server** — two
processes. So the whole cheap gate rests on greedy output surviving a restart.
llama.cpp does that under an ordinary serve command (measured: two boots an hour
apart, 4/4 byte-identical). vLLM does **not** — unless it is pinned.

Measured 2026-07-28 on Qwen3.6-27B-NVFP4, vLLM 0.26.0. A first probe on a single
short prompt came back identical across three boots, with cudagraphs on and with
`--enforce-eager`. **A wider four-boot run over four prompts did not agree with
that**, and the fuller picture is the one that matters:

| prompt | boot 1 (cold cache) | boot 2 | boot 3 | boot 4 |
|---|---|---|---|---|
| code, 93 tok | `ec2e7194` | `efc4efe8` | `efc4efe8` | `efc4efe8` |
| prose, 73 tok | `35c3d229` | `f1ed7ada` | `f1ed7ada` | `f1ed7ada` |
| **code, 1,715 tok** | `d758565d` | `ad33fd5a` | **`c0f7047a`** | `ad33fd5a` |
| doc, 6,863 tok | `a8351da1` | `c5f8b3c6` | `c5f8b3c6` | `c5f8b3c6` |

Two things fall out.

**The first boot against a cold `VLLM_CACHE_ROOT` is not comparable.** Boot 1
differs on every prompt; boots 2–4 share a warm cache and mostly agree. This is
the request-warmup rule one level up, and the harness now discards a cache-warmup
boot for the same reason it discards a warmup request.

**Cross-boot identity is per-prompt, not global.** Three of the four prompts are
rock stable across warm boots — including the 6,863-token one — while the
1,715-token prompt flips between two hashes. So it is not a length effect; it is a
numerical knife-edge on one particular input, the same family as batch size
flipping the argmax at `n=2`.

A prompt whose argmax is a coin flip cannot test anything — it would fail honest
submissions at random. So `baseline` **screens each prompt** across boots and
keeps only the stable ones, recording what it excluded and why in `goldens.json`.
An excluded prompt is evidence about the model's numerical stability, not a
failure to hide.

Cudagraph capture is not implicated either way, so the arena does not have to pay
`--enforce-eager`.

The pins: `--max-num-seqs 1`, `--kv-cache-memory-bytes`, `--max-model-len`,
`--gpu-memory-utilization`, and a `VLLM_CACHE_ROOT` shared across boots. They
live in the target, and changing any of them invalidates its goldens.

## Optimization vs. recipe vs. config

Three things that look similar and are not:

- **Optimization** — same config, same output, faster. That is what this ranks.
- **Config tuning** — different flags. That is a *recipe*, and it belongs on
  [howtospark.com](https://howtospark.com).
- **Output-changing** — different precision, quantization, sampling, speculation.
  Also a recipe, and it needs eval evidence rather than a token diff.

Holding those apart is what keeps verification here cheap enough to actually
happen.

## Targets

| Target | Shape | Why it is first |
|---|---|---|
| `qwen3-6-27b-nvfp4` | 27B NVFP4, single Spark | Its cross-boot behaviour is *measured* — including the one prompt in four that is not stable, which is why prompt screening exists |

**Multi-node is in scope but not open yet.** Plenty of Spark owners have two or
more, and the clustered recipes are the ones that most need kernel help. The
blocker is that nobody has measured whether greedy token identity survives an
NCCL allreduce — two attempts on 2026-07-28 died on `RPC call to sample_tokens
timed out` before producing an answer, for reasons unrelated to determinism.
Until that is settled, a TP target cannot be ranked, because the gate would be
comparing across a collective whose reduction order is not known to be stable.

## Status

Early, and honest about it. The contract, harness and first target are here. The
baseline is not recorded yet: the first two attempts *failed their own stability
screen*, which is how the cold-cache and knife-edge-prompt findings above were
made. That is the harness working — a baseline that cannot reproduce itself has
no business defining a gate.

Submissions are not open until there is a screened baseline to beat.
