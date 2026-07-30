# Runbook — running the vLLM arena

Same shape as the [llama.cpp arena runbook](https://github.com/Sapid-Labs/llamacpp-spark-arena/blob/main/RUNBOOK.md).
Three differences, all of them consequences of the substrate.

**No build step.** A submission is `patches/<name>/sitecustomize.py`, put on
`PYTHONPATH` for the candidate arm only. Iteration is fast; model load dominates
instead.

**The serve config is pinned by the target and is not yours to tune.** Changing
a flag changes the measurement, so gate 1 checks the wheel, the model and the
serve arguments are identical between arms.

**Gate 3 times the held-out arm as well as comparing it.** The submission is
arbitrary Python and could memoize completions; identity alone would let that
through. Pass `--claimed-speedup` from your bench record so the check can run.

---

## Once per node

```bash
python3 harness/arena.py baseline --target qwen3-6-27b-nvfp4 --repeats 4
```

Four boots: the first is discarded as **compile-cache warmup** (a cold
`VLLM_CACHE_ROOT` changes kernel selection), and the remaining three screen each
prompt for cross-boot stability. Prompts that flip are excluded and recorded in
`goldens.json` — a prompt whose argmax is a coin flip would fail honest
submissions at random.

## The loop

```bash
cp -r patches/example-noop patches/my-idea && vim patches/my-idea/sitecustomize.py

python3 harness/arena.py bench --target qwen3-6-27b-nvfp4 --patch my-idea
# prints: gate 3 runs on the referee's node — pass --claimed-speedup <N>

python3 harness/arena.py heldout --target qwen3-6-27b-nvfp4 --patch my-idea \
    --claimed-speedup <N from the bench record> --referee joe

python3 harness/arena.py promote --target qwen3-6-27b-nvfp4 \
    --record results/qwen3-6-27b-nvfp4-<stamp>.json \
    --held-out-record results/qwen3-6-27b-nvfp4/heldout-<stamp>.json \
    --author <gh-handle> --note "..." --referee joe

python3 harness/arena.py leaderboard --target qwen3-6-27b-nvfp4
```

Promotion refuses a gate-3 record that verified a *different patch*, one that
failed, one whose speedup did not generalize, or none at all.

It also refuses a bench record measured on **another node**. When you referee a
submission, re-run `bench` here and promote your own record: the ratio that goes
on the frontier has to be one this node measured, since the site presents it as
verified. `--force` waives the check and says so in the output.

## Kernel submissions

A submission may also own engine **source** files, via `patches/<name>/kernels/`.
The overlay puts a symlink farm over the wheel's `csrc`, copies the owned files
over it, and rebinds `jit_env.FLASHINFER_CSRC_DIR` — so nothing in
`site-packages` is ever modified and wins compound as an accumulating set of
owned files. See `patches/flashinfer-dsv4-pbs256/` for a worked example, and
`kernels/MANIFEST.json` for the upstream-hash check that stops a stale copy
silently reverting an upstream fix.

## Publishing

Identical to the other arena — from `howtospark`:

```bash
npm run arena:pull && git diff data/arena/ && npm run arena:sync
```

## Node-specific traps (all measured, all silent)

- **`ray stop --force` does not reliably kill the raylet.** A later `ray start`
  *attaches* to the survivor and inherits its environment. This manufactures
  convincing fake bugs — it produced both an `ActorHandleNotFoundError` and a
  `CUDA_VISIBLE_DEVICES` failure that had nothing to do with the code under
  test. Verify `pgrep -cf '[r]aylet --'` is 0 **before** starting.
- **`oom-guard.sh` kills vLLM and Ray with no traceback** when MemAvailable
  drops below 8 GiB. It has killed a run by a 321 MiB margin. Check
  `journalctl --user | grep oom-guard` before blaming the engine.
- **`ulimit -l` is 8 MB**, so NCCL cannot use RDMA and needs
  `NCCL_IB_DISABLE=1` until the memlock fix is applied as root.
- **Guard every `pkill -f` pattern** (`[r]aylet`, not `raylet`) or you kill your
  own ssh session mid-teardown.


---

## The automated loop (added 2026-07-30)

### One attempt, unattended

```bash
python3 harness/arena.py attempt \
  --target qwen3-6-27b-nvfp4 --patch my-change \
  --author <gh-handle> --model "Claude Opus 5" \
  --note "one line on what it does" --referee joe
```

Runs `probe -> bench -> heldout -> promote` and stops at the first stage that
says no. It writes a ledger entry to `results/_attempts/` for **every** outcome,
including a pass — that file is the experiment log, and it is what stops an
automated searcher retrying the same idea forever.

Useful flags: `--no-promote` (run every gate, leave the frontier alone),
`--bench-anyway` (measure a change the probe says will fail gate 2, to price it),
`--skip-probe` (only when the effect is already known).

### The probe, on its own

```bash
python3 harness/arena.py probe --target <t> --rebaseline      # once per pinned config
python3 harness/arena.py probe --target <t> --patch <name>
```

One boot instead of four. It compares the **dispatch fingerprint** — the set of
kernel-selection lines the engine printed — plus the output hashes.

| verdict | meaning |
|---|---|
| `no-effect` | fingerprint and output both identical. Do not spend a bench. |
| `will-fail-gate-2` | output differs. A bench would reject it on token identity. |
| `promising` | dispatch changed, output held, throughput moved. Bench it. |
| `inconclusive` | dispatch changed, output held, movement inside the one-boot noise floor. |

**Re-run `--rebaseline` whenever the pinned serve config changes.** The fingerprint
is a property of the config, not only of the engine.

A probe is a filter, never a score. Only the paired bench measures.

### What the probe cannot tell you

It runs one arm, so it cannot separate a small win from drift — that is what
`probeNoise` in the contract is for. And a changed fingerprint is not a
guarantee of a win; it only means the engine made a different decision, which is
the precondition for one.
