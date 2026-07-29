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
