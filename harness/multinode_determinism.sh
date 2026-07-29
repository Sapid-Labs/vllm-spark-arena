#!/usr/bin/env bash
# Multi-node greedy determinism for DeepSeek-V4-Flash under TP2.
#
# MUST be run on Spark-1 through `ssh localhost` (real OpenSSH), never through a
# Tailscale SSH session: tailscaled does not run PAM, so pam_limits never fires
# and every child inherits memlock=8MB, which silently forces NCCL off RDMA and
# onto TCP sockets. That is the transport the previous two attempts ran on.
#
# Two boots, same pinned config, same prompts, greedy. Compare hashes.
set -uo pipefail

ARENA=$HOME/Dev/vllm-spark-arena
VENV=$HOME/venvs/vllm-026
WORKER=192.168.0.28            # Spark-2 over LAN OpenSSH (NOT the tailnet IP)
HEAD=192.168.100.1             # 200G ConnectX link
NIC=enp1s0f1np1
OUT=/tmp/dsv4_det
RAY=$VENV/bin/ray
SSHW="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new $WORKER"

mkdir -p $OUT

ENVS=(
  # vLLM's has_flashinfer() returns False when nvcc is absent, because FlashInfer
  # JITs its kernels -- and a non-interactive shell does not get /usr/local/cuda
  # on PATH. The symptom is "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's
  # sparse MLA decode API", which points at the overlay and is not about the
  # overlay at all. The llama.cpp harness already guards this; this one did not.
  # PREPEND, never replace: FlashInfer also shells out to `ninja`, which lives
  # in the venv, so a hardcoded PATH that drops $VENV/bin trades one missing
  # build tool for another.
  "PATH=/usr/local/cuda/bin:$VENV/bin:$HOME/.local/bin:$PATH"
  "RAY_local_fs_capacity_threshold=1"
  "RAY_memory_monitor_refresh_ms=0"
  "RAY_memory_usage_threshold=0.99"
  "PYTHONPATH=$ARENA/patches/flashinfer-dsv4-pbs256"
  "VLLM_CACHE_ROOT=$HOME/.cache/vllm-dsv4"
  "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900"
  "MAX_JOBS=4"
  "NCCL_SOCKET_IFNAME=$NIC"
  "GLOO_SOCKET_IFNAME=$NIC"
  "NCCL_DEBUG=INFO"
  "NCCL_DEBUG_SUBSYS=INIT,NET"
)
ENVSTR="${ENVS[*]}"

# /etc/hosts on both nodes maps the hostname to 127.0.0.1 BEFORE its ConnectX
# address, so anything that resolves its own hostname binds loopback. Gloo did
# exactly that and rank 1 on the other node got "connection refused" to
# 127.0.0.1 -- which reads like an interconnect fault and is not one. Pin the
# host IP per node; it must differ between them, so it cannot live in ENVSTR.
HEAD_ENV="VLLM_HOST_IP=192.168.100.1"
WORKER_ENV="VLLM_HOST_IP=192.168.100.2"

log() { echo "[$(date +%H:%M:%S)] $*"; }

teardown() {
  log "teardown"
  # [r]aylet character class: an unguarded pattern matches the ssh command line
  # running it and kills the session mid-teardown.
  $SSHW "pkill -9 -f '[v]llm serve' 2>/dev/null; $RAY stop --force >/dev/null 2>&1; pkill -9 -f '[r]aylet --' 2>/dev/null; pkill -9 -f '[g]cs_server' 2>/dev/null; true"
  pkill -9 -f '[v]llm serve' 2>/dev/null
  $RAY stop --force >/dev/null 2>&1
  pkill -9 -f '[r]aylet --' 2>/dev/null
  pkill -9 -f '[g]cs_server' 2>/dev/null
  sleep 5
  # pgrep -c prints 0 AND exits non-zero, so `|| echo 0` yields "0\n0".
  local h w
  h=$(pgrep -cf '[r]aylet --'); h=${h:-0}
  w=$($SSHW "pgrep -cf '[r]aylet --'"); w=${w:-0}
  log "surviving raylets: head=$h worker=$w"
  [ "$h" = "0" ] && [ "$w" = "0" ]
}

boot() {
  local n=$1
  log "=== BOOT $n ==="
  teardown || { log "BOOT $n: stale raylet survived teardown — refusing to attach"; return 1; }

  log "boot $n: memlock head=$(ulimit -l) worker=$($SSHW 'ulimit -l')"

  env $ENVSTR $HEAD_ENV $RAY start --head --node-ip-address=$HEAD --port=6379 \
    --num-gpus=1 --disable-usage-stats >$OUT/ray-head-$n.log 2>&1 || return 1
  sleep 5
  $SSHW "env $ENVSTR $WORKER_ENV $RAY start --address=$HEAD:6379 --node-ip-address=192.168.100.2 --num-gpus=1" \
    >$OUT/ray-worker-$n.log 2>&1 || return 1
  sleep 8

  # A cluster that quietly came up with one GPU produces a plausible number from
  # the wrong hardware, which is worse than an error.
  local gpus
  gpus=$(env $ENVSTR $VENV/bin/python -c "import ray; ray.init(address='$HEAD:6379'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1)
  log "boot $n: ray sees ${gpus:-?} GPU(s)"
  [ "${gpus:-0}" = "2" ] || { log "boot $n: expected 2 GPUs across 2 nodes"; return 1; }

  # Assert the kernel path the target depends on is actually reachable, on BOTH
  # nodes, before spending 4 minutes loading 222 GB of weights to find out.
  for where in head worker; do
    local got
    if [ $where = head ]; then
      got=$(env $ENVSTR $VENV/bin/python -c "from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120 as f; print(f())" 2>/dev/null | tail -1)
    else
      got=$($SSHW "env $ENVSTR $VENV/bin/python -c \"from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120 as f; print(f())\"" 2>/dev/null | tail -1)
    fi
    log "boot $n: sparse MLA sm120 available on $where = ${got:-?}"
    [ "$got" = "True" ] || { log "boot $n: $where cannot reach the DSv4 decode kernel"; return 1; }
  done

  env $ENVSTR $HEAD_ENV $VENV/bin/vllm serve $HOME/models/hf/DeepSeek-V4-Flash \
    --served-model-name dsv4 \
    --distributed-executor-backend ray --tensor-parallel-size 2 \
    --block-size 256 --kv-cache-dtype fp8 \
    --max-model-len 8192 --gpu-memory-utilization 0.70 \
    --kv-cache-memory-bytes 6442450944 \
    --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --trust-remote-code >$OUT/serve-$n.log 2>&1 &
  local pid=$!

  local ok=0
  for i in $(seq 1 90); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || { log "boot $n: server died after ${i}0s"; break; }
    sleep 10
  done
  [ $ok = 1 ] || { log "boot $n: never healthy"; tail -30 $OUT/serve-$n.log; return 1; }
  log "boot $n: healthy"

  # Is the collective actually on RDMA? This is the whole point of the rerun.
  if grep -qE 'NET/IB|NET/IB.*RoCE' $OUT/serve-$n.log; then
    log "boot $n: NCCL transport = IB/RoCE  $(grep -oE 'NET/IB[^ ]*' $OUT/serve-$n.log | head -1)"
  elif grep -q 'NET/Socket' $OUT/serve-$n.log; then
    log "boot $n: NCCL transport = SOCKET (RDMA did NOT engage)"
  else
    log "boot $n: NCCL transport = unknown"
  fi

  $VENV/bin/python $OUT/probe.py $n >$OUT/hashes-$n.txt 2>&1
  local rc=$?
  cat $OUT/hashes-$n.txt
  teardown
  return $rc
}

cat >$OUT/probe.py <<'PY'
import hashlib, json, sys, urllib.request
BOOT = sys.argv[1]
PROMPTS = [
    ("a-capital", "What is the capital of France? Answer in one word."),
    ("b-code", "Write a Python function that reverses a linked list. Code only."),
    ("c-prose", "Explain in three sentences why RDMA is faster than TCP sockets."),
    ("d-math", "Compute 17 * 23 and show each step of the multiplication."),
]
def ask(prompt, max_tokens):
    body = json.dumps({
        "model": "dsv4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]

# The first request after a boot differs on both engines. Discard it or every
# comparison reports a phantom mismatch.
try:
    ask("Say OK.", 8)
    print("warmup ok (discarded)")
except Exception as e:
    print("warmup FAILED:", e); sys.exit(1)

for name, p in PROMPTS:
    try:
        out = ask(p, 128)
    except Exception as e:
        print(f"{name}\tERROR\t{e}"); sys.exit(1)
    print(f"{name}\t{hashlib.sha256(out.encode()).hexdigest()[:16]}\t{len(out)}")
PY

rc=0
boot 1 || rc=1
boot 2 || rc=1

if [ -s $OUT/hashes-1.txt ] && [ -s $OUT/hashes-2.txt ]; then
  echo "=== COMPARISON ==="
  if diff <(grep -P '^\w-' $OUT/hashes-1.txt) <(grep -P '^\w-' $OUT/hashes-2.txt) >/dev/null; then
    echo "IDENTICAL across boots — TP2 greedy output survived the NCCL collective"
  else
    echo "DIVERGED across boots:"
    diff <(grep -P '^\w-' $OUT/hashes-1.txt) <(grep -P '^\w-' $OUT/hashes-2.txt)
  fi
fi
exit $rc
