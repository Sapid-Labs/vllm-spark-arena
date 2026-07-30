#!/usr/bin/env bash
# Find the prompt size at which a TP2 request stops returning.
#
# Both executors (ray and mp) stalled on the SAME request in the isolation test:
# number 2, the ~800-token prompt. That is a threshold, not randomness. One boot,
# then prompts in increasing size until one hangs -- the engine usually dies with
# the first hang, so increasing order gets the answer in a single boot.
set -uo pipefail
MODEL=${1:-$HOME/models/hf/Qwen3.6-27B-NVFP4}
VENV=$HOME/venvs/vllm-026
W=192.168.0.28
HEAD=192.168.100.1
NIC=enp1s0f1np1
PORT=${2:-8146}
MPORT=${3:-25320}

E="PATH=/usr/local/cuda/bin:$VENV/bin:$PATH
NCCL_SOCKET_IFNAME=$NIC
GLOO_SOCKET_IFNAME=$NIC
NCCL_IB_HCA=rocep1s0f1
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_CROSS_NIC=0
NCCL_CUMEM_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1
VLLM_CACHE_ROOT=$HOME/.cache/vllm-tp2mp
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600"
ENVSTR=$(echo $E | tr '\n' ' ')

say() { echo "[$(date +%H:%M:%S)] $*"; }

GID_H=$(bash $HOME/Dev/vllm-spark-arena/harness/gid_index.sh rocep1s0f1 1 $HEAD)
GID_W=$(ssh -n -o BatchMode=yes $W "bash \$HOME/Dev/vllm-spark-arena/harness/gid_index.sh rocep1s0f1 1 192.168.100.2" </dev/null)
say "GID index: head=$GID_H worker=$GID_W"

COMMON="--tensor-parallel-size 2 --pipeline-parallel-size 1 \
--max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.60 \
--no-async-scheduling --trust-remote-code \
--nnodes 2 --master-addr $HEAD --master-port $MPORT"

teardown() {
  [ -n "${WSSH:-}" ] && kill $WSSH 2>/dev/null
  # vLLM RENAMES its children to VLLM::EngineCore / VLLM::Worker_TP0, so a
  # pattern of 'vllm serve' misses every one of them. Measured: leftovers held
  # 11 GiB on the head and 71 GiB on the worker across runs, and a stale worker
  # still attached to the same shm_broadcast segment consumes blocks the new run
  # is waiting for. Guard the pattern too -- [V]LLM:: unguarded matches the ssh
  # command line carrying it and kills your own session.
  ssh -n -o BatchMode=yes $W "pkill -9 -f '[V]LLM::'; pkill -9 -f '[v]llm'; true" </dev/null
  pkill -9 -f '[V]LLM::' 2>/dev/null
  pkill -9 -f '[v]llm serve' 2>/dev/null
  sleep 6
  for _n in 1 2 3; do
    local h w
    h=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    w=$(ssh -n -o BatchMode=yes $W "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c ." </dev/null || true)
    [ "${h:-0}" = "0" ] && [ "${w:-0}" = "0" ] && break
    say "  waiting for GPUs to clear (head=$h worker=$w)"; sleep 6
  done
}
teardown

env $ENVSTR NCCL_IB_GID_INDEX=$GID_H VLLM_HOST_IP=$HEAD $VENV/bin/vllm serve $MODEL \
  --served-model-name iso $COMMON --node-rank 0 --host 127.0.0.1 --port $PORT \
  >/tmp/bis-head.log 2>&1 &
sleep 15
ssh -n -o BatchMode=yes $W \
  "cd \$HOME && env $ENVSTR NCCL_IB_GID_INDEX=$GID_W VLLM_HOST_IP=192.168.100.2 \
   $VENV/bin/vllm serve $MODEL --served-model-name iso $COMMON --node-rank 1 --headless \
   > /tmp/bis-worker.log 2>&1 </dev/null" </dev/null &
WSSH=$!

for i in $(seq 1 90); do
  curl -fsS --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || {
  say "never healthy"; grep -aiE "Error" /tmp/bis-head.log | tail -3 | cut -c1-180; teardown; exit 1; }
say "healthy"

ask() {  # <approx prompt tokens> <timeout>
  local n=$1 to=$2
  local p; p=$(python3 -c "print('the quick brown fox jumps over the lazy dog. ' * ($n // 9))")
  local body; body=$(python3 -c "
import json,sys
print(json.dumps({'model':'iso','messages':[{'role':'user','content':sys.argv[1]}],
                  'max_tokens':32,'temperature':0,'top_p':1,'seed':0}))" "$p")
  local t0=$(date +%s)
  local out; out=$(curl -sS --max-time $to http://127.0.0.1:$PORT/v1/chat/completions \
                   -H 'Content-Type: application/json' -d "$body" 2>&1)
  local dt=$(( $(date +%s) - t0 ))
  if echo "$out" | grep -q '"choices"'; then
    # Report the real prompt-token count the server saw, not my estimate.
    local pt; pt=$(echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)
    say "  size~$n -> OK in ${dt}s (prompt_tokens=$pt)"
    return 0
  fi
  say "  size~$n -> HUNG (${dt}s)"
  return 1
}

ask 20 120 || true    # warmup, discarded

say "sweeping prompt size upward"
for n in 100 200 300 400 500 600 700 800 900 1000 1200 1600 2000; do
  ask $n 150 || { say "THRESHOLD: first hang at ~$n prompt tokens"; break; }
done

say "shm long-waits: $(grep -ac 'No available shared memory' /tmp/bis-head.log)"
teardown
