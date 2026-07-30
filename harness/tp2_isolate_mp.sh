#!/usr/bin/env bash
# Same TP2 stall test, but over vLLM's HEADLESS mp backend instead of Ray.
#
# Why: the published dual-Spark recipe -- which works, at 42-76 tok/s -- uses
#   --distributed-executor-backend mp --nnodes 2 --node-rank N --master-addr ... --headless
# and no Ray at all. The arena used the Ray executor, whose message path
# (ray_executor_v2 -> shm_broadcast) is where every stall was found. This tests
# the recipe's own transport, plus the NCCL pins from its .env that the arena
# never carried over.
set -uo pipefail
MODEL=${1:-$HOME/models/hf/Qwen3.6-27B-NVFP4}
REQS=${2:-24}
VENV=$HOME/venvs/vllm-026
W=192.168.0.28              # worker over real OpenSSH (LAN, never the tailnet)
HEAD=192.168.100.1
NIC=enp1s0f1np1
PORT=8135
MPORT=25210

# Straight from the working recipe's docker-compose + .env.
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
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,ENV"
ENVSTR=$(echo $E | tr '\n' ' ')


COMMON="--tensor-parallel-size 2 --pipeline-parallel-size 1 \
--max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.60 \
--no-async-scheduling --trust-remote-code \
--nnodes 2 --master-addr $HEAD --master-port $MPORT"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# Derived per node, never shared. The IPv4 RoCE v2 GID sits at a different index
# on each of these two machines (3 and 5), because one has an extra IPv6
# link-local on the interface. One shared index points at an IPv6 GID on the
# other node and ibv_modify_qp fails INIT->RTR with EINVAL.
GID_H=$(bash $HOME/Dev/vllm-spark-arena/harness/gid_index.sh rocep1s0f1 1 $HEAD)
GID_W=$(ssh -n -o BatchMode=yes $W "bash \$HOME/Dev/vllm-spark-arena/harness/gid_index.sh rocep1s0f1 1 192.168.100.2" </dev/null)
say "GID index: head=$GID_H worker=$GID_W"

teardown() {
  [ -n "${WSSH:-}" ] && kill $WSSH 2>/dev/null
  ssh -n -o BatchMode=yes $W "pkill -9 -f '[V]LLM::'; pkill -9 -f '[v]llm'; true" </dev/null
  pkill -9 -f '[V]LLM::' 2>/dev/null
  pkill -9 -f '[v]llm serve' 2>/dev/null
  sleep 6
}

teardown
# HEAD FIRST: rank 0 hosts the torch.distributed TCPStore at master-port, and
# the worker blocks on connecting to it (600 s, then dies).
say "head: rank 0 + API server (hosts the TCPStore)"
env $ENVSTR NCCL_IB_GID_INDEX=$GID_H VLLM_HOST_IP=$HEAD $VENV/bin/vllm serve $MODEL --served-model-name iso \
  $COMMON --node-rank 0 --host 127.0.0.1 --port $PORT >/tmp/mp-head.log 2>&1 &

sleep 15
say "worker: headless rank 1"
# Backgrounded LOCALLY. A remote `nohup ... &` does not make ssh return on this
# fleet even with -n and </dev/null, and waiting for it means the head never
# launches -- which is what made the worker time out on the TCPStore.
ssh -n -o BatchMode=yes $W \
  "cd \$HOME && env $ENVSTR NCCL_IB_GID_INDEX=$GID_W VLLM_HOST_IP=192.168.100.2 $VENV/bin/vllm serve $MODEL \
   --served-model-name iso $COMMON --node-rank 1 --headless \
   > /tmp/mp-worker.log 2>&1 </dev/null" </dev/null &
WSSH=$!

for i in $(seq 1 90); do
  curl -fsS --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || {
  say "never healthy"; tail -25 /tmp/mp-head.log; teardown; exit 1; }
say "healthy"

curl -sS --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"iso","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0,"seed":0}' \
  >/tmp/mp-warm.json 2>&1
grep -q '"choices"' /tmp/mp-warm.json || { say "WARMUP FAILED"; head -3 /tmp/mp-warm.json; teardown; exit 1; }
say "warmup ok — sending $REQS requests"

fails=0 slow=0
for i in $(seq 1 $REQS); do
  case $((i % 4)) in
    0) n=20 ;; 1) n=200 ;; 2) n=800 ;; 3) n=2000 ;;
  esac
  p=$(python3 -c "print('the quick brown fox jumps over the lazy dog. ' * ($n // 9))")
  t0=$(date +%s)
  out=$(curl -sS --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'iso','messages':[{'role':'user','content':sys.argv[1]}],
                  'max_tokens':64,'temperature':0,'top_p':1,'seed':0}))" "$p")" 2>&1)
  dt=$(( $(date +%s) - t0 ))
  if echo "$out" | grep -q '"choices"'; then
    if [ $dt -ge 30 ]; then slow=$((slow+1)); say "req $i (~$n tok): SLOW ${dt}s"; fi
  else
    fails=$((fails+1)); say "req $i (~$n tok): FAILED after ${dt}s"; echo "$out" | head -3; break
  fi
done

say "RESULT: $fails failure(s), $slow slow, out of $REQS requests"
say "shm long-waits: $(grep -c 'No available shared memory' /tmp/mp-head.log)"
say "GID warnings:   $(grep -c 'GID table changed' /tmp/mp-head.log)"
teardown
