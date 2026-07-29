#!/usr/bin/env bash
# Does the TP2 shm_broadcast stall need DeepSeek, or does any model do it?
#
# The DeepSeek target costs ~5 minutes per boot and 222 GB of reads. If a 3 GB
# model stalls the same way, the bug belongs to the fleet + vLLM and can be
# chased in 2-minute iterations. If it does not, the bug needs DeepSeek and the
# next suspects are its own kernels.
set -uo pipefail
MODEL=${1:-$HOME/models/hf/Laguna-S-2.1-DFlash-NVFP4}
REQS=${2:-40}
VENV=$HOME/venvs/vllm-026
RAY=$VENV/bin/ray
W=192.168.0.28
HEAD=192.168.100.1
NIC=enp1s0f1np1
PORT=8123

E="PATH=/usr/local/cuda/bin:$VENV/bin:$PATH
NCCL_SOCKET_IFNAME=$NIC
GLOO_SOCKET_IFNAME=$NIC
NCCL_IB_HCA=rocep1s0f1
RAY_local_fs_capacity_threshold=1
RAY_memory_monitor_refresh_ms=0
VLLM_CACHE_ROOT=$HOME/.cache/vllm-tp2iso
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300"
ENVSTR=$(echo $E | tr '\n' ' ')

say() { echo "[$(date +%H:%M:%S)] $*"; }

teardown() {
  ssh -o BatchMode=yes $W "$RAY stop --force >/dev/null 2>&1; pkill -9 -f '[r]aylet --' 2>/dev/null; pkill -9 -f '[v]llm' 2>/dev/null; true"
  pkill -9 -f '[v]llm serve' 2>/dev/null
  $RAY stop --force >/dev/null 2>&1
  pkill -9 -f '[r]aylet --' 2>/dev/null
  sleep 4
}

teardown
say "ray up"
env $ENVSTR VLLM_HOST_IP=$HEAD $RAY start --head --node-ip-address=$HEAD --port=6379 \
  --num-gpus=1 --disable-usage-stats >/tmp/iso-rayhead.log 2>&1
sleep 4
ssh -o BatchMode=yes $W "env $ENVSTR VLLM_HOST_IP=192.168.100.2 $RAY start --address=$HEAD:6379 \
  --node-ip-address=192.168.100.2 --num-gpus=1" >/tmp/iso-rayworker.log 2>&1
sleep 6

say "serve $MODEL TP2"
env $ENVSTR VLLM_HOST_IP=$HEAD $VENV/bin/vllm serve $MODEL --served-model-name iso \
  --distributed-executor-backend ray --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.60 \
  --no-async-scheduling --trust-remote-code \
  --host 127.0.0.1 --port $PORT >/tmp/iso-serve.log 2>&1 &

for i in $(seq 1 60); do
  curl -fsS --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || {
  say "never healthy"; tail -20 /tmp/iso-serve.log; teardown; exit 1; }
say "healthy — warmup first (generous timeout; the first request after a boot is\
 never comparable, and here it may also compile)"
wt0=$(date +%s)
curl -sS --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"iso","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0,"seed":0}' \
  >/tmp/iso-warm.json 2>&1
say "warmup took $(( $(date +%s) - wt0 ))s"
grep -q '"choices"' /tmp/iso-warm.json || { say "WARMUP FAILED"; head -3 /tmp/iso-warm.json; teardown; exit 1; }

say "sending $REQS requests of varying size"

fails=0
for i in $(seq 1 $REQS); do
  # Vary prompt length: the DeepSeek stalls moved between prompts, so size is a
  # suspect. 40 requests across 4 sizes, each timed.
  case $((i % 4)) in
    0) n=20   ;;
    1) n=200  ;;
    2) n=800  ;;
    3) n=2000 ;;
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
    [ $dt -ge 30 ] && say "req $i (~$n tok prompt): SLOW ${dt}s"
  else
    fails=$((fails+1))
    say "req $i (~$n tok prompt): FAILED after ${dt}s"
    echo "$out" | head -3
    break
  fi
done

say "done: $fails failure(s) in $REQS requests"
say "shm_broadcast long-waits in serve log: $(grep -c 'No available shared memory' /tmp/iso-serve.log)"
say "GID table warnings: $(grep -c 'GID table changed' /tmp/iso-serve.log)"
teardown
