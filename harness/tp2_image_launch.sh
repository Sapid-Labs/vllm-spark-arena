#!/usr/bin/env bash
# Launch the image on both nodes and LEAVE the containers in place, so their
# logs survive a failure. No teardown here on purpose.
set -uo pipefail
IMG=aidendle94/sparkrun-vllm-ds4-gb10:production-ready
MODEL=/models/hf/DeepSeek-V4-Flash
W=192.168.0.28
HEAD=192.168.100.1
PORT=8000
MPORT=25510
NAME=arena-img

say() { echo "[$(date +%H:%M:%S)] $*"; }

GID_H=$(bash $HOME/Dev/vllm-spark-arena/harness/gid_index.sh rocep1s0f1 1 $HEAD)
GID_W=$(ssh -n -o BatchMode=yes $W "bash \$HOME/Dev/vllm-spark-arena/harness/gid_index.sh rocep1s0f1 1 192.168.100.2" </dev/null)
say "GID: head=$GID_H worker=$GID_W"

docker rm -f $NAME >/dev/null 2>&1
ssh -n -o BatchMode=yes $W "docker rm -f $NAME >/dev/null 2>&1; true" </dev/null
sleep 4

D="-d --name $NAME --network host --ipc host --shm-size 64gb --gpus all --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME/models:/models:ro -v $HOME/.cache/arena-dsv4-img:/cache -e HF_HOME=/cache/hf -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN"

S="serve $MODEL --served-model-name dsv4 --trust-remote-code --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --tensor-parallel-size 2 --pipeline-parallel-size 1 --nnodes 2 --master-addr $HEAD --master-port $MPORT --block-size 256 --kv-cache-dtype fp8 --max-model-len 8192 --gpu-memory-utilization 0.70 --kv-cache-memory-bytes 6442450944 --max-num-seqs 1 --max-num-batched-tokens 8192"

say "worker rank 1"
ssh -n -o BatchMode=yes $W "docker run $D -e NCCL_IB_GID_INDEX=$GID_W -e VLLM_HOST_IP=192.168.100.2 --entrypoint /usr/local/bin/dsv4-vllm-entrypoint $IMG $S --node-rank 1 --headless" </dev/null >/dev/null

say "head rank 0"
docker run $D -e NCCL_IB_GID_INDEX=$GID_H -e VLLM_HOST_IP=$HEAD --entrypoint /usr/local/bin/dsv4-vllm-entrypoint $IMG $S --node-rank 0 --host 127.0.0.1 --port $PORT >/dev/null

say "launched; containers left running for inspection"
