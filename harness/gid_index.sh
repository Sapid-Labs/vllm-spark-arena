#!/usr/bin/env bash
# Print the RoCE v2 GID index on <device>:<port> that carries <ipv4>.
#
#   gid_index.sh rocep1s0f1 1 192.168.100.2   ->  5
#
# WHY THIS IS NOT A CONSTANT. A RoCE GID table holds one entry per address per
# type on the port, so the index of the IPv4 RoCE v2 entry depends on how many
# other addresses that interface has. Measured 2026-07-29 on this fleet:
#
#   Spark-1  idx 3 = RoCE v2  ::ffff:c0a8:6401   (192.168.100.1)
#   Spark-2  idx 5 = RoCE v2  ::ffff:c0a8:6402   (192.168.100.2)
#
# They differ because Spark-2 has an IPv6 link-local on that interface and
# Spark-1 does not, which inserts two entries and shifts everything after them.
# Pinning one NCCL_IB_GID_INDEX on both nodes therefore points at an IPv6 GID on
# one of them, and the queue-pair transition INIT->RTR fails:
#
#   NCCL WARN Call to ibv_modify_qp failed with 22 Invalid argument,
#             on dev rocep1s0f1:1, curr state INIT, next state RTR
#   RuntimeError: NCCL error: unhandled system error
#
# The index also MOVES when an address appears or disappears -- and this fleet
# runs Tailscale and IPv6 privacy addresses -- so it must be derived at launch
# on each node, never copied from a working run on another node or another day.
set -uo pipefail
DEV=${1:?device, e.g. rocep1s0f1}
PORT=${2:-1}
IP=${3:?ipv4 address that must carry the traffic}

# 192.168.100.2 -> c0a8:6402, the tail of the IPv4-mapped IPv6 GID.
hex=$(printf '%02x%02x:%02x%02x' $(echo "$IP" | tr '.' ' '))
D=/sys/class/infiniband/$DEV/ports/$PORT

for i in $(seq 0 31); do
  g=$(cat "$D/gids/$i" 2>/dev/null) || continue
  [ -z "$g" ] && continue
  t=$(cat "$D/gid_attrs/types/$i" 2>/dev/null)
  # RoCE v2 only: v1 is not routable and the peers must agree on the version.
  [ "$t" = "RoCE v2" ] || continue
  case "$g" in
    *ffff:$hex) echo "$i"; exit 0 ;;
  esac
done

echo "no RoCE v2 GID for $IP on $DEV:$PORT" >&2
exit 1
