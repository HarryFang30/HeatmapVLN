#!/usr/bin/env bash
# Forward the Ascend box's model and VO server ports into the client box.
#
# This is the client half of the split deployment (docs/ops/deploy_ascend_910b.md):
# it runs where the Habitat clients run -- inside the fjl-habitat container on the
# RTX 4090 box, which is on a bridge network with no published ports, so the tunnel
# has to be started inside the container rather than on its host.
#
# Slot k maps 52400+k (model) and 52500+k (VO) to the same local port, because the
# client addresses them as 127.0.0.1:<port> on both platforms and nothing in the
# client path changes between them.
#
# The gRPC channels have no authentication at either end, so they must only ever
# travel inside this tunnel; both ends bind loopback only.  The key is a dedicated
# tunnel key, authorized on the Ascend box with restrict,port-forwarding and
# permitopen for just these ports (scripts/ascend/bootstrap_instance.sh writes that
# line), so it can neither open a shell nor forward anywhere else.  Revoke it by
# deleting its line from ~/.ssh/authorized_keys there.  The private key lives only
# on the client box and is never in this repository.
#
#   PPA_TUNNEL_HOST=<notebook address> PPA_TUNNEL_PORT=<notebook port> \
#     scripts/ascend/start_tunnel.sh 2          # slots 0 and 1
#   PPA_TUNNEL_HOST=... PPA_TUNNEL_PORT=... \
#     scripts/ascend/start_tunnel.sh --slot 1   # just slot 1, next to a running tunnel
#
# Reconnects for ever: the evaluation client has no RPC retry of its own, so a
# tunnel that is down for one call takes the whole shard with it.  Adding a slot to
# a running evaluation therefore uses --slot rather than restarting this script with
# a larger count, which would drop the live forwards.
#
# Stop it by PID (pkill -f would also match the shell running it).

set -u

usage() {
  printf 'usage: %s [slots] | %s --slot <index>\n' "$0" "$0" >&2
  exit 2
}

TUNNEL_DIR="${PPA_TUNNEL_DIR:-/workspace/ppa_tunnel}"
KEY="${PPA_TUNNEL_KEY:-$TUNNEL_DIR/id_ed25519}"
KNOWN_HOSTS="${PPA_TUNNEL_KNOWN_HOSTS:-$TUNNEL_DIR/known_hosts}"
HOST="${PPA_TUNNEL_HOST:?set PPA_TUNNEL_HOST to the Ascend instance address}"
PORT="${PPA_TUNNEL_PORT:?set PPA_TUNNEL_PORT to the Ascend instance SSH port}"
LOGIN="${PPA_TUNNEL_USER:-ma-user}"
MODEL_PORT_BASE="${PPA_EVAL_MODEL_PORT_BASE:-52400}"
VO_PORT_BASE="${PPA_EVAL_VO_PORT_BASE:-52500}"

declare -a SLOTS=()
case "${1:-1}" in
  --slot)
    [[ "${2:-}" =~ ^[0-7]$ ]] || usage
    SLOTS=("$2")
    label="slot $2"
    ;;
  *)
    [[ "${1:-1}" =~ ^[1-8]$ ]] || usage
    for slot in $(seq 0 $(( ${1:-1} - 1 ))); do SLOTS+=("$slot"); done
    label="slots ${SLOTS[*]}"
    ;;
esac

[[ -r "$KEY" ]] || { printf '[tunnel] ERROR: no readable key at %s\n' "$KEY" >&2; exit 2; }

# A container shell may carry a proxy that does not exist here, and ssh would use it.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

declare -a FORWARDS=()
for slot in "${SLOTS[@]}"; do
  FORWARDS+=(-L "127.0.0.1:$((MODEL_PORT_BASE + slot)):127.0.0.1:$((MODEL_PORT_BASE + slot))")
  FORWARDS+=(-L "127.0.0.1:$((VO_PORT_BASE + slot)):127.0.0.1:$((VO_PORT_BASE + slot))")
done

printf '[tunnel] %s -> %s:%s as %s\n' "$label" "$HOST" "$PORT" "$LOGIN"

while true; do
  # ExitOnForwardFailure: a port already taken locally must fail loudly, not leave a
  # live ssh session with no forwards while the client happily connects to whatever
  # else holds the port.
  ssh -N -i "$KEY" -p "$PORT" \
    -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$KNOWN_HOSTS" \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ControlMaster=no -o ControlPath=none \
    "${FORWARDS[@]}" "$LOGIN@$HOST"
  printf '[tunnel] %s: ssh exited (%s); reconnecting in 5s\n' "$label" "$?" >&2
  sleep 5
done
