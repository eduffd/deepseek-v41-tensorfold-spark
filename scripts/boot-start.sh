#!/usr/bin/env bash
# Boot-start fast path for the DeepSeek-V4.1 service. Run once at boot by scripts/systemd/dsv41-boot-start.service on
# the head (a user unit with linger on); the watchdog stays the fallback. It runs on the config it is given: sourcing
# scripts/serve.sh pulls STATE_DIR, BOOT_LEASE, BOOT_LEASE_MIN and the recipe's keys from $CONFIG, so a second profile
# (CONFIG=config/prod-zig.env) reads its own stop marker and its own lease, not the Python recipe's.
# Decides from the state the previous boot left, through the recipe's pair_boot_decision (the Python recipe's:
#   - BOOT_LEASE (optional file) younger than BOOT_LEASE_MIN minutes -> do nothing (you are working on the pair)
#   - rank 0 already running                                           -> do nothing
#   - rank 0 container present (exited: power loss / crash)            -> start
#   - rank 0 absent and serve.sh's stop marker is from an earlier boot -> start
#   - rank 0 absent and no such marker (removed by hand, or stopped during this boot) -> do nothing
# Before the start it waits (bounded, BOOT_WAIT s) for docker, the worker over ssh with docker up, and the RoCE ports
# ACTIVE on both nodes, through serve.sh's nodes_ready; then `scripts/serve.sh restart` (drops caches, memory gate,
# slot check, canary).
# Log: $STATE_DIR/boot-start.log. BOOT_DRY_RUN=1 prints the decision and waits, starts nothing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
CONFIG=${CONFIG:-config/prod.env}
# Sourcing serve.sh applies its `set -e` to this shell, and boot-start must run past a failed start to log what
# serving answers, so it comes off.
SERVE_LIB=1
# shellcheck disable=SC1090
source scripts/serve.sh
set +e
BOOT_WAIT=${BOOT_WAIT:-300}
BOOT_LEASE=${BOOT_LEASE:-}
BOOT_LEASE_MIN=${BOOT_LEASE_MIN:-20}
mkdir -p "$STATE_DIR"
[[ "${BOOT_STDOUT:-0}" == 1 ]] || exec >> "$STATE_DIR/boot-start.log" 2>&1
up() { awk '{printf "%.0f", $1}' /proc/uptime; }
log() { echo "[$(date '+%F %T')] dsv41 boot-start (+$(up)s): $*"; }

if [[ -n "$BOOT_LEASE" && -f "$BOOT_LEASE" ]] && (( $(date +%s) - $(stat -c %Y "$BOOT_LEASE") < BOOT_LEASE_MIN * 60 )); then
    log "$BOOT_LEASE is fresh: not starting (the watchdog takes over when it ages)"; exit 0
fi
nodes_ready "$BOOT_WAIT" || { log "worker ($WORKER_SSH) / docker / RoCE not ready after ${BOOT_WAIT}s: leaving it to the watchdog"; exit 1; }
pair_boot_decision || exit 0
[[ "${BOOT_DRY_RUN:-0}" == 1 ]] && { log "BOOT_DRY_RUN=1: not starting"; exit 0; }
# restart, not start: it removes both rank containers first (after a head-only reboot the worker's rank 1 still runs)
log "serve.sh restart ($CONFIG)"
CONFIG="$CONFIG" scripts/serve.sh restart; rc=$?
log "serve.sh restart rc=$rc; serving: $(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | cut -c1-80)"
exit $rc
