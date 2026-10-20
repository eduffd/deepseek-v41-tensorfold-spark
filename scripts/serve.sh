#!/usr/bin/env bash
# Start / stop / inspect the two DeepSeek-V4.1-Flash TensorFold ranks. Run on the head Spark (rank 0).
#
#   scripts/serve.sh build       build the image here (docker/Dockerfile), ship it to the worker (save | ssh load), then
#                                prebuild (PREBUILD=0 skips it)
#   scripts/serve.sh prebuild    build the CUDA extensions into CACHE_VOL on both nodes, no weights loaded
#                                (scripts/prebuild_ext.py; stale build locks removed first); the server must be stopped
#                                then `cache` (below)
#   scripts/serve.sh cache       the cache volume's data files on both nodes: config/pfdense-table.json at
#                                TF_DSV41_PF_DENSE_TABLE, and the image routing bias at TF_DSV41_BIAS_VL (IMAGES=native)
#   scripts/serve.sh preflight [static]   read-only checks of both nodes (image, weights, Engram shards, RoCE ports;
#                                not static: the HTTP / rendezvous ports, GPUs idle, the RoCE-failed marker)
#   scripts/serve.sh start       preflight, drop caches, memory gate, rank 1 on the worker then rank 0 here, wait for
#                                /v1/models, slot check, drop caches (the boot's), canary, drop caches again
#   scripts/serve.sh restart | stop | status | logs [0|1] | canary | watch [--once] | args [0|1]
#   scripts/serve.sh run MODULE [ARGS...]   an engine module on both ranks instead of the server (benchmarks:
#                                tensorfold.families.deepseek_v41.cuda.m2bench, .gate); rank 0 writes to OUT=dir (/out)
#
# Configuration: config/prod.env (cp config/prod.env.example config/prod.env; CONFIG=path for another). A non-empty
# caller export of a config key wins over the file, e.g. a test server next to nothing else:
#   PORT=8001 NAME=dsv41-tf-test MASTER_PORT=29561 scripts/serve.sh start
# Recipe: RECIPE (config key, default python) picks scripts/serve/$RECIPE.sh, sourced after the config; it answers
# five calls (pair_required_keys, pair_exit <rank>, pair_launch <rank>, pair_stop <grace_s>, pair_boot_decision).
# python = config/prod.env today; zig = the pair scripts/run-zig.sh launches (its adapter names its own keys).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-config/prod.env}"
[[ -f "$CONFIG" ]] || { echo "[dsv41-tf] no config file $CONFIG (cp config/prod.env.example config/prod.env and fill it in)" >&2; exit 2; }
export CONFIG
caller_env=$(env | grep -E '^(CONTEXT|PARALLEL|IMAGE|PORT|HOST|NAME|SERVED_NAME|ALIASES|MAX_TOKENS|KV_DTYPE|MASTER_PORT|CANARY|DROP_CACHES|MEM_GATE_[A-Z]+|READY_TIMEOUT|PREFLIGHT|HEAD_STATE|WORKER_STATE|HEAD_PREPARED|WORKER_PREPARED|CACHE_VOL|STATE_DIR|LOG_MAX_[A-Z]+|RECIPE|BOOT_[A-Z_]+|WATCH_[A-Z_]+|OUT|TF_DSV41_[A-Z0-9_]+|GLM53_TF_[A-Z0-9_]+|(MI)?MALLOC_[A-Z_]+)=.' || true)
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a
while IFS= read -r kv; do [[ -n "$kv" ]] && export "${kv?}"; done <<< "$caller_env"
# The config boundary: the recipe, its adapter file, the pair's addresses, then the recipe's own key check. The ports,
# the head address and the slot-line regex are keys / derived values here, not adapter calls: the Zig profile names
# them TF_TP_PORT and TF_TP_MASTER, so each takes the other's value when its own is unset.
RECIPE="${RECIPE:-python}"
case "$RECIPE" in
    python|zig) ;;
    *) echo "[dsv41-tf] $CONFIG: RECIPE=$RECIPE: use python or zig" >&2; exit 2 ;;
esac
adapter="scripts/serve/$RECIPE.sh"
[[ -f "$adapter" ]] || { echo "[dsv41-tf] $CONFIG: no adapter for RECIPE=$RECIPE (write $adapter)" >&2; exit 2; }
NAME="${NAME:-dsv41-tf}"                            # containers $NAME-r0 (head) / $NAME-r1 (worker), both recipes
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-${TF_TP_PORT:-29571}}"; TF_TP_PORT="${TF_TP_PORT:-$MASTER_PORT}"
HEAD_IP="${HEAD_IP:-${TF_TP_MASTER:-}}"
SLOT_LINE_RE="${SLOT_LINE_RE:-serving: [0-9]+ slot\(s\)}"
ROCE_STATE="${ROCE_STATE:-/sys/class/infiniband}"
# shellcheck disable=SC1090
source "$adapter"
pair_required_keys

HOST="${HOST:-127.0.0.1}"
BASE="http://127.0.0.1:$PORT"
PARALLEL="${PARALLEL:-4}"
CACHE_VOL="${CACHE_VOL:-dsv41-tf-cache}"
CANARY="${CANARY:-strict}"
DROP_CACHES="${DROP_CACHES:-1}"
MEM_GATE_GIB="${MEM_GATE_GIB:-0}"
MEM_GATE_TIMEOUT="${MEM_GATE_TIMEOUT:-600}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
PREFLIGHT="${PREFLIGHT:-strict}"
LOG_MAX_SIZE="${LOG_MAX_SIZE:-200m}"
LOG_MAX_FILE="${LOG_MAX_FILE:-3}"
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/dsv41-tf}"
log() { echo "[dsv41-tf] $*"; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
on_node() { local n=$1; shift; if [[ "$n" == head ]]; then bash -c "$*"; else wssh "$*"; fi; }

roce_cmd() { # the RoCE leg as one command string, so the head and the worker run the same test (printf keeps $d for
    # the node's own shell): every NCCL_IB_HCA device's port 1 must read ACTIVE
    printf 'for d in %s; do grep -q ACTIVE %s/$d/ports/1/state || exit 1; done' "${NCCL_IB_HCA//,/ }" "$ROCE_STATE"
}

nodes_ready() { # both nodes usable for the pair: docker here, docker on the worker over ssh, RoCE ACTIVE on both.
    # $1 = seconds to keep trying (0: one look). This is the repo's one readiness gate: preflight, boot-start.sh and
    # every heal run through it, so the RoCE test lives in exactly one place.
    local end
    end=$(( $(date +%s) + ${1:-0} ))
    while :; do
        if docker info >/dev/null 2>&1 && wssh "docker info >/dev/null 2>&1 && $(roce_cmd)" >/dev/null 2>&1 && bash -c "$(roce_cmd)"; then
            return 0
        fi
        (( $(date +%s) < end )) || { log "nodes not ready: docker on this node, docker over ssh ($WORKER_SSH), RoCE ACTIVE (${NCCL_IB_HCA//,/ })"; return 1; }
        sleep 3
    done
}

gpu_busy() { # a CUDA process on either node: another workload is still up
    { nvidia-smi --query-compute-apps=pid --format=csv,noheader; wssh nvidia-smi --query-compute-apps=pid --format=csv,noheader; } \
        2>/dev/null | grep -q '[0-9]'
}

running() { # $1 = rank: true | false | absent | unreachable
    local out rc
    if [[ "$1" == 0 ]]; then
        out=$(docker inspect -f '{{.State.Running}}' "$NAME-r0" 2>/dev/null) || out=absent
    else
        out=$(wssh docker inspect -f "'{{.State.Running}}'" "$NAME-r1" 2>/dev/null) && rc=0 || rc=$?
        if [[ $rc == 255 ]]; then out=unreachable; elif [[ $rc != 0 ]]; then out=absent; fi
    fi
    echo "${out:-absent}"
}

memfree() { on_node "$1" "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || true; }

drop_caches() { # both nodes: page cache only (`echo 3`: clean pages + slab), never swap
    [[ "$DROP_CACHES" == 1 ]] || return 0
    local node
    for node in head worker; do
        on_node "$node" "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null \
            && log "dropped the $node's page cache (MemFree now $(memfree "$node") GiB)" \
            || log "drop_caches refused on the $node"
    done
}

mem_gate() { # MemFree (not MemAvailable: unified memory) >= MEM_GATE_GIB on both nodes, or MEM_GATE_TIMEOUT
    [[ "$MEM_GATE_GIB" -gt 0 ]] || return 0
    local t0 h w
    t0=$(date +%s)
    while :; do
        h=$(memfree head); w=$(memfree worker)
        if [[ "${h:-0}" -ge "$MEM_GATE_GIB" && "${w:-0}" -ge "$MEM_GATE_GIB" ]]; then
            log "MemFree ${h} / ${w} GiB (head / worker) >= $MEM_GATE_GIB"; return 0
        fi
        if (( $(date +%s) - t0 >= MEM_GATE_TIMEOUT )); then
            log "MemFree ${h:-?} / ${w:-?} GiB still under $MEM_GATE_GIB after ${MEM_GATE_TIMEOUT}s; not starting"; return 1
        fi
        log "waiting for memory: MemFree ${h:-?} / ${w:-?} GiB (head / worker), want $MEM_GATE_GIB"
        DROP_CACHES=1 drop_caches >/dev/null; sleep 10
    done
}

check_slots() { # rank 0's `[tensorfold] serving: N slot(s) x ...` line (deepseek_v41 stack.py) must show PARALLEL
    local n="" _i
    for _i in $(seq 1 60); do
        n=$(docker logs "$NAME-r0" 2>&1 | grep -oE "$SLOT_LINE_RE" | tail -1 | grep -oE '[0-9]+' | tail -1 || true)
        [[ -n "$n" ]] && break; sleep 1
    done
    if [[ -z "$n" ]]; then log "no 'serving: N slot(s)' line from rank 0 within 60 s"; return 3; fi
    if [[ "$n" -lt "$PARALLEL" ]]; then log "rank 0 started with $n slot(s), PARALLEL=$PARALLEL"; return 3; fi
    log "request slots: $n"
}

listening() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"; }

# An image's content key: its creation time and the digests of its root filesystem layers. Not .Id: two nodes'
# docker stores can give the same loaded image different ids (same layers and creation time), so comparing ids
# would refuse a good pair.
IMAGE_KEY_FMT='{{.Created}} {{range .RootFS.Layers}}{{.}} {{end}}'
image_key() { docker image inspect -f "$IMAGE_KEY_FMT" "$1" 2>/dev/null | sha256sum | cut -c1-16; }
wimage_key() { wssh "docker image inspect -f '$IMAGE_KEY_FMT' '$1' 2>/dev/null | sha256sum | cut -c1-16"; }

preflight() { # read-only checks of both nodes; non-zero on a problem. $1 = static: skip ports / GPU / RoCE marker
    local bad=0 node f h w
    nodes_ready || { log "preflight: the readiness gate failed (docker, ssh, RoCE)"; return 1; }
    h=$(docker image inspect -f '{{.Id}}' "$IMAGE" >/dev/null 2>&1 && image_key "$IMAGE" || true)
    w=$(wssh docker image inspect "$IMAGE" >/dev/null 2>&1 && wimage_key "$IMAGE" || true)
    [[ -n "$h" ]] || { log "preflight: no image $IMAGE here (scripts/serve.sh build)"; bad=1; }
    [[ -n "$w" ]] || { log "preflight: no image $IMAGE on the worker (scripts/serve.sh build ships it)"; bad=1; }
    [[ -z "$h" || -z "$w" || "$h" == "$w" ]] || { log "preflight: $IMAGE differs between the nodes (content keys $h / $w)"; bad=1; }
    for node in head worker; do
        local model engram rank prep
        if [[ "$node" == head ]]; then model="$HEAD_MODEL"; engram="$HEAD_ENGRAM"; prep="$HEAD_PREPARED"; rank=0
        else model="$WORKER_MODEL"; engram="$WORKER_ENGRAM"; prep="$WORKER_PREPARED"; rank=1; fi
        for f in config.json tokenizer.json tokenizer_config.json; do
            on_node "$node" "test -f '$model/$f'" || { log "preflight: $model/$f missing on the $node"; bad=1; }
        done
        for f in "engram-l1-r${rank}of2.bin" "engram-l14-r${rank}of2.bin"; do
            on_node "$node" "test -f '$engram/$f'" || { log "preflight: $engram/$f missing on the $node (scripts/pack_engram.py)"; bad=1; }
        done
        on_node "$node" "ls -d '$prep'/*/*/rank$rank >/dev/null 2>&1" \
            || log "preflight: warning: no prepared rank$rank folder under $prep on the $node (the first start writes ~95 GB, slow)"
    done
    if [[ "${1:-}" == static ]]; then return $bad; fi    # static: what can be checked while something else serves
    listening "$PORT" && { log "preflight: something listens on :$PORT already (the other stack?)"; bad=1; }
    listening "$MASTER_PORT" && { log "preflight: the rendezvous port $MASTER_PORT is in use"; bad=1; }
    docker run --rm -v "$CACHE_VOL:/cache" --entrypoint sh "$IMAGE" -c 'test -e /cache/roce-failed' 2>/dev/null \
        && log "preflight: warning: /cache/roce-failed exists (both ranks start on NCCL; delete it to retry RoCE)"
    if gpu_busy; then log "preflight: a CUDA process is running on a node: stop it first"; bad=1; fi
    return $bad
}

boot_id() { echo "${WATCH_BOOT_ID:-$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)}"; }
boot_start_epoch() { # when this boot began: an exit record's epoch is compared with it to tell this boot's deaths
    awk -v n="$(date +%s)" '{printf "%d", n - $1}' /proc/uptime 2>/dev/null || echo 0
}
mark_stopped() { mkdir -p "$STATE_DIR" && echo "$(boot_id) $(date +%s)" > "$STATE_DIR/stopped"; }
clear_stopped() { rm -f "$STATE_DIR/stopped" 2>/dev/null || true; }

run_canary() { # 0: passed (or off)
    [[ "$CANARY" == off ]] && return 0
    if timeout 600 python3 scripts/canary.py --base "$BASE" ${SERVED_NAME:+--model "$SERVED_NAME"}; then return 0; fi
    [[ "$CANARY" == strict ]] && return 1
    log "canary failed (CANARY=warn: still serving)"
}

start_once() { # 0 ready; 1 failed; 2 canary failed; 3 short of slots
    pair_stop 0
    drop_caches
    mem_gate || return 1
    IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo "$IMAGE")
    pair_launch 1 >/dev/null &            # rank 1 over ssh, overlapping rank 0's start as it always has
    local wpid=$!
    pair_launch 0 >/dev/null
    wait "$wpid" || { log "rank 1 did not start"; return 1; }
    log "waiting for rank 0 on :$PORT"
    local t0
    t0=$(date +%s)
    until curl -sf -m 5 "$BASE/v1/models" >/dev/null; do
        [[ "$(running 0)" == true ]] || { log "rank 0 exited"; docker logs --tail 40 "$NAME-r0" 2>&1; return 1; }
        case "$(running 1)" in true|unreachable) ;; *) log "rank 1 exited"; wssh docker logs --tail 40 "$NAME-r1" 2>&1; return 1 ;; esac
        (( $(date +%s) - t0 < READY_TIMEOUT )) || { log "not ready after ${READY_TIMEOUT}s"; return 1; }
        sleep 2
    done
    log "ready after $(( $(date +%s) - t0 )) s: $(curl -s "$BASE/v1/models")"
    check_slots || return 3
    drop_caches                                     # the boot's page cache before the canary (admission waits on MemFree)
    run_canary || return 2
    drop_caches                                     # the canary's page cache out again
}

take_lock() { # one start at a time; the watchdog stands down while it is held
    [[ "${DSV41_TF_LOCKED:-0}" == 1 ]] && return 0
    DSV41_TF_LOCKED=1
    mkdir -p "$STATE_DIR"; exec 8>"$STATE_DIR/lock"
    flock -n 8 || { log "another start (or a watchdog heal) is running"; exit 1; }
}

cmd_start() {
    take_lock
    clear_stopped
    if gpu_busy; then log "a CUDA process is running on a node; stop it first"; exit 1; fi
    case "$PREFLIGHT" in
        off) ;;
        strict) preflight || { log "preflight failed; not starting"; exit 1; } ;;
        *) preflight || log "preflight: problems above (PREFLIGHT=warn: starting anyway)" ;;
    esac
    local rc=0 try
    for try in 1 2; do                              # a short-of-slots start (page cache at load) gets one more try
        rc=0; start_once || rc=$?
        [[ $rc == 3 && $try == 1 ]] || break
        log "short of request slots: dropping the caches and starting again"
    done
    case $rc in
        0) log "serving on :$PORT ($SERVED_NAME, image $IMAGE)" ;;
        2) log "canary failed: stopping both ranks (logs in $STATE_DIR)"; mkdir -p "$STATE_DIR"
           docker logs --tail 200 "$NAME-r0" > "$STATE_DIR/canary-fail-r0.log" 2>&1 || true
           wssh docker logs --tail 200 "$NAME-r1" > "$STATE_DIR/canary-fail-r1.log" 2>&1 || true
           pair_stop 0; exit 1 ;;
        3) log "short of request slots twice: leaving it up for 'logs'; stop / restart after dropping caches"; exit 3 ;;
        *) log "start failed; 'logs' to inspect, 'stop' removes the containers"; exit 1 ;;
    esac
}

# -- watchdog ---------------------------------------------------------------------------------------------------------------
# Stands down while a start runs (lock) and while WATCH_LEASE (optional: a file you touch while you benchmark or
# maintain the pair) is younger than WATCH_LEASE_MIN minutes. classify is the tick's one merge site: it reads both
# ranks' presence, the recipe's exit witnesses (pair_exit: "<code> <epoch> <boot_id>", empty when there is none) and
# the stop marker, and answers the verdict with its class.
#   definitive — the engine's own words, so no debounce: a fail-fast exit 70 (TF_DSV41_FAILFAST: a failure on either
#     rank ends both in ~1 s) witnessed on this boot and newer than the last heal, a pair death witnessed on this
#     boot, and a /health body labelled by the engine ("status":"stalled" with a stall_reason, "ok":false with
#     "fatal"). Heals on the FIRST tick, at most once every WATCH_FF_MIN_HEAL s.
#   debounced — the actor's own client observations: a plain non-200, 000 past WATCH_GRACE, a rank gone with no
#     witness. WATCH_FAILS bad ticks in a row heal (WATCH_HEAL=1: restart in the background) at most once every
#     WATCH_MIN_HEAL s.
# The stop marker owns the human teardown, and only that: it stands the tick down when nothing died after it (newer
# than both witnesses); a marker from an earlier boot is bad (a `stop` before a reboot does not survive it;
# boot-start.sh starts first); both ranks absent with no witness and no marker means the pair never started on this
# node. No threshold in the actor is a stall threshold: the 503 labels and the 70 are the engine's, re-derived here.
WATCH_GRACE="${WATCH_GRACE:-1800}"; WATCH_FAILS="${WATCH_FAILS:-3}"; WATCH_HEAL="${WATCH_HEAL:-0}"
WATCH_MIN_HEAL="${WATCH_MIN_HEAL:-1800}"; WATCH_FF_MIN_HEAL="${WATCH_FF_MIN_HEAL:-120}"; WATCH_ALERT="${WATCH_ALERT:-}"
WATCH_LEASE="${WATCH_LEASE:-}"; WATCH_LEASE_MIN="${WATCH_LEASE_MIN:-20}"
lease_fresh() { [[ -n "$WATCH_LEASE" && -f "$WATCH_LEASE" ]] && (( $(date +%s) - $(stat -c %Y "$WATCH_LEASE") < WATCH_LEASE_MIN * 60 )); }
alert() { log "ALERT: $*"; [[ -z "$WATCH_ALERT" ]] || "$WATCH_ALERT" "$*" || true; }
STALLED_RE='"status"[[:space:]]*:[[:space:]]*"stalled"'
FATAL_RE='"ok"[[:space:]]*:[[:space:]]*false'

witness_class() { # $1 code $2 epoch $3 boot: exit 70 on this boot, newer than the last heal, is the engine reporting
    # a dead-and-consistent pair; every other code is the actor reading what a container left behind.
    [[ "$1" == 70 && "$3" == "$(boot_id)" && "${2:-0}" -gt "$last" ]] && { echo definitive; return; }
    echo debounced
}

classify() { # one merge site, reading watch_tick's locals: sets bad (the verdict) and cls (definitive | debounced |
    # standdown, with stand as its line). Empty bad with no cls = healthy.
    bad=""; cls=""; stand=""
    if [[ "$r0" == absent && "$r1" == absent ]]; then
        if [[ -n "$mb" && "$mb" != "$cb" ]]; then bad="both ranks absent after a reboot (stopped before it: $STATE_DIR/stopped)"; cls=debounced
        elif [[ -n "$mb" && "$mt" -gt "${t0:-0}" && "$mt" -gt "${t1:-0}" ]]; then cls=standdown; stand="both ranks absent (stopped on purpose); standing down"
        elif [[ "$b0" == "$cb" || "$b1" == "$cb" ]]; then
            bad="both ranks absent, exit witnesses from this boot (r0 ${c0:-none} r1 ${c1:-none})"
            [[ "$c0" == 70 || "$c1" == 70 ]] && bad="$bad (fail-fast exit 70)"
            cls=definitive
        elif [[ -z "$c0" && -z "$c1" && -z "$mb" ]]; then cls=standdown; stand="both ranks absent with no exit witness and no stop marker (never started on this node); standing down"
        else bad="both ranks absent, exit witnesses from an earlier boot (r0 ${c0:-none} r1 ${c1:-none})"; cls=debounced
        fi
        return 0
    fi
    if [[ "$r0" != true ]]; then
        bad="rank 0 $r0"; cls=$(witness_class "$c0" "$t0" "$b0")
        [[ "$cls" == definitive ]] && bad="$bad (fail-fast exit 70)"
        return 0
    fi
    if [[ "$r1" == unreachable ]]; then cls=standdown; stand="worker unreachable over ssh (not counted)"; return 0; fi
    if [[ "$r1" != true ]]; then
        bad="rank 1 $r1"; cls=$(witness_class "$c1" "$t1" "$b1")
        [[ "$cls" == definitive ]] && bad="$bad (fail-fast exit 70)"
        return 0
    fi
    age=$(( now - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' "$NAME-r0")" +%s) ))
    body=$(curl -s -m 10 -w '\n%{http_code}' "$BASE/health" || true)
    code=${body##*$'\n'}; body=${body%$'\n'*}
    if [[ "$code" == 200 ]]; then return 0; fi
    if [[ "$code" == 000 && $age -lt $WATCH_GRACE ]]; then cls=standdown; stand="loading (${age}s)"; return 0; fi
    bad="/health $code ${body:0:300}"
    cls=debounced
    [[ "$body" =~ $STALLED_RE && "$body" == *stall_reason* ]] && cls=definitive
    [[ "$body" =~ $FATAL_RE && "$body" == *fatal* ]] && cls=definitive
    return 0
}

watch_tick() {
    mkdir -p "$STATE_DIR"
    local fails_f="$STATE_DIR/fails" heal_f="$STATE_DIR/last_heal" fails r0 r1 age code body now last bad cls stand
    local e0 e1 c0 t0 b0 c1 t1 b1 mb mt cb need min
    fails=$(cat "$fails_f" 2>/dev/null || echo 0); now=$(date +%s); last=$(cat "$heal_f" 2>/dev/null || echo 0)
    if lease_fresh; then echo 0 > "$fails_f"; log "watch: $WATCH_LEASE is fresh; standing down"; return 0; fi
    if ! flock -n "$STATE_DIR/lock" true; then log "watch: a start is in progress; standing down"; return 0; fi
    r0=$(running 0); r1=$(running 1)
    c0=""; t0=0; b0=""; c1=""; t1=0; b1=""; mb=""; mt=0        # a missing witness or marker leaves these at "none"
    e0=$(pair_exit 0); e1=$(pair_exit 1)
    read -r c0 t0 b0 <<< "${e0:-}"
    read -r c1 t1 b1 <<< "${e1:-}"
    read -r mb mt <<< "$(cat "$STATE_DIR/stopped" 2>/dev/null || true)"; mt="${mt:-0}"
    cb=$(boot_id)
    classify
    if [[ "$cls" == standdown ]]; then echo 0 > "$fails_f"; log "watch: $stand"; return 0; fi
    if [[ -z "$bad" ]]; then echo 0 > "$fails_f"; return 0; fi
    need=$WATCH_FAILS; min=$WATCH_MIN_HEAL
    [[ "$cls" == definitive ]] && { need=1; min=$WATCH_FF_MIN_HEAL; }
    fails=$((fails + 1)); echo "$fails" > "$fails_f"
    log "watch: bad tick $fails of $need: $bad"
    (( fails >= need )) || return 0
    if [[ "$WATCH_HEAL" != 1 ]]; then alert "unhealthy ($bad); WATCH_HEAL=0, not restarting"; return 1; fi
    if (( now - last < min )); then alert "unhealthy ($bad); healed $((now - last))s ago, waiting"; return 1; fi
    echo "$now" > "$heal_f"; echo 0 > "$fails_f"
    alert "unhealthy ($bad); restarting both ranks"
    setsid env DSV41_TF_LOCKED=1 flock "$STATE_DIR/lock" "$0" restart >>"$STATE_DIR/heal.log" 2>&1 < /dev/null &
    return 1
}

cmd_build() { # the image here from docker/Dockerfile (vendor/TensorFold + patches/), then onto the worker
    [[ -f vendor/TensorFold/pyproject.toml ]] || { log "vendor/TensorFold is empty: git submodule update --init"; exit 1; }
    local commit; commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
    docker build -f docker/Dockerfile --build-arg TF_SRC_COMMIT="$commit" -t "$IMAGE" .
    log "shipping $IMAGE to the worker ($WORKER_SSH)"
    docker save "$IMAGE" | wssh docker load
    local h w; h=$(image_key "$IMAGE"); w=$(wimage_key "$IMAGE")
    [[ "$h" == "$w" ]] && log "$IMAGE on both nodes (content key $h)" || { log "image contents differ: head $h worker $w"; exit 1; }
    if [[ "${PREBUILD:-1}" != 0 ]]; then cmd_prebuild; else log "PREBUILD=0: the first start compiles the extensions"; fi
}

cmd_prebuild() { # both nodes: the CUDA extensions into CACHE_VOL with nothing else in memory (a build beside the
    # weights once took the worker to 1.26 GiB MemAvailable). The script goes in on stdin, so nothing is copied.
    if [[ "$(running 0)" == true || "$(running 1)" == true ]]; then log "the server is running: scripts/serve.sh stop first"; exit 1; fi
    if gpu_busy; then log "a CUDA process is running on a node; stop it first"; exit 1; fi
    local env_args="-e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton"
    env_args+=" -e CUDA_CACHE_PATH=/cache/nv/ComputeCache -e PYTHONDONTWRITEBYTECODE=1"
    local unlock="docker run --rm -v $CACHE_VOL:/cache --entrypoint find $IMAGE /cache/torch_extensions -name lock -delete"
    local run="docker run --rm -i --name $NAME-prebuild --gpus all --ipc=host -v $CACHE_VOL:/cache $env_args --entrypoint python $IMAGE -"
    local tmp rc0 rc1; tmp=$(mktemp -d)
    bash -c "$unlock" 2>/dev/null || true
    wssh "$unlock" 2>/dev/null || true
    log "prebuild: the CUDA extensions on both nodes (a few minutes on a fresh volume)"
    # shellcheck disable=SC2029
    timeout 1500 ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$run" < scripts/prebuild_ext.py > "$tmp/r1.txt" 2>&1 &
    local p1=$!
    timeout 1500 bash -c "$run" < scripts/prebuild_ext.py > "$tmp/r0.txt" 2>&1 && rc0=0 || rc0=$?
    wait "$p1" && rc1=0 || rc1=$?
    grep -E '^(FAIL|prebuild:)' "$tmp/r0.txt" | sed 's/^/[head] /' || true
    grep -E '^(FAIL|prebuild:)' "$tmp/r1.txt" | sed 's/^/[worker] /' || true
    if [[ $rc0 != 0 || $rc1 != 0 ]]; then
        log "prebuild failed (head rc=$rc0, worker rc=$rc1); full output in $tmp"; exit 1
    fi
    rm -rf "$tmp"; log "prebuild: every extension built on both nodes"
    cache_files
}

cache_files() { # the cache volume's data files on both nodes: the pfdense tuning table (TF_DSV41_PF_DENSE_TABLE) and,
    # for TF_DSV41_IMAGES=native, the image routing bias the EXL3 packs dropped (TF_DSV41_BIAS_VL; 66 KB fetched by
    # range requests from deepseek-ai/DeepSeek-V4.1-Flash, so both nodes need network access once)
    local tbl="${TF_DSV41_PF_DENSE_TABLE:-}" bvl="${TF_DSV41_BIAS_VL:-}" put
    if [[ -n "$tbl" && "$tbl" == /cache/* && -f config/pfdense-table.json ]]; then
        put="docker run --rm -i -v $CACHE_VOL:/cache --entrypoint sh $IMAGE -c 'mkdir -p \$(dirname $tbl) && cat > $tbl'"
        bash -c "$put" < config/pfdense-table.json && wssh "$put" < config/pfdense-table.json \
            && log "cache: config/pfdense-table.json -> $tbl on both nodes" || { log "cache: copying the pfdense table failed"; exit 1; }
    fi
    if [[ "${TF_DSV41_IMAGES:-}" == native && -n "$bvl" && "$bvl" == /cache/* ]]; then
        local fetch="docker run --rm -v $CACHE_VOL:/cache -e HF_HUB_OFFLINE=0 --entrypoint sh $IMAGE -c 'test -f $bvl/bias_vl.safetensors || python -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch $bvl'"
        bash -c "$fetch" && wssh "$fetch" && log "cache: $bvl/bias_vl.safetensors on both nodes" \
            || { log "cache: fetching the image routing bias failed (TF_DSV41_IMAGES=native needs it)"; exit 1; }
    fi
}

cmd_run() { # MODULE [ARGS...]: an engine module on both ranks (rank 1 on the worker in the background, rank 0 here)
    local mod="${1:?usage: serve.sh run MODULE [ARGS...]}"; shift
    local out="${OUT:-$PWD/results/run-$(date +%Y%m%d-%H%M%S)}" wout="${WORKER_OUT:-/tmp/dsv41-run}" rc0 rc1
    if [[ "$(running 0)" == true ]]; then log "the server is running: scripts/serve.sh stop first"; exit 1; fi
    if gpu_busy; then log "a CUDA process is running on a node; stop it first"; exit 1; fi
    mkdir -p "$out" "$HEAD_STATE/sessions"; drop_caches; mem_gate || exit 1
    # shellcheck disable=SC2046
    wssh "mkdir -p '$wout' '$WORKER_STATE/sessions' && docker run --rm --name $NAME-run-r1 $(container_args 1) -v $wout:/out --entrypoint python \
          $IMAGE -m $mod --model /model --rank 1 --master $HEAD_IP --port $MASTER_PORT $*" > "$out/r1.log" 2>&1 &
    local p1=$!
    sleep 2
    # shellcheck disable=SC2046
    docker run --rm --name "$NAME-run-r0" $(container_args 0) -v "$out:/out" --entrypoint python "$IMAGE" -m "$mod" \
        --model /model --rank 0 --master "$HEAD_IP" --port "$MASTER_PORT" "$@" 2>&1 | tee "$out/r0.log"
    rc0=${PIPESTATUS[0]}
    wait "$p1"; rc1=$?
    log "rank 0 rc=$rc0, rank 1 rc=$rc1; logs and reports in $out"
    return "$rc0"
}

if [[ "${SERVE_LIB:-0}" == 1 ]]; then return 0; fi     # sourced as a library (scripts/boot-start.sh): functions only
case "${1:-}" in
build) cmd_build ;;
prebuild) cmd_prebuild ;;
cache) cache_files ;;
run) [[ "$RECIPE" == python ]] || { log "run: the engine-module runner belongs to the python recipe"; exit 2; }
     shift; cmd_run "$@" ;;
start) cmd_start ;;
restart) take_lock; pair_stop 0; log "stopped"; cmd_start ;;
stop) pair_stop 0; mark_stopped; log "stopped (stop marker: $STATE_DIR/stopped)" ;;
status)
    docker ps -a --filter "name=^$NAME-r" --format '{{.Names}} {{.Status}}'
    wssh "docker ps -a --filter name=^$NAME-r --format '{{.Names}} {{.Status}}'" || true
    curl -s -m 10 "$BASE/v1/models" || true; echo
    curl -s -m 10 "$BASE/health" || true; echo ;;
logs) if [[ "${2:-0}" == 1 ]]; then wssh docker logs --tail "${TAIL:-80}" "$NAME-r1"; else docker logs --tail "${TAIL:-80}" "$NAME-r0"; fi ;;
canary) CANARY=strict run_canary ;;
preflight) preflight "${2:-}" && log "preflight ok${2:+ ($2)}" ;;
watch)
    if [[ "${2:-}" == --once ]]; then watch_tick; exit $?; fi
    while :; do watch_tick || true; sleep "${WATCH_INTERVAL:-60}"; done ;;
args) IMAGE_ID="${IMAGE_ID:-$IMAGE}" pair_args "${2:-0}"; echo ;;
*) sed -n '2,18p' "$0"; exit 2 ;;
esac
