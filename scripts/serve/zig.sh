#!/usr/bin/env bash
# The RECIPE=zig pair: the two containers scripts/run-zig.sh launches, tensorfold-zig-r0 (head, `tensorfold-dsv41
# serve`, the API) and tensorfold-zig-r1 (worker, `tf-dsv41-m1 follow`, no HTTP surface). scripts/serve.sh sources
# this after the config and calls five things: pair_required_keys, pair_exit <rank>, pair_launch <rank>,
# pair_stop <grace_s>, pair_boot_decision. The ports (PORT, TF_TP_PORT), the head address (TF_TP_MASTER) and the
# slot-line regex are config keys / derived values in serve.sh, not adapter calls.
# pair_required_keys here and scripts/run-zig.sh's own key check list the same keys on purpose (each entry point
# validates its own contract): a key change edits both, and this file is the one place the recipe's keys are named.
# The exit witness: this pair runs `docker run --rm --init`, so a rank leaves no container for docker inspect to
# answer. scripts/run-zig.sh writes "$STATE_DIR/exit-r<rank>" as "<code> <epoch> <boot_id>" on every invocation, and
# rank 1's record lives in the worker's own STATE_DIR. The Python recipe's containers persist and read inspect
# instead (scripts/serve/python.sh).
WATCH_EXIT_LOOKBACK_S="${WATCH_EXIT_LOOKBACK_S:-180}"
WORKER_STATE_DIR=""

worker_config_key() { # $1 = key: its value in the worker's own copy of config/prod-zig.env, the file its
    # run-zig.sh reads when no CONFIG is passed over ssh. The head cannot see that file, and the pair's paths differ
    # per node (engram-r0 here, engram-r1 there), so the worker's paths can only be read from the worker.
    [[ -n "${WORKER_REPO:-}" ]] || return 0
    wssh "cd '$WORKER_REPO' && sed -n 's/^$1=//p' config/prod-zig.env | tail -1" 2>/dev/null | tr -d '"' || true
}

worker_state_dir() { # the worker's own STATE_DIR, from the config its run-zig.sh reads: the directory rank 1's exit
    # record and durable log live in. Read once per process, because a tick reads rank 1's witness and may launch it.
    [[ -n "$WORKER_STATE_DIR" ]] && { echo "$WORKER_STATE_DIR"; return 0; }
    WORKER_STATE_DIR=$(worker_config_key STATE_DIR)
    echo "$WORKER_STATE_DIR"
}

kill_supervisor() { # $1 = head|worker, $2 = that node's supervisor pid file: end the supervised relaunch loop
    # before its containers are removed, by the pid the loop recorded. By pid, never by a command-line pattern: a
    # pattern matches the ssh shell carrying it, which reads back as a worker that is unreachable, and a stop that
    # kills the channel stops the pair on one node only. The pid is checked against the launcher before the kill,
    # because the file outlives the loop it names.
    local node=$1 file=$2
    [[ -n "$file" ]] || return 0
    on_node "$node" "pid=\$(cat '$file' 2>/dev/null); rm -f '$file'
pid=\${pid//[^0-9]/}
[[ -n \"\$pid\" ]] && ps -p \"\$pid\" -o args= 2>/dev/null | grep -q 'run-zig.sh' && kill \"\$pid\" 2>/dev/null
true"
}

pair_required_keys() { # the keys this recipe's config must set (serve.sh has checked RECIPE and sourced this)
    local _k
    for _k in WORKER_SSH WORKER_REPO IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_DIR ENGRAM_DIR \
              ASSETS_DIR PREPARED_DIR STATE_DIR CACHE_DIR; do
        if [[ -z "${!_k:-}" || "${!_k}" == *"<"*">"* ]]; then echo "[dsv41-tf] $CONFIG: set $_k" >&2; exit 2; fi
    done
    # The launcher names its containers by its own prefix, and that name is the witness. A different NAME aims every
    # read at containers nothing runs.
    [[ "$NAME" == tensorfold-zig ]] || { echo "[dsv41-tf] $CONFIG: NAME=$NAME, scripts/run-zig.sh launches tensorfold-zig-r0 and tensorfold-zig-r1: set NAME=tensorfold-zig" >&2; exit 2; }
}

pair_exit() { # $1 = rank: its exit witness "<code> <epoch> <boot_id>", empty when there is none. The record format is
    # parsed here and only here. A record from an earlier boot answers no boot id, which is what makes the tick read it
    # as an earlier boot. One function, one fallback chain: the launcher's record, then the die event (a hand-run
    # `docker run` or an earlyoom SIGKILL has no cooperating launcher, so it leaves the event and nothing else).
    local rank=$1 rec code epoch rec_boot out t
    if [[ "$rank" == 0 ]]; then
        rec=$(cat "$STATE_DIR/exit-r0" 2>/dev/null || true)
    else
        rec=$(wssh "cat '$(worker_state_dir)'/exit-r1 2>/dev/null" || true)
    fi
    if [[ -n "$rec" ]]; then
        read -r code epoch rec_boot <<< "$rec"
        [[ "$rec_boot" == "$(boot_id)" ]] && { echo "$code ${epoch:-0} $rec_boot"; return 0; }
        echo "$code ${epoch:-0} "
        return 0
    fi
    local fmt='{{.Time}} {{.Attributes.exitCode}}' ev
    if [[ "$rank" == 0 ]]; then
        ev="docker events --since ${WATCH_EXIT_LOOKBACK_S}s --until 0s --filter event=die --filter container=$NAME-r0 --format '$fmt'"
        out=$(bash -c "$ev 2>/dev/null | tail -1" || true)
    else
        ev="docker events --since ${WATCH_EXIT_LOOKBACK_S}s --until 0s --filter event=die --filter container=$NAME-r1 --format '$fmt'"
        out=$(wssh "$ev 2>/dev/null | tail -1" || true)
    fi
    [[ -n "$out" ]] || { echo ""; return 0; }
    read -r t code <<< "$out"
    echo "$code $(epoch_of "$t") $(boot_id)"
}

pair_launch() { # $1 = rank: launch it detached through scripts/run-zig.sh --detached, the way bin/up.sh does it (the
    # ssh channel returns; the launcher lives on its node, into the durable log of its own state dir). Answer 0 once
    # docker ps shows the container, 1 with the log tail when it never appears.
    local rank=$1 rank_log _i st
    mkdir -p "$STATE_DIR"
    if [[ "$rank" == 1 ]]; then
        rank_log="$(worker_state_dir)/run-zig-r1.log"
        wssh "cd '$WORKER_REPO' && setsid nohup scripts/run-zig.sh 1 --detached >> '$rank_log' 2>&1 < /dev/null & sleep 1; exit 0" >/dev/null 2>&1 || true
    else
        setsid nohup scripts/run-zig.sh 0 --detached >> "$STATE_DIR/run-zig-r0.log" 2>&1 < /dev/null &
    fi
    for _i in $(seq 1 15); do
        if [[ "$rank" == 1 ]]; then st=$(wssh "docker ps -a --filter name=$NAME-r1 -q" 2>/dev/null || true)
        else st=$(docker ps -a --filter name="$NAME-r0" -q 2>/dev/null || true); fi
        [[ -n "$st" ]] && return 0
        sleep 2
    done
    log "rank $rank: no container within 30 s; the last 40 lines of its launcher log:"
    if [[ "$rank" == 1 ]]; then wssh "tail -40 '$rank_log'" 2>/dev/null || true
    else tail -40 "$STATE_DIR/run-zig-r0.log" 2>/dev/null || true; fi
    return 1
}

pair_stop() { # $1 = grace seconds: the bounded drain (rank 0 answers 503 and gives requests in progress
    # TF_DSV41_DRAIN_S), then the force, on both nodes, idempotent. A worker unreachable over ssh is reported and
    # stays non-fatal: rank 1 carries no client traffic.
    local grace=${1:-0} it gone=0 wrc=0 wdir
    # The supervisors first: a relaunched rank would come back a second after its container is removed.
    kill_supervisor head "$STATE_DIR/supervisor-r0.pid"
    wdir=$(worker_state_dir)
    if [[ -n "$wdir" ]]; then kill_supervisor worker "$wdir/supervisor-r1.pid"; else wrc=255; fi
    (( wrc == 255 )) && log "rank 1: the worker is unreachable over ssh; its container and launcher are untouched from here"
    if (( grace > 0 )); then
        docker stop -t "$grace" "$NAME-r0" >/dev/null 2>&1 || true
        for (( it = 0; it < (grace + 30) / 5; it++ )); do
            if wssh docker inspect "$NAME-r1" >/dev/null 2>&1; then sleep 5; else gone=1; break; fi
        done
        [[ "$gone" == 1 ]] || log "rank 1 still present after the ${grace}s drain: stopping it directly (it carries no client traffic)"
        wssh docker stop -t "$grace" "$NAME-r1" >/dev/null 2>&1 || true
    fi
    docker rm -f "$NAME-r0" >/dev/null 2>&1 || true
    (( wrc == 255 )) || wssh docker rm -f "$NAME-r1" >/dev/null 2>&1 || true
}

pair_boot_decision() { # 0 = start the pair, 1 = leave it. The rows, in the order the pair's own witnesses decide:
    # rank 0 running; an exit record from an earlier boot; this boot's stop marker; a marker from an earlier boot;
    # nothing at all, which is the pair never having started on this node.
    local state mb c0 t0 b0 e0
    state=$(docker inspect -f '{{.State.Status}}' "$NAME-r0" 2>/dev/null || echo absent)
    case "$state" in
        running) log "rank 0 already running: nothing to do"; return 1 ;;
        absent) ;;
        *) log "rank 0 container $state (a container the boot left behind): starting"; return 0 ;;
    esac
    e0=$(pair_exit 0); read -r c0 t0 b0 <<< "${e0:-}"
    read -r mb _ <<< "$(cat "$STATE_DIR/stopped" 2>/dev/null || true)"
    if [[ -n "$c0" && "$b0" != "$(boot_id)" ]]; then
        log "rank 0 absent with an exit witness from an earlier boot (exit $c0): starting"; return 0
    fi
    if [[ -n "$mb" && "$mb" == "$(boot_id)" ]]; then
        log "rank 0 absent and stopped on purpose during this boot (marker $STATE_DIR/stopped): nothing to do"; return 1
    fi
    [[ -z "$mb" ]] || { log "rank 0 absent, stopped before the reboot (marker $STATE_DIR/stopped): starting"; return 0; }
    if [[ -n "$c0" ]]; then
        log "rank 0 absent with an exit witness from this boot (exit $c0): starting"; return 0
    fi
    log "rank 0 absent with no exit record, no die event and no stop marker (never started on this node): nothing to do"
    return 1
}
