#!/usr/bin/env bash
# The RECIPE=zig pair: the two containers scripts/run-zig.sh launches, tensorfold-zig-r0 (head, `tensorfold-dsv41
# serve`, the API) and tensorfold-zig-r1 (worker, `tf-dsv41-m1 follow`, no HTTP surface). scripts/serve.sh sources
# this after the config and calls the five pair calls: pair_required_keys, pair_exit <rank>, pair_launch <rank>,
# pair_stop <grace_s>, pair_boot_decision, plus worker_config_key, the read of a key in the worker's own config that
# no head-side key answers. The ports (PORT, TF_TP_PORT), the head address (TF_TP_MASTER) and the slot-line regex are
# config keys / derived values in serve.sh, not adapter calls.
# pair_required_keys here and scripts/run-zig.sh's own key check cover the same pair-facing keys on purpose (each
# entry point validates its own contract; run-zig.sh needs no ssh keys, it is the local rank): a key change edits
# both, and this file is the one place the recipe's keys are named.
# The exit witness: this pair runs `docker run --rm --init`, so a rank leaves no container for docker inspect to
# answer. scripts/run-zig.sh writes "$STATE_DIR/exit-r<rank>" as "<code> <epoch> <boot_id>" on every invocation, and
# rank 1's record lives in the worker's own STATE_DIR, stamped with the worker's boot id: the pair's two nodes
# reboot independently. The Python recipe's containers persist and read inspect instead (scripts/serve/python.sh).
WATCH_EXIT_LOOKBACK_S="${WATCH_EXIT_LOOKBACK_S:-180}"
WORKER_STATE_DIR=""
WORKER_BOOT=""

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

worker_boot_id() { # the worker's own boot id, read over ssh. The pair's two nodes reboot independently, so a
    # witness from rank 1's node is placed in time against that node's boot, never the head's. Read once per process
    # like the worker's state dir. Empty when the worker does not answer, which is the tick's stand-down row.
    [[ -n "$WORKER_BOOT" ]] && { echo "$WORKER_BOOT"; return 0; }
    WORKER_BOOT=$(wssh "cat /proc/sys/kernel/random/boot_id 2>/dev/null" || true)
    echo "$WORKER_BOOT"
}

kill_supervisor() { # $1 = head|worker, $2 = that node's supervisor pid file: end the supervised relaunch loop
    # before its containers are removed, by the pid the loop recorded. By pid, never by a command-line pattern of
    # the whole command: a pattern matches the ssh shell carrying it, which reads back as a worker that is
    # unreachable, and a stop that kills the channel stops the pair on one node only. The pid is checked against the
    # loop's own flag before the kill, because the file outlives the loop it names and pids get reused. The kill
    # takes the process group: a loop killed while its launch is in flight leaves that launch running, and the rank
    # comes back behind the stop.
    local node=$1 file=$2
    [[ -n "$file" ]] || return 0
    on_node "$node" "pid=\$(cat '$file' 2>/dev/null); rm -f '$file'
pid=\${pid//[^0-9]/}
[[ -n \"\$pid\" ]] && ps -p \"\$pid\" -o args= 2>/dev/null | grep -q -- '--detached' && kill -- -\"\$pid\" 2>/dev/null
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

exit_lookback_since() { # the die-event window's start, printed as an epoch, the one form both docker dialects answer.
    # The pair's nodes run Docker 29.6.2, where the CLI parses --since and --until itself as relative durations
    # (measured on the head 2026-10-20: --since 180s --until 0s answered a 3-second-old die event, --since 1s did not).
    # The older daemon parses them as seconds with strconv.ParseInt (moby api/types/time/timestamp.go, v20.10 through
    # v28.5.2, source-verified), where an interval is a parse error and the query answers nothing. Epochs are honored
    # in both, so the window is printed as epochs and stays portable across the two. The window opens at the pair's
    # last human stop *on this boot*, because an event from before that stop is a death that stop already answered, not
    # this one. The marker's own boot id is the check: a stop from before the reboot says nothing about this boot's
    # pair, and anchoring on it would stretch the window across the reboot and read a pre-reboot death as this boot.
    # The daemon's lower bound is inclusive, so the window cannot be the place that excludes the stop's own kills:
    # mark_stopped stamps the marker one second after them, which leaves them out of the window and keeps the death a
    # rank dies a beat behind its peer inside it, where classify answers it as the stop's.
    local mb mt now s
    read -r mb mt <<< "$(cat "$STATE_DIR/stopped" 2>/dev/null || true)"
    now=$(date +%s)
    [[ "$mb" == "$(boot_id)" && "$mt" =~ ^[0-9]+$ ]] || { echo "$(( now - WATCH_EXIT_LOOKBACK_S ))"; return 0; }
    s=$(( now - mt )); (( s < 0 )) && s=0
    echo "$(( now - s ))"
}

pair_exit() { # $1 = rank: its exit witness "<code> <epoch> <boot_id>", empty when there is none. The record format is
    # parsed here and only here, and the boot id is answered in the head's terms: the head's own when the death is
    # from this boot on the node the rank ran on, empty when it is from an earlier boot there. Rank 1 ran on the
    # worker, whose boot id is its own, and this adapter is the one place that knows which node a rank's witness
    # lives on. The fallback to the die event is for a rank with no cooperating launcher (a hand-run `docker run`, an
    # earlyoom SIGKILL), which leaves the event and nothing else. A die event carries no boot id, so the fallback places
    # it by its window alone: bounded below by this boot's stop marker, and by WATCH_EXIT_LOOKBACK_S when there is
    # none, so a "this boot" stamp on a die event is coarse by at most the lookback. A coarse definitive read heals a
    # pair that is down, which is what the read exists for. The read speaks its own rows on stderr: stdout here is the
    # witness, and every caller captures it.
    local rank=$1 rec code epoch rec_boot out t node_boot fmt ev since wdir
    node_boot=$(boot_id)
    [[ "$rank" == 1 ]] && node_boot=$(worker_boot_id)
    rec=""
    if [[ "$rank" == 0 ]]; then
        rec=$(cat "$STATE_DIR/exit-r0" 2>/dev/null || true)
    else
        wdir=$(worker_state_dir)
        if [[ -n "$wdir" ]]; then rec=$(wssh "cat '$wdir'/exit-r1 2>/dev/null" || true)
        else log_err "rank 1: the worker's own config names no STATE_DIR, so rank 1's exit record cannot be read"; fi
    fi
    if [[ -n "$rec" ]]; then
        read -r code epoch rec_boot <<< "$rec"
        [[ -n "$node_boot" && "$rec_boot" == "$node_boot" ]] && { echo "$code ${epoch:-0} $(boot_id)"; return 0; }
        echo "$code ${epoch:-0} "
        return 0
    fi
    fmt='{{.Time}} {{.Actor.Attributes.exitCode}}'
    since=$(exit_lookback_since)
    ev="docker events --since $since --until $(date +%s) --filter event=die --filter container=$NAME-r$rank --format '$fmt'"
    if [[ "$rank" == 0 ]]; then out=$(bash -c "$ev 2>/dev/null | tail -1" || true)
    else out=$(wssh "$ev 2>/dev/null | tail -1" || true); fi
    [[ -n "$out" ]] || { echo ""; return 0; }
    read -r t code <<< "$out"
    echo "$code $(epoch_of "$t") $(boot_id)"
}

pair_launch() { # $1 = rank: launch it detached through scripts/run-zig.sh --detached, the way bin/up.sh does it (the
    # ssh channel returns; the launcher lives on its node, into the durable log of its own state dir). Answer 0 once
    # docker ps shows the container, 1 with the log tail when it never appears. rank 1 reads the config of its own
    # node, named here by path so a profile config (WORKER_CONFIG) reaches the worker's launcher. That knob splits the
    # witness chain today: worker_config_key and everything built on it parse the worker's config/prod-zig.env, so a
    # profile here puts rank 1's record, log and pid file where the watchdog does not read (issue #5).
    local rank=$1 rank_log _i st wdir wconfig
    mkdir -p "$STATE_DIR"
    if [[ "$rank" == 1 ]]; then
        wdir=$(worker_state_dir); wconfig="$WORKER_REPO/${WORKER_CONFIG:-config/prod-zig.env}"
        [[ -n "$wdir" ]] || { log "rank 1: the worker's own config names no STATE_DIR, so its launcher has no log directory to launch into and no place to write its exit record: fix the worker's config"; return 1; }
        rank_log="$wdir/run-zig-r1.log"
        wssh "mkdir -p '$wdir' && cd '$WORKER_REPO' && CONFIG='$wconfig' setsid nohup scripts/run-zig.sh 1 --detached >> '$rank_log' 2>&1 < /dev/null & sleep 1; exit 0" >/dev/null 2>&1 || true
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
    # stays non-fatal: rank 1 carries no client traffic. Reachability is a probe of its own (`wssh true`), not an
    # inference from the state dir being unreadable: an empty answer is a config problem, and a pair stopped by
    # inference leaves rank 1's container up. A removal the worker's own daemon refuses is alerted, not swallowed: the
    # marker written after a stop makes the watchdog stand down beside a rank 1 that is still running, so the row is
    # the only thing that says so. The alert requires the container to be seen after the failed removal: `docker rm
    # -f` answers a non-zero rc for a name the daemon cannot find, and a rank 1 whose peer stopped leaves on its own
    # (its launcher's fate plus --rm), so the return code alone says nothing about a container being left running.
    local grace=${1:-0} it gone=0 wrc=0 wdir
    # The supervisors first: a relaunched rank would come back a second after its container is removed.
    kill_supervisor head "$STATE_DIR/supervisor-r0.pid"
    wssh true >/dev/null 2>&1 && wrc=0 || wrc=$?
    if (( wrc == 255 )); then
        log "rank 1: the worker is unreachable over ssh; its container and launcher are untouched from here"
    else
        wdir=$(worker_state_dir)
        kill_supervisor worker "$wdir/supervisor-r1.pid"
    fi
    if (( grace > 0 )); then
        docker stop -t "$grace" "$NAME-r0" >/dev/null 2>&1 || true
        if (( wrc != 255 )); then
            for (( it = 0; it < (grace + 30) / 5; it++ )); do
                if wssh docker inspect "$NAME-r1" >/dev/null 2>&1; then sleep 5; else gone=1; break; fi
            done
            [[ "$gone" == 1 ]] || log "rank 1 still present after the ${grace}s drain: stopping it directly (it carries no client traffic)"
            wssh docker stop -t "$grace" "$NAME-r1" >/dev/null 2>&1 || true
        fi
    fi
    docker rm -f "$NAME-r0" >/dev/null 2>&1 || true
    if (( wrc != 255 )) && ! wssh docker rm -f "$NAME-r1" >/dev/null 2>&1 && wssh docker inspect "$NAME-r1" >/dev/null 2>&1; then
        alert "rank 1: docker rm -f failed on the worker: $NAME-r1 is left running, and the stop marker written after this says the pair is stopped"
    fi
}

pair_boot_decision() { # 0 = start the pair, 1 = leave it. The rows, in the order the pair's own witnesses decide:
    # rank 0 running; an exit record from an earlier boot; this boot's stop marker; a marker from an earlier boot;
    # nothing at all, which is the pair never having started on this node.
    local state mb c0 b0 e0
    state=$(docker inspect -f '{{.State.Status}}' "$NAME-r0" 2>/dev/null || echo absent)
    case "$state" in
        running) log "rank 0 already running: nothing to do"; return 1 ;;
        absent) ;;
        *) log "rank 0 container $state (a container the boot left behind): starting"; return 0 ;;
    esac
    e0=$(pair_exit 0); read -r c0 _ b0 <<< "${e0:-}"
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
