#!/usr/bin/env bash
# The RECIPE=python pair: `python -m tensorfold serve` on $NAME-r0 (head) / $NAME-r1 (worker). scripts/serve.sh
# sources this after the config and calls five things: pair_required_keys, pair_exit <rank>, pair_launch <rank>,
# pair_stop <grace_s>, pair_boot_decision. The ports (PORT, TF_TP_PORT), the head address and the slot-line regex are
# config keys / derived values in serve.sh, not adapter calls.
# pair_required_keys here and scripts/run-zig.sh's own key check list the same keys on purpose (each entry point
# validates its own contract): a key change edits both, and this file is the one place the recipe's keys are named.
# The exit witness: this pair's containers persist, so docker inspect holds it; the Zig pair's `docker run --rm`
# leaves nothing to inspect and reads the launcher's $STATE_DIR/exit-r<rank> record (scripts/serve/zig.sh).

container_args() { # $1 = rank: `docker run` options of that rank's container (no image, no command)
    local rank=$1 model engram prep state a
    if [[ "$rank" == 0 ]]; then model=$HEAD_MODEL engram=$HEAD_ENGRAM prep=$HEAD_PREPARED state=$HEAD_STATE
    else model=$WORKER_MODEL engram=$WORKER_ENGRAM prep=$WORKER_PREPARED state=$WORKER_STATE; fi
    a="--gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1"
    a+=" --cap-add IPC_LOCK --log-opt max-size=$LOG_MAX_SIZE --log-opt max-file=$LOG_MAX_FILE"
    a+=" -v $model:/model:ro -v $engram:/engram:ro -v $prep:/prepared"
    a+=" -v $state:/state -v $state/sessions:/sessions -v $CACHE_VOL:/cache -e PYTHONDONTWRITEBYTECODE=1"
    a+=" -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton"
    a+=" -e CUDA_CACHE_PATH=/cache/nv/ComputeCache -e TF_DSV41_IMAGE_ID=${IMAGE_ID:-$IMAGE}"
    a+=" -e TF_DSV41_LAUNCH_T0=$(date +%s.%N)"
    a+=" $(env | grep -E '^(TF_DSV41_[A-Z0-9_]+|GLM53_TF_[A-Z0-9_]+|(MI)?MALLOC_[A-Z_]+)=' | grep -vE '^TF_DSV41_(LAUNCH_T0|IMAGE_ID)=' | sort | sed 's/^/-e /' | tr '\n' ' ')"
    a+=" -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e NCCL_IB_HCA=$NCCL_IB_HCA -e GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    # GB10: NCCL's cuMem allocations exhausted the driver's pool at boot (NV_ERR_NO_MEMORY), failing later starts
    a+=" -e NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0} -e NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}"
    echo "$a"
}

pair_args() { # $1 = rank: `docker run` arguments of that rank's server container (`python -m tensorfold serve`)
    local rank=$1 a
    a="--name $NAME-r$rank -d $(container_args "$rank")"
    a+=" --entrypoint python $IMAGE -m tensorfold serve /model --tp 2 --rank $rank --master $HEAD_IP"
    a+=" --master-port $MASTER_PORT --context ${CONTEXT:-300000} --parallel $PARALLEL --kv-dtype ${KV_DTYPE:-fp8}"
    a+=" --no-update-check"
    if [[ "$rank" == 0 ]]; then
        a+=" --host $HOST --port $PORT${SERVED_NAME:+ --name $SERVED_NAME}${MAX_TOKENS:+ --max-tokens $MAX_TOKENS}"
        local al; for al in ${ALIASES:-}; do a+=" --alias $al"; done
    fi
    echo "$a"
}

pair_launch() { # $1 = rank: run that rank's container (rank 1 on the worker: the pair is built rank 1 first)
    if [[ "$1" == 1 ]]; then
        # shellcheck disable=SC2046
        wssh "mkdir -p '$WORKER_STATE/sessions' && docker run $(pair_args 1)"
    else
        mkdir -p "$HEAD_STATE/sessions"
        # shellcheck disable=SC2046
        docker run $(pair_args 0)
    fi
}

pair_stop() { # both ranks, now: this recipe stops with a force remove (it has never had a drain step; the grace is
    # the Zig pair's drain, and the Python pair's drain semantics live in its own stop path, not here)
    docker rm -f "$NAME-r0" >/dev/null 2>&1 &
    local p0=$!
    wssh docker rm -f "$NAME-r1" >/dev/null 2>&1 &
    local p1=$!
    wait "$p0" || true; wait "$p1" || true
}

pair_exit() { # $1 = rank: its exit witness "<code> <epoch> <boot_id>", empty when its container is gone.
    # The record format is parsed here and only here: the code is docker inspect's ExitCode, the epoch its FinishedAt,
    # and the boot id only when that finish lies on this boot (an exit from an earlier boot names no boot).
    local out rc code fin fin_epoch
    if [[ "$1" == 0 ]]; then out=$(docker inspect -f '{{.State.ExitCode}} {{.State.FinishedAt}}' "$NAME-r0" 2>/dev/null) && rc=0 || rc=$?
    else out=$(wssh docker inspect -f "'{{.State.ExitCode}} {{.State.FinishedAt}}'" "$NAME-r1" 2>/dev/null) && rc=0 || rc=$?; fi
    [[ $rc == 0 ]] || { echo ""; return 0; }
    read -r code fin <<< "$out"
    fin_epoch=$(date -d "$fin" +%s 2>/dev/null || echo 0)
    if (( fin_epoch >= $(boot_start_epoch) )); then echo "$code $fin_epoch $(boot_id)"; else echo "$code $fin_epoch "; fi
}

pair_required_keys() { # the keys this recipe's config must set (serve.sh has checked RECIPE and sourced this)
    local _k
    for _k in WORKER_SSH HEAD_IP NCCL_SOCKET_IFNAME NCCL_IB_HCA HEAD_MODEL WORKER_MODEL HEAD_ENGRAM WORKER_ENGRAM \
              HEAD_PREPARED WORKER_PREPARED HEAD_STATE WORKER_STATE IMAGE; do
        if [[ -z "${!_k:-}" || "${!_k}" == *"<"*">"* ]]; then echo "[dsv41-tf] $CONFIG: set $_k" >&2; exit 2; fi
    done
}

pair_boot_decision() { # 0 = start the pair, 1 = leave it: what rank 0's container and the stop marker say at boot
    local state mb
    state=$(docker inspect -f '{{.State.Status}}' "$NAME-r0" 2>/dev/null || echo absent)
    case "$state" in
        running) log "rank 0 already running: nothing to do"; return 1 ;;
        absent)
            read -r mb _ <<< "$(cat "$STATE_DIR/stopped" 2>/dev/null || true)"
            if [[ -z "$mb" || "$mb" == "$(boot_id)" ]]; then log "rank 0 absent and not stopped before this boot: nothing to do"; return 1; fi
            log "rank 0 absent, stopped before the reboot (marker $STATE_DIR/stopped): starting" ;;
        *) log "rank 0 container $state (the previous boot was serving): starting" ;;
    esac
}
