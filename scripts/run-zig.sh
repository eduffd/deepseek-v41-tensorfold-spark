#!/usr/bin/env bash
# Run one local rank. Invoke separately on each node with the same serving env.
#
#   scripts/run-zig.sh 0|1                  the rank in this terminal; its streams go to this terminal
#   scripts/run-zig.sh 0|1 --detached       self-supervise in the background, into $STATE_DIR/run-zig-r<rank>.log,
#                                           recording its pid in $STATE_DIR/supervisor-r<rank>.pid (serve.sh stop ends it)
#   scripts/run-zig.sh 0|1 --dry-run        print the docker run arguments, launch nothing
#   scripts/run-zig.sh 0|1 --check-aot      check this env's Triton variants against $ASSETS_DIR/aot (docs/ZIG-SERVE.md)
#
# Every invocation ends by writing "$STATE_DIR/exit-r<rank>" as "<code> <epoch> <boot_id>": the containers run with
# --rm, so that record is the watchdog's exit witness (scripts/serve/zig.sh pair_exit reads it).
# One stack per pair: this refuses while its own rank's container exists.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rank=${1:?usage: run-zig.sh 0|1 [--dry-run|--check-aot|--detached]}
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo 'rank must be 0 or 1' >&2; exit 2; }
config=${CONFIG:-config/prod-zig.env}
[[ -f "$config" ]] || { echo "missing $config; copy config/prod-zig.env.example first" >&2; exit 2; }
set -a
source "$config"
set +a
for key in IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_DIR ENGRAM_DIR ASSETS_DIR PREPARED_DIR STATE_DIR CACHE_DIR; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set $key in $config" >&2; exit 2; }
done
# Pass values by environment name, preserving spaces in values.
envs=()
while IFS='=' read -r key value; do
    case "$key" in TF_DSV41_*|GLM53_TF_*|NCCL_*|TF_TP_ROCE_HCA) envs+=(-e "$key");; esac
done < <(env)
# --check-aot: every Triton variant this env launches, checked against this rank's aot/ (host only, no GPU work).
# Run it before the first start and after any serving-setting change; it needs the fill's sigs.json in ASSETS_DIR.
if [[ "${2:-}" == --check-aot ]]; then
    [[ -f "$ASSETS_DIR/sigs.json" && -f "$ASSETS_DIR/aot/aot.json" ]] || {
        echo "--check-aot needs $ASSETS_DIR/sigs.json (triton_fill.py sigs) and $ASSETS_DIR/aot/aot.json" >&2; exit 2; }
    mkdir -p "$STATE_DIR"
    docker run --rm --init -v "$ASSETS_DIR:/assets:ro" -v "$STATE_DIR:/state" "${envs[@]}" -e TF_TP_WORLD=2 "$IMAGE" \
        tf-dsv41-m1 aot-needs /assets/sigs.json "/state/aot-needs-r$rank.json" /assets/aot || {
        echo "rank $rank: aot/ does not cover this env (missing variants above; every needed one in $STATE_DIR/aot-needs-r$rank.json)" >&2
        echo "compile them: triton_fill.py compile --needs <that file> --have <aot> --out <aot> (docs/ZIG-SERVE.md)" >&2
        exit 1; }
    exit
fi
args=(docker run --rm --init --name "tensorfold-zig-r$rank" --gpus all --ipc=host --network host
      --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$ASSETS_DIR:/assets:ro"
      -v "$PREPARED_DIR:/prepared" -v "$STATE_DIR:/state" -v "$STATE_DIR/sessions:/sessions"
      -v "$CACHE_DIR:/cache" "${envs[@]}")
args+=(-e TF_TP_WORLD=2 -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0
       -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=${TF_TP_PORT:-29571}"
       -e "TF_COMM_BACKEND=${TF_COMM_BACKEND:-roce}"
       -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin "$IMAGE")
if [[ "$rank" == 0 ]]; then
    args+=(tensorfold-dsv41 serve /model --host "${HOST:-localhost}" --port "${PORT:-8000}"
           --parallel "$TF_DSV41_SLOTS" --context "$TF_DSV41_CONTEXT"
           --name DeepSeek-V4.1-Flash-TF --max-tokens 32768)
    if [[ "${TF_DSV41_THINKING:-1}" == 0 ]]; then args+=(--no-thinking); fi  # the default; requests still override
else
    args+=(tf-dsv41-m1 follow /model /assets)
fi
if [[ "${2:-}" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit; fi
for dir in "$MODEL_DIR" "$ENGRAM_DIR" "$ASSETS_DIR" "$CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "missing directory $dir" >&2; exit 2; }
done
for asset in rope.bin engram-host.bin aot; do
    [[ -e "$ASSETS_DIR/$asset" ]] || { echo "missing asset $asset" >&2; exit 2; }
done
# rope.bin ("DSV41RP1", u32 rows, ...) covers the context plus the 2,048 prefill rows the engine reads past it
rope_rows=$(od -An -t u4 -j 8 -N 4 "$ASSETS_DIR/rope.bin" | tr -d ' ')
need_rows=$(( ${TF_DSV41_CONTEXT:-4096} + 2048 ))
if (( rope_rows < need_rows )); then
    echo "rope.bin holds $rope_rows rows; TF_DSV41_CONTEXT ${TF_DSV41_CONTEXT:-4096} needs $need_rows. Regenerate it:" >&2
    echo "  PYTHONPATH=\$PY_SOURCE python engine/zig/tools/zig/dsv41_rope_tables.py --config $MODEL_DIR/config.json \\" >&2
    echo "    --rows $need_rows --out $ASSETS_DIR/rope.bin   (docs/ZIG-SERVE.md, Generate runtime assets)" >&2
    exit 2
fi
# image input with the routing bias: the folder TF_DSV41_BIAS_VL names inside /cache must hold the fetched tensors
bias=${TF_DSV41_BIAS_VL:-}
if [[ "${TF_DSV41_IMAGES:-}" == native && "$bias" == /cache/* ]] &&
    ! compgen -G "$CACHE_DIR/${bias#/cache/}/*.safetensors" > /dev/null; then
    echo "TF_DSV41_BIAS_VL=$bias: no *.safetensors in $CACHE_DIR/${bias#/cache/}; the engine would refuse to boot." >&2
    echo "  fetch bias_vl.safetensors there (docs/IMAGES.md, The Zig serving path), or unset TF_DSV41_BIAS_VL" >&2
    exit 2
fi
# the published pfdense tuning table, placed where the env points inside /cache when it is not there yet
table=${TF_DSV41_PF_DENSE_TABLE:-}
if [[ "$table" == /cache/* && ! -e "$CACHE_DIR/${table#/cache/}" ]]; then
    mkdir -p "$(dirname "$CACHE_DIR/${table#/cache/}")"
    cp config/pfdense-table.json "$CACHE_DIR/${table#/cache/}"
    echo "copied config/pfdense-table.json to $CACHE_DIR/${table#/cache/} (TF_DSV41_PF_DENSE_TABLE=$table)" >&2
fi
mkdir -p "$PREPARED_DIR" "$STATE_DIR/sessions"
name="tensorfold-zig-r$rank"
if docker inspect "$name" >/dev/null 2>&1; then
    echo "rank $rank: container $name exists: this pair runs one stack. Stop it first (.claude/skills/verify/bin/down.sh from the Mac, scripts/serve.sh stop on the head), then run-zig.sh $rank." >&2
    exit 1
fi
boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)

run_rank() { # the container, then its exit witness. The temp + mv -f keeps a reader from ever seeing a half-written
    # record, which is the only thing standing between a crash and a tick that reads no witness at all.
    local code tmp
    "${args[@]}" && code=0 || code=$?
    tmp="$STATE_DIR/.exit-r$rank.$$.tmp"
    printf '%s %s %s\n' "$code" "$(date +%s)" "$boot_id" > "$tmp"
    mv -f "$tmp" "$STATE_DIR/exit-r$rank"
    return "$code"
}

rank_log="$STATE_DIR/run-zig-r$rank.log"

cap_rank_log() { # keep the newest 8 launch sections: the log a heal reads tails of must not grow without bound. In
    # place, because this loop's own stdout is that file.
    local n="" start tmp
    n=$(grep -c '^=== run-zig rank ' "$rank_log" 2>/dev/null || true); [[ "$n" =~ ^[0-9]+$ ]] || n=0
    (( n > 8 )) || return 0
    start=$(grep -n '^=== run-zig rank ' "$rank_log" | tail -n 8 | cut -d: -f1 | head -1)
    [[ -n "$start" ]] || return 0
    tmp="$STATE_DIR/.run-zig-r$rank.$$.tmp"
    tail -n "+$start" "$rank_log" > "$tmp" && cat "$tmp" > "$rank_log"
    rm -f "$tmp"
}

if [[ "${DSV41_ZIG_DETACHED:-0}" == 1 ]]; then         # the supervised loop: already detached, log open, rank's own node
    echo $$ > "$STATE_DIR/supervisor-r$rank.pid"       # the pid a human stop ends the loop by (scripts/serve/zig.sh)
    code=0
    while :; do
        cap_rank_log
        echo "=== run-zig rank $rank $(date -u +%Y-%m-%dT%H:%M:%S) ==="
        run_rank && code=0 || code=$?
        echo "rank $rank: the container exited $code"
        # A clean exit and a fail-fast 70 are the pair ending (a drained stop, or a death on either rank ending both),
        # and the pair's watchdog relaunches a pair. This loop relaunches only a rank that died alone.
        [[ "$code" == 0 || "$code" == 70 ]] && break
        sleep 5
    done
    exit "$code"
fi

if [[ "${2:-}" == --detached ]]; then
    setsid nohup env DSV41_ZIG_DETACHED=1 bash "${BASH_SOURCE[0]}" "$rank" --detached \
        >> "$rank_log" 2>&1 < /dev/null &
    exit 0
fi

code=0
run_rank && code=0 || code=$?
exit "$code"
