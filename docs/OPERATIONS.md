# Operations

## What a start does (`scripts/serve.sh start`)

1. Takes a lock (one start at a time; the watchdog stands down while it is held) and clears the stop marker.
2. Refuses if a CUDA process runs on either node.
3. Preflight (`PREFLIGHT=strict`): ssh to the worker; the image on both nodes with equal content keys (creation time
   + layer digests); `config.json` / tokenizer in both model folders; both Engram shards on each node; a prepared
   folder (warning only); the RoCE ports ACTIVE; nothing on `PORT` / `MASTER_PORT`; the `/cache/roce-failed` marker.
4. Drops the page cache on both nodes and waits until MemFree >= `MEM_GATE_GIB` (104) on both (`MEM_GATE_TIMEOUT`).
   On GB10 the page cache is GPU memory: a load next to a full page cache can start with fewer request slots.
5. Starts rank 1 on the worker, then rank 0 here (`python -m tensorfold serve /model --tp 2 ...`), and waits for
   `/v1/models` (`READY_TIMEOUT`).
6. Checks rank 0's `serving: N slot(s)` line against `PARALLEL` (one retry after a cache drop if short).
7. Drops the caches again, runs `scripts/canary.py` (a greedy chat, a thinking reply with separated reasoning, a forced
   tool call, a JSON-schema reply, `/tokenize`), drops the caches once more. `CANARY=strict` stops both ranks on a
   failed canary and saves their last logs in `STATE_DIR`.

`scripts/serve.sh args 0|1` prints the exact `docker run` arguments without starting anything.

A start does not build the CUDA extensions on purpose: `scripts/serve.sh build` ends with `scripts/serve.sh prebuild`,
which compiles them into `CACHE_VOL` on both nodes with no weights loaded (a build next to ~100 GB of weights once
took the worker to 1.26 GiB MemAvailable). Run `scripts/serve.sh prebuild` yourself after `PREBUILD=0 scripts/serve.sh
build`, after deleting the cache volume, or before turning on a lever whose extension was never built. A start that
finds an extension missing builds it at load time, slowly and beside the weights. This is the Python recipe's build
step: the Zig recipe has no CUDA extensions, so `build` skips it and `prebuild` exits 2 naming the check the Zig
profile has, `scripts/run-zig.sh <rank> --check-aot`.

## Knobs

Every `TF_DSV41_*`, `GLM53_TF_*`, `MALLOC_*` and `MIMALLOC_*` line of the config reaches both ranks. A non-empty caller export wins
over the file: `CONTEXT=196608 scripts/serve.sh restart`.

| knob | production | off / fallback |
| --- | --- | --- |
| `TF_DSV41_PREFILL` | `replay` (~2x the kit) | `full`: the exact prefill (with `FULL_CONE=1`: 1.45-2.09x the kit) |
| `TF_DSV41_PREFILL_CHUNK` / `_ROWS` | 2048 / 2048 | 1024 / 1024 (G16-G18 production: replay 32K 1,996 vs 2,310) |
| `TF_DSV41_PREFILL_ADAPT` / `_ADAPT_GIB` | 1 (default) / 4.5: rank 0 steps each round's rows down (2,048 / 1,024 / 512, never below `_MIN` 512) when the tighter rank would keep less than this after the round's priced transient; back up with `_UP_GIB` (0.5) more | `_ADAPT=0`: fixed rows. The engine default `_ADAPT_GIB` (5) flaps on GB10 |
| `TF_DSV41_PF_COPIES`, `TF_DSV41_MHC_PF` | 1, 1 (G17, same bits) | 0 |
| `TF_DSV41_PF_DENSE` / `_TABLE` | `fused` / `/cache/pfdense-table.json` (G17, same bits; `scripts/serve.sh cache` puts `config/pfdense-table.json` there) | `off`. The table file must exist in the cache (the Python recipe's `CACHE_VOL`, the Zig profile's per-node `CACHE_DIR`; a missing file fails the prefill); unset `_TABLE` for the engine's built-in tiles (+1.5-3.4% instead of +5.4-6.9%) |
| `TF_DSV41_FULL_CONE` | 1: `full` prefill runs the decoder only over its dependency cone (same bits as plain `full`) | 0 |
| `TF_DSV41_INDEX_BUDGET_MIB` / `_STREAM_MIN` | 64 / 4096 (G16 memory cap, exact) | 256 / 16384 (+1.6 GiB of cached allocator segments in long prefills) |
| `TF_DSV41_POOL_TOKENS` | 614400: the 4 slots share 614K tokens of KV (a request waits for pages, never refused) | unset: 4 x CONTEXT (+0.6 GiB a rank) |
| `MALLOC_ARENA_MAX` | 4 | unset: glibc's default arenas |
| `TF_DSV41_IMAGES` | `native` (+ `TF_DSV41_BIAS_VL`, `_VISION_PREP_MB=64`) | `placeholder` (the engine default: a text notice per image) or `reject` (HTTP 400) |
| `TF_DSV41_PREFILL_KERNELS` | `fast` | `exact` |
| `TF_DSV41_FAST_EXPERTS` | `gm` | `grouped`; `tc` is 3-5x slower |
| `TF_DSV41_EXPERT_TOPP` / `_MIN_K` / `_RENORM` | 0.85 / 3 / kept (lossy, +5%) | delete the three lines: the unpruned model |
| `TF_DSV41_MHC_FN` | `bf16` | unset: fp32 mixing weights (prose -2.4%) |
| `TF_DSV41_L2PF` / `_MB` | 1 / 12 | 0 |
| `TF_DSV41_MHC_CUDA` | 1: an mHC boundary of a decode window as one CUDA launch (G13, same bits) | 0 (the engine's default): the Triton boundary |
| `TF_DSV41_ATTN_CUDA` | 1: CSA2's decode attention core and the indexer's top-k in CUDA (G13, same bits) | 0: Triton |
| `TF_DSV41_DENSE_V3` | 1: dense EXL3 linears over a 16-byte-coalesced repack of the trellis (G13, same bits) | 0: x3seg |
| `TF_DSV41_PLAN_LINK` / `_PLAN_PIN` | `rdma` / `auto` (G14: the round plan through the RoCE host mailbox) | `tcp` (MASTER_PORT + 7) |
| `TF_DSV41_BRANCHES` / `_PRIO` | 1 / `side` (G14, same bits) | 0 |
| `TF_DSV41_L2PF_PACE_GBPS` | 150 (G14, same bits) | 0: unpaced |
| `GLM53_TF_ROCE_FAST` | 1 (G14, the same bytes) | 0 |
| `TF_DSV41_PDL` | 0: must stay 0 with `DENSE_V3=1` (not bit-exact together; refused at boot) | |
| `TF_DSV41_GRAPH_MODE` | `rows` | `mix`: C2 / C4 slower |
| `TF_DSV41_ROUTER` | `gemv` | `fused` / `split` |
| `GLM53_TF_COMM_BACKEND` | `roce` | `nccl` (-11% / -17%). A RoCE failure at run time writes `/cache/roce-failed` in the cache (`CACHE_VOL`, the Zig profile's `CACHE_DIR`) and the next start of both ranks uses NCCL; delete it to retry RoCE |
| `CONTEXT` | 300000 (4 x 300K) | 196608 keeps the worker further from its memory floor |
| `TF_DSV41_THINKING` / `_DEFAULT_EFFORT` | 1 / `high` (75) | requests override |
| `TF_DSV41_GRAMMAR` | 1 | 0: `response_format` and strict tools ignored |
| `MIMALLOC_ALLOW_THP` | 0: no transparent huge pages in torch's CPU allocator (an embedded mimalloc), read at process start | unset: mimalloc's default (on); long prefills then stall in synchronous memory compaction (G12) |

Knobs left at their defaults that you may meet in the code: `TF_DSV41_DEPTH` / `TF_DSV41_DRAFT_DEPTH` (draft depth:
the default policy reads the boot calibration and caps at 5; `static` + 3 is the kit's), `TF_DSV41_DRAFT_HEAD` (full),
`TF_DSV41_VERIFY_BUDGET` (unset; lossy, do not use), `TF_DSV41_TREE_PC` (0), `TF_DSV41_DSPARK_DELTA` (unset).
From G13: `TF_DSV41_MOE_FUSED` (0: the shortened decode MoE chain, exact but +0.7 ms on a 1-row window),
`TF_DSV41_ATTN_CUDA_PARTS` (`attn,topk`; either alone for an A/B) and `TF_DSV41_MHC_CUDA_ROWS` (16, the largest
window the mHC kernel takes; 8 measured the same). From G15-G17, all off: `TF_DSV41_XMAP`, `TF_DSV41_DEPTH_JOINT=2`,
`TF_DSV41_PF_XOVL`, `TF_DSV41_PREFILL_TILES`, `TF_DSV41_PF_4K` (refused at boot without `TF_DSV41_PF_4K_MEMORY_OK=1`),
`TF_DSV41_SPEC_NUCLEUS`, `TF_DSV41_ALLOC_CEIL_GIB` (0), `TF_DSV41_SWITCH_MS` (unset).

Serving safety, on by default (G16-G19):

- `TF_DSV41_FAILFAST` (1): an exception in either rank's round fails every in-flight request with a 500, tells the
  peer over the plan link's TCP connection, and exits both ranks with code 70 after `TF_DSV41_FAILFAST_GRACE_S` (1.0)
  s. `scripts/serve.sh watch` heals an exit 70 on its first tick. 0 restores the old behaviour (the peer waits in its
  exchange: minutes).
- `TF_DSV41_FLOOR_PRICE` (1): admission prices a prompt's prefill transient (window rows, the indexer's blocks, host
  buffers) and admits it only if the tighter rank keeps `TF_DSV41_FLOOR_HARD_GIB` after it; rank 1 reports its usable
  memory every `TF_DSV41_PEER_MEM_S` (0.5) s. A prompt that does not fit waits; with nothing else running it is
  refused with a 503 after `TF_DSV41_FLOOR_REFUSE_S` (30) s (for a streaming request that refusal arrives as an
  in-stream error event after the 200).
- `TF_DSV41_GRAPHS_MAX` (48): the LRU CUDA graph cache; context buckets grow 1.25x past 32K
  (`TF_DSV41_GRAPH_BUCKET_GROW`; 1 restores equal 2,048-token steps); `TF_DSV41_GRAPH_DIP_ROUNDS` (64): captures
  refused within that many rounds of a prefill round evict nothing.
- `TF_DSV41_VISION_MAX_IMAGES` (8): more images in one request are refused (HTTP 400); clients with long agent
  histories should drop old screenshots.

Host memory and stalls (G12; defaults in brackets):

- `TF_DSV41_STALL_S` (300; 0 = off): rank 0's engine watch prints a stall report when a round has run longer than
  that (rank, round, phase, the round's and the phase's seconds, the host's running and waiting counts and plan sizes),
  again every 4x that. `TF_DSV41_STALL_EXIT_S` (0 = never; both shipped Zig profiles set 300) exits 70 when work is in
  flight and the current round has run that long, so the watchdog restarts both ranks. The watch owns its round clock
  and reads it without the lane host's mutex, so it does not depend on `/health`; the vendor Python family's
  `stall.py` `Watch` also prints every thread's stack, which Zig std cannot, so the report names the phase and both
  elapsed times and `gdb -p` remains the stack tool. `/health` keeps its own semantics. Its `stalled[]` list reads
  `TF_DSV41_STALL_S` (default 0), and its stall labels read `TF_DSV41_STALL_ROUND_S` (120) and
  `TF_DSV41_STALL_SLOW_S` (180). `GLM53_TF_STALL_S` and `GLM53_TF_STALL_EXIT_S` win over the `TF_DSV41_` spellings, and
  the replay gates pass the watch off. The fail-fast exit 70 (`TF_DSV41_FAILFAST`) is separate, and the watchdog heals
  either on its first tick. Rank 1 has no watch of its own yet (fork issue
  [eduffd/deepseek-v41-tensorfold-spark#3](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/3)); peer
  waits ([#2](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/2)) and the stale mailbox
  ([#4](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/4)) stay open
  ([KNOWN-ISSUES](KNOWN-ISSUES.md)).
- `TF_DSV41_ENGRAM_WAIT_S` (300; 0 = none): deadline on an Engram read and on the Engram gate's worker.
- `TF_DSV41_NUMPY_HUGEPAGE` (0): NumPy's huge-page request on its large buffers, off on the ranks.
- `TF_DSV41_HOST_BUFFERS` (6): reused NumPy buffers for a prompt segment's Engram rows.
- `TF_DSV41_HASH_MEMO_ROWS` (16384): rows the Engram hash memo holds. `TF_DSV41_HOST_TRIM_ROWS` (32768; 0 = off):
  `malloc_trim` every that many prompt rows (it reaches glibc only; harmless).
- `TF_DSV41_MEMPROBE=N` (0 = off): one memory line a rank every N rounds (RssAnon / RssShmem, MemAvailable, torch's
  allocator); `TF_DSV41_MEMPROBE_TENSORS=K` and `_SMAPS=1` add live CPU tensors and anonymous RSS by mapping (slow,
  diagnostics only).

## Memory

- **Before a load:** MemFree >= 104 GiB on both nodes.
- **While serving:** `TF_DSV41_FLOOR_GIB=5` is the MemAvailable target the load-time budget plans for;
  `TF_DSV41_FLOOR_HARD_GIB=4` is the level below which nothing new is admitted, checked on both ranks with each
  prompt's prefill transient priced in.
- **Measured (G19, 2,048-row adaptive prefill):** 299K stress minimum head 5.07 / worker 4.81 GiB; 1 h soak minimum
  head 4.63 / worker 3.99 GiB (at `_ADAPT_GIB=4.0`; production runs 4.5), RssAnon falling over the hour. The worker
  binds: 2 GiB less MemTotal than the head, and 6.2-8.4 GiB at serving between identical boots. If it runs short,
  raise `TF_DSV41_PREFILL_ADAPT_GIB` or go back to 1,024-row windows. [RESULTS.md](RESULTS.md) section 5.
- **earlyoom:** if the nodes run earlyoom (`-m 2,1` here: SIGTERM below ~2.4 GiB), it is the last backstop under these
  floors and kills a rank; fail-fast then ends the pair in ~1 s and the watchdog restarts it. Never build CUDA
  extensions beside loaded weights (`scripts/serve.sh prebuild` exists for that: G16 lost both ranks to it once).
- Keep other workloads (desktop sessions, other containers, a second model) off both nodes.

## Reported configurations

- **1M context, 1 slot** (reported on this path by @ZackO2o in #5): `KV_DTYPE=fp8`, `CONTEXT=1048576`, `PARALLEL=1`
  booted and served. `/v1/models` reported 1,048,576, retrieval at 60K tokens hit, and prefill ran at 2,048 / 2,104 /
  1,934 tok/s for 16.6K / 135K / 270K-token prompts. Our own runs cover 4 x 300K and 4 x 196K. One slot serves one
  request at a time.
- **The same on the Zig path** ([ZIG-SERVE](ZIG-SERVE.md#split-kv-and-the-pool)): the profile already serves one 1M
  request. `TF_DSV41_CONTEXT=1048576` with `TF_DSV41_KV_SPLIT=1` and the recommended flash17 profile's
  `TF_DSV41_POOL_TOKENS=2100000` holds one slot's whole 1M, two slots at 1M each, or four live slots at about 525K
  each (totals across live slots). The q28-v2 rollback profile's `TF_DSV41_POOL_TOKENS=3000000` holds four live slots
  at about 750K each. For a dedicated
  one-slot server set `TF_DSV41_SLOTS=1` and check the AOT set again (`scripts/run-zig.sh 0 --check-aot`); drafting
  still runs, and `TF_DSV41_SLOT_DRAFTS` then has no effect. `rope.bin` needs `TF_DSV41_CONTEXT` + 2,048 rows. The
  boot check refuses a pool the measured memory cannot hold, and a request whose prompt plus `max_tokens` exceeds the
  context is refused with the context-length error.

## Watchdog and start at boot

The watchdog runs on the head, on one config at a time. `scripts/serve.sh watch --once` is the tick, and
`scripts/systemd/dsv41-watchdog@<config-basename>.timer` runs it every minute, so
`dsv41-watchdog@prod-zig.timer` ticks with `CONFIG=config/prod-zig.env`. The Python profile keeps the legacy `dsv41-tf-watchdog.timer` and
`dsv41-boot-start.service` with `config/prod.env`, untouched. Each profile keeps its own `STATE_DIR`, so no lock, lease,
counter, marker, log or heal folder is shared between them.

- The tick is bad when a rank is gone, `/health` answers other than 200, or a probe gets no completion (below). `000`
  while rank 0 is younger than `WATCH_GRACE` (1800 s) is loading, not bad.
- The verdicts split in two. Definitive is the engine's own words: a fail-fast exit 70 witnessed on this boot, a pair
  death witnessed on this boot, and a `/health` body the engine labelled (`"status":"stalled"` with a `stall_reason`,
  `"ok":false` with `"fatal"`). Those heal on the first tick, at most once every `WATCH_FF_MIN_HEAL` (120 s). Debounced
  is the actor's own client observation (a plain non-200, `000` past grace, a rank gone with no witness). `WATCH_FAILS`
  (3) bad ticks in a row heal, at most once every `WATCH_MIN_HEAL` (1800 s), with `WATCH_HEAL=1`. No threshold in the
  actor is a stall threshold. The engine's stall knobs are in the table above; the tick re-derives none of them.
- The two recipes witness a rank's exit differently, and only the recipe knows which. The Python pair's containers
  persist, so `docker inspect` answers their exit code and finish time. The Zig pair runs `docker run --rm`, so a rank
  leaves no container to inspect. `scripts/run-zig.sh` writes `<code> <epoch> <boot_id>` to `$STATE_DIR/exit-r<rank>`
  on every invocation, rank 1's record living on the worker's own `STATE_DIR` and stamped with the *worker's* boot id:
  the two nodes reboot independently, and rank 1's death belongs to its node's boot, never the head's. The adapter
  compares each rank's record with the boot of the node it ran on and answers the tick in the head's terms. When no
  record exists, which covers a hand-run `docker run` and an earlyoom SIGKILL, the `die` event is the witness, read
  with the template the CLI validates (`{{.Time}} {{.Actor.Attributes.exitCode}}`; a top-level `.Attributes` is not a
  field of `events.Message`, and the CLI exits 64 printing nothing, measured on both Sparks) and over a window given in
  epoch seconds, the form both dialects answer: the nodes' Docker 29.6.2 parses `--since`/`--until` client-side as
  relative durations, measured 2026-10-20, and the older daemon parses them as epoch seconds only (moby
  `api/types/time/timestamp.go`, source-verified). The window reaches `WATCH_EXIT_LOOKBACK_S` (180 s)
  back, or from the last stop marker forward when that marker is from this boot, so a death that stop answered is not
  read as this one, and a stop from before the reboot anchors nothing. The stop's own kills are outside that window
  because of how the marker is stamped, not how the window is cut: the daemon's lower bound is inclusive, so
  `mark_stopped` writes the marker one second *after* the force kill, and epochs are whole seconds. A rank that dies a
  beat behind its peer lands in the marker's own second and is inside the window, where the tick answers it as the
  stop's. A witness from an earlier boot is an earlier boot's death, never this one's.
- The tick stands down while the start lock is held (a start, a heal's restart, a stop), while rank 0 is loading
  (`WATCH_GRACE`), after a deliberate
  `scripts/serve.sh stop` in the same boot (the stop marker, and the marker's own second counts, which is the second a
  rank dying a beat behind its peer leaves its witness in), when both ranks are absent with no witness and no marker
  (the pair never started on this node), and while `WATCH_LEASE` is fresh (`WATCH_LEASE_MIN` 20 min, re-touchable). The
  marker owns the human teardown, and only that. A worker that does not answer ssh is not counted either, and it is
  alerted by name: from the head, rank 1 is unwatched, and half a pair answers `/health` 200 by itself. A worker that
  answers ssh and names no `STATE_DIR` in its own config is said the same way, on stderr beside a debounced bad tick:
  rank 1's record cannot be read, which is a config fault, not a rank with no death.
- The start lock is fd 8 of the shell that took it, and bash children inherit open fds, so the launcher closes it at
  its own entry and `pair_launch` closes it on both spawn paths. A supervised loop that carried it stands every later
  tick down and refuses every later `stop` for the life of the rank: that is what the pair measured on 2026-10-20,
  where `fuser $STATE_DIR/lock` named `bash scripts/run-zig.sh 0 --detached` and the `docker run` client it spawned as
  the lock's holders. A lock held by a launcher is not "a start is running"; it is the actor switching itself off.
- A start the canary refused is a pair stopped on purpose: the canary-failed start writes the stop marker, so the
  watchdog leaves the pair the canary rejected for the human to read (`$STATE_DIR/canary-fail-r0.log`,
  `canary-fail-r1.log`). A `stop` and a heal both take the start lock, non-blocking: neither queues behind a start it
  would race, and each says so in its log. A `stop` alerts only on a removal the worker's daemon refused on a
  container that is *still there* after the failed removal, which it checks by inspecting it: a rank 1 whose peer
  stopped leaves on its own (its launcher's fate plus `--rm`), and `docker rm -f` answers non-zero for a name the
  daemon cannot find, so an alert keyed on the return code alone fires on every routine stop and stops being read.
- A heal counts. `WATCH_MAX_HEALS` (5) consecutive heals with no clean tick between them alert and stop healing
  (`0` turns the cap off); a clean tick resets the count in `$STATE_DIR/heals`. A pair that keeps coming back wedged
  needs a human look, not a sixth restart.
- A heal captures before it stops. `capture_evidence` writes `$STATE_DIR/heal-<utc>/` with `verdict.txt` (the class,
  the verdict line, both witnesses), `health.json`, both ranks' log tails, and `nvidia-smi` from both nodes, and keeps
  the newest `WATCH_EVIDENCE_KEEP` (20) folders. The force stop that follows destroys the wedged pair's containers,
  and the root cause of the 2026-10-19 wedge is open ([KNOWN-ISSUES](KNOWN-ISSUES.md)), so the folder is that evidence.
- The idle probe (`WATCH_PROBE=1`) is one tiny greedy completion, asked only when a 200 `/health` body reports
  `inflight` 0 and `queued` 0, at most every `WATCH_PROBE_EVERY_S` (120 s), waiting `WATCH_PROBE_S` (120 s) for the
  answer. A timeout is one debounced bad tick. It exists for the state no engine signal reports, a wedged rank 1
  beside an idle rank 0. Its cost is one 8-token completion per idle pair every two minutes, and a completion that
  never arrives is healed only after `WATCH_FAILS` of them. The scripts' default is off; both shipped Zig profiles set
  it on, since the idle wedge is the family the incident log has and the engine cannot report.
- `scripts/serve.sh heal` prints what to read after a heal: the last heal and its age, `fails`, the consecutive heal
  count, the lease and lock state, both exit witnesses, the stop marker, the `heal.log` tail and both ranks' log tails.
  `WATCH_ALERT=<cmd>` is called with the text of every alert, and a tick that keeps alerting calls it every tick: a
  worker that does not answer ssh alerts once a minute, so a hook that pages has to dedup.
- `scripts/boot-start.sh` is the `dsv41-boot-start.service` oneshot, and
  `dsv41-boot-start@<config-basename>.service` names a profile. After a reboot, it starts the service if it was serving
  (or crashed) before, once docker, the worker and the RoCE ports are up. `BOOT_DRY_RUN=1` shows the decision.
  It reads `STATE_DIR`, the stop marker and `BOOT_LEASE` from the config it is given. The watchdog is the fallback.
- Before benchmarks with `scripts/serve.sh run`, stop the server and the timer
  (`systemctl --user stop dsv41-watchdog@prod-zig.timer`), or touch `WATCH_LEASE`. Any launcher you run beside the
  watchdog should touch the lease first: this pair's Mac-side launch and teardown scripts do, so a start or a stop is
  not raced by the tick.

## Logs and state

- `docker logs dsv41-tf-r0` / `scripts/serve.sh logs 1`; the `[boot]` lines time each start phase.
- `HEAD_STATE` / `WORKER_STATE`: the request log (`requests.jsonl`) and the session NVMe tier (`sessions/`).
- `STATE_DIR` (default `~/.local/state/dsv41-tf`; each profile sets its own): the start lock, watchdog counters
  (`fails`, `heals`, `last_heal`), the probe throttle stamp (`last_probe`), the stop marker, the lease, the heal and
  boot-start logs, canary-failure logs, and one `heal-<utc>/` evidence folder per heal (newest `WATCH_EVIDENCE_KEEP`
  kept).
- On the Zig profile, each node's `STATE_DIR` also holds that rank's launcher files: `exit-r<rank>` (the exit witness,
  one line `<code> <epoch> <boot_id>`, written on every `scripts/run-zig.sh` invocation), `run-zig-r<rank>.log` (the
  durable rank log a `--detached` launch appends to, capped at the newest 8 launch sections) and
  `supervisor-r<rank>.pid` (the pid of a `--detached` loop, which `scripts/serve.sh stop` ends by its process group: a
  stop that left the loop alive would relaunch the rank it was asked to keep down, and a kill that missed the launch
  in flight leaves that launch running). The loop deletes the pid file as it exits, and a second `--detached` on a
  rank whose loop is live is refused. A `--rm` container has no `docker logs` after it exits, so these are the rank's
  own record of its last boots.
