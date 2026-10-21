# Changes since G13

The production engine went `767ad9f` (G13) -> `356a188` (G14) -> `7954c1d` (G15a) -> `da5ae43` (G16) -> `cd4245a`
(G18) -> **`7bd2d67`** (G19). Each window's write-up is in [`docs/campaign/`](campaign/README.md).

## Bugs fixed

- **`/health` flags a pair wedged issuing its first round, and rank 1's progress reaches rank 0** (this update, fork
  [issue #3](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/3)). The 503 chain skipped every state
  where no round had ever completed, so a pair frozen mid-first-round answered 200 forever (measured on the pair: 364 s
  of 200 with four hung requests). The verdict now reads rank 0's own evidence, and it fires when a request has been
  quiet past `TF_DSV41_STALL_ROUND_S` (120 s) plus the prompt's prefill estimate at
  `GLM53_TF_STALL_PREFILL_TPS` (200 tok/s), so a long first prefill is never refused for being long. Measured on the
  pair: 503 at 265 s quiet with a 26k-token first prompt, the reason string naming both numbers, and the pair heals to
  200 when the wedge clears. Rank 1 now ships its transport counters on a progress frame over the Fate channel, at the
  existing 0.5 s report cadence, header and payload in one whole-buffer send. `/health` grows a `peer` section (age,
  freeze, plans, outstanding, seconds since the pair last kept pace, and the fault record only when one landed), and
  `/metrics` grows `tensorfold_peer_age_seconds`, `tensorfold_peer_frames_outstanding` and `tensorfold_peer_failed`.
  A progress frame never dies the pair, and a half-wedge is export only. A frozen rank 1 beside an idle rank 0 stays
  a 200 with a growing `frozen_s` until a measured wave names the peer-keyed threshold. The wedge's root cause stays
  open in [KNOWN-ISSUES](KNOWN-ISSUES.md).
- **Rank 0's Zig engine exits 70 on a wedged round** (this update, fork
  [issue #1](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/1)). The engine's own stall watch reads
  `TF_DSV41_STALL_S` (default 300 s) and `TF_DSV41_STALL_EXIT_S` (`GLM53_TF_` wins, default 0; both Zig profiles now
  set 300), starts before the lane host's engine thread, and polls a round clock the engine thread publishes without
  the host mutex, so it does not depend on `/health`. A round in flight past the exit knob prints one report and one
  exit line to stderr, then exits 70, so the watchdog restarts both ranks; a normal drain ends its round first and
  never fires. The report names the round, the phase and both elapsed times, plus the host's running and waiting
  counts and plan sizes; it has no thread stacks, which Zig std cannot dump, so `gdb -p` remains the stack tool.
  `/health`'s stall labels are unchanged, and the replay gates (`serve_cli`, `sess4`, `mdraft`, `slots`) pass the
  watch off. The `/health` chain blind to a first round that never ended is fixed in this update
  ([#3](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/3), the bullet above; its wedge root cause
  stays open), and peer-wait deadlines ([#2](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/2)) and
  the stale mailbox ([#4](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/4)) remain open. The 300 s is above every
  measured round, but a cold whole-prompt prefill is armed from admission and p100 is unmeasured, so a legitimate
  prefill over 300 s restarts the pair.
- **A Zig pair that dies or wedges now restarts itself** (this update, the head-side watchdog on the Zig profile).
  `scripts/run-zig.sh` records `<code> <epoch> <boot_id>` in `$STATE_DIR/exit-r<rank>` on every invocation, so a
  `docker run --rm` rank's fail-fast exit 70 is a witnessed death and heals on the watchdog's first tick, at most once
  every `WATCH_FF_MIN_HEAL` (120 s). Recovery is arithmetic, and for the wedge it is a floor: at most `WATCH_INTERVAL`
  (60 s) to notice, the heal's `WATCH_HEAL_GRACE` (20 s) drain, the measured 122 s warm relaunch, so about 3.5 minutes.
  Only the 122 s is a measurement, and it was measured on a pair drained by `bin/up.sh`, not on a wedge heal. The
  heal's own path is not in the sum: the worker's drain poll runs up to 50 s when a wedged rank 1 lingers
  (`scripts/serve/zig.sh`'s `(grace + 30) / 5` loop), and the force stop, the evidence capture and the preflight follow
  it. The drill that times the sum on the nodes is open.
  The 2026-10-19 wedge is the before. No actor watched it, and it ran ~80 minutes to a human stop and start. A heal
  captures the wedged pair into `$STATE_DIR/heal-<utc>/` before the force stop destroys its
  containers, `WATCH_MAX_HEALS` (5) stops the healing of a pair that keeps coming back wedged, `scripts/serve.sh heal`
  prints the witnesses and the log tails, and the tick probes a pair whose `/health` body reports itself idle, the
  state a wedged rank 1 leaves rank 0 reporting. Units, knobs and stand-downs are in
  [OPERATIONS](OPERATIONS.md) ("Watchdog and start at boot") and [ZIG-SERVE](ZIG-SERVE.md) ("Keep it supervised"); the
  wedge's root cause stays open in [KNOWN-ISSUES](KNOWN-ISSUES.md).
- **The watchdog's die-event witness answered nothing** (this update, from an independent review of the watchdog
  wave, and measured on the nodes on 2026-10-20). The fallback read events with the template `{{.Time}}
  {{.Attributes.exitCode}}`, and `Attributes` is not a field of the daemon's `events.Message`: the CLI rejects the
  template, exits 64 and prints nothing. Measured on both Sparks (Docker 29.6.2, build dfc4efb), and it is why a rank
  with no cooperating launcher (a hand-run `docker run`, an earlyoom SIGKILL) had no witness: the read answered nothing
  before it reached the daemon, and the scripts sent that failure to `/dev/null`. The window is epochs now
  (`--since <epoch> --until <epoch>`), which is measured honored on 29.6.2 and is the form the older daemon parses as
  seconds (`api/types/time/timestamp.go`, moby read across v20.10 through v28.5.2, source-verified and not
  node-measured). The interval form `main` emitted (`--since 180s --until 0s`) is not a failure on these nodes, and
  the claim that it was is refuted by measurement: 29.6.2 parses `--since`/`--until` client-side as relative durations,
  so a window of `180s` to `0s` was honored. Measured on Spark1 with a die event made by the leg itself, `--since 1s`
  excluded it at 3 s old, `--since 60s` and `--since 180s` included it, all rc 0. The epoch form is kept for that
  reason, portability and not a repair: the older daemon parses only seconds, and epochs work on both dialects. This
  repo's `docker events` shim implements the daemon dialect (epoch seconds, an interval answered as
  `strconv.ParseInt: parsing "180s": invalid syntax`), which is a stricter reader than the nodes run. A marker from an earlier boot no longer anchors the
  window (it is a file, and it outlives the boot that wrote it). Two rows a stop owes are now printed: a removal the
  worker's daemon refuses alerts with the container named, and a worker config naming no `STATE_DIR` says so instead of
  reading rank 1 as a plain absence. The fixture's shared `world.json` reads take its lock, which is the mechanism
  behind the canary test's 1-in-5 red.
  A second review pass, on a third model family, found the pair's own stop standing inside that window. The daemon's
  lower bound is inclusive, and a force stop killed the ranks inside the same whole second as the marker it writes, so
  the window witnessed the stop's own SIGKILLs and the tick relaunched a pair the operator had just stopped. The fix
  is at the marker, not at the readers that lean on it: `mark_stopped` stamps one second after the force kill, which
  puts the stop's own kills outside the window, and the tick's comparison against the marker is inclusive at the
  marker's own second, because that is the second a rank dying a beat behind its peer leaves its witness in. The
  stop's alert also requires the container to be seen after a failed removal: `docker rm -f` answers a non-zero code
  for a name the daemon cannot find, and rank 1 leaves on its own when its peer's stop ends its launcher, so a routine
  stop says nothing. `mount_words` quotes per argument, where one `%q` over the joined pair made a cache path with a
  space one unparseable argument for docker.
- **The watchdog's start lock outlived the start** (this update, found by running the drills on the pair, 2026-10-20).
  `take_lock` opens `$STATE_DIR/lock` as fd 8 of the acting shell, and bash children inherit open fds. The Zig start
  launches its rank through `scripts/run-zig.sh --detached`, and that supervised loop lives as long as the rank does,
  so the loop and the `docker run` client it spawns held the lock for the life of the container. Measured on the head:
  `fuser` named `bash scripts/run-zig.sh 0 --detached` and its `docker run` as the holders, and every later
  `scripts/serve.sh stop` answered `a start, a heal or a stop is running` with no marker written while
  `scripts/serve.sh watch --once` answered the same row and stood down. That is a pair no human can stop, and a
  watchdog that switched itself off right after the first heal it performed. The launcher closes fd 8 at its own entry
  and `pair_launch` closes it on both spawn paths, so a launcher that outlives the change carries no lock. The suite
  could not see this: its `docker` shim answers and exits, so no fixture launcher was ever alive to inherit the fd, and
  a leg now holds a shim container up 12 s, finishes the start inside that window, and requires a stop and a tick to
  get through while the launcher is alive.
- **A duplicate image in one request killed both ranks** (G16). The vision row store counted a reference per
  occurrence of an image, so an agent history that resends the same screenshot released it early; the next request
  with that image raised `KeyError` inside rank 0's prefill forward, rank 1 waited in the exchange, and both ranks
  died. The store now counts each image once per request table. Two tests that fail on the old code.
- **The serving memory leak was the CUDA graph cache** (G16). The verify-graph key carries the context bucket (then
  2,048 tokens), so agent sessions growing to 120K tokens kept capturing new graphs (132 held in one soak, ~15 MiB of
  driver host memory each), until MemAvailable sat at the capture floor. Now an LRU cache capped at 48
  (`TF_DSV41_GRAPHS_MAX`) with context buckets 1.25x apart past 32K (26 buckets to 300K instead of 147; replay ==
  eager as before). Host trims also moved off the round thread (they were not the leak).
- **Graph-eviction cascade** (G17-G19). Under the capture floor the cache evicted a quarter of its graphs at every
  refused capture, so the long-prefill dips of one soak walked the cap from 48 to 8. Now one eviction a memory dip (then the freed pool
  is released and 64 capture attempts are skipped), and captures refused within 64 rounds of a prefill round evict
  nothing (`TF_DSV41_GRAPH_DIP_ROUNDS`).
- **NVMe session-index memory drift** (G18). The NVMe session tier's in-RAM index kept every parked entry's token ids as
  a Python `list[int]` (~36 bytes a token): +0.4 GiB/h of anonymous memory a rank under agent traffic, and 0.5-2.5 GiB
  a rank once the 128 GB tier fills (the slow drift seen since G6). The index now keeps (length, page chain, the < 256
  tokens past the last full page). RssAnon slope after the fix: -0.4 GiB/h.
- **Fail fast across both ranks** (G17-G18, `TF_DSV41_FAILFAST=1`, default). An exception on one rank inside a round
  used to leave the other waiting in an exchange: 120 s (RoCE), forever (NCCL), then the plan link's 300 s, then three
  watchdog ticks. Now both ranks fail the in-flight requests with a 500, tell each other over the plan link's TCP
  connection, and exit with code 70 within ~1-3 s (a CUDA out-of-memory on rank 1: both gone in 3.1 s, the client's
  500 names the error). `scripts/serve.sh watch` heals an exit 70 on its first tick (at most every 2 min).
- **A two-rank, priced memory floor** (G17, `TF_DSV41_FLOOR_PRICE=1`, default). Admission ran on rank 0 only with a
  zero price, so a 299K prompt was admitted at 6.2 GiB on the worker (never consulted) and its own prefill took it to
  3.5. Now rank 1 reports its usable memory every 0.5 s and a prompt is admitted only if its priced prefill transient
  fits on the tighter rank; otherwise it waits.
- **The "empty reply" was a capacity refusal priced too high** (G19). The floor priced an 18K prompt at the full
  2,048-row transient and refused it after 30 s (as an in-stream error: the 200 had gone out). Admission now prices at
  the smallest adaptive step; the soak client records in-stream errors.
- **Adaptive prefill rows** (G19, `TF_DSV41_PREFILL_ADAPT=1`, default). Rank 0 picks each round's prompt rows (2,048 /
  1,024 / 512) from the tighter rank's memory, so 2,048-row windows (+13-16% prefill) no longer push the worker under
  the floor in long prompts. Bit-identical outputs at any row count.
- **Fail-fast exit code race** (G18): rank 1 could exit 1 instead of 70 when rank 0 died first, which the watchdog's
  fast heal did not recognise.
- **Image input on CUDA** (G15): the ViT's attention handed 3-D tensors to the fused SDPA kernels (every image -> HTTP
  500); a typed `<｜deepseek_image｜>` in text is escaped instead of refused.
- **Build and packaging** (this update): the image build failed on `xgrammar.__version__` (xgrammar 0.2.8 has none;
  the Dockerfile now reads the package metadata); an installed engine lacked `csa2/*.cu` / `*.cpp` (ATTN_CUDA could not
  build) and the `pfdense` headers (PF_DENSE=fused could not build): packaging fix `66d0dcd`, and the Dockerfile checks
  the files; `scripts/pack_engram.py` failed with `KeyError: 'engram_layer_ids'` on the EXL3 packs' configs, which
  nest the text model's keys under `text_config`.
- earlyoom note: with earlyoom running on the nodes (`-m 2,1`), building CUDA extensions beside the loaded weights
  once pushed MemAvailable under its line and it killed both ranks (G16). `scripts/serve.sh prebuild` builds every
  extension before any weights are loaded; keep it that way.

Thanks to **WireLLM** for issue #3 (host memory growth on rank 0 until admission stalled: the graph cache and the
session index above), **ZackO2o** for PR #4 (the missing `csa2` kernel sources in an installed engine) and
**flashosophy** for PR #6, whose four-Spark work surfaced the `xgrammar.__version__` and `text_config` bugs. The fixes
here are our own; the PRs themselves are under review.

## New levers

| lever | window | what | measured | in the config |
| --- | --- | --- | --- | --- |
| `TF_DSV41_PLAN_LINK=rdma`, `PLAN_PIN` | G14 | the round plan through a RoCE host mailbox, pinned plan threads | with the three below: 1-row window 23.4 -> 21.8 ms; code +3.7%, prose +3.4%, structured +3.3%, C2 +5.4%, C4 +3.3% | on |
| `TF_DSV41_BRANCHES=1` (`_PRIO=side`) | G14 | CSA2 indexer / compressor on a dedicated high-priority side stream | 1-row -0.53 ms over 3 boots | on |
| `TF_DSV41_L2PF_PACE_GBPS=150` | G14 | the L2 prefetch paced beside the exchanges | 1-row -1.09 ms | on |
| `GLM53_TF_ROCE_FAST=1` | G14 | a shorter critical path in the RoCE all-gather kernel | 1-row -0.16, 16-row -0.51 ms | on |
| calibration VERSION 4 | G15 | every verify-table row measured (no fitted lines) | prose +2.8%, the rest flat | default |
| `TF_DSV41_IMAGES=native` | G15-G16 | DeepSeek's ViT + aligner on rank 0, `bias_vl` routing, Engram shut at image positions | VQA 20 / 20; text replies byte-identical to text-only builds | on |
| `TF_DSV41_PF_COPIES=1` | G16-G17 | one prefill weight copy | replay +1.7-3.3%, full +0.2-6.3% | on |
| `TF_DSV41_MHC_PF=1` | G16-G17 | the mHC prefill kernel | replay +3.8-4.7%, full -0.1 to +11.3% | on |
| `TF_DSV41_PF_DENSE=fused` + table | G16-G17 | the dense EXL3 prefill GEMM with the trellis decoded inside it (mma.sync, no fp16 weight workspace), tile table from a sweep | replay +5.4-6.9% with the swept table (+1.5-3.4% with the built-in one); the three together +12.8-13.6% | on |
| `TF_DSV41_FULL_CONE=1` | G17 | full-mode prefill: the decoder runs only over its dependency cone (the last ~2,541 rows), the same bits as plain full | full 32K 853 -> 1,619 alone, 1,828 with the three above (1,024 rows); 2,025 at 2,048 rows | on |
| adaptive 2,048-row prefill | G19 | `PREFILL_CHUNK` / `_ROWS` 2,048 with rows chosen a round from memory (`_ADAPT_GIB=4.5`) | replay 32K 1,996 -> 2,310 | on |
| `TF_DSV41_PF_XOVL`, `PREFILL_TILES`, 4,096-row windows, `SPEC_NUCLEUS`, `XMAP`, `DEPTH_JOINT=2`, `ALLOC_CEIL_GIB`, `SWITCH_MS` | G15-G17 | (built, exact or opt-in) | slower, no gain, or not needed: [`docs/RESULTS.md`](RESULTS.md) section 4 | off |

The dense-prefill tile table (G17's sweep on GB10, 10 shapes): best 68-78 TFLOP/s at 2,048 rows on the wide layers
(Engram 78, `wo_b` 75, shared gate 70, `wq_a` 68), 54-57 on the narrow ones; the whole table is
[`config/pfdense-table.json`](../config/pfdense-table.json) and `results/campaign/G17-20261004/sweep.txt`.

**Still open:** the items under [What is not solved](KNOWN-ISSUES.md).
