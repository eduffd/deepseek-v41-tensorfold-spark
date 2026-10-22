# What is not solved

- **A wedged pair heals itself, and the wedge's root cause is open.** On 2026-10-19 the Zig pair stopped issuing
  rounds: `/health` answered 503 with the stall labels, `rounds_total` froze at 383,335, in-flight requests grew from 22
  to 24, rank 1's GPU sat at 96%, and clients saw a server that accepted work and answered nothing (evidence in
  `results/verify/20261019-213006/`). A round in flight never returns, and rank 1 waits in the TP exchange. Rank 0's
  engine watch now exits 70 when its round in flight passes `TF_DSV41_STALL_EXIT_S` (300 in both Zig profiles), so the
  watchdog restarts both ranks ([issue
  #1](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/1), fixed). Issue
  [#3](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/3) closed for its `/health` scope. The verdict
  now flags a first round that never ends (measured 503 at 265 s quiet on the pair's 26k-token first prompt, against
  200 for 364 s at HEAD), and rank 1's transport progress rides the Fate channel into the `peer` section and
  `/metrics`, so a rank-1-side wedge is visible there as a growing `frozen_s` and outstanding plans. The wedge itself
  is the open half of #3, and a peer-keyed verdict waits on a measured threshold (OPERATIONS "Knobs").
  A deadline now sits on every peer wait:
  `TF_DSV41_PEER_WAIT_S` (600 in both shipped Zig profiles) arms one clock per rank inside a round, and the rank that
  stops moving names the wait it was in, prints the knob and the bound, and exits 70 beside its peer
  ([#2](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/2), fixed). The pair no longer waits forever on
  the rank-1 side of that wedge; an idle pair stays armed off. Still open: a mailbox timeout
  that leaves stale buffers live
  ([#4](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/4)). That day's recovery was a human: stop the
  pair, start it again, ready in 122 s, about 80 minutes after the pair went dead. The self-heal watchdog
  ([OPERATIONS](OPERATIONS.md) "Watchdog and start at boot") bounds the outage instead. An engine-labelled 503 is a
  definitive verdict, so the pair heals on the tick that sees it, at most once every `WATCH_FF_MIN_HEAL` (120 s), and
  serving returns in about 3.5 minutes of arithmetic: at most `WATCH_INTERVAL` (60 s) to notice, the heal's
  `WATCH_HEAL_GRACE` drain (20 s as both shipped Zig profiles set it; the scripts default is 15), and the measured
  122 s warm relaunch. That sum is a floor for the wedge family, not its expected time: the 122 s was measured on a
  pair drained by `bin/up.sh` and not on a wedge heal, and the heal's own path is outside the sum (the worker's drain
  poll runs up to 50 s when a wedged rank 1 lingers, then the force stop, the evidence capture and the preflight).
  The 122 s is measured on the pair; the notice and drain terms are the config's own numbers, and
  no end-to-end outage drill has timed the sum on the nodes yet.
  `WATCH_MAX_HEALS` (5) stops the
  healing of a pair that keeps coming back wedged, and each heal writes its evidence to `$STATE_DIR/heal-<utc>/` before
  the force stop destroys the pair. The idle probe (`WATCH_PROBE`) covers the shape rank 0 cannot report, a wedged
  rank 1 beside an idle rank 0. The wedge itself is unfixed. Upstream
  [issue #38](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/38) reports the same last two engine
  lines and the same rank-0 stack.

- **4 streams on the mixed cell are below the 120 target.** C4 mixed decode aggregate is ~101 tok/s (136 while all
  four streams are live). The cell's wall is mostly its prose stream finishing alone; 4 distinct code streams are
  estimated at ~125-135 but that cell has not been measured. The built-but-off `TF_DSV41_SPEC_NUCLEUS` measured
  +1.3% on C2 steady and -0.6% on C4 steady: not adopted.
- **Prose drafting.** DSpark keeps ~1.7 tokens a round on prose (3.7 on code), so prose (1.44x) and 2 streams (1.59x)
  are not at 2x. Self-distilling the drafter on our own logs helped prose and hurt code; not shipped.
- **The graph cache still shrinks under repeated memory dips.** Long prefills at 2,048 rows dip the worker's memory;
  G19 keeps the graphs through dips inside 64 decode-only rounds of a prefill, but in the 1 h soak five longer dips
  still took the LRU cap 48 -> 11 (G18: 48 -> 8). Fewer cached graphs cost speed until the next restart.
- **Capacity refusals of streaming requests arrive as in-stream errors.** When the priced admission floor refuses a
  streaming request after its `200` headers went out (it waited up to 30 s for memory first), the client gets an SSE
  `{"error": ...}` event, not an HTTP 503. Sending the headers lazily would fix it. G19's pricing at the smallest
  adaptive step removed the case the soak hit (0 refusals since).
- **The adaptive-row threshold.** Production runs `TF_DSV41_PREFILL_ADAPT_GIB=4.5`; the 1 h soak ran 4.0 (worker
  minimum 3.99 GiB, 9 MiB under our own 4.0 line, no errors). 4.5 has more margin but was not soaked, and not measured
  at 64K / 128K. The engine default (5) makes rows flap on GB10 (replay 32K 1,912 tok/s).
- **The worker is the binding node.** Its MemTotal is 2 GiB below the head's (firmware), and its MemAvailable at
  serving varies 6.2-8.4 GiB between identical boots, outside torch. earlyoom on the nodes (`-m 2,1` here) is the last
  backstop under our 4-5 GiB floors: it kills a rank below ~2.4 GiB, and fail-fast then takes the pair down in ~1 s
  for the watchdog to restart.
- **Two G13 rewrites ship off.** The shortened decode MoE chain (`TF_DSV41_MOE_FUSED`) is exact but measured +0.7 ms
  on a 1-row window in the graph. (`TF_DSV41_BRANCHES`, off in G13, is on since G14 on dedicated priority streams.)
- **Strict mode on the current engine** (every precision trade off, `full` prefill with the cone) has not been
  measured; the strict numbers above are G13's.
- **`/health`** reports `drafted_total` / `accepted_total` as 0 for this family (the counters are not wired).
- **No trimmed draft-head vocabulary is shipped** (`TF_DSV41_DRAFT_HEAD=trim`, off by default and not adopted). The
  development ranking was counted from private chat transcripts and is excluded, so `trim` needs
  `TF_DSV41_DRAFT_VOCAB=<file>` and `tests/test_dsv41_draft_head.py::test_shipped_ranking` fails. A ranking of your
  own traffic (`scripts/campaign/draftvocab.py`) or of public text (the GLM recipe's `bench/draftvocab_public.py`
  method on this tokenizer) can be used.
- **The watchdog's stand-downs are 20 minutes, not ownership.** Two open issues on the actor side, both found in
  review of the watchdog wave. The pair's Mac-side teardown touches only `WATCH_LEASE`, and a lease mutes the tick for
  `WATCH_LEASE_MIN` (20 min), while the both-absent verdict is definitive on any exit code, a clean exit 0 included,
  so a pair taken down by hand is relaunched once the lease ages
  ([#6](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/6)); the stop marker is the ownership
  transfer, and the teardown path does not write it. And `WORKER_CONFIG` is a knob whose two halves disagree: the
  worker's launcher takes the profile's config, while every head-side read of the worker parses
  `config/prod-zig.env`, so rank 1's record, log and supervisor pid land where the watchdog does not look
  ([#5](https://github.com/eduffd/deepseek-v41-tensorfold-spark/issues/5)).
