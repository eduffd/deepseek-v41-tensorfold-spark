"""The Zig pair's supervision: the heal's evidence, the heal cap, the idle probe, the launcher's own rules, and
the boot decision's rows. The tick's verdicts and the incident repro live in test_pair_watchdog.py.

    python3 -m pytest -q tests/scripts
    python3 -m unittest discover -s tests/scripts -t tests/scripts -p 'test_*.py'   # the same suite, no pytest

Every assertion is a log line or a state file a maintainer reads on the node.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from conftest import Rig

IDLE_BODY = '{"ok":true,"rounds_total":383335,"inflight":0,"queued":0}'
STALLED_BODY = ('{"ok":false,"status":"stalled","stall_reason":"no round for 4772 s with work in flight",'
                '"inflight":22,"queued":21,"decoding":0}')
BUSY_BODY = '{"ok":true,"rounds_total":383335,"inflight":3,"queued":2}'
COMPLETION = ('{"id":"c1","object":"chat.completion","model":"Test-Model","choices":[{"index":0,'
              '"message":{"role":"assistant","content":"42"},"finish_reason":"stop"}]}')
SECTION = "=== run-zig rank 1 2020-01-0"


class ShellCase(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = self.new_rig()

    def new_rig(self) -> Rig:
        tmp = tempfile.TemporaryDirectory(prefix="dsv41-sup-")
        self.addCleanup(tmp.cleanup)
        return Rig(Path(tmp.name))

    def pair_down_fast(self, rig: Rig) -> None:
        """The incident's state: both ranks gone, this boot's fail-fast witnesses left behind."""
        r0, r1 = rig.zig_names()
        rig.absent(r0, r1)
        rig.exit_record(0, "70", time.time() - 60)
        rig.exit_record(1, "70", time.time() - 60)

    def run_to_ready(self, rig: Rig) -> str:
        return rig.wait_for_state_file_containing("heal.log", "serving on :8199", timeout=90)


class TestProbeGate(ShellCase):
    def test_probe_fires_when_a_healthy_body_reports_the_pair_idle(self) -> None:
        """An idle pair that reports 200 is asked one tiny greedy completion, and a probe that gets no completion is a
        debounced bad tick: the idle-wedge family has no engine signal, so only a request reaches it."""
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("200", IDLE_BODY)
        self.rig.completions("000", "")

        proc = self.rig.tick("zig", WATCH_PROBE="1", WATCH_PROBE_S="2", WATCH_FAILS="3")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 3: probe " in line for line in lines),
                        f"a probe that gets no completion is the actor's own observation, so it rides WATCH_FAILS: {lines}")
        self.assertEqual(self.rig.state_file("fails"), "1", f"the probe counts one bad tick, not two: {lines}")
        self.assertEqual(proc.returncode, 0, f"3 ticks are needed to heal: {proc.stdout}{proc.stderr}")
        self.assertNotEqual(self.rig.state_file("last_probe"), "", "the probe stamp throttles the next probe")

    def test_probe_answers_a_completion_and_leaves_the_tick_clean(self) -> None:
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("200", IDLE_BODY)
        self.rig.completions("200", COMPLETION)

        proc = self.rig.tick("zig", WATCH_PROBE="1", WATCH_PROBE_S="5")

        self.assertEqual(self.rig.stdout_lines(proc), [], f"a pair that answers the probe is healthy: {proc.stdout}{proc.stderr}")
        self.assertEqual(self.rig.state_file("fails"), "0", "an answered probe clears nothing, because it counted nothing")

    def test_probe_is_not_fired_under_traffic(self) -> None:
        """A body with work in flight is not the idle wedge, and a busy pair needs no extra request."""
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("200", BUSY_BODY)

        proc = self.rig.tick("zig", WATCH_PROBE="1", WATCH_PROBE_S="2")

        self.assertEqual(self.rig.stdout_lines(proc), [], f"inflight 3 / queued 2 is traffic, not an idle pair: {proc.stdout}{proc.stderr}")
        self.assertEqual(self.rig.state_file("fails"), "0", f"{proc.stdout}{proc.stderr}")
        self.assertEqual(self.rig.state_file("last_probe"), "", "no probe ran, so the throttle stamp stays unwritten")

    def test_probe_writes_no_second_probe_inside_WATCH_PROBE_EVERY_S(self) -> None:
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("200", IDLE_BODY)
        self.rig.completions("000", "")
        self.rig.tick("zig", WATCH_PROBE="1", WATCH_PROBE_S="2", WATCH_PROBE_EVERY_S="120")
        stamp = self.rig.state_file("last_probe")

        second = self.rig.tick("zig", WATCH_PROBE="1", WATCH_PROBE_S="2", WATCH_PROBE_EVERY_S="120")

        self.assertEqual(self.rig.stdout_lines(second), [], f"a probe every {120}s, not one per tick: {second.stdout}{second.stderr}")
        self.assertEqual(self.rig.state_file("last_probe"), stamp, "the throttle stamp is not rewritten by a skipped probe")

    def test_watch_probe_off_keeps_a_healthy_idle_pair_silent(self) -> None:
        """WATCH_PROBE defaults off, so the Python recipe's config (which names no WATCH_PROBE) ticks as before."""
        r0, r1 = self.rig.py_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("200", IDLE_BODY)

        proc = self.rig.tick("python")

        self.assertEqual(self.rig.stdout_lines(proc), [], f"{proc.stdout}{proc.stderr}")
        self.assertEqual(self.rig.state_file("last_probe"), "", "no probe, no stamp")


class TestEvidenceBeforeTheStop(ShellCase):
    """A heal stops the pair before it can read anything: `docker run --rm` and a force stop leave no container and
    no log. So capture_evidence runs first, and the folder it writes is what the open root cause is studied from.
    The wedged pair here is alive and answering 503, which is the state the incident was caught in."""

    def wedged_pair(self, rig: Rig) -> None:
        r0, r1 = rig.zig_names()
        rig.container(r0); rig.container(r1)
        rig.log_line(r0, "[tensorfold] serving: 4 slot(s) x 1048576 tokens; wedged-rank-0")
        rig.log_line(r1, "[tf-dsv41-m1] follow: wedged-rank-1 doorbell spinning")
        rig.health("503", STALLED_BODY)

    def test_the_heal_writes_its_verdict_health_logs_and_gpus_before_it_stops_the_pair(self) -> None:
        self.wedged_pair(self.rig)

        proc = self.rig.tick("zig", WATCH_HEAL="1", WATCH_HEAL_GRACE="0")
        heal_log = self.run_to_ready(self.rig)

        self.assertIn("serving on :8199", heal_log, f"the heal must finish before the evidence is judged: {proc.stdout}")
        dirs = self.rig.heal_dirs()
        self.assertEqual(len(dirs), 1, f"one evidence folder per heal, in {self.rig.state}: {heal_log}")
        files = sorted(p.name for p in dirs[0].iterdir())
        self.assertEqual(files, ["gpu.txt", "health.json", "r0.log", "r1.log", "verdict.txt"], f"{dirs[0]}")
        verdict = (dirs[0] / "verdict.txt").read_text()
        self.assertIn("class=definitive", verdict, f"the tick's class travels with the heal, not the heal's guess: {verdict}")
        self.assertIn('note=/health 503 {"ok":false,"status":"stalled"', verdict, f"the verdict line, unit-free: {verdict}")
        self.assertIn("witness_r0=none", verdict, f"a pair that is standing has no exit witness: {verdict}")
        self.assertIn("witness_r1=none", verdict, f"a pair that is standing has no exit witness: {verdict}")
        self.assertIn("health_http=503", verdict, f"the /health status the heal saw: {verdict}")
        self.assertIn('"stall_reason":"no round for 4772 s with work in flight"', (dirs[0] / "health.json").read_text(),
                      "health.json holds the body the engine answered")
        self.assertIn("wedged-rank-0", (dirs[0] / "r0.log").read_text(),
                      "rank 0's tail, from the pair that was standing when the heal began")
        self.assertIn("wedged-rank-1", (dirs[0] / "r1.log").read_text(), "the worker's tail, read over ssh")
        captured = (dirs[0] / "r0.log").read_text()
        after = self.rig.run(self.rig.serve, "logs", "0")
        self.assertNotIn("wedged-rank-0", after.stdout,
                         f"the relaunched pair answers its own boot line, so nothing read after the stop is the wedge: {after.stdout}")
        self.assertIn("fresh-boot", after.stdout, f"{after.stdout}")
        self.assertNotIn("fresh-boot", captured,
                         "the capture ran before the relaunch, which is the only reason a --rm wedge has any log at all")

    def test_the_evidence_folder_count_is_capped(self) -> None:
        """WATCH_EVIDENCE_KEEP bounds the folders: a recurring wedge must not fill the state dir."""
        self.pair_down_fast(self.rig)
        self.rig.run_hold_s(2)
        for _ in range(2):
            proc = self.rig.tick("zig", WATCH_HEAL="1", WATCH_EVIDENCE_KEEP="1", WATCH_FF_MIN_HEAL="0",
                                 WATCH_MAX_HEALS="9", WATCH_HEAL_GRACE="0")
            self.run_to_ready(self.rig)
            self.rig.state_file("last_heal", "0")
            self.rig.state_file("fails", "0")
            time.sleep(3.0)      # the relaunched pair dies inside its hold, so the next tick sees a pair down again

        dirs = self.rig.heal_dirs()
        self.assertEqual(self.rig.state_file("heals"), "2", f"two heals ran: {proc.stdout}")
        self.assertEqual(len(dirs), 1, f"WATCH_EVIDENCE_KEEP=1 keeps the newest folder only: {[d.name for d in dirs]}")


class TestHealCap(ShellCase):
    def test_the_cap_alerts_and_refuses_to_heal(self) -> None:
        """A wedge that keeps taking the pair down is the pair needing a human look, not a sixth restart."""
        self.pair_down_fast(self.rig)
        self.rig.state_file("heals", "5")
        self.rig.state_file("last_heal", "0")

        proc = self.rig.tick("zig", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("5 heals in a row with no clean tick" in line and "WATCH_MAX_HEALS=5" in line
                            for line in lines), f"past the cap the alert names the cap: {lines}")
        self.assertFalse(any("restarting both ranks" in line for line in lines), f"the cap stops the heal: {lines}")
        self.assertEqual(self.rig.state_file("heals"), "5", "the count does not move on a refused heal")
        self.assertEqual(self.rig.state_file("last_heal"), "0", "a refused heal stamps no last_heal")
        self.assertEqual(self.rig.state_file("fails"), "1", f"the tick still counts: {lines}")

    def test_WATCH_MAX_HEALS_zero_turns_the_cap_off(self) -> None:
        self.pair_down_fast(self.rig)
        self.rig.state_file("heals", "9")
        self.rig.state_file("last_heal", "0")

        proc = self.rig.tick("zig", WATCH_HEAL="1", WATCH_MAX_HEALS="0")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("restarting both ranks" in line for line in lines),
                        f"0 = off is the documented meaning: {lines}")

    def test_a_clean_tick_resets_the_consecutive_heal_count(self) -> None:
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("200", IDLE_BODY)
        self.rig.state_file("heals", "5")

        proc = self.rig.tick("zig")

        self.assertEqual(self.rig.state_file("heals"), "0", f"a clean tick is the reset: {proc.stdout}{proc.stderr}")


class TestBootDecisionRows(ShellCase):
    """scripts/serve/zig.sh pair_boot_decision, through scripts/boot-start.sh with BOOT_DRY_RUN=1: the rows the head
    reads at boot, in the order its own witnesses decide them."""

    def decide(self, rig: Rig) -> tuple[int, str]:
        rig.make_roce_active()
        rig.write_config("zig")
        proc = rig.run(rig.boot_start, env_extra={"BOOT_DRY_RUN": "1", "BOOT_WAIT": "5"})
        return proc.returncode, rig.state_file("boot-start.log")

    def test_rank0_running_stands_down(self) -> None:
        self.rig.container(self.rig.zig_names()[0])
        rc, log = self.decide(self.rig)
        self.assertEqual(rc, 0, log)
        self.assertIn("rank 0 already running: nothing to do", log)

    def test_a_record_from_an_earlier_boot_starts(self) -> None:
        self.rig.absent(*self.rig.zig_names())
        self.rig.exit_record(0, "70", time.time() - 7200, boot="boot-from-the-previous-encarnation")
        rc, log = self.decide(self.rig)
        self.assertEqual(rc, 0, log)
        self.assertIn("exit witness from an earlier boot (exit 70): starting", log)
        self.assertIn("BOOT_DRY_RUN=1: not starting", log)

    def test_a_this_boot_stop_marker_stands_down(self) -> None:
        self.rig.absent(*self.rig.zig_names())
        self.rig.marker(self.rig.boot, time.time() - 30)
        rc, log = self.decide(self.rig)
        self.assertEqual(rc, 0, log)
        self.assertIn("stopped on purpose during this boot", log)
        self.assertIn("nothing to do", log)

    def test_a_marker_from_an_earlier_boot_starts(self) -> None:
        self.rig.absent(*self.rig.zig_names())
        self.rig.marker("boot-from-the-previous-encarnation", 1_700_000_000)
        rc, log = self.decide(self.rig)
        self.assertEqual(rc, 0, log)
        self.assertIn("stopped before the reboot (marker", log)
        self.assertIn("BOOT_DRY_RUN=1: not starting", log)

    def test_no_record_no_event_no_marker_stands_down(self) -> None:
        self.rig.absent(*self.rig.zig_names())
        rc, log = self.decide(self.rig)
        self.assertEqual(rc, 0, log)
        self.assertIn("no exit record, no die event and no stop marker (never started on this node): nothing to do", log)

    def test_a_die_event_inside_the_lookback_starts(self) -> None:
        """A rank killed by something that is not the launcher (an earlyoom SIGKILL, a hand-run docker run) leaves a
        die event and no record, so the tick's stand-down row must not swallow it."""
        r0, _ = self.rig.zig_names()
        self.rig.absent(*self.rig.zig_names())
        self.rig.die_event(r0, "137", 30)
        rc, log = self.decide(self.rig)
        self.assertEqual(rc, 0, log)
        self.assertIn("exit witness from this boot (exit 137): starting", log)


class TestRank1WitnessOverSsh(ShellCase):
    def test_rank1_absent_with_a_this_boot_record_heals_on_the_first_tick(self) -> None:
        """rank 1's record lives on the worker's own STATE_DIR and is read there, so a rank-1-only death is witnessed
        on this boot and heals on the first tick, where the pair's `--rm` containers leave inspect nothing to say."""
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0)
        self.rig.absent(r1)
        self.rig.exit_record(1, "70", time.time() - 60)

        proc = self.rig.tick("zig", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 1: rank 1 absent (fail-fast exit 70)" in line for line in lines),
                        f"the witness came from the worker's state dir: {lines} / {proc.stderr}")
        self.assertIn("serving on :8199", self.run_to_ready(self.rig),
                      "the heal relaunches the pair (rank 1 first), so a half-up pair comes back whole")


class TestLauncherRules(ShellCase):
    def test_a_rank_whose_container_exists_refuses_and_names_the_stop(self) -> None:
        self.rig.write_config("zig")
        self.rig.make_zig_dirs()
        self.rig.container("tensorfold-zig-r1")

        proc = self.rig.run(self.rig.run_zig, "1")

        self.assertEqual(proc.returncode, 1, f"one stack per pair ({proc.stdout}{proc.stderr})")
        self.assertIn("tensorfold-zig-r1 exists", proc.stderr, f"name the rank's container: {proc.stderr}")
        self.assertIn("down.sh", proc.stderr, f"name the stop to run: {proc.stderr}")
        self.assertEqual(self.rig.state_file("exit-r1"), "", "a refusal launches nothing, so it leaves no witness")

    def test_detached_launch_records_the_exit_and_bounds_its_log(self) -> None:
        """The durable rank log is what a heal reads tails of, so it keeps the newest launch sections and no more."""
        self.rig.write_config("zig")
        self.rig.make_zig_dirs()
        seeded = "".join(f"{SECTION}{i} ===\nsection {i}\n" for i in range(1, 10))
        self.rig.state_file("run-zig-r1.log", seeded)
        self.rig.run_rc(137)

        proc = self.rig.run(self.rig.run_zig, "1", "--detached")
        record = self.rig.wait_for_state_file_containing("exit-r1", "137", timeout=30)
        log = self.rig.wait_for_state_file_containing("run-zig-r1.log", "=== run-zig rank 1 202", timeout=30)
        self.rig.run_rc(0)          # the loop's next launch exits clean, which ends it (the pair watchdog owns 0 and 70)

        self.assertEqual(proc.returncode, 0, f"--detached returns at once and leaves the loop running: {proc.stdout}{proc.stderr}")
        self.assertEqual(record.split()[:1], ["137"], f"the supervised launch writes its witness: {record!r}")
        headers = [line for line in log.splitlines() if line.startswith("=== run-zig rank ")]
        self.assertLessEqual(len(headers), 9, f"8 sections kept, then this launch's header: {len(headers)}")
        self.assertIn("section 9", log, "the newest seeded section survives")
        self.assertNotIn("section 1", log, "the oldest section is the one that goes")


class TestHealVerb(ShellCase):
    def test_heal_prints_the_last_heal_the_counters_the_witnesses_and_the_logs(self) -> None:
        self.pair_down_fast(self.rig)
        self.rig.state_file("last_heal", str(int(time.time()) - 300))
        self.rig.state_file("fails", "2")
        self.rig.state_file("heals", "1")
        self.rig.state_file("heal.log", "[dsv41-tf] serving on :8199 (tensorfold-zig, image test-zig-image:1)")
        self.rig.write_config("zig")

        proc = self.rig.run(self.rig.serve, "heal")
        out = proc.stdout

        self.assertEqual(proc.returncode, 0, f"the verb is read-only ({proc.stdout}{proc.stderr})")
        self.assertIn("last heal: ", out, f"the last heal and its age: {out}")
        self.assertIn("fails: 2 of WATCH_FAILS=3; heals in a row: 1 of WATCH_MAX_HEALS=5", out, f"{out}")
        self.assertIn("lock: free", out, f"{out}")
        self.assertIn("witness r0: 70 ", out, f"both witnesses, as the tick reads them: {out}")
        self.assertIn("witness r1: 70 ", out, f"both witnesses, as the tick reads them: {out}")
        self.assertIn("serving on :8199", out, f"the heal.log tail: {out}")
        self.assertIn("rank 0 log tail:", out, f"both rank log tails: {out}")
        self.assertIn("rank 1 log tail:", out, f"both rank log tails: {out}")
        self.assertEqual(self.rig.state_file("fails"), "2", "the verb writes no state")

    def test_heal_reports_a_fresh_lease_as_a_stand_down(self) -> None:
        self.rig.write_config("zig")
        lease = self.rig.lease()

        proc = self.rig.run(self.rig.serve, "heal")

        self.assertIn(f"lease: {lease} is fresh; the tick stands down", proc.stdout, f"{proc.stdout}{proc.stderr}")


class TestAdapterBoundary(ShellCase):
    def test_a_NAME_that_is_not_the_launchers_prefix_is_a_config_error(self) -> None:
        """The launcher's container name is the witness, so a NAME aimed at other containers is refused at the config
        boundary, naming both the NAME it was given and the prefix it needs."""
        self.rig.write_config("zig", NAME="dsv41-tf")

        proc = self.rig.run(self.rig.serve, "watch", "--once")

        self.assertEqual(proc.returncode, 2, f"the config boundary answers 2 ({proc.stdout}{proc.stderr})")
        self.assertIn("NAME=dsv41-tf", proc.stderr, f"{proc.stderr}")
        self.assertIn("tensorfold-zig-r0", proc.stderr, f"name the prefix the launcher uses: {proc.stderr}")
        self.assertEqual(self.rig.state_file("fails"), "", "a config error touches no tick state")

    def test_a_config_missing_a_recipe_key_names_the_key(self) -> None:
        self.rig.write_config("zig", WORKER_SSH="")

        proc = self.rig.run(self.rig.serve, "watch", "--once")

        self.assertEqual(proc.returncode, 2, f"{proc.stdout}{proc.stderr}")
        self.assertIn("set WORKER_SSH", proc.stderr, f"{proc.stderr}")


class TestShippedZigConfig(ShellCase):
    def test_the_shipped_example_config_passes_preflight(self) -> None:
        """`cp config/prod-zig.env.example config/prod-zig.env`, fill the placeholders, arm the watchdog: the head's
        start path runs preflight, and preflight speaks head/worker path keys. The Zig config names none of them: it
        names MODEL_DIR / ENGRAM_DIR / PREPARED_DIR per node, and the pair's per-node dirs differ (engram-r0 on the
        head, engram-r1 on the worker), so the worker's must be read from the config the worker's launcher reads."""
        self.rig.write_example_config()
        self.rig.make_roce_active()

        proc = self.rig.run(self.rig.serve, "preflight", "static")

        self.assertNotIn("unbound variable", proc.stdout + proc.stderr,
                         f"preflight reads only keys the shipped config names ({proc.stdout}{proc.stderr})")
        self.assertEqual(proc.returncode, 0, f"the shipped config shape must pass the gate the heal runs ({proc.stdout}{proc.stderr})")
        self.assertIn("preflight ok (static)", proc.stdout, f"{proc.stdout}")


class TestDieEventWitness(ShellCase):
    def test_a_die_event_answered_in_epoch_seconds_keeps_the_first_tick_heal(self) -> None:
        """`docker events --format '{{.Time}} {{.Attributes.exitCode}}'` answers epoch seconds, not a timestamp
        string. The die event is the 70's only witness for a pair launched by hand (no cooperating launcher), so its
        epoch must reach the tick's boot / newer-than-the-last-heal comparison intact, not collapse to 0 and ride
        WATCH_FAILS."""
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0)
        self.rig.absent(r1)
        self.rig.die_event(r1, "70", 60, dialect="epoch")

        proc = self.rig.tick("zig", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 1: rank 1 absent (fail-fast exit 70)" in line for line in lines),
                        f"an epoch-seconds die event is the engine's report, so it is definitive: {lines} / {proc.stderr}")
        self.assertIn("serving on :8199", self.run_to_ready(self.rig),
                      f"the first tick launches the heal: {lines}")


class TestSupervisedRank(ShellCase):
    def test_a_human_stop_ends_the_supervised_relaunch_loop(self) -> None:
        """`--detached` self-relaunches a rank that died alone, which is what keeps a half-up pair up. A human
        `serve.sh stop` must end that loop, not leave it re-launching the rank behind the stop, and it ends the loop
        by the pid the loop records: a command-line pattern matches the ssh shell carrying it, which reads back as a
        worker that is unreachable."""
        self.rig.write_config("zig")
        self.rig.make_zig_dirs()
        self.rig.run_rc(137)
        self.rig.run_hold_s(1)

        launched = self.rig.run(self.rig.run_zig, "1", "--detached")
        self.assertEqual(launched.returncode, 0, f"--detached returns at once ({launched.stdout}{launched.stderr})")
        first = self.rig.wait_for_state_file_containing("exit-r1", "137", timeout=30).split()
        self.assertEqual(len(first), 3, f"the supervised launch writes its witness: {first!r}")
        epoch = int(first[1])
        deadline = time.time() + 30
        while time.time() < deadline and int(self.rig.state_file("exit-r1").split()[1]) == epoch:
            time.sleep(0.5)
        self.assertGreater(int(self.rig.state_file("exit-r1").split()[1]), epoch,
                           "the loop is live: it relaunched the rank that died alone")
        self.assertNotEqual(self.rig.state_file("supervisor-r1.pid"), "",
                            "the loop records the pid a stop has to end")

        stop = self.rig.run(self.rig.serve, "stop")
        self.assertEqual(stop.returncode, 0, f"{stop.stdout}{stop.stderr}")
        epoch = int(self.rig.state_file("exit-r1").split()[1])
        self.assertEqual(self.rig.state_file("supervisor-r1.pid"), "", "a stop consumes the pid it killed")
        time.sleep(8)      # longer than the loop's 5 s relaunch

        self.assertEqual(int(self.rig.state_file("exit-r1").split()[1]), epoch,
                         "no relaunch after the stop: the witness stopped moving")
        self.assertNotIn("tensorfold-zig-r1", self.rig.get_world()["containers"],
                         "a stopped rank stays down")


if __name__ == "__main__":
    unittest.main()
