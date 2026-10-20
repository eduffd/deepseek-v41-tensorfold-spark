"""The Zig pair's supervision: the heal's evidence, the heal cap, the idle probe, the launcher's own rules, and
the boot decision's rows. The tick's verdicts and the incident repro live in test_pair_watchdog.py.

    python3 -m pytest -q tests/scripts
    python3 -m unittest discover -s tests/scripts -t tests/scripts -p 'test_*.py'   # the same suite, no pytest

Every assertion is a log line or a state file a maintainer reads on the node.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from conftest import EXAMPLE_FLASH17, EXAMPLE_ZIG, Rig

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
        """The durable rank log is what a heal reads tails of, so it keeps the newest launch sections and no more.
        Rank 1 runs on the worker, on the worker's config, so its record and log land in the worker's state dir."""
        self.rig.write_config("zig")
        self.rig.make_zig_dirs()
        seeded = "".join(f"{SECTION}{i} ===\nsection {i}\n" for i in range(1, 10))
        (self.rig.worker_state / "run-zig-r1.log").write_text(seeded)
        self.rig.run_rc(137)

        proc = self.rig.run_as_worker(self.rig.run_zig, "1", "--detached")

        def worker_state(name: str, needle: str) -> str:
            deadline = time.time() + 30
            while time.time() < deadline:
                text = self.rig.worker_state_file(name)
                if needle in text:
                    return text
                time.sleep(0.25)
            return self.rig.worker_state_file(name)

        record = worker_state("exit-r1", "137")
        log = worker_state("run-zig-r1.log", "=== run-zig rank 1 202")
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

    def test_the_recommended_profile_passes_the_same_gate(self) -> None:
        """config/prod-zig-flash17.env.example is the profile the docs recommend, so the heal's gate has to answer it,
        not only the rollback profile."""
        self.rig.write_example_config(example=EXAMPLE_FLASH17)
        self.rig.make_roce_active()

        proc = self.rig.run(self.rig.serve, "preflight", "static")

        self.assertEqual(proc.returncode, 0, f"the recommended profile must pass the gate the heal runs ({proc.stdout}{proc.stderr})")
        self.assertIn("preflight ok (static)", proc.stdout, f"{proc.stdout}")

    def test_both_shipped_profiles_ship_the_supervision_keys(self) -> None:
        """A profile that names no RECIPE is the Python pair, and a profile with no WATCH_* keys is a pair nobody
        watches. Both shipped Zig examples carry the supervision block, with the name the launcher uses."""
        keys = ("RECIPE", "NAME", "WORKER_SSH", "WORKER_REPO", "WATCH_HEAL", "WATCH_FAILS", "WATCH_HEAL_GRACE",
                "WATCH_MAX_HEALS", "WATCH_EVIDENCE_KEEP", "WATCH_LEASE", "WATCH_LEASE_MIN", "WATCH_PROBE",
                "WATCH_EXIT_LOOKBACK_S", "BOOT_LEASE")
        for example in (EXAMPLE_ZIG, EXAMPLE_FLASH17):
            with self.subTest(example=example.name):
                shipped = Rig(Path(tempfile.mkdtemp())).example_keys(example)
                self.assertEqual([k for k in keys if k not in shipped], [], f"{example.name} names none of these")
                self.assertEqual(shipped["RECIPE"], "zig", f"{example.name}")
                self.assertEqual(shipped["NAME"], "tensorfold-zig",
                                 f"{example.name}: scripts/run-zig.sh launches tensorfold-zig-r0 and -r1")
                self.assertGreaterEqual(int(shipped["WATCH_HEAL_GRACE"]), int(shipped["TF_DSV41_DRAIN_S"]),
                                        f"{example.name}: the drain window is not shorter than the engine's own drain")


class TestDieEventWitness(ShellCase):
    def test_a_die_event_answered_in_epoch_seconds_keeps_the_first_tick_heal(self) -> None:
        """The die-event read is the 70's only witness for a pair launched by hand (no cooperating launcher), so the
        read must answer on this Docker: `events.Message` carries the exit code under `.Actor.Attributes`, and the
        CLI validates the template and prints nothing when it cannot resolve it. `.Time` answers epoch seconds, and
        that epoch must reach the tick's boot / newer-than-the-last-heal comparison intact, not collapse to 0 and
        ride WATCH_FAILS."""
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0)
        self.rig.absent(r1)
        self.rig.die_event(r1, "70", 60)

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
        worker that is unreachable. It takes the loop's process group, because a loop killed while its launch is in
        flight leaves that launch running and the rank comes back behind the stop."""
        self.rig.write_config("zig")
        self.rig.make_zig_dirs()
        self.rig.run_rc(137)
        self.rig.run_hold_s(1)

        launched = self.rig.run_as_worker(self.rig.run_zig, "1", "--detached")
        self.assertEqual(launched.returncode, 0, f"--detached returns at once ({launched.stdout}{launched.stderr})")
        deadline = time.time() + 30
        while time.time() < deadline and not self.witness(1):
            time.sleep(0.25)
        first = self.witness(1)
        self.assertEqual(len(first), 3, f"the supervised launch writes its witness: {first!r}")
        self.assertEqual(first[2], self.rig.worker_boot,
                         f"the witness is stamped with the boot of the node the rank ran on: {first!r}")
        epoch = int(first[1])
        deadline = time.time() + 30
        while time.time() < deadline and int(self.witness(1)[1]) == epoch:
            time.sleep(0.5)
        self.assertGreater(int(self.witness(1)[1]), epoch,
                           "the loop is live: it relaunched the rank that died alone")
        self.assertNotEqual(self.rig.worker_state_file("supervisor-r1.pid"), "",
                            "the loop records the pid a stop has to end")

        second = self.rig.run_as_worker(self.rig.run_zig, "1", "--detached")
        self.assertEqual(second.returncode, 1,
                         f"a second supervised loop on the same rank is refused, not a second pair: {second.stdout}{second.stderr}")
        self.assertIn("supervised loop", second.stderr, f"name the loop that owns the rank: {second.stderr}")

        stop = self.rig.run(self.rig.serve, "stop")
        self.assertEqual(stop.returncode, 0, f"{stop.stdout}{stop.stderr}")
        epoch = int(self.witness(1)[1])
        self.assertEqual(self.rig.worker_state_file("supervisor-r1.pid"), "", "a stop consumes the pid it killed")
        time.sleep(8)      # longer than the loop's 5 s relaunch

        self.assertEqual(int(self.witness(1)[1]), epoch,
                         "no relaunch after the stop: the witness stopped moving")
        self.assertNotIn("tensorfold-zig-r1", self.rig.get_world()["containers"],
                         "a stopped rank stays down")

    def witness(self, rank: int) -> list[str]:
        state = self.rig.worker_state_file(f"exit-r{rank}") if rank == 1 else self.rig.state_file(f"exit-r{rank}")
        return state.split()


class TestDieEventWindow(ShellCase):
    """The die-event read is bounded by the pair's own stop marker, not only by a fixed lookback: a death after a
    human stop is this death, and a death the stop already answered is not."""

    def test_a_death_after_an_old_stop_is_witnessed(self) -> None:
        r0, r1 = self.rig.zig_names()
        self.rig.absent(r0, r1)
        self.rig.marker(self.rig.boot, time.time() - 600)
        self.rig.die_event(r0, "137", 300)

        proc = self.rig.tick("zig", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 1: both ranks absent, exit witnesses from this boot (r0 137"
                            in line for line in lines),
                        f"a pair death witnessed on this boot is definitive, and the marker is older than it: "
                        f"{lines} / {proc.stderr}")
        self.assertFalse(any("standing down" in line for line in lines), f"{lines}")

    def test_a_death_the_stop_answered_is_no_witness(self) -> None:
        r0, r1 = self.rig.zig_names()
        self.rig.absent(r0, r1)
        self.rig.marker(self.rig.boot, time.time() - 300)
        self.rig.die_event(r0, "137", 600)

        proc = self.rig.tick("zig", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertEqual(lines, ["[dsv41-tf] watch: both ranks absent (stopped on purpose); standing down"],
                         f"an event from before the stop is a death the stop already answered: {lines} / {proc.stderr}")


class TestHealYieldsToAStart(ShellCase):
    """A start holds the lock for its whole run, and a heal that queued on that lock would stop the pair the human is
    mid-start on. The tick gives up when the lock is held, and the heal it launches takes the lock the same way:
    never queued. The fixture cannot schedule a start into the gap between a tick's check and its launch, so these
    are the two ends of the rule, each deterministic."""

    def test_a_tick_during_a_heal_stands_down(self) -> None:
        """A heal is a start: it owns the pair from its stop to its ready. A second tick must not stack a heal on it."""
        self.pair_down_fast(self.rig)
        self.rig.run_hold_s(15)          # the launched rank stays up, so the heal is mid-start

        proc = self.rig.tick("zig", WATCH_HEAL="1", WATCH_FF_MIN_HEAL="0")
        lines = self.rig.stdout_lines(proc)
        self.assertTrue(any("restarting both ranks" in line for line in lines), f"the first tick launched the heal: {lines}")

        time.sleep(2)                    # the heal has its lock and its launch in flight by now

        proc2 = self.rig.run(self.rig.serve, "watch", "--once")

        self.assertEqual(self.rig.stdout_lines(proc2), ["[dsv41-tf] watch: a start, a heal or a stop is running; standing down"],
                         f"a tick runs no second heal beside a heal: {proc2.stdout}{proc2.stderr}")
        self.assertEqual(self.rig.state_file("heals"), "1", "the second tick healed nothing")

    def test_a_heal_that_finds_the_lock_held_says_so_and_touches_nothing(self) -> None:
        """The heal's restart takes the lock non-blocking. Queuing behind a human start is not what it does: it says
        so in heal.log and leaves the pair the human is starting alone."""
        self.pair_down_fast(self.rig)
        holder = self.hold_lock(25)

        proc = self.rig.run(self.rig.serve, "restart",
                             env_extra={"DSV41_TF_HEAL": "1", "DSV41_TF_GRACE": "0", "CANARY": "off"})
        holder.wait(30)

        self.assertEqual(proc.returncode, 1, f"the heal gives up on a held lock ({proc.stdout}{proc.stderr})")
        self.assertIn("a start, a heal or a stop is running", proc.stdout, f"{proc.stdout}")
        self.assertNotIn("tensorfold-zig-r0", self.rig.get_world()["containers"],
                         "the pair the human is starting stays as the human left it")
        self.assertEqual(self.rig.state_file("stopped"), "", "a refused heal marks nothing")

    def test_a_stop_under_a_held_lock_refuses_and_writes_no_marker(self) -> None:
        """`stop` is the human's word that the pair is down on purpose. Run it while a start owns the pair and it
        would stop that start's containers and mark the pair stopped on a node mid-boot."""
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)
        holder = self.hold_lock(25)

        proc = self.rig.run(self.rig.serve, "stop")
        holder.wait(30)

        self.assertEqual(proc.returncode, 1, f"a stop races no start ({proc.stdout}{proc.stderr})")
        self.assertIn("a start, a heal or a stop is running", proc.stdout, f"{proc.stdout}")
        self.assertEqual(self.rig.state_file("stopped"), "", "a refused stop marks nothing")
        self.assertIn(r0, self.rig.get_world()["containers"], "the running pair is untouched")

    def hold_lock(self, seconds: int) -> subprocess.Popen:
        holder = subprocess.Popen(["bash", "-c", f"exec 8>'{self.rig.state}/lock'; flock 8; sleep {seconds}"],
                                  env=self.rig.env())
        time.sleep(1.5)
        return holder


class TestCanaryRefusalStandsDown(ShellCase):
    def test_a_canary_failed_start_marks_the_pair_stopped(self) -> None:
        """The canary is the human's gate: a pair that fails it is a pair the operator must look at, and the
        watchdog relaunching it every tick is the actor answering a verdict that is not its own."""
        self.rig.write_config("python", CANARY="strict")
        self.rig.completions("500", '{"detail":"Internal Server Error"}')
        self.rig.run_hold_s(2)

        proc = self.rig.run(self.rig.serve, "start", timeout=180)

        self.assertEqual(proc.returncode, 1, f"CANARY=strict fails the start ({proc.stdout}{proc.stderr})")
        self.assertIn("canary failed", proc.stdout, f"{proc.stdout}{proc.stderr}")
        self.assertNotEqual(self.rig.state_file("stopped"), "",
                            "a start the canary refused is a pair stopped on purpose")

        tick = self.rig.run(self.rig.serve, "watch", "--once")
        lines = self.rig.stdout_lines(tick)
        self.assertTrue(any("both ranks absent (stopped on purpose); standing down" in line for line in lines),
                        f"the next tick stands down instead of relaunching: {lines} / {tick.stderr}")
        self.assertEqual(self.rig.state_file("last_heal"), "", "the watchdog healed nothing")


class TestStopReachesTheWorker(ShellCase):
    def test_a_stop_reaches_the_worker_whose_state_dir_is_unreadable(self) -> None:
        """The worker's config naming no readable STATE_DIR means the head cannot find the worker's supervisor pid.
        That is a config problem on a reachable node, not an unreachable node: inferring unreachability from it
        leaves rank 1's container up after a stop the operator ran on the pair."""
        self.rig.write_config("zig")
        (self.rig.root / "config" / "prod-zig.env").write_text(
            "".join(f"{line}\n" for line in (self.rig.root / "config" / "prod-zig.env").read_text().splitlines()
                    if not line.startswith("STATE_DIR=")))
        r0, r1 = self.rig.zig_names()
        self.rig.container(r0); self.rig.container(r1)

        stop = self.rig.run(self.rig.serve, "stop")

        self.assertEqual(stop.returncode, 0, f"{stop.stdout}{stop.stderr}")
        self.assertNotIn("unreachable", stop.stdout, f"the worker answered ssh: {stop.stdout}")
        self.assertNotIn(r1, self.rig.get_world()["containers"],
                         f"the force stop reached rank 1's container: {stop.stdout}")


class TestZigRecipeVerbs(ShellCase):
    def test_the_args_verb_prints_the_zig_launchers_arguments(self) -> None:
        """`serve.sh args 1` is the operator's look at what a rank would be launched with. On the Zig recipe that is
        scripts/run-zig.sh's own print, not the Python recipe's container builder."""
        self.rig.write_config("zig")

        proc = self.rig.run(self.rig.serve, "args", "1")

        self.assertEqual(proc.returncode, 0, f"the verb answers on the Zig recipe ({proc.stdout}{proc.stderr})")
        self.assertIn("TF_TP_RANK=1", proc.stdout, f"{proc.stdout}")
        self.assertIn("tf-dsv41-m1", proc.stdout, f"rank 1's own command: {proc.stdout}")

    def test_prebuild_refuses_the_zig_recipe_and_names_the_check_it_replaces(self) -> None:
        """The CUDA-extension prebuild is a Python-recipe build. A Zig profile that ran it would report built
        extensions it never built; the Zig check is the launcher's --check-aot."""
        self.rig.write_config("zig")

        proc = self.rig.run(self.rig.serve, "prebuild")

        self.assertEqual(proc.returncode, 2, f"prebuild is not this recipe's verb ({proc.stdout}{proc.stderr})")
        self.assertIn("--check-aot", proc.stdout, f"name the check the Zig recipe has: {proc.stdout}")

    def test_cache_fills_the_cache_the_zig_launcher_mounts(self) -> None:
        """The Zig recipe mounts a per-node directory at /cache (scripts/run-zig.sh -v $CACHE_DIR:/cache), not the
        Python recipe's docker volume. A cache verb pointed at the volume the profile never names fills nothing."""
        self.rig.write_config("zig", TF_DSV41_PF_DENSE_TABLE="/cache/tables/pfdense.json")

        proc = self.rig.run(self.rig.serve, "cache")
        calls = self.rig.run_calls()

        self.assertEqual(proc.returncode, 0, f"{proc.stdout}{proc.stderr}")
        self.assertEqual(len(calls), 2, f"the table goes to both nodes: {calls}")
        for n, call in zip(("head", "worker"), calls):
            self.assertIn(f"-v {self.rig.root / 'cache'}:/cache", call,
                          f"{n}: the mount names the CACHE_DIR the config of that node names, not the python "
                          f"recipe's volume: {call}")
            self.assertNotIn("-v :/cache", call, f"{n}: an unset CACHE_VOL is not a mount: {call}")


if __name__ == "__main__":
    unittest.main()
