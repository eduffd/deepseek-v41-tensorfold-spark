"""The watchdog tick's verdicts (scripts/serve.sh) and the Zig launcher's exit witness (scripts/run-zig.sh).

    python3 -m pytest -q tests/scripts
    python3 -m unittest discover -s tests/scripts -t tests/scripts -p 'test_*.py'   # the same suite, no pytest

Two tests are the 2026-10-19 pair wedge's repro, and both exercise the Zig lane: a pair down with this-boot exit-70
records healing on the first tick, and a plain scripts/run-zig.sh invocation writing its rank's exit record. The rest
is the contract's behavior, which must not move: the Python recipe's verdicts, the recipe key's validation, and the
definitive / debounced classes.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from conftest import Rig

PY_DOWN_STOPPED = "[dsv41-tf] watch: both ranks absent (stopped on purpose); standing down"
PY_DOWN_NEVER = ("[dsv41-tf] watch: both ranks absent with no exit witness and no stop marker "
                 "(never started on this node); standing down")
STALLED_BODY = ('{"ok":false,"status":"stalled","stall_reason":"no round for 4772 s with work in flight",'
                '"inflight":22,"queued":21,"decoding":0}')
UNAVAILABLE_BODY = '{"detail":"Service Unavailable"}'


class ShellCase(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = self.new_rig()

    def new_rig(self) -> Rig:
        tmp = tempfile.TemporaryDirectory(prefix="dsv41-tick-")
        self.addCleanup(tmp.cleanup)
        return Rig(Path(tmp.name))


class TestStalledVerdict(ShellCase):
    def test_health_503_stalled_is_a_bad_tick_naming_the_engine_words(self) -> None:
        """A stalled /health 503 is the engine's own verdict: bad on the first look, quoting its labels verbatim."""
        r0, r1 = self.rig.py_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.health("503", STALLED_BODY)

        proc = self.rig.tick("python")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 1: /health 503" in line for line in lines),
                        f"an engine-stalled 503 is definitive (no threshold in the actor is a stall threshold): {lines}")
        self.assertTrue(any('"status":"stalled"' in line and "no round for 4772 s with work in flight" in line
                            for line in lines),
                        f"the log line must carry the engine's words verbatim: {lines}")
        self.assertEqual(self.rig.state_file("fails"), "1", f"a definitive verdict counts the first tick: {lines}")


class TestIncidentRepro(ShellCase):
    def test_pair_down_with_this_boot_exit70_records_heals_on_the_first_tick(self) -> None:
        """Incident repro (2026-10-19 pair wedge): a pair down whose exit-70 witnesses are from this boot heals on
        the first tick. The Zig pair runs `docker run --rm`, so its witnesses are the launcher's exit records, read
        by the Zig adapter."""
        r0, r1 = self.rig.zig_names()
        self.rig.absent(r0, r1)
        self.rig.exit_record(0, "70", time.time() - 60)
        self.rig.exit_record(1, "70", time.time() - 60)

        proc = self.rig.tick("zig", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 1:" in line and "(fail-fast exit 70)" in line for line in lines),
                        "incident repro: both ranks absent with this-boot exit-70 records and no stop marker must heal "
                        "on the first tick, where scripts/serve.sh at a8806b9 (:290-291) answered 'standing down': "
                        f"{lines} / {proc.stderr}")
        self.assertTrue(any("restarting both ranks" in line for line in lines),
                        f"incident repro: the first tick must launch the heal, not log a stand-down: {lines}")
        heal_log = self.rig.wait_for_state_file_containing("heal.log", "serving on :8199")
        self.assertIn("serving on :8199", heal_log,
                      "incident repro: the heal must relaunch the pair to ready (rank 1 first)")

    def test_plain_run_zig_invocation_writes_the_exit_record(self) -> None:
        """Incident repro: every scripts/run-zig.sh invocation must leave "<code> <epoch> <boot_id>" as the
        watchdog's pair witness. A plain invocation (no --detached) is the shape bin/up.sh launches, and a pair-down
        tick has only that record to read."""
        self.rig.write_config("zig")
        self.rig.make_zig_dirs()
        self.rig.run_rc(70)

        proc = self.rig.run(self.rig.run_zig, "1")
        record = self.rig.state_file("exit-r1")
        fields = record.split()

        self.assertEqual(fields[:1], ["70"],
                         "incident repro: a plain run-zig.sh 1 invocation (its rank exited 70) must write "
                         f'$STATE_DIR/exit-r1 as "<code> <epoch> <boot_id>": record={record!r} rc={proc.returncode} '
                         f"out={proc.stdout} err={proc.stderr}")
        self.assertEqual(len(fields), 3,
                         f"the witness needs its epoch and boot id for the tick's boot/marker comparison: {record!r}")
        self.assertGreaterEqual(int(fields[1]), time.time() - 300, f"the witness's epoch is the invocation's time: {record!r}")


class TestHeal(ShellCase):
    def test_a_definitive_verdict_heals_on_the_first_tick(self) -> None:
        """A definitive verdict with WATCH_HEAL=1 heals on the first tick and comes back serving: rank 1 launched
        first, rank 0, /v1/models ready, the slot line proven. The Python pair's containers persist, so docker
        inspect is its witness; the Zig pair's witness is the launcher's record (the repro above)."""
        r0, r1 = self.rig.py_names()
        self.rig.container(r0, running="false", status="exited", exit_code="70", finished_age_s=60)
        self.rig.container(r1, running="false", status="exited", exit_code="70", finished_age_s=60)
        self.rig.log_line(r0, "[tensorfold] serving: 4 slot(s) x 300000 tokens; health strict")

        proc = self.rig.tick("python", WATCH_HEAL="1")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 1: rank 0 false (fail-fast exit 70)" in line for line in lines),
                        f"the engine's 70 is definitive: {lines}")
        self.assertTrue(any("restarting both ranks" in line for line in lines), f"the first tick launches the heal: {lines}")
        heal_log = self.rig.wait_for_state_file_containing("heal.log", "serving on :8199")
        self.assertIn("request slots: 4", heal_log, f"the heal proves the slot line before it calls the pair serving: {heal_log}")
        self.assertIn("serving on :8199", heal_log, f"the heal brings the pair back ready: {heal_log}")
        self.assertNotEqual(self.rig.state_file("last_heal"), "", "the heal stamps last_heal at its start")
        self.assertEqual(self.rig.state_file("fails"), "0", "the heal clears the bad-tick streak")


class TestStandDowns(ShellCase):
    def test_fresh_lease_stands_the_tick_down_and_resets_fails(self) -> None:
        """A lease touched during a soak or a maintenance stand stands the watchdog down mid-death."""
        r0, r1 = self.rig.py_names()
        self.rig.container(r0, running="false", exit_code="1", finished_age_s=120)
        self.rig.container(r1, running="false", exit_code="1", finished_age_s=120)
        self.rig.state_file("fails", "2")
        lease = self.rig.lease()

        proc = self.rig.tick("python")

        self.assertEqual(self.rig.stdout_lines(proc), [f"[dsv41-tf] watch: {lease} is fresh; standing down"])
        self.assertEqual(self.rig.state_file("fails"), "0", "standing down clears the bad-tick streak")
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}{proc.stderr}")

    def test_pair_down_with_no_witness_and_no_marker_stands_down(self) -> None:
        """Both ranks absent, no exit record, no stop marker: the pair never started on this node, so the tick
        stands down. The marker owns the human teardown, and only the human teardown."""
        r0, r1 = self.rig.py_names()
        self.rig.absent(r0, r1)

        proc = self.rig.tick("python")

        self.assertEqual(self.rig.stdout_lines(proc), [PY_DOWN_NEVER], f"{proc.stdout}{proc.stderr}")
        self.assertEqual(self.rig.state_file("fails"), "0", "standing down clears the bad-tick streak")
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}{proc.stderr}")


class TestDefinitiveAndDebounced(ShellCase):
    def test_exit70_older_than_the_last_heal_is_debounced(self) -> None:
        """A 70 witness the last heal already answers for is not a fresh report: it rides WATCH_FAILS, it does not
        heal on its own first tick."""
        r0, r1 = self.rig.py_names()
        self.rig.container(r0, running="false", status="exited", exit_code="70", finished_age_s=300)
        self.rig.container(r1, running="false", status="exited", exit_code="1", finished_age_s=300)
        self.rig.state_file("last_heal", str(int(time.time()) - 100))

        proc = self.rig.tick("python")
        lines = self.rig.stdout_lines(proc)

        self.assertTrue(any("watch: bad tick 1 of 3: rank 0 false" in line for line in lines),
                        f"a witness older than the last heal is the actor's inference, not the engine's report: {lines}")
        self.assertFalse(any("(fail-fast exit 70)" in line for line in lines),
                         f"the stale witness must not claim the fast heal: {lines}")


class TestPythonRecipeIdentity(ShellCase):
    def test_python_recipe_tick_verdicts_match_today(self) -> None:
        """The Python recipe's tick, line for line: scripts/serve/python.sh moved today's pair internals behind the
        adapter contract, and moved nothing a log line shows. The table below is HEAD's behavior written out."""

        def healthy(r):
            r0, r1 = r.py_names()
            r.container(r0); r.container(r1)

        def rank1_absent(r):
            r0, r1 = r.py_names()
            r.container(r0); r.absent(r1)

        def pair_down_older_boot(r):
            r0, r1 = r.py_names()
            r.absent(r0, r1)
            r.marker("boot-from-the-previous-encarnation", 1_700_000_000)

        def pair_down_this_boot(r):
            r0, r1 = r.py_names()
            r.absent(r0, r1)
            r.marker(r.boot, time.time() - 10)

        def health_503_plain(r):
            r0, r1 = r.py_names()
            r.container(r0); r.container(r1)
            r.health("503", UNAVAILABLE_BODY)

        def health_000_young(r):
            r0, r1 = r.py_names()
            r.container(r0, age_s=3); r.container(r1)
            r.health("000", "")

        def rank0_exit70(r):
            r0, r1 = r.py_names()
            r.container(r0, running="false", status="exited", exit_code="70", finished_age_s=60)
            r.container(r1, running="false", status="exited", exit_code="70", finished_age_s=60)

        def rank0_exit1(r):
            r0, r1 = r.py_names()
            r.container(r0, running="false", status="exited", exit_code="1", finished_age_s=60)
            r.container(r1)

        def worker_unreachable(r):
            r0, r1 = r.py_names()
            r.container(r0)
            r.ssh_rc(255)

        def bad_tick(bad):
            return [f"[dsv41-tf] watch: bad tick 1 of 3: {bad}"]

        table = {
            "healthy": (healthy, lambda r: ([], "0", 0)),
            "rank1_absent": (rank1_absent, lambda r: (bad_tick("rank 1 absent"), "1", 0)),
            "pair_down_marker_older_boot": (pair_down_older_boot, lambda r: (
                bad_tick(f"both ranks absent after a reboot (stopped before it: {r.state}/stopped)"), "1", 0)),
            "pair_down_marker_this_boot": (pair_down_this_boot, lambda r: ([PY_DOWN_STOPPED], "0", 0)),
            "health_503_plain": (health_503_plain, lambda r: (bad_tick(f"/health 503 {UNAVAILABLE_BODY}"), "1", 0)),
            "health_000_young": (health_000_young, lambda r: (["[dsv41-tf] watch: loading (3s)"], "0", 0)),
            "rank0_exit70": (rank0_exit70, lambda r: (
                ["[dsv41-tf] watch: bad tick 1 of 1: rank 0 false (fail-fast exit 70)",
                 "[dsv41-tf] ALERT: unhealthy (rank 0 false (fail-fast exit 70)); WATCH_HEAL=0, not restarting"],
                "1", 1)),
            "rank0_exit1": (rank0_exit1, lambda r: (bad_tick("rank 0 false"), "1", 0)),
            "worker_unreachable": (worker_unreachable, lambda r: (
                ["[dsv41-tf] watch: worker unreachable over ssh (not counted)"], "0", 0)),
        }

        for state, (setup, expect) in table.items():
            with self.subTest(state=state):
                rig = self.new_rig()
                setup(rig)
                proc = rig.tick("python")
                want_lines, want_fails, want_rc = expect(rig)

                self.assertEqual(rig.stdout_lines(proc), want_lines, f"{state} ({proc.stderr})")
                self.assertEqual(rig.state_file("fails"), want_fails, f"{state}: fails file")
                self.assertEqual(proc.returncode, want_rc, f"{state}: rc ({proc.stdout}{proc.stderr})")


class TestBootStart(ShellCase):
    def test_boot_start_takes_its_state_dir_and_lease_from_the_config(self) -> None:
        """The boot fast path reads STATE_DIR and BOOT_LEASE from the CONFIG it is given (it hardcoded the
        dsv41-tf state dir before, so a second profile read the stop marker from the wrong directory): the lease
        that config names stands it down, and its log lands in the state dir that config names."""
        lease = self.rig.state / "lease"
        lease.touch()
        self.rig.write_config("python", BOOT_LEASE=str(lease), BOOT_LEASE_MIN="20")

        proc = self.rig.run(self.rig.boot_start)
        log = self.rig.state_file("boot-start.log")

        self.assertEqual(proc.returncode, 0, f"the lease path is a clean stand-down ({proc.stdout}{proc.stderr})")
        self.assertIn(f"{lease} is fresh: not starting", log)
        self.assertIn("the watchdog takes over when it ages", log)

    def test_boot_start_gates_then_asks_the_recipe_for_the_decision(self) -> None:
        """The gate (docker, the worker's docker, RoCE ACTIVE) and the boot decision are serve.sh's
        nodes_ready and the recipe's pair_boot_decision: a pair stopped before a reboot reads as a start, and
        BOOT_DRY_RUN=1 stops it there."""
        r0, r1 = self.rig.py_names()
        self.rig.absent(r0, r1)
        self.rig.marker("boot-from-the-previous-encarnation", 1_700_000_000)
        self.rig.make_roce_active()
        self.rig.write_config("python")

        proc = self.rig.run(self.rig.boot_start, env_extra={"BOOT_DRY_RUN": "1", "BOOT_WAIT": "5"})
        log = self.rig.state_file("boot-start.log")

        self.assertEqual(proc.returncode, 0, f"the dry run answers 0 ({proc.stdout}{proc.stderr})")
        self.assertIn(f"rank 0 absent, stopped before the reboot (marker {self.rig.state}/stopped): starting", log)
        self.assertIn("BOOT_DRY_RUN=1: not starting", log)
        self.assertNotIn("not ready after", log, f"the fixture gate passes: {log}")


class TestConfigBoundary(ShellCase):
    def test_recipe_key_is_validated_at_the_config_boundary(self) -> None:
        """RECIPE is validated once, at the config boundary: a value outside python|zig exits 2 with a config
        error, and no pair work starts."""
        self.rig.write_config("python", RECIPE="bogus")

        proc = self.rig.run(self.rig.serve, "watch", "--once")

        self.assertEqual(proc.returncode, 2, f"the config boundary answers 2 ({proc.stdout}{proc.stderr})")
        self.assertIn("RECIPE=bogus", proc.stderr)
        self.assertIn("python or zig", proc.stderr)
        self.assertEqual(self.rig.state_file("fails"), "", "a config error does not touch the tick's state")

    def test_a_config_without_the_recipe_key_is_the_python_pair(self) -> None:
        """config/prod.env names no RECIPE: the default is python, so the Python pair needs no new keys."""
        r0, r1 = self.rig.py_names()
        self.rig.container(r0); self.rig.container(r1)
        self.rig.write_config("python")
        self.rig.config.write_text("".join(f"{line}\n" for line in self.rig.config.read_text().splitlines()
                                           if not line.startswith("RECIPE=")))

        proc = self.rig.run(self.rig.serve, "watch", "--once")

        self.assertEqual(self.rig.stdout_lines(proc), [], f"{proc.stdout}{proc.stderr}")
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
