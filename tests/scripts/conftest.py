"""CPU tests of the shell layer: scripts/serve.sh's watchdog tick, its recipe adapters, scripts/run-zig.sh.

    python3 -m pytest -q tests/scripts
    python3 -m unittest discover -s tests/scripts -t tests/scripts -p 'test_*.py'   # the same suite, no pytest

No GPU, no docker, no ssh, no root, no network: a test builds a fixture dir (world.json plus a bin/ of PATH shims for
docker, ssh and curl, and for the Linux-only flock / setsid / stat -c / date -d) and runs the real script against it,
asserting the log lines and the state files a maintainer reads. The shims answer only from world.json; portability is
the harness's job, never the scripts'. tests/scripts/shims/_world.py is the fixture schema.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIM_DIR = Path(__file__).parent / "shims"
SERVE_SH = ROOT / "scripts" / "serve.sh"
RUN_ZIG_SH = ROOT / "scripts" / "run-zig.sh"
EXAMPLE_ZIG = ROOT / "config" / "prod-zig.env.example"
EXAMPLE_FLASH17 = ROOT / "config" / "prod-zig-flash17.env.example"

IMAGE_PY = "test-image:1"
IMAGE_ZIG = "test-zig-image:1"

HEALTH_OK = '{"ok":true,"rounds_total":383335,"inflight":0,"queued":0}'

# rank 0's boot line, in the form engine/zig/zig/src/server/serve.zig prints it (serve.sh's slot check reads it).
ZIG_SLOT_LINE = ("[tensorfold] serving: 4 slot(s) x 1048576 tokens; wire spark; health strict; "
                 "request log /state/requests.jsonl; disconnect on")

PY_KEYS = {
    "RECIPE": "python",
    "NAME": "dsv41-tf-test",
    "IMAGE": IMAGE_PY,
    "WORKER_SSH": "worker@127.0.0.1",
    "HEAD_IP": "10.1.0.1",
    "NCCL_SOCKET_IFNAME": "lo0",
    "NCCL_IB_HCA": "fakea,fakeb",
    "MASTER_PORT": "29599",
    "HEAD_MODEL": "/fixture/head/model",
    "WORKER_MODEL": "/fixture/worker/model",
    "HEAD_ENGRAM": "/fixture/head/engram",
    "WORKER_ENGRAM": "/fixture/worker/engram",
    "HEAD_PREPARED": "/fixture/head/prepared",
    "WORKER_PREPARED": "/fixture/worker/prepared",
    "HEAD_STATE": "/fixture/head/state",
    "WORKER_STATE": "/fixture/worker/state",
    "CACHE_VOL": "test-cache",
    "SERVED_NAME": "Test-Model",
    "PORT": "8199",
    "PARALLEL": "4",
    "CONTEXT": "4096",
    "CANARY": "off",
    "DROP_CACHES": "0",
    "MEM_GATE_GIB": "0",
    "PREFLIGHT": "off",
    "READY_TIMEOUT": "15",
    "WATCH_HEAL": "0",
    "WATCH_FAILS": "3",
    "WATCH_GRACE": "1800",
    "WATCH_MIN_HEAL": "1800",
    "WATCH_FF_MIN_HEAL": "120",
    "WATCH_LEASE_MIN": "20",
}

ZIG_KEYS = {
    "RECIPE": "zig",
    "NAME": "tensorfold-zig",
    "IMAGE": IMAGE_ZIG,
    "WORKER_SSH": "worker@127.0.0.1",
    "TF_TP_MASTER": "10.1.0.1",
    "TF_TP_PORT": "29599",
    "TF_COMM_BACKEND": "roce",
    "NCCL_SOCKET_IFNAME": "lo0",
    "NCCL_IB_HCA": "fakea,fakeb",
    "HOST": "localhost",
    "PORT": "8199",
    "TF_DSV41_CONTEXT": "1024",
    "TF_DSV41_SLOTS": "4",
    "TF_DSV41_ASSETS": "/assets",
    "CANARY": "off",
    "DROP_CACHES": "0",
    "MEM_GATE_GIB": "0",
    "PREFLIGHT": "off",
    "READY_TIMEOUT": "15",
    "WATCH_HEAL": "1",
    "WATCH_FAILS": "3",
    "WATCH_GRACE": "1800",
    "WATCH_MIN_HEAL": "1800",
    "WATCH_FF_MIN_HEAL": "120",
    "WATCH_LEASE_MIN": "20",
}


def boot_id() -> str:
    """The same answer scripts/serve.sh's boot_id() gives: the kernel's id, or 'unknown' off Linux."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


def iso(epoch: float) -> str:
    """A docker-shaped UTC timestamp (the tick reads StartedAt / FinishedAt through `date -d`)."""
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class Rig:
    """One test's world: a fixture dir, its PATH shims, the config the script reads, and the state dir it shares."""

    def __init__(self, root: Path):
        self.root = root
        self.serve = SERVE_SH
        self.run_zig = RUN_ZIG_SH
        self.boot_start = ROOT / "scripts" / "boot-start.sh"
        self.boot = boot_id()
        # The pair's two nodes reboot independently, so the worker has a boot id of its own, and its own state dir:
        # rank 1's exit record, durable log and supervisor pid land there (config/prod-zig.env on the worker names
        # them, and the head reads them over ssh). One filesystem can answer two nodes, one boot id cannot.
        self.worker_boot = f"{self.boot}-worker"
        self.state = root / "state"
        self.state.mkdir()
        self.worker_state = root / "worker-state"
        self.worker_state.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        for shim in sorted(SHIM_DIR.iterdir()):
            if not shim.name.startswith("_"):
                (self.bin / shim.name).symlink_to(shim.resolve())
        self.world_path = root / "world.json"
        self.config = root / "config.env"
        self.world()
        self.write_config("python")

    # -- the simulated pair -------------------------------------------------------------------

    def world(self, **patch):
        data = json.loads(json.dumps({
            "containers": {},
            "images": {IMAGE_PY: {"id": "sha256:img", "created": "2026-01-01T00:00:00.000000000Z",
                                  "layers": "sha256:layera sha256:layerb"},
                       IMAGE_ZIG: {"id": "sha256:zigimg", "created": "2026-01-02T00:00:00.000000000Z",
                                   "layers": "sha256:ziglayera sha256:ziglayerb"}},
            "logs": {f"{ZIG_KEYS['NAME']}-r0": ZIG_SLOT_LINE},
            "die_events": {},
            "run_calls": [],
            "events_calls": [],
            "worker_boot_id": self.worker_boot,
            "http": {"/health": {"code": "200", "body": HEALTH_OK},
                     "/v1/models": {"code": "200", "body": '{"data":[{"id":"Test-Model"}]}'}},
            "docker_run_rc": 0, "docker_run_hold_s": 30, "docker_info_rc": 0, "worker_docker_info_rc": 0, "ssh_rc": 0,
        }))
        data.update(patch)
        self._write_world(data)
        return data

    def _world_lock(self, mode: int):
        """world.json.lock, the same lock the shims take: a shim's write truncates the file, so a reader that lands in
        the middle of one reads an empty world. Tests read and write the world the shims are writing."""
        lk = open(str(self.world_path) + ".lock", "a+")
        fcntl.flock(lk, mode)
        return lk

    def get_world(self) -> dict:
        lk = self._world_lock(fcntl.LOCK_SH)
        try:
            return json.loads(self.world_path.read_text())
        finally:
            lk.close()

    def _write_world(self, data: dict) -> None:
        lk = self._world_lock(fcntl.LOCK_EX)
        try:
            self.world_path.write_text(json.dumps(data))
        finally:
            lk.close()

    def container(self, name: str, *, running="true", exit_code="0", status="running",
                  age_s: float = 300.0, finished_age_s: float | None = None) -> str:
        """A rank's container as docker inspect answers it: running true/false, its exit code, its timestamps."""
        now = time.time()
        data = self.get_world()
        data["containers"][name] = {
            "running": running,
            "exit_code": exit_code,
            "status": status,
            "started_at": iso(now - age_s),
            "finished_at": iso(now - finished_age_s) if finished_age_s is not None else "0001-01-01T00:00:00Z",
            "id": "shim-container-id",
        }
        self._write_world(data)
        return name

    def absent(self, *names: str) -> None:
        data = self.get_world()
        for name in names:
            data["containers"].pop(name, None)
        self._write_world(data)

    def health(self, code: str, body: str) -> None:
        data = self.get_world()
        data["http"]["/health"] = {"code": code, "body": body}
        self._write_world(data)

    def ssh_rc(self, rc: int) -> None:
        data = self.get_world()
        data["ssh_rc"] = rc
        self._write_world(data)

    def log_line(self, name: str, line: str) -> None:
        data = self.get_world()
        data["logs"][name] = line
        self._write_world(data)

    def run_rc(self, rc: int) -> None:
        data = self.get_world()
        data["docker_run_rc"] = rc
        self._write_world(data)

    def worker_rm_rc(self, rc: int) -> None:
        """The worker's docker answers a `docker rm -f` with this return code. Its own key, because a node's daemon
        fails on its own: a stop whose worker refuses the removal leaves rank 1 running."""
        data = self.get_world()
        data["worker_docker_rm_rc"] = rc
        self._write_world(data)

    def run_hold_s(self, seconds: float) -> None:
        """How long a launched rank's container stays up in the world before the shim's `docker run` returns."""
        data = self.get_world()
        data["docker_run_hold_s"] = seconds
        self._write_world(data)

    def die_event(self, name: str, code: str, age_s: float) -> None:
        """The die event `docker events` answers for a container with no cooperating launcher, at `age_s` seconds
        ago. `{{.Time}}` prints epoch seconds, so that is what the fixture stores."""
        data = self.get_world()
        data.setdefault("die_events", {})[name] = {"code": code, "time": int(time.time() - age_s)}
        self._write_world(data)

    def completions(self, code: str, body: str) -> None:
        """The /v1/chat/completions answer the watchdog's probe reads."""
        data = self.get_world()
        data["http"]["/v1/chat/completions"] = {"code": code, "body": body}
        self._write_world(data)

    # -- the state dir the scripts share ------------------------------------------------------

    def state_file(self, name: str, content: str | None = None) -> str:
        """One of the head state dir's one-line dialect files (fails, last_heal, stopped, exit-r0, heal.log). Rank 1's
        record and pid file live in the worker's state dir (worker_state_file). Written with the newline `echo`
        gives, because serve.sh reads them with `read`."""
        path = self.state / name
        if content is None:
            return path.read_text().strip() if path.exists() else ""
        path.write_text(content + "\n")
        return content

    def marker(self, boot: str, epoch: float) -> None:
        """scripts/serve.sh's stop marker: '<boot_id> <epoch>' (mark_stopped's format)."""
        self.state_file("stopped", f"{boot} {int(epoch)}")

    def exit_record(self, rank: int, code: str, epoch: float, boot: str | None = None) -> None:
        """The launcher's exit witness for a rank: '<code> <epoch> <boot_id>' (scripts/run-zig.sh writes it), in the
        state dir of the node the rank ran on, stamped with that node's boot id."""
        node_state, node_boot = (self.worker_state, self.worker_boot) if rank == 1 else (self.state, self.boot)
        (node_state / f"exit-r{rank}").write_text(f"{code} {int(epoch)} {boot or node_boot}\n")

    def worker_state_file(self, name: str) -> str:
        path = self.worker_state / name
        return path.read_text().strip() if path.exists() else ""

    def lease(self) -> Path:
        path = self.state / "lease"
        path.touch()
        return path

    # -- the config and the run -------------------------------------------------------------

    # A fill beginning with "=" is a literal value; the rest are fixture paths, relative to the fixture root
    # ("." is the root itself, which is where the worker's copy of the checkout sits).
    EXAMPLE_FILL = {
        "ZIG_COMMIT": "=deadbeef",
        "IMAGE": f"={IMAGE_ZIG}",
        "TF_TP_MASTER": "=10.1.0.1",
        "NCCL_SOCKET_IFNAME": "=lo0",
        "NCCL_IB_HCA": "=fakea,fakeb",
        "WORKER_SSH": "=worker@127.0.0.1",
        "MODEL_DIR": "model",
        "ENGRAM_DIR": "engram",
        "ASSETS_DIR": "assets",
        "PREPARED_DIR": "prepared",
        "CACHE_DIR": "cache",
        "STATE_DIR": "state",
        "WORKER_REPO": ".",
        "WATCH_LEASE": "state/lease",
        "BOOT_LEASE": "state/lease",
    }

    def write_config(self, recipe: str = "python", **overrides) -> Path:
        keys = dict(PY_KEYS if recipe == "python" else ZIG_KEYS)
        keys["STATE_DIR"] = str(self.state)
        keys["ROCE_STATE"] = str(self.root / "roce")
        keys["WATCH_LEASE"] = str(self.state / "lease")
        if recipe == "python":
            # The Python recipe's config names a head and a worker copy of every pair path. The Zig recipe names
            # each node's paths in that node's own config, so a Zig config gets no head/worker keys at all: that is
            # what the shipped config/prod-zig.env.example names, and the scripts must answer it.
            for node in ("head", "worker"):
                for kind in ("model", "engram", "prepared", "state"):
                    keys[f"{node.upper()}_{kind.upper()}"] = str(self.root / node / kind)
        if recipe == "zig":
            keys.update({"MODEL_DIR": str(self.root / "model"), "ENGRAM_DIR": str(self.root / "engram"),
                         "ASSETS_DIR": str(self.root / "assets"), "PREPARED_DIR": str(self.root / "prepared"),
                         "CACHE_DIR": str(self.root / "cache"), "WORKER_REPO": str(self.root)})
        for key, value in overrides.items():
            keys[key] = str(value)
        self.config.write_text("".join(f"{key}={value}\n" for key, value in keys.items()))
        if recipe == "zig":
            self.make_zig_dirs()
            self.write_worker_config()
        return self.config

    def write_worker_config(self) -> None:
        """The worker's copy of the checkout and of config/prod-zig.env, so a head-side launch can
        `cd $WORKER_REPO && scripts/run-zig.sh 1 --detached` exactly as it does on the nodes. The pair names each
        node's paths in that node's own config, so the copy differs from the head's where the paths differ: the
        state dir, which is where rank 1's exit record, durable log and supervisor pid land."""
        text = "".join(f"{line}\n" for line in self.config.read_text().splitlines()
                       if not line.startswith("STATE_DIR="))
        (self.root / "config").mkdir(exist_ok=True)
        (self.root / "config" / "prod-zig.env").write_text(text + f"STATE_DIR={self.worker_state}\n")
        if not (self.root / "scripts").exists():
            (self.root / "scripts").symlink_to(ROOT / "scripts")

    def example_keys(self, example: Path = EXAMPLE_ZIG) -> dict:
        """A shipped Zig example config as written, every placeholder filled with a fixture path: what an operator
        gets from `cp config/prod-zig.env.example config/prod-zig.env`. Only the test knobs (fixture ports, no
        canary, no memory gate, short timeouts, the context the fixture's rope.bin covers) are added, so a script
        that reads a key the example does not name fails here, not on a node."""
        keys = {}
        for raw in example.read_text().splitlines():
            line = raw.strip().removeprefix("export ")
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.split("#")[0].strip().strip('"')
            if not value:
                continue
            if value.startswith("<"):
                if key not in self.EXAMPLE_FILL:
                    raise AssertionError(f"{example.name} names a new placeholder for {key}: "
                                         f"name its fixture value in Rig.EXAMPLE_FILL")
                rel = self.EXAMPLE_FILL[key]
                value = rel[1:] if rel.startswith("=") else str(self.root if rel == "." else self.root / rel)
            keys[key] = value
        return keys

    def write_example_config(self, example: Path = EXAMPLE_ZIG, **overrides) -> Path:
        keys = self.example_keys(example)
        keys.update({"PORT": "8199", "CANARY": "off", "DROP_CACHES": "0", "MEM_GATE_GIB": "0",
                     "READY_TIMEOUT": "15", "TF_DSV41_CONTEXT": "1024", "TF_DSV41_SLOTS": "4",
                     "ROCE_STATE": str(self.root / "roce"), "WATCH_HEAL": "1",
                     "WATCH_MIN_HEAL": "0", "WATCH_FF_MIN_HEAL": "0"})
        keys.update({key: str(value) for key, value in overrides.items()})
        self.config.write_text("".join(f"{key}={value}\n" for key, value in keys.items()))
        self.make_zig_dirs()
        self.make_pair_files()
        self.write_worker_config()
        return self.config

    def make_pair_files(self) -> None:
        """What preflight tests for on a node: the model's tokenizer files, this rank's Engram shards, and a
        prepared rank folder. The fixture keeps one tree, so it carries both ranks' files."""
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.root / "model" / name).write_text("{}")
        for rank in (0, 1):
            for kind in ("l1", "l14"):
                (self.root / "engram" / f"engram-{kind}-r{rank}of2.bin").write_bytes(b"\0")
            (self.root / "prepared" / "fill" / "shape" / f"rank{rank}").mkdir(parents=True, exist_ok=True)

    def make_zig_dirs(self) -> None:
        """The folders and assets scripts/run-zig.sh checks before it launches a rank, with a rope.bin big enough."""
        import struct

        for name in ("model", "engram", "assets", "prepared", "cache"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "assets" / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<I", 1024 + 2048))
        (self.root / "assets" / "engram-host.bin").write_bytes(b"\0")
        (self.root / "assets" / "aot").mkdir(exist_ok=True)

    def env(self) -> dict:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        env["CONFIG"] = str(self.config)
        env["FIXTURE"] = str(self.world_path)
        env["SHIM_DIR"] = str(SHIM_DIR)
        return env

    def run(self, script: Path, *args: str, env_extra: dict | None = None,
            timeout: float = 90.0) -> subprocess.CompletedProcess:
        env = self.env()
        env.update({key: str(value) for key, value in (env_extra or {}).items()})
        return subprocess.run(["bash", str(script), *args], cwd=str(ROOT), env=env,
                              capture_output=True, text=True, timeout=timeout)

    def sh(self, line: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        """A shell line run with the fixture's PATH shims and FIXTURE, which is how the scripts run it."""
        return subprocess.run(["bash", "-c", line], cwd=str(ROOT), env=self.env(),
                              capture_output=True, text=True, timeout=timeout)

    def run_as_worker(self, script: Path, *args: str, env_extra: dict | None = None,
                      timeout: float = 90.0) -> subprocess.CompletedProcess:
        """Run a script the way the worker runs it: in the worker's checkout, as the worker node role, on the config
        the worker's launcher reads (the copy at config/prod-zig.env, which names the worker's state dir)."""
        env = self.env()
        env["NODE_ROLE"] = "worker"
        env["CONFIG"] = str(self.root / "config" / "prod-zig.env")
        env.update({key: str(value) for key, value in (env_extra or {}).items()})
        return subprocess.run(["bash", str(script), *args], cwd=str(self.root), env=env,
                              capture_output=True, text=True, timeout=timeout)

    def run_calls(self) -> list[str]:
        """The `docker run` argument strings the shim has been asked to run, in order."""
        return self.get_world().get("run_calls", [])

    def events_calls(self) -> list[str]:
        """The `docker events` argument strings the shim has been asked to run, in order: the window a witness read
        used is part of what the scripts emit, not a detail of the read."""
        return self.get_world().get("events_calls", [])

    def make_roce_active(self) -> None:
        """The sysfs tree nodes_ready reads: every NCCL_IB_HCA device the config names, port 1 ACTIVE, under
        ROCE_STATE. Read from the config, so the shipped example's device names work as well as the fixture's."""
        line = [l for l in self.config.read_text().splitlines() if l.startswith("NCCL_IB_HCA=")]
        devices = (line[-1].split("=", 1)[1] if line else PY_KEYS["NCCL_IB_HCA"]).split(",")
        for dev in devices:
            port = self.root / "roce" / dev / "ports" / "1"
            port.mkdir(parents=True, exist_ok=True)
            (port / "state").write_text("0: ACTIVE\n")

    def tick(self, recipe: str = "python", **overrides) -> subprocess.CompletedProcess:
        self.write_config(recipe, **overrides)
        return self.run(self.serve, "watch", "--once")

    def stdout_lines(self, proc: subprocess.CompletedProcess) -> list[str]:
        return proc.stdout.splitlines()

    def py_names(self) -> tuple[str, str]:
        return f"{PY_KEYS['NAME']}-r0", f"{PY_KEYS['NAME']}-r1"

    def zig_names(self) -> tuple[str, str]:
        return f"{ZIG_KEYS['NAME']}-r0", f"{ZIG_KEYS['NAME']}-r1"

    def heal_dirs(self) -> list[Path]:
        """The evidence folders a heal wrote: newest first, as `ls -1dt` orders them."""
        dirs = sorted((p for p in self.state.glob("heal-*") if p.is_dir()), reverse=True)
        return dirs

    def wait_for_state_file_containing(self, name: str, needle: str, timeout: float = 60.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.state_file(name)
            if needle in text:
                return text
            time.sleep(0.25)
        return self.state_file(name)
