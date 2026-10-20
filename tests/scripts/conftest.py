"""CPU tests of the shell layer: scripts/serve.sh's watchdog tick, its recipe adapters, scripts/run-zig.sh.

    python3 -m pytest -q tests/scripts
    python3 -m unittest discover -s tests/scripts -t tests/scripts -p 'test_*.py'   # the same suite, no pytest

No GPU, no docker, no ssh, no root, no network: a test builds a fixture dir (world.json plus a bin/ of PATH shims for
docker, ssh and curl, and for the Linux-only flock / setsid / stat -c / date -d) and runs the real script against it,
asserting the log lines and the state files a maintainer reads. The shims answer only from world.json; portability is
the harness's job, never the scripts'. tests/scripts/shims/_world.py is the fixture schema.
"""
from __future__ import annotations

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

IMAGE_PY = "test-image:1"
IMAGE_ZIG = "test-zig-image:1"

HEALTH_OK = '{"ok":true,"rounds_total":383335,"inflight":0,"queued":0}'

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
        self.state = root / "state"
        self.state.mkdir()
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
                                  "layers": "sha256:layera sha256:layerb"}},
            "logs": {},
            "http": {"/health": {"code": "200", "body": HEALTH_OK},
                     "/v1/models": {"code": "200", "body": '{"data":[{"id":"Test-Model"}]}'}},
            "docker_run_rc": 0, "docker_info_rc": 0, "worker_docker_info_rc": 0, "ssh_rc": 0,
        }))
        data.update(patch)
        self.world_path.write_text(json.dumps(data))
        return data

    def get_world(self) -> dict:
        return json.loads(self.world_path.read_text())

    def _write_world(self, data: dict) -> None:
        self.world_path.write_text(json.dumps(data))

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

    # -- the state dir the scripts share ------------------------------------------------------

    def state_file(self, name: str, content: str | None = None) -> str:
        """One of the state dir's one-line dialect files (fails, last_heal, stopped, exit-r<rank>, heal.log).
        Written with the newline `echo` gives, because serve.sh reads them with `read`."""
        path = self.state / name
        if content is None:
            return path.read_text().strip() if path.exists() else ""
        path.write_text(content + "\n")
        return content

    def marker(self, boot: str, epoch: float) -> None:
        """scripts/serve.sh's stop marker: '<boot_id> <epoch>' (mark_stopped's format)."""
        self.state_file("stopped", f"{boot} {int(epoch)}")

    def exit_record(self, rank: int, code: str, epoch: float, boot: str | None = None) -> None:
        """The launcher's exit witness for a rank: '<code> <epoch> <boot_id>' (scripts/run-zig.sh writes it)."""
        self.state_file(f"exit-r{rank}", f"{code} {int(epoch)} {boot or self.boot}")

    def lease(self) -> Path:
        path = self.state / "lease"
        path.touch()
        return path

    # -- the config and the run -------------------------------------------------------------

    def write_config(self, recipe: str = "python", **overrides) -> Path:
        keys = dict(PY_KEYS if recipe == "python" else ZIG_KEYS)
        keys["STATE_DIR"] = str(self.state)
        keys["ROCE_STATE"] = str(self.root / "roce")
        keys["WATCH_LEASE"] = str(self.state / "lease")
        for node in ("head", "worker"):
            for kind in ("model", "engram", "prepared", "state"):
                keys[f"{node.upper()}_{kind.upper()}"] = str(self.root / node / kind)
        if recipe == "zig":
            keys.update({"MODEL_DIR": str(self.root / "model"), "ENGRAM_DIR": str(self.root / "engram"),
                         "ASSETS_DIR": str(self.root / "assets"), "PREPARED_DIR": str(self.root / "prepared"),
                         "CACHE_DIR": str(self.root / "cache")})
        for key, value in overrides.items():
            keys[key] = str(value)
        self.config.write_text("".join(f"{key}={value}\n" for key, value in keys.items()))
        return self.config

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

    def make_roce_active(self) -> None:
        """The sysfs tree nodes_ready reads: every NCCL_IB_HCA device's port 1 ACTIVE, under ROCE_STATE."""
        for dev in PY_KEYS["NCCL_IB_HCA"].split(","):
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

    def wait_for_state_file_containing(self, name: str, needle: str, timeout: float = 60.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.state_file(name)
            if needle in text:
                return text
            time.sleep(0.25)
        return self.state_file(name)
