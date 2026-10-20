#!/usr/bin/env python3
"""One answer for the docker / ssh / curl PATH shims, read from the fixture's world.json.

The fixture is the simulated pair: containers keyed by name (absent = not a key), images, per-container log bodies,
http bodies keyed by path, and the return codes of docker run / docker info / ssh. A shim never invents state: what
serve.sh sees is exactly what the fixture says.
"""

import json
import os
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


FIELDS = (
    ("ExitCode", "exit_code"),
    ("FinishedAt", "finished_at"),
    ("StartedAt", "started_at"),
    ("Running", "running"),
    ("Status", "status"),
    (".Id", "id"),
    ("Created", "created"),
    ("RootFS", "layers"),
)


def main():
    op, path = sys.argv[1], sys.argv[2]
    world = load(path)

    if op == "inspect":
        fmt, name = sys.argv[3], sys.argv[4]
        c = world["containers"].get(name)
        if c is None:
            sys.stderr.write(f"Error: No such object: {name}\n")
            return 1
        if not fmt:
            return 0
        print(" ".join(str(c.get(key, "")) for token, key in FIELDS if token in fmt))
        return 0

    if op == "image":
        fmt, name = sys.argv[3], sys.argv[4]
        c = world["images"].get(name)
        if c is None:
            sys.stderr.write(f"Error: No such image: {name}\n")
            return 1
        if not fmt:
            return 0
        print(" ".join(str(c.get(key, "")) for token, key in FIELDS if token in fmt))
        return 0

    if op == "logs":
        body = world["logs"].get(sys.argv[3], "")
        sys.stdout.write(body + ("\n" if body else ""))
        return 0

    if op == "run":
        print("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        return int(world.get("docker_run_rc", 0))

    if op == "info":
        node = "worker_docker_info_rc" if os.environ.get("NODE_ROLE") == "worker" else "docker_info_rc"
        print("shim docker info")
        return int(world.get(node, world.get("docker_info_rc", 0)))

    if op == "rm":
        return int(world.get("docker_rm_rc", 0))

    if op == "stop":
        return int(world.get("docker_stop_rc", 0))

    if op == "ssh_rc":
        return int(world.get("ssh_rc", 0))

    if op == "http":
        url, wflag, fail = sys.argv[3], sys.argv[4], sys.argv[5] == "1"
        rest = url.split("://", 1)[-1]
        path_key = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
        rec = world["http"].get(path_key, {"code": "000", "body": ""})
        code, body = rec["code"], rec["body"]
        sys.stdout.write(body)
        if "%{http_code}" in wflag:
            sys.stdout.write("\n" + code)
        if code == "000":
            return 7
        if fail and int(code) >= 400:
            return 22
        return 0

    sys.stderr.write(f"shim: unhandled op {op}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
