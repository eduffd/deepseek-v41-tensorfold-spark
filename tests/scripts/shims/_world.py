#!/usr/bin/env python3
"""One answer for the docker / ssh / curl PATH shims, read from the fixture's world.json.

The fixture is the simulated pair: containers keyed by name (absent = not a key), images, per-container log bodies,
http bodies keyed by path, die events keyed by container name, and the return codes of docker run / docker info / ssh.
A shim never invents state: what serve.sh sees is exactly what the fixture says. `docker run` with a --name adds that
running container to the world and holds it for `docker_run_hold_s` seconds (0: return at once), which is how a launched
rank becomes visible to `docker ps` / `docker inspect`; a non-zero docker_run_rc answers at once, the way a rank that
dies immediately does. Writes take a lock: a launch runs rank 1 and rank 0 side by side.
"""

import fcntl
import json
import os
import re
import sys
import time


def load(path):
    with open(path) as f:
        return json.load(f)


def mutate(path, change):
    """Read-modify-write on world.json, serialized: two ranks launch at the same moment."""
    with open(path + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = load(path)
        result = change(data)
        with open(path, "w") as f:
            json.dump(data, f)
    return result


def iso(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + ".000000Z"


def matched(data, filters):
    """Containers whose name matches any `name=<regex>` filter (docker's filter is a regex, and `^name` anchors)."""
    names = [f[5:] for f in filters if f.startswith("name=")]
    return [n for n in sorted(data["containers"]) if not names or any(re.search(p, n) for p in names)]


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
        args = sys.argv[3:]
        name = args[args.index("--name") + 1] if "--name" in args else ""
        rc = int(world.get("docker_run_rc", 0))
        print("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        if rc:
            return rc
        if name:
            # A launch boots a fresh container, so its log starts fresh: a capture written before a force stop keeps
            # the wedged pair's own lines, and this is the line the relaunched pair answers afterwards.
            fresh = "[tensorfold] serving: 4 slot(s) x 1048576 tokens; fresh-boot"
            mutate(path, lambda d: (d["containers"].__setitem__(name, {
                "running": "true", "exit_code": "0", "status": "running",
                "started_at": iso(time.time()), "finished_at": "0001-01-01T00:00:00Z",
                "id": "shim-container-id"}), d["logs"].__setitem__(name, fresh)))
            hold = int(world.get("docker_run_hold_s", 0))
            if hold:
                time.sleep(hold)
                if "--rm" in args:
                    mutate(path, lambda d: d["containers"].pop(name, None))
        return 0

    if op == "info":
        node = "worker_docker_info_rc" if os.environ.get("NODE_ROLE") == "worker" else "docker_info_rc"
        print("shim docker info")
        return int(world.get(node, world.get("docker_info_rc", 0)))

    if op in ("rm", "stop"):
        # --rm containers leave on their own, and `docker rm -f` removes what is left: the world drops them.
        def drop(d, names=sys.argv[3:]):
            for n in names:
                d["containers"].pop(n, None)
        mutate(path, drop)
        key = "docker_rm_rc" if op == "rm" else "docker_stop_rc"
        return int(world.get(key, 0))

    if op == "ps":
        quiet = sys.argv[3] == "1"
        for n in matched(world, sys.argv[4:]):
            c = world["containers"][n]
            if quiet:
                print(c.get("id", "shim-container-id"))
            elif c.get("running") == "true":
                print(f"{n} Up (shim)")
            else:
                print(f"{n} Exited ({c.get('exit_code')}) (shim)")
        return 0

    if op == "events":
        # docker events --format '{{.Time}} {{.Attributes.exitCode}}' with --filter event=die --filter container=<name>
        want = [f[10:] for f in sys.argv[4:] if f.startswith("container=")]
        for n, ev in sorted((world.get("die_events") or {}).items()):
            if want and n not in want:
                continue
            print(f"{ev['time']} {ev['code']}")
        return 0

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
