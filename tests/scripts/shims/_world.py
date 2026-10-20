#!/usr/bin/env python3
"""One answer for the docker / ssh / curl PATH shims, read from the fixture's world.json.

The fixture is the simulated pair: containers keyed by name (absent = not a key), images, per-container log bodies,
http bodies keyed by path, die events keyed by container name, and the return codes of docker run / docker info / ssh.
A shim never invents state: what serve.sh sees is exactly what the fixture says. `docker run` with a --name adds that
running container to the world and holds it for `docker_run_hold_s` seconds (0: return at once), which is how a launched
rank becomes visible to `docker ps` / `docker inspect`; a non-zero docker_run_rc answers at once, the way a rank that
dies immediately does. Reads and writes take world.json.lock: a launch runs rank 1 and rank 0 side by side, and a tick
reads the world a shim is writing.
"""

import fcntl
import json
import os
import re
import sys
import time


def load(path):
    """Read of world.json. A mutate truncates the file before it dumps, so a reader that lands in between reads an
    empty file, and the shim that read it answers a rank as absent, a stop as reaching nothing. Reads take the same
    lock's shared side; a tick, a probe and a stop read the world side by side."""
    with open(path + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_SH)
        with open(path) as f:
            return json.load(f)


def mutate(path, change):
    """Read-modify-write on world.json, serialized: two ranks launch at the same moment. The read is taken here,
    inside the lock, so no writer ever reads a half-dumped file."""
    with open(path + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            data = json.load(f)
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


# The fields `events.Message` answers, each measured on docker 29.8.2: `--format '{{.Time}}'` is accepted and the
# CLI goes on to contact the daemon, while `{{.Source}}`, `{{.Nope}}` and `{{.Attributes.exitCode}}` are rejected
# with exit 64, no output, and "can't evaluate field ... in type *events.Message".
FIELDS_OK = ("Status", "ID", "Type", "Action", "Actor", "Time", "TimeNano")


def render(fmt, event):
    """The template's values for one die event. `events.Message` answers Time, TimeNano and the Actor's attribute map;
    the CLI validates the template before it prints, so a field it cannot resolve never reaches the output."""
    def value(path):
        parts = path.strip(".").split(".")
        if parts == ["Time"]:
            return str(event["time"])
        if parts == ["TimeNano"]:
            return str(event["time"] * 10**6)
        if parts == ["Actor", "Attributes", "exitCode"]:
            return event["code"]
        if parts in (["Action"], ["Status"]):
            return "die"
        if parts == ["ID"]:
            return event["name"]
        return ""

    return re.sub(r"\{\{(.*?)\}\}", lambda m: value(m.group(1)), fmt)


def events_op(world, path, argv):
    """docker events --format <tmpl> --filter event=die --filter container=<name> [--since X] [--until X].

    The template is validated against the fields `events.Message` answers, and a template naming a field it does not
    answer is exit 64 with no output, which is what the CLI's own check does.

    `--since` / `--until` bound the window the way the daemon bounds it, because the CLI forwards them untouched
    (docker/cli cli/command/system/events.go): the daemon parses each with strconv.ParseInt of `seconds[.nanoseconds]`
    (moby api/types/time/timestamp.go, v20.10.24 through v28.5.2, reached from the events route's eventTime), so an
    epoch is the only form that answers. `180s` and `0s` are parse errors: the daemon answers 400, the CLI prints the
    parse error, and the read yields no events. Accepting interval forms here is what let a window built from them pass
    the suite while answering nothing on a node. Each read is recorded in the world, so a test can name the window the
    scripts emitted."""
    fmt, fmtflag = argv[0], argv[1] == "1"
    args = argv[2:]
    if fmtflag:
        for expr in re.findall(r"\{\{(.*?)\}\}", fmt):
            parts = expr.strip().strip(".").split(".")
            if not re.fullmatch(r"\.[\w.]+", expr.strip()):
                sys.stderr.write(f"Error parsing format: template: :1:1: syntax error: {expr}\n")
                return 64
            if parts[0] not in FIELDS_OK:
                sys.stderr.write(f'Error parsing format: template: :1:2: executing "" at <{expr.strip()}>: '
                                 f'can\'t evaluate field {parts[0]} in type *events.Message\n')
                return 64
    since, until, want = None, None, []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--since", "--until"):
            value = args[i + 1]
            i += 2
            if a == "--since":
                since = value
            else:
                until = value
            continue
        if a.startswith("container="):
            want.append(a[10:])
        i += 1

    def bound(value, default):
        """The daemon's parse: seconds, an optional `.nanoseconds`, nothing else. None means the value does not parse."""
        if value is None:
            return default
        parts = value.split(".")
        if not parts[0].isdigit() or (len(parts) > 1 and not parts[1].isdigit()):
            return None
        frac = int(parts[1]) * 10 ** (9 - len(parts[1])) if len(parts) > 1 else 0
        return int(parts[0]) + frac / 1e9

    mutate(path, lambda d: d.setdefault("events_calls", []).append(" ".join(args)))
    lo, hi = bound(since, 0), bound(until, float("inf"))
    if lo is None or hi is None:
        bad = since if lo is None else until
        sys.stderr.write(f'Error response from daemon: strconv.ParseInt: parsing "{bad}": invalid syntax\n')
        return 1
    for n, ev in sorted((world.get("die_events") or {}).items()):
        if want and n not in want:
            continue
        if not lo <= ev["time"] <= hi:
            continue
        print(render(fmt, {"name": n, **ev}) if fmtflag else f"{n} die {ev['time']} exit={ev['code']}")
    return 0


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
        mutate(path, lambda d: d.setdefault("run_calls", []).append(" ".join(args)))
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
        # --rm containers leave on their own, and `docker rm -f` removes what is left: the world drops them, and drops
        # them only when the daemon accepted the removal. A node's docker answers on its own, so the worker's failure
        # is its own key: a stop whose worker's daemon refuses the removal leaves rank 1 running, and the scripts have
        # to be able to show that row.
        def drop(d, names=sys.argv[3:]):
            for n in names:
                d["containers"].pop(n, None)
        verb = "rm" if op == "rm" else "stop"
        worker = os.environ.get("NODE_ROLE") == "worker"
        key = f"worker_docker_{verb}_rc" if worker else f"docker_{verb}_rc"
        rc = int(world.get(key, world.get(f"docker_{verb}_rc", 0)))
        if rc == 0:
            mutate(path, drop)
        return rc

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
        return events_op(world, path, sys.argv[3:])

    if op == "boot_id":
        print(world.get("worker_boot_id", ""))
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
