#!/usr/bin/env python3
"""
Start/stop helper for a long-running, blocking energy-measurement command
(e.g. `ncm-control -t 0`) that needs to run in the background, pinned to a
single core, with stdout/stderr redirected to a file -- and later be
stopped with SIGINT once your benchmark has finished.

Designed to be called as two separate, independent invocations (e.g. once
from a "preprocess" step and once from a "postprocess" step), since a
pidfile is used to hand off the running process's PID between calls.

Usage
-----
Start (in your preprocess step). By default this runs the standard
--pre commands (ncm-control -P/-M enable calls) and then launches
`ncm-control -t 0` as the monitor command:
    ./ncm_monitor.py start --core 0 --output /scratch/$(whoami)/energy.log

You can still override the command and/or the --pre commands explicitly:
    ./ncm_monitor.py start --core 0 --output /scratch/$(whoami)/energy.log \
        --pre "ncm-control -P 1" --pre "ncm-control -P 2" \
        --pre "ncm-control -M 1" --pre "ncm-control -M 2" \
        -- ncm-control -t 0

Stop (in your postprocess step). By default this stops the monitor and
then runs the standard --post commands (ncm-control -m/-p disable calls):
    ./ncm_monitor.py stop

You can still override the --post commands explicitly:
    ./ncm_monitor.py stop \
        --post "ncm-control -m 1" --post "ncm-control -m 2" \
        --post "ncm-control -p 1" --post "ncm-control -p 2"

Options:
    --pidfile PATH   where to remember the running PID (default: /tmp/ncm_monitor.pid)
    --timeout SEC    (stop only) seconds to wait after SIGINT before
                      escalating to SIGTERM/SIGKILL (default: 10)
    --pre CMD        (start only, repeatable) command run synchronously,
                      in order given, BEFORE the monitor is launched.
                      Defaults to the four ncm-control -P/-M enable calls
                      if not given at all.
    --post CMD       (stop only, repeatable) command run synchronously,
                      in order given, AFTER the monitor has stopped.
                      Defaults to the four ncm-control -m/-p disable calls
                      if not given at all.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PIDFILE = "/tmp/ncm_monitor.pid"
DEFAULT_COMMAND = "ncm-control -t 0"
DEFAULT_PRE = [
    "ncm-control -P 1",
    "ncm-control -P 2",
    "ncm-control -M 1",
    "ncm-control -M 2",
]
DEFAULT_POST = [
    "ncm-control -m 1",
    "ncm-control -m 2",
    "ncm-control -p 1",
    "ncm-control -p 2",
]


def pin_to_core(core: int):
    def _pin():
        os.sched_setaffinity(0, {core})
    return _pin


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run_step(cmd: str, best_effort: bool = False) -> bool:
    """Run a single shell command synchronously. Returns True on success.
    If best_effort is True, a failure is logged but not raised (used for
    --post cleanup commands, so one failure doesn't skip the rest)."""
    print(f"$ {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if best_effort:
            print(f"WARNING: command failed (exit {e.returncode}): {cmd}")
            return False
        raise


def cmd_start(args):
    command = " ".join(args.command) if args.command else DEFAULT_COMMAND

    pre_cmds = args.pre if args.pre is not None else DEFAULT_PRE
    for pre_cmd in pre_cmds:
        run_step(pre_cmd)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(output_path, "wb")

    print(f"$ {command}   (background, core {args.core}, output -> {output_path})")
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,       # own process group, so it can be signaled cleanly
        preexec_fn=pin_to_core(args.core),
    )

    pidfile = Path(args.pidfile)
    pidfile.write_text(str(proc.pid))
    print(f"Started monitor: pid={proc.pid}  pidfile={pidfile}")


def cmd_stop(args):
    pidfile = Path(args.pidfile)
    if not pidfile.exists():
        print(f"No pidfile at {pidfile}; nothing to stop.")
        return

    pid = int(pidfile.read_text().strip())

    if not pid_alive(pid):
        print(f"Process {pid} is not running.")
        pidfile.unlink()
        return

    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        print(f"Process {pid} is not running.")
        pidfile.unlink()
        return

    print(f"Sending SIGINT to process group {pgid} (pid {pid})...")
    os.killpg(pgid, signal.SIGINT)

    stopped_cleanly = False
    for sig, name in [(None, None), (signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")]:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            if not pid_alive(pid):
                if name:
                    print(f"Monitor exited after {name}.")
                else:
                    print("Monitor exited cleanly after SIGINT.")
                stopped_cleanly = True
                break
            time.sleep(0.2)
        if stopped_cleanly:
            break
        if sig is not None:
            print(f"Still alive after previous signal, escalating to {name}...")
            os.killpg(pgid, sig)

    if not stopped_cleanly:
        print(f"WARNING: process {pid} may still be alive.")

    pidfile.unlink()

    post_cmds = args.post if args.post is not None else DEFAULT_POST
    ok = True
    for post_cmd in post_cmds:
        if not run_step(post_cmd, best_effort=True):
            ok = False
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="Start the monitor in the background")
    p_start.add_argument("--core", type=int, required=True, help="CPU core to pin the monitor to")
    p_start.add_argument("--output", required=True, help="File to redirect stdout+stderr to")
    p_start.add_argument("--pidfile", default=DEFAULT_PIDFILE, help=f"Where to store the PID (default: {DEFAULT_PIDFILE})")
    p_start.add_argument("--pre", action="append", metavar="CMD",
                          help="Command to run synchronously before starting the monitor "
                               "(repeatable, in order). If omitted entirely, defaults to "
                               "the four ncm-control -P/-M enable calls.")
    p_start.add_argument("command", nargs=argparse.REMAINDER,
                          help=f"Command to run, after '--'. If omitted, defaults to "
                               f"'{DEFAULT_COMMAND}'")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop a previously started monitor")
    p_stop.add_argument("--pidfile", default=DEFAULT_PIDFILE, help=f"Where the PID was stored (default: {DEFAULT_PIDFILE})")
    p_stop.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait per signal before escalating (default: 10)")
    p_stop.add_argument("--post", action="append", metavar="CMD",
                         help="Command to run synchronously after the monitor has stopped "
                              "(repeatable, in order, best-effort). If omitted entirely, "
                              "defaults to the four ncm-control -m/-p disable calls.")
    p_stop.set_defaults(func=cmd_stop)

    args = parser.parse_args()

    # strip a leading "--" from REMAINDER if present
    if args.action == "start" and args.command and args.command[0] == "--":
        args.command = args.command[1:]

    args.func(args)


if __name__ == "__main__":
    main()