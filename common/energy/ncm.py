#!/usr/bin/env python3
"""
Start/stop helper for a long-running, blocking energy-measurement command
(e.g. `ncm-control -t 0`) that needs to run in the background, pinned to a
single core, with stdout/stderr redirected to a file -- and later be
stopped with SIGINT once your benchmark has finished.

Designed to be called as two separate, independent invocations (e.g. once
from a "preprocess" step and once from a "postprocess" step). A small JSON
state file (by default at the --pidfile path) is used to hand off
everything the "stop" step needs to know from the "start" step.

Two modes are supported:

  * ACTIVE / "--detailed" mode. `ncm-control -t 0` is launched in the
    background at `start` time (pinned to a core), and its full,
    continuous stream of detailed energy samples is redirected straight
    into --output. `stop` simply signals that background process to
    exit (SIGINT, escalating to SIGTERM/SIGKILL if needed) and then runs
    the --post commands.

        ./ncm_monitor.py start --detailed --core 0 \
            --output /scratch/$(whoami)/energy.log
        ...
        ./ncm_monitor.py stop

  * PASSIVE mode (the default -- no --detailed flag). At `start` time
    only the --pre commands are run (ncm-control -P/-M enable calls);
    nothing is launched in the background and --output is not touched
    yet. At `stop` time, `ncm-control -t 0` is launched (pinned to the
    core recorded at start time), its output is read line-by-line, and
    we watch the first column (a timestamp/counter) for a jump -- the
    device tends to first flush a batch of stale/buffered samples
    before its readings become "live", which shows up as a big jump in
    that counter. A jump is defined as a skip of more than 100 in that
    counter. The first sample after that jump is the one we want: it is
    written (alone) to --output, then the monitor process is killed and
    the --post commands are run.

        ./ncm_monitor.py start --core 0 \
            --output /scratch/$(whoami)/energy.log
        ...
        ./ncm_monitor.py stop [--print-tot-energy]

    By default the trailing "...J" (Joules / cumulative energy) field
    of the stable sample is stripped before it is written out; pass
    --print-tot-energy on the `stop` call to keep it.

Options:
    --pidfile PATH        where the JSON state is stored
                           (default: /tmp/ncm_monitor.pid)
    --timeout SEC         (stop only) seconds to wait after SIGINT before
                           escalating to SIGTERM/SIGKILL (default: 10)
    --pre CMD              (start only, repeatable) command run
                           synchronously, in order given, BEFORE the
                           monitor is launched. Defaults to the four
                           ncm-control -P/-M enable calls if not given.
    --post CMD             (stop only, repeatable) command run
                           synchronously, in order given, AFTER the
                           monitor has stopped. Defaults to the four
                           ncm-control -m/-p disable calls if not given.
    --detailed             (start only) use ACTIVE mode instead of the
                           default PASSIVE mode (see above).
    --stable-timeout SEC   (stop only, passive mode) give up waiting for
                           a jump/stable sample after this many seconds
                           (default: 30).
    --print-tot-energy     (stop only, passive mode) keep the trailing
                           Joules field in the sample written to output.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PIDFILE = "/tmp/ncm_monitor.pid"
DEFAULT_COMMAND = ""

# By default just call "ncm-control" and rely on $PATH. If NCM_PATH is set,
# use it as the absolute path to the binary instead.
NCM_BIN = os.environ.get("NCM_PATH") or "ncm-control"

DEFAULT_COMMAND_DETAILED = f"{NCM_BIN} -t 0"
DEFAULT_PRE = [
    f"{NCM_BIN} -P 1",
    f"{NCM_BIN} -P 2",
    f"{NCM_BIN} -M 1",
    f"{NCM_BIN} -M 2",
]
DEFAULT_POST = [
    f"{NCM_BIN} -m 1",
    f"{NCM_BIN} -m 2",
    f"{NCM_BIN} -p 1",
    f"{NCM_BIN} -p 2",
]
JUMP_THRESHOLD = 100  # a skip of more than this many timestamps counts as "the jump"
DEFAULT_STABLE_TIMEOUT = 2.0


def default_core() -> int:
    """Return the highest-numbered CPU available to this process."""
    try:
        # Respects cpusets/affinity restrictions.
        return max(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for platforms without sched_getaffinity().
        return (os.cpu_count() or 1) - 1


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


def run_step(cmd: str) -> bool:
    """Run a single shell command synchronously.
    Failures are logged to stderr but do not raise exceptions."""
    print(f"$ {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"WARNING: command failed (exit {result.returncode}): {cmd}",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:
        print(
            f"WARNING: failed to execute command '{cmd}': {e}",
            file=sys.stderr,
        )
        return False


def write_state(pidfile: Path, state: dict):
    try:
        pidfile.write_text(json.dumps(state))
    except OSError as e:
        print(f"Warning: could not write state file {pidfile}: {e}")


def read_state(pidfile: Path) -> dict:
    """Read the JSON state file. Falls back to treating the contents as a
    bare PID for backwards compatibility with older state files."""
    text = pidfile.read_text().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"mode": "active", "pid": int(text)}


def terminate_process_group(pid: int, timeout: float) -> bool:
    """Send SIGINT to the process group of `pid`, escalating to SIGTERM
    then SIGKILL if it doesn't exit within `timeout` seconds at each
    step. Returns True if the process exited cleanly (or was already
    gone)."""
    if not pid_alive(pid):
        return True

    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return True

    print(f"Sending SIGINT to process group {pgid} (pid {pid})...")
    os.killpg(pgid, signal.SIGINT)

    stopped_cleanly = False
    for sig, name in [(None, None), (signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")]:
        deadline = time.time() + timeout
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

    return stopped_cleanly


def cmd_start(args):
    pre_cmds = args.pre if args.pre is not None else DEFAULT_PRE
    for pre_cmd in pre_cmds:
        run_step(pre_cmd)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pidfile = Path(args.pidfile)

    if not args.detailed:
        # Passive mode: nothing is launched now. Just remember everything
        # `stop` will need in order to sample the device itself later.
        write_state(pidfile, {
            "mode": "passive",
            "output": str(output_path),
            "core": args.core,
        })
        print(
            f"Passive mode: recorded state to {pidfile}. "
            f"Energy will be sampled at 'stop' time."
        )
        return

    # Active/detailed mode: launch the monitor now and stream its full
    # output straight to --output.
    command = " ".join(args.command) if args.command else DEFAULT_COMMAND_DETAILED

    try:
        log_file = open(output_path, "wb")
    except OSError as e:
        print(f"Failed to open output file {output_path}: {e}")
        return

    print(f"$ {command}   (background, core {args.core}, output -> {output_path})")

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=pin_to_core(args.core),
        )
    except Exception as e:
        print(f"Failed to start command: {e}")
        log_file.close()
        return

    write_state(pidfile, {"mode": "active", "pid": proc.pid})

    print(f"Started monitor: pid={proc.pid}  pidfile={pidfile}")


def _find_stable_sample(proc, stable_timeout: float):
    """Read lines from `proc.stdout` until the first-column counter jumps
    by more than JUMP_THRESHOLD. Returns the first line seen *after*
    that jump, or None if no jump was seen within `stable_timeout`
    seconds / before the process's output ended."""
    prev_ts = None
    deadline = time.time() + stable_timeout

    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue

        line = line.rstrip("\n")
        tokens = line.split()
        if not tokens:
            continue
        try:
            ts = int(tokens[0])
        except ValueError:
            continue

        # print(f'Line: {tokens}')
        if prev_ts is not None and (ts - prev_ts) > JUMP_THRESHOLD:
            # print(f'JUMP!!')
            return line

        prev_ts = ts

    return None


def cmd_stop(args):
    pidfile = Path(args.pidfile)
    if not pidfile.exists():
        print(f"No state file at {pidfile}; nothing to stop.")
        return

    state = read_state(pidfile)
    mode = state.get("mode", "active")

    if mode == "passive":
        _stop_passive(args, pidfile, state)
    else:
        _stop_active(args, pidfile, state)


def _stop_active(args, pidfile: Path, state: dict):
    pid = state.get("pid")
    if pid is None:
        print("State file did not contain a pid; nothing to stop.")
        pidfile.unlink()
        return

    if not pid_alive(pid):
        print(f"Process {pid} is not running.")
    else:
        terminate_process_group(pid, args.timeout)

    pidfile.unlink()
    _run_post(args)


def _stop_passive(args, pidfile: Path, state: dict):
    output_path = Path(state["output"])
    core = state.get("core", default_core())

    command = DEFAULT_COMMAND_DETAILED
    print(f"$ {command}   (sampling for a stable reading, core {core})")

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
            preexec_fn=pin_to_core(core),
        )
    except Exception as e:
        print(f"Failed to start sampling command: {e}")
        pidfile.unlink()
        return

    try:
        stable_line = _find_stable_sample(proc, args.stable_timeout)
    finally:
        terminate_process_group(proc.pid, args.timeout)
        try:
            proc.stdout.close()
        except Exception:
            pass

    if stable_line is None:
        print(
            "WARNING: never observed a timestamp jump within "
            f"{args.stable_timeout}s; no measurement written to {output_path}.",
            file=sys.stderr,
        )
    else:
        tokens = stable_line.split()
        if not args.print_tot_energy and tokens and tokens[-1].endswith("J"):
            tokens = tokens[:-1]
        result_line = " ".join(tokens)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(result_line + "\n")

        print(f"STABLE ENERGY MEASUREMENT: {result_line}")

    pidfile.unlink()
    _run_post(args)


def _run_post(args):
    post_cmds = args.post if args.post is not None else DEFAULT_POST
    ok = True
    for post_cmd in post_cmds:
        if not run_step(post_cmd):
            ok = False
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="Start the monitor (or, in passive mode, just run --pre)")
    p_start.add_argument("--core", type=int, default=default_core(), help="CPU core to pin the monitor to (default: highest available core)")
    p_start.add_argument("--output", required=True, help="File to write energy measurement(s) to")
    p_start.add_argument("--pidfile", default=DEFAULT_PIDFILE, help=f"Where to store run state (default: {DEFAULT_PIDFILE})")
    p_start.add_argument("--pre", action="append", metavar="CMD",
                          help="Command to run synchronously before starting the monitor "
                               "(repeatable, in order). If omitted entirely, defaults to "
                               "the four ncm-control -P/-M enable calls.")
    p_start.add_argument("--detailed", action="store_true",
                          help="Use ACTIVE mode: launch 'ncm-control -t 0' now and stream "
                               "its full continuous output to --output. Default is PASSIVE "
                               "mode: only --pre is run now; a single stable sample is "
                               "captured and written to --output at 'stop' time.")
    p_start.add_argument("command", nargs=argparse.REMAINDER,
                          help="(active mode only) command to run, after '--'. If omitted, "
                               f"defaults to '{DEFAULT_COMMAND_DETAILED}'")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop a previously started monitor")
    p_stop.add_argument("--pidfile", default=DEFAULT_PIDFILE, help=f"Where run state was stored (default: {DEFAULT_PIDFILE})")
    p_stop.add_argument("--timeout", type=float, default=1.0, help="Seconds to wait per signal before escalating (default: 10)")
    p_stop.add_argument("--stable-timeout", type=float, default=DEFAULT_STABLE_TIMEOUT,
                         help="(passive mode only) give up waiting for a stable sample "
                              f"after this many seconds (default: {DEFAULT_STABLE_TIMEOUT})")
    p_stop.add_argument("--print-tot-energy", action="store_true",
                         help="(passive mode only) keep the trailing Joules field in the "
                              "sample written to --output (dropped by default)")
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