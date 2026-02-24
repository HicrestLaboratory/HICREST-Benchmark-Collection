import subprocess
import time
import sys
import sbatchman as sbm
import argparse
import signal
from collections import Counter

WAIT_SECONDS = 60
sleep_interrupted = False


def handle_sigint(signum, frame):
    """Handle CTRL+C (SIGINT) → break sleep only."""
    global sleep_interrupted
    print("\nCTRL+C detected. Skipping wait...")
    sleep_interrupted = True


def handle_sigtstp(signum, frame):
    """Handle CTRL+Z (SIGTSTP) → terminate program."""
    print("\nCTRL+Z detected. Terminating program.")
    sys.exit(0)


def run_command(cmd):
    """Run a command and return its exit code."""
    result = subprocess.run(cmd)
    return result.returncode


def is_all_done() -> bool:
    jobs = list(sbm.jobs_list(from_active=True, from_archived=False))
    statuses = [j.status for j in jobs]
    status_counts = Counter(statuses)

    print("\nJob Status Histogram:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    for j in jobs:
        if j.status != sbm.Status.COMPLETED.value:
            return False
    return True


def interruptible_sleep(seconds):
    """Sleep but allow CTRL+C to break early."""
    global sleep_interrupted
    sleep_interrupted = False

    start = time.time()
    while time.time() - start < seconds:
        if sleep_interrupted:
            break
        time.sleep(1)


def main():
    signal.signal(signal.SIGINT, handle_sigint)   # CTRL+C
    signal.signal(signal.SIGTSTP, handle_sigtstp) # CTRL+Z

    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_file", help="Path to the YAML jobs file")
    args = parser.parse_args()
    loop_i = 0

    MAIN_COMMAND = ["sbatchman", "launch", "--ignore-conf-in-dup-check", "--ignore-commands-in-dup-check", "-f", args.jobs_file]

    while True:
        loop_i += 1
        print('='*50)
        print(f'\t\t Loop #{loop_i}')
        print('='*50)
        print("Launching jobs...")
        proc = subprocess.Popen(MAIN_COMMAND)

        # Wait until sbatchman launch finishes
        ret = proc.wait()
        if ret != 0:
            print(f"Launch command failed with exit code {ret}")

        print(f"Waiting {WAIT_SECONDS} seconds... (CTRL+C to try again, CTRL+Z to exit)")
        interruptible_sleep(WAIT_SECONDS)

        print("Checking if all is done...")
        if is_all_done():
            print("Job finished successfully. Exiting.")
            sys.exit(0)
        else:
            print("Job not finished. Running deleting failed jobs...")
            run_command(["sbatchman", "delete-jobs", "-na", "-s", "FAILED", "-s", "CANCELLED"])
            print("Restarting...\n")


if __name__ == "__main__":
    main()