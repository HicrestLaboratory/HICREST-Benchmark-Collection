import subprocess
import time
import sys
import sbatchman as sbm
import argparse
from collections import Counter

WAIT_SECONDS = 60

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_file", help="Path to the YAML jobs file")
    args = parser.parse_args()

    MAIN_COMMAND = ["sbatchman", "launch", "-f", args.jobs_file]
    while True:
        print("Starting main command...")
        subprocess.Popen(MAIN_COMMAND)

        print(f"Waiting {WAIT_SECONDS} seconds...")
        time.sleep(WAIT_SECONDS)

        print("Checking if all is done...")
        if is_all_done():
            print("Job finished successfully. Exiting.")
            sys.exit(0)
        else:
            print("Job not finished. Running deleting failed jobs...")
            run_command(["sbatchman", "delete-jobs", "-s", "FAILED"])
            print("Restarting...\n")


if __name__ == "__main__":
    main()
