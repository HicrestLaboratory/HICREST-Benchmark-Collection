import argparse
from collections import Counter, defaultdict

import sbatchman as sbm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archives",
        action="store_true",
        help="Include archived jobs grouped by archive name",
    )

    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable histogram of job statuses",
    )

    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable runtime statistics",
    )

    return parser.parse_args()


def format_time(seconds: float) -> str:
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    args = parse_args()

    print(f"System name: {sbm.get_cluster_name()}")

    jobs = sbm.jobs_list(
        from_active=True,
        from_archived=args.archives,
    )

    # ---------------------------
    # Group jobs
    # ---------------------------
    grouped = defaultdict(list)

    for job in jobs:
        if args.archives:
            key = job.archive_name or "active"
        else:
            key = "active"
        grouped[key].append(job)

    # ---------------------------
    # Iterate and print jobs
    # ---------------------------
    all_runtimes = []
    status_counter = Counter()

    for group, group_jobs in grouped.items():
        print(f"\n=== {group} ===")

        for job in group_jobs:
            tag = str(job.tag)
            status = str(job.status)
            runtime = job.get_run_time()

            print(
                f"  [Job {job.job_id}]\n"
                f"    tag:     {tag}\n"
                f"    status:  {status}\n"
                f"    runtime: {format_time(runtime)}\n"
            )

            # collect stats
            status_counter[status] += 1
            if runtime is not None:
                all_runtimes.append(runtime)

    # ---------------------------
    # Histogram of statuses (ON by default)
    # ---------------------------
    if not args.no_hist:
        print("\n=== Status Histogram ===")
        for status, count in status_counter.items():
            bar = "#" * count
            print(f"{status:15} | {bar} ({count})")

    # ---------------------------
    # Runtime statistics (ON by default)
    # ---------------------------
    if not args.no_stats and all_runtimes:
        total = sum(all_runtimes)
        avg = total / len(all_runtimes)
        mx = max(all_runtimes)
        mn = min(all_runtimes)

        print("\n=== Runtime Stats ===")
        print(f"Jobs counted : {len(all_runtimes)}")
        print(f"Total runtime: {format_time(total)}")
        print(f"Average      : {format_time(avg)}")
        print(f"Min          : {format_time(mn)}")
        print(f"Max          : {format_time(mx)}")


if __name__ == "__main__":
    main()