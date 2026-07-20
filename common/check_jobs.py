import argparse
import json
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

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
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
    # Collect data
    # ---------------------------
    all_runtimes = []
    status_counter = Counter()

    json_output = {
        "system": sbm.get_cluster_name(),
        "groups": {},
    }

    for group, group_jobs in grouped.items():
        group_entries = []

        for job in group_jobs:
            tag = str(job.tag)
            status = str(job.status)
            runtime = job.get_run_time()

            group_entries.append(
                {
                    "job_id": job.job_id,
                    "tag": tag,
                    "status": status,
                    "runtime_seconds": runtime,
                    "runtime": format_time(runtime),
                }
            )

            # collect stats
            status_counter[status] += 1
            if runtime is not None:
                all_runtimes.append(runtime)

        json_output["groups"][group] = group_entries

    # ---------------------------
    # Stats
    # ---------------------------
    if not args.no_hist:
        json_output["status_histogram"] = dict(status_counter)

    if not args.no_stats and all_runtimes:
        total = sum(all_runtimes)
        avg = total / len(all_runtimes)
        mx = max(all_runtimes)
        mn = min(all_runtimes)

        json_output["runtime_stats"] = {
            "jobs_counted": len(all_runtimes),
            "total_runtime_seconds": total,
            "average_runtime_seconds": avg,
            "min_runtime_seconds": mn,
            "max_runtime_seconds": mx,
            "total_runtime": format_time(total),
            "average_runtime": format_time(avg),
            "min_runtime": format_time(mn),
            "max_runtime": format_time(mx),
        }

    # ---------------------------
    # Output
    # ---------------------------
    if args.json:
        print(json.dumps(json_output, indent=2))
        return

    # ---------------------------
    # Human-readable output
    # ---------------------------
    print(f"System name: {json_output['system']}")

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

    if not args.no_hist:
        print("\n=== Status Histogram ===")
        for status, count in status_counter.items():
            bar = "#" * count
            print(f"{status:15} | {bar} ({count})")

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