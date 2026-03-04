import sbatchman as sbm
import argparse
from collections import defaultdict


def compute_job_stats(job):
    vars_ = job.variables
    runtime_sec = job.get_run_time()

    nodes = vars_.get("nodes", 1)
    gpus_per_node = vars_.get("gpus_per_node", 0)

    runtime_hours = runtime_sec / 3600.0
    node_hours = runtime_hours * nodes
    gpu_hours = runtime_hours * nodes * gpus_per_node

    return {
        "runtime_sec": runtime_sec,
        "runtime_hours": runtime_hours,
        "node_hours": node_hours,
        "gpu_hours": gpu_hours,
    }


def make_group_key(vars_, group_by):
    if not group_by:
        return ("ALL",)

    return tuple(vars_.get(k, "MISSING") for k in group_by)


def format_group_key(key_tuple, group_by):
    if not group_by:
        return "ALL JOBS"

    return ", ".join(f"{k}={v}" for k, v in zip(group_by, key_tuple))


def main():
    parser = argparse.ArgumentParser(
        description="Compute runtime and compute-hour statistics from sbatchman jobs."
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        help="List of job variable names to group by (e.g. implementation primitive nodes)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-job statistics"
    )

    args = parser.parse_args()

    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED],
        from_active=True,
        from_archived=False
    )
    # jobs = sbm.jobs_list(
    #     status=[sbm.Status.COMPLETED],
    #     from_active=False,
    #     from_archived=True,
    #     archive_name='archive_name'
    # )


    total_runtime = 0.0
    total_node_hours = 0.0
    total_gpu_hours = 0.0

    grouped_stats = defaultdict(lambda: {
        "count": 0,
        "runtime_sec": 0.0,
        "node_hours": 0.0,
        "gpu_hours": 0.0,
    })

    if args.verbose:
        print("\n=== Per Job Stats ===\n")

    for job in jobs:
        stats = compute_job_stats(job)
        vars_ = job.variables

        total_runtime += stats["runtime_sec"]
        total_node_hours += stats["node_hours"]
        total_gpu_hours += stats["gpu_hours"]

        key = make_group_key(vars_, args.group_by)
        grouped_stats[key]["count"] += 1
        grouped_stats[key]["runtime_sec"] += stats["runtime_sec"]
        grouped_stats[key]["node_hours"] += stats["node_hours"]
        grouped_stats[key]["gpu_hours"] += stats["gpu_hours"]

        if args.verbose:
            print(f"Job variables: {vars_}")
            print(f"  Runtime:    {stats['runtime_sec']:.2f} sec")
            print(f"  Node-hours: {stats['node_hours']:.4f}")
            print(f"  GPU-hours:  {stats['gpu_hours']:.4f}")
            print()

    print("\n=== Global Summary ===\n")
    print(f"Total jobs:       {len(jobs)}")
    print(f"Total runtime:    {total_runtime:.2f} sec")
    print(f"Total node-hours: {total_node_hours:.4f}")
    print(f"Total GPU-hours:  {total_gpu_hours:.4f}")

    print("\n=== Grouped Statistics ===\n")

    for key, data in sorted(grouped_stats.items()):
        avg_runtime = data["runtime_sec"] / data["count"]
        print(format_group_key(key, args.group_by))
        print(f"  Jobs:        {data['count']}")
        print(f"  Avg runtime: {avg_runtime:.2f} sec")
        print(f"  Node-hours:  {data['node_hours']:.4f}")
        print(f"  GPU-hours:   {data['gpu_hours']:.4f}")
        print()


if __name__ == "__main__":
    main()