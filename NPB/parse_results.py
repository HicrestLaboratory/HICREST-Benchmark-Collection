from pathlib import Path
import re
import pandas as pd
import sbatchman as sbm

# ---------------------------
# Regex for NPB output
# ---------------------------

RE_TIME = re.compile(r"Time in seconds\s*=\s*([\d.]+)")
RE_MOPS = re.compile(r"Mop/s total\s*=\s*([\d.]+)")
RE_CLASS = re.compile(r"Class\s*=\s*(\w)")
RE_VERIFY = re.compile(r"Verification\s*=\s*(\w+)")

RE_TAG = re.compile(r"(\w+)\.(\w)\.x_(\d+)cpus")

# ---------------------------
# Parser
# ---------------------------

def parse_npb_outputs(jobs):

    records = []

    for job in jobs:

        m = RE_TAG.match(job.tag)
        if not m:
            print("[WARN] skipping", job.tag)
            continue

        benchmark = m.group(1)
        cls = m.group(2)
        cores = int(m.group(3))

        device = job.config_name.split("_")[0]

        output = job.get_stdout()
        if not output:
            continue

        time = None
        mops = None
        verify = None

        for line in output.splitlines():

            if not time:
                m = RE_TIME.search(line)
                if m:
                    time = float(m.group(1))

            if not mops:
                m = RE_MOPS.search(line)
                if m:
                    mops = float(m.group(1))

            if not verify:
                m = RE_VERIFY.search(line)
                if m:
                    verify = m.group(1)

        records.append({
            "system": device,
            "benchmark": benchmark,
            "class": cls,
            "cores": cores,
            "time_sec": time,
            "mops": mops,
            "verification": verify
        })

        print("Parsed:", job.tag)

    return pd.DataFrame(records)


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED])
    system = sbm.get_cluster_name()

    df = parse_npb_outputs(jobs)

    if df.empty:
        print("No results found")
        exit()

    df.sort_values(["benchmark", "cores"], inplace=True)

    out_path = Path('results') / f"npb_{system}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved to: {out_path.resolve().absolute()}")