import csv
import json
import sbatchman as sbm

OUTPUT_DIR = "./results"

def job_to_json(stdout: str, metadata: dict, cluster_name: str, indent: int = 4) -> str:
    """
    Convert job output (in CSV format) and metadata to a single JSON string.
    Dynamically handles changing headers and automatically converts numbers.
    """
    # Clean output and filter out empty lines
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("No valid lines found in stdout.")

    reader = csv.DictReader(lines)
    data = []

    for row in reader:
        parsed_row = {}
        for key, val in row.items():
            # Try converting values to int or float automatically
            try:
                parsed_row[key] = int(val)
            except ValueError:
                try:
                    parsed_row[key] = float(val)
                except ValueError:
                    parsed_row[key] = val

        # Merge metadata and add cluster_name
        parsed_row.update(metadata)
        parsed_row['cluster_name'] = cluster_name
        data.append(parsed_row)

    return json.dumps(data, indent=indent)


if __name__ == "__main__":
    jobs = sbm.jobs_list()

    for job in jobs:
        metadata = job.variables
        stdout = job.get_stdout()
        cluster_name = job.cluster_name
        json_data = job_to_json(stdout, metadata, cluster_name)
        # Save the JSON data to a file
        with open(f"{OUTPUT_DIR}/job_{job.job_id}.json", "w+") as f:
            f.write(json_data)

