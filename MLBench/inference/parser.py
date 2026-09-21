import csv
import json
from typing import Dict, Optional
import sbatchman as sbm

def parse(job: sbm.Job) -> Optional[Dict[str, Dict]]:
    if job.status != sbm.Status.COMPLETED.value:
        return None

    # i need to save only few info from the json file, cpu name, samples_ts, samples_ns, model_name, model_size
    data = {k:v for k,v in (job.variables or {}).items()}
    data['cluster'] = job.cluster_name
    data['tot_runtime'] = job.get_run_time()

    stdout = job.get_stdout()
    try:
        json_data = json.loads(stdout)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from job {job.job_id}: {e}")
        return None

    if not isinstance(json_data, list):
        return None

    # Process the JSON data to extract the required information
    for item in json_data:
        if isinstance(item, dict):
            data['cpu_info'] = item.get('cpu_info', '')
            data['samples_ts'] = item.get('samples_ts', [])
            data['samples_ns'] = item.get('samples_ns', [])
            data['model_filename'] = item.get('model_filename', '')
            data['model_size'] = item.get('model_size', 0)

    return data