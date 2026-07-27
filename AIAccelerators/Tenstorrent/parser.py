import json
import re
from typing import Any, Dict, Optional

import sbatchman as sbm


RESULT_RE = re.compile(
    r"^HICREST_RESULT\s+(\{.*\})\s*$",
    re.MULTILINE,
)


def parse(job: sbm.Job) -> Optional[Dict[str, Dict[str, Any]]]:
    """Parse one Tenstorrent HICREST_RESULT record from job stdout."""
    if not job.tag.startswith("tt_"):
        return None

    if job.status != sbm.Status.COMPLETED.value:
        return None

    stdout = job.get_stdout() or ""
    matches = RESULT_RE.findall(stdout)

    if not matches:
        raise ValueError(
            f"Completed Tenstorrent job {job.tag!r} has no HICREST_RESULT"
        )

    if len(matches) != 1:
        raise ValueError(
            f"Tenstorrent job {job.tag!r} emitted {len(matches)} "
            "HICREST_RESULT records; expected exactly one"
        )

    data = dict(job.variables or {})
    data.update(json.loads(matches[0]))
    data["cluster"] = job.cluster_name
    data["tag"] = job.tag
    data["sbatchman_runtime_s"] = job.get_run_time()

    return {"tenstorrent": data}

