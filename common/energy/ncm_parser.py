import re
from pathlib import Path

NCM_LOG_RE = re.compile(
    r"^\s*"
    r"(?P<timestamp>\d+)\s+"
    r"(?P<probe>\d+\.\d+)\s+"
    r"(?P<flags>\S+)\s+"
    r"(?P<temperature>[\d.]+)dC\s+"
    r"(?P<voltage>[\d.]+)V\s+"
    r"(?P<current>[\d.]+)A\s+"
    r"(?P<energy>[\d.]+)J\s*$"
)

def parse_ncm_energy_log(filename: Path):
    """Parse an ncm-control log file."""
    samples = []
    with open(filename) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            m = NCM_LOG_RE.match(line)
            if not m:
                continue
                # raise ValueError(f"Malformed line {lineno}: {line}")

            probe = float(m["probe"])
            probe_id, channel = map(int, m["probe"].split("."))

            samples.append({
                "timestamp_ms": int(m["timestamp"]),
                "probe": probe,
                "probe_id": probe_id,
                "channel": channel,
                "flags": m["flags"],
                "temperature_C": float(m["temperature"]),
                "voltage_V": float(m["voltage"]),
                "current_A": float(m["current"]),
                "energy_J": float(m["energy"]),
            })

    return samples

def parse_ncm_tot_energy_print(stdout: str):
    # Example target line: "STABLE ENERGY MEASUREMENT: 11469 0.0 0xff 35.3dC 12.128V 0.1989A 47.316J"
    for line in stdout.splitlines():
        if line.startswith('STABLE ENERGY MEASUREMENT'):
            return float(line.split(': ')[1].split(' ')[-1][:-1])
    return None