#!/usr/bin/env python3

import subprocess
import math
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / 'common'))
from utils.plots import format_bytes

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

GEN_SCRIPT = "./gen_dat_file.sh"
HPL_DAT = Path("HPL.dat")

# Hard-coded (nodes, tasks) pairs
# CONFIGS = [(n,n) for n in [1,2,4,8]]
CONFIGS = [(n,n*4) for n in [1,2,4,8]]

# Assume log base 2 for HPL broadcasts
def logP(P):
    return math.log2(P)

# ----------------------------------------------------------------------
# HPL helpers
# ----------------------------------------------------------------------

def run_gen_dat(nodes: int, tasks: int):
    subprocess.run(
        [GEN_SCRIPT, str(nodes), str(tasks)],
        check=True
    )
    if not HPL_DAT.exists():
        raise FileNotFoundError("HPL.dat was not generated")


def parse_hpl_dat(path: Path):
    """
    Extract N, NB, P, Q from a standard HPL.dat file.
    """
    with path.open() as f:
        lines = [l.strip() for l in f if l.strip()]

    def next_int_after(keyword):
        for i, l in enumerate(lines):
            if keyword in l:
                return int(float(lines[i + 1].split()[0]))
        raise ValueError(f"Keyword '{keyword}' not found")

    return {
        "N":  next_int_after("# of problems sizes"),
        "NB": next_int_after("# of NBs"),
        "P":  next_int_after("Ps"),
        "Q":  next_int_after("Qs"),
    }


def compute_beta_terms(params):
    """
    Return β coefficients (message volume, in words).
    """
    N  = params["N"]
    NB = params["NB"]
    P  = params["P"]
    Q  = params["Q"]

    beta_pfact = 2 * N * logP(P) + (N * N) / P
    beta_update = 3 * (N * NB) / Q
    beta_backs = 2 * N

    return {
        "beta_pfact": beta_pfact,
        "beta_update": beta_update,
        "beta_backs": beta_backs,
        "beta_total": beta_pfact + beta_update + beta_backs,
    }

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------

DATATYPE_SIZE_BYTES = 8 # 64-bit double

def main():
    rows = []

    for nodes, tasks in CONFIGS:
        run_gen_dat(nodes, tasks)
        params = parse_hpl_dat(HPL_DAT)
        betas = compute_beta_terms(params)

        rows.append({
            "nodes": nodes,
            "tasks": tasks,
            "N": params["N"],
            "NB": params["NB"],
            "P": params["P"],
            "Q": params["Q"],
            r"$\beta_{\mathrm{pfact}}$":  format_bytes(DATATYPE_SIZE_BYTES * betas["beta_pfact"],  binary=True, precision=0, space_between_size_and_unit=True),
            r"$\beta_{\mathrm{update}}$": format_bytes(DATATYPE_SIZE_BYTES * betas["beta_update"], binary=True, precision=0, space_between_size_and_unit=True),
            r"$\beta_{\mathrm{backs}}$":  format_bytes(DATATYPE_SIZE_BYTES * betas["beta_backs"],  binary=True, precision=0, space_between_size_and_unit=True),
            r"$\beta_{\mathrm{total}}$":  format_bytes(DATATYPE_SIZE_BYTES * betas["beta_total"],  binary=True, precision=0, space_between_size_and_unit=True),
        })

    df = pd.DataFrame(rows)

    # Optional: nicer ordering
    df = df.sort_values(["nodes", "tasks"]).reset_index(drop=True)
    print(df)

    # Print LaTeX table
    print(
        df.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,   # needed for math-mode column names
            caption=r"HPL message volume analysis ($\beta$ terms only)",
            label="tab:hpl-beta"
        )
    )


if __name__ == "__main__":
    main()
