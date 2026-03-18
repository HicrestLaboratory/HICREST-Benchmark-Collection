"""
print_experiments.py
====================
Pretty-print an experiment JSON file produced by experiments_generator.py.

The output is identical to what the generator prints to stdout, reconstructed
entirely from the JSON document — no dependency on the generator module.

Usage
-----
    python print_experiments.py experiments.json
    python print_experiments.py experiments.json --no-experiments
    python print_experiments.py experiments.json --section summary
    python print_experiments.py experiments.json --section baseline
    python print_experiments.py experiments.json --section patterns
    python print_experiments.py experiments.json --section experiments
    python print_experiments.py experiments.json --section placement-registry

Sections
--------
  all                 Print every section (default)
  baseline            Baseline set  |B|
  baseline-topology   Topology-annotated baseline  |B_topo|  (if present)
  placement-registry  Placement-class vocabulary + per-strategy map (if present)
  patterns            Pattern set  |P|
  experiments         Flat experiment set  |E|
  hier-experiments    Hierarchical experiment set  |E_hier|  (if present)
  summary             Summary block

The --no-experiments / --no-hier flags suppress the verbose per-experiment
blocks while still printing all other sections.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional

# ── Terminal width for section separators ────────────────────────────────────
SEP = 140

# ── Colour helpers ───────────────────────────────────────────────────────────
BLUE   = "\033[34m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[0m"

# ── Bin display order ────────────────────────────────────────────────────────
ENTROPY_BINS   = ("low", "medium", "high")
PLACEMENT_BINS = ("low", "medium", "high")


# ===========================================================================
# Section: Baseline
# ===========================================================================

def print_baseline(doc: dict) -> None:
    baseline = doc.get("baseline_set", [])
    print(f"\n{BLUE}{'='*SEP}")
    print(f"BASELINE SET  |B| = {len(baseline)}")
    print(f"{'='*SEP}{RESET}")

    by_strategy: dict[str, list[int]] = {}
    for run in baseline:
        by_strategy.setdefault(run["strategy"], []).append(run["gpus"])
    for name, gpus in by_strategy.items():
        print(f"  {name:20s}  gpus = {sorted(gpus)}")


# ===========================================================================
# Section: Topology baseline
# ===========================================================================

def print_baseline_topology(doc: dict) -> None:
    baseline_topo = doc.get("baseline_set_topology")
    if baseline_topo is None:
        return
    print(f"\n{BLUE}{'='*SEP}")
    print(f"TOPOLOGY BASELINE SET  |B_topo| = {len(baseline_topo)}")
    print(f"{'='*SEP}{RESET}")
    for bwp in baseline_topo:
        run_str = f"({bwp['run']['strategy']}:{bwp['run']['gpus']})"
        label   = bwp["placement_class"]
        rep     = bwp["replicate_index"]
        seed    = bwp["seed"]
        print(
            f"  {run_str:<30}  class={label:<22}  "
            f"rep={rep}  seed={seed}"
        )


# ===========================================================================
# Section: Placement-class registry
# ===========================================================================

def print_placement_registry(doc: dict) -> None:
    params = doc.get("parameters", {})
    class_defs  = params.get("placement_class_defs")
    strat_map   = params.get("strategy_placement_map")

    if not class_defs and not strat_map:
        # No topology info in this file.
        return

    print(f"\n{BLUE}{'='*SEP}")
    print("PLACEMENT CLASS REGISTRY")
    print(f"{'='*SEP}{RESET}")

    if class_defs:
        # Build a label lookup for the strategy-map section.
        name_to_label = {d["name"]: d["label"] for d in class_defs}
        print(f"  {'Name':<25}  {'Label':<28}  Score")
        print(f"  {'-'*25}  {'-'*28}  -----")
        for d in class_defs:
            print(f"  {d['name']:<25}  {d['label']:<28}  {d['score']}")
    else:
        name_to_label = {}

    if strat_map:
        print(f"\n  Per-strategy allowed classes:")
        for strat, pc_names in strat_map.items():
            labels = [name_to_label.get(n, n) for n in pc_names]
            print(f"  {strat:<20}  {', '.join(labels)}")


# ===========================================================================
# Section: Patterns
# ===========================================================================

def print_patterns(doc: dict) -> None:
    patterns = doc.get("pattern_set", [])
    G        = doc["parameters"]["G"]
    print(f"\n{BLUE}{'='*SEP}")
    print(f"PATTERN SET  |P| = {len(patterns)}")
    print(f"{'='*SEP}{RESET}")
    for tp in patterns:
        slots = tp["slots"]
        s     = tp["total_gpus"]
        fam   = tp["family"]
        k     = tp["k"]
        print(
            f"  [{fam}] {str(tuple(slots)):<80}  "
            f"totGPUs={s:<4}  util={int(s/G*100):<3}%  k={k}"
        )


# ===========================================================================
# Formatting helpers shared by experiment printers
# ===========================================================================

def _fmt_run(run: dict) -> str:
    return f"({run['strategy']}:{run['gpus']})"


def _fmt_config(cfg: dict, G: int) -> str:
    slots = ", ".join(_fmt_run(r) for r in cfg["runs"])
    util  = int(cfg["utilization"] * 100)
    nodes = int(cfg["total_gpus"] / 4)
    gpus  = cfg["total_gpus"]
    return f"[{slots}]\n  util={util:3}% nodes: {nodes} ({gpus} GPUs)"


def _fmt_experiment(exp: dict, G: int) -> str:
    pattern  = tuple(exp["pattern"])
    h_bin    = exp["entropy_bin"]
    cfg_str  = _fmt_config(exp["config"], G)
    return (
        f"  pattern={str(pattern)}\n"
        f"  H-bin={h_bin}\n"
        f"  config={cfg_str}\n"
    )


def _fmt_hier_experiment(he: dict, G: int) -> str:
    base_str = _fmt_experiment(he, G)           # shares all flat fields
    kappa    = he.get("placement_class_vector", [])
    p_bin    = he.get("placement_bin", "?")
    score    = he.get("placement_score", 0.0)
    seed     = he.get("placement_seed", 0)
    return (
        f"{base_str}"
        f"  κ={kappa}\n"
        f"  P-bin={p_bin}  score={score:.2f}  seed={seed}\n"
    )


# ===========================================================================
# Section: Flat experiments
# ===========================================================================

def print_experiments(doc: dict, verbose: bool = True) -> None:
    experiments = doc.get("experiments", [])
    G           = doc["parameters"]["G"]

    print(f"\n{BLUE}{'='*SEP}")
    print(f"EXPERIMENT SET  |E| = {len(experiments)}")
    print(f"{'='*SEP}{RESET}")

    if not verbose:
        return

    by_bin: dict[str, list] = {b: [] for b in ENTROPY_BINS}
    for exp in experiments:
        by_bin[exp["entropy_bin"]].append(exp)

    for bin_name in ENTROPY_BINS:
        exps = by_bin[bin_name]
        if not exps:
            continue
        print(f"\n  {YELLOW}── H-bin: {bin_name} ({len(exps)} experiments) ──{RESET}\n")
        for exp in exps:
            print(_fmt_experiment(exp, G))


# ===========================================================================
# Section: Hierarchical experiments
# ===========================================================================

def print_hier_experiments(doc: dict, verbose: bool = True) -> None:
    hier = doc.get("hierarchical_experiments")
    if hier is None:
        return
    G = doc["parameters"]["G"]

    print(f"\n{BLUE}{'='*SEP}")
    print(f"HIERARCHICAL EXPERIMENT SET  |E_hier| = {len(hier)}")
    print(f"{'='*SEP}{RESET}")

    if not verbose:
        return

    by_pbin: dict[str, list] = {b: [] for b in PLACEMENT_BINS}
    for he in hier:
        by_pbin[he["placement_bin"]].append(he)

    for pbin in PLACEMENT_BINS:
        if not by_pbin[pbin]:
            continue
        print(f"\n  {YELLOW}── P-bin: {pbin} ({len(by_pbin[pbin])} experiments) ──{RESET}\n")
        for he in by_pbin[pbin]:
            print(_fmt_hier_experiment(he, G))


# ===========================================================================
# Section: Summary
# ===========================================================================

def print_summary(doc: dict) -> None:
    params  = doc.get("parameters", {})
    summary = doc.get("summary", {})
    meta    = doc.get("meta", {})

    G                = params.get("G", summary.get("log2_G", 0))
    n_strategies     = summary.get("n_strategies", len(doc.get("strategies", [])))
    n_baseline       = summary.get("n_baseline_runs", len(doc.get("baseline_set", [])))
    n_baseline_topo  = summary.get("n_baseline_topo_runs")
    n_patterns       = summary.get("n_patterns", len(doc.get("pattern_set", [])))
    n_flat           = summary.get("n_flat_experiments", len(doc.get("experiments", [])))
    n_hier           = summary.get("n_hier_experiments")
    min_exp          = params.get("min_experiments", 0)
    max_exp          = params.get("max_experiments")
    log2_G           = summary.get("log2_G", math.log2(G) if G > 0 else 0)

    print(f"\n{BLUE}{'='*SEP}")
    print("SUMMARY")
    print(f"{'='*SEP}{RESET}")
    print(f"  Generated at           : {meta.get('generated_at', 'unknown')}")
    print(f"  G (total GPUs)         : {G}")
    print(f"  Strategies  |S|        : {n_strategies}")
    print(f"  Baseline set |B|       : {n_baseline}")
    if n_baseline_topo is not None:
        print(f"  Topology baseline |B_t|: {n_baseline_topo}")
    print(f"  Pattern set |P|        : {n_patterns}")
    print(f"  Flat experiments |E|   : {n_flat}")
    if n_hier is not None:
        print(f"  Hier experiments |E_h| : {n_hier}")

    bounds = (f"min={min_exp}" if min_exp else "no min") + " / " + \
             (f"max={max_exp}" if max_exp is not None else "no max")
    print(f"  Size bounds            : {bounds}")
    print(f"  O(log G) bound         : log₂({G}) ≈ {log2_G:.1f}")

    # Distribution breakdown — derive from the actual experiment lists.
    hier_list = doc.get("hierarchical_experiments")
    flat_list = doc.get("experiments", [])
    primary   = hier_list if hier_list is not None else flat_list

    if hier_list is not None:
        pb_counts  = {b: 0 for b in PLACEMENT_BINS}
        hb_counts  = {b: 0 for b in ENTROPY_BINS}
        fam_counts: dict[str, int] = {}
        for he in hier_list:
            pb_counts[he["placement_bin"]] += 1
            hb_counts[he["entropy_bin"]] += 1
            fam_counts[he["pattern_family"]] = fam_counts.get(he["pattern_family"], 0) + 1
        print(f"  Placement-bin counts   : {dict(pb_counts)}")
        print(f"  Entropy-bin counts     : {dict(hb_counts)}")
        print(f"  Family counts          : {dict(fam_counts)}")
    else:
        hb_counts  = {b: 0 for b in ENTROPY_BINS}
        fam_counts = {}
        for e in flat_list:
            hb_counts[e["entropy_bin"]] += 1
            fam_counts[e["pattern_family"]] = fam_counts.get(e["pattern_family"], 0) + 1
        print(f"  Entropy-bin counts     : {dict(hb_counts)}")
        print(f"  Family counts          : {dict(fam_counts)}")


# ===========================================================================
# CLI
# ===========================================================================

SECTIONS_ALL = (
    "baseline",
    "baseline-topology",
    "placement-registry",
    "patterns",
    "experiments",
    "hier-experiments",
    "summary",
)

SECTION_HELP = (
    "baseline            – baseline set |B|\n"
    "baseline-topology   – topology-annotated baseline (if present)\n"
    "placement-registry  – placement-class vocabulary (if present)\n"
    "patterns            – pattern set |P|\n"
    "experiments         – flat experiment set |E|\n"
    "hier-experiments    – hierarchical experiment set |E_hier| (if present)\n"
    "summary             – summary block\n"
    "all                 – all of the above (default)"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="print_experiments.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pretty-print an experiment JSON file produced by experiments_generator.py.",
        epilog=f"Sections:\n{SECTION_HELP}",
    )
    p.add_argument(
        "json_file",
        metavar="FILE",
        help="Path to the experiments JSON file.",
    )
    p.add_argument(
        "--section", "-s",
        metavar="SECTION",
        default="all",
        choices=list(SECTIONS_ALL) + ["all"],
        help="Which section to print (default: all).",
    )
    p.add_argument(
        "--no-experiments",
        action="store_true",
        default=False,
        help="Print section headers and counts but suppress per-experiment detail lines.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Load JSON
    try:
        with open(args.json_file, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        print(f"error: file not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.json_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.no_experiments

    # Determine which sections to print.
    sections: tuple[str, ...]
    if args.section == "all":
        sections = SECTIONS_ALL
    else:
        sections = (args.section,)

    # Dispatch.
    for sec in sections:
        if sec == "baseline":
            print_baseline(doc)
        elif sec == "baseline-topology":
            print_baseline_topology(doc)
        elif sec == "placement-registry":
            print_placement_registry(doc)
        elif sec == "patterns":
            print_patterns(doc)
        elif sec == "experiments":
            print_experiments(doc, verbose=verbose)
        elif sec == "hier-experiments":
            print_hier_experiments(doc, verbose=verbose)
        elif sec == "summary":
            print_summary(doc)


if __name__ == "__main__":
    main()