"""
Leonardo Supercomputer Nodelist Generator
Generates optimized nodelists for SLURM job scheduling based on network topology constraints
"""

import os
from pathlib import Path
import pprint
import warnings
import pandas as pd
import re
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import itertools

PARTITION_TO_ID_MAP = {
    "boost_usr_prod": 1,
    "dcgp_usr_prod": 2,
}

PARTITION_NAME_MAP = {
    "1": "boost_usr_prod",
    "2": "dcgp_usr_prod",
    "boost_usr_prod": "boost_usr_prod",
    "dcgp_usr_prod": "dcgp_usr_prod",
}


class TopologyConstraint(Enum):
    """Network topology constraints for node selection"""

    SAME_RACK = "same_rack"  # All nodes in same rack
    SAME_CELL = "same_cell"  # All nodes in same cell
    SAME_SWITCH = "same_switch"  # All nodes on same L1 switch
    DIFFERENT_CELLS = "different_cells"  # Nodes spread across different cells
    DIFFERENT_RACKS = "different_racks"  # Nodes spread across different racks
    MINIMIZE_L2_HOPS = "minimize_l2"  # Minimize inter-cell (L2) communication
    NO_CONSTRAINT = "no_constraint"  # No topology constraint


@dataclass
class SlurmResources:
    """SLURM resource requirements"""

    num_nodes: int
    tasks_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    gpus_per_node: Optional[int] = None
    partition: Optional[int] = None  # 1=Booster, 2=DCGP

    def __post_init__(self):
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")


def load_leonardo_system_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Leonardo system data from CSV or create sample data.

    Args:
        csv_path: Path to CSV file with system topology

    Returns:
        DataFrame with system topology
    """
    print(f"Loading system data from {csv_path}...")
    pattern = re.compile(
        r"NODE\s+(\d+)\s+RACK\s+(\d+)\s+CELL\s+(\d+)\s+ROW\s+(\d+)\s+PARTITION\s+(\d+)\s+SWITCH\s+(\d+)"
    )

    records = []
    with open(csv_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                records.append(match.groups())

    return pd.DataFrame(
        records, columns=["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
    ).astype(int)


class LeonardoNodelistGenerator:
    """
    Generates optimized nodelists for Leonardo supercomputer based on
    network topology and SLURM resource requirements.
    """

    def __init__(
        self,
        system_df: Union[None, pd.DataFrame] = None,
        csv_path: Union[None, str] = None,
        verify_with_sinfo: bool = False,
        sinfo_states: Optional[List[str]] = None,
        sinfo_partitions: Optional[List[str]] = None,
    ):
        """
        Initialize with system topology dataframe.

        Args:
            system_df: DataFrame with columns [NODE, RACK, CELL, ROW, PARTITION, SWITCH]
            csv_path: Path to topology file (used if system_df is None)
            verify_with_sinfo: If True, verify node availability using sinfo
            sinfo_states: List of acceptable node states (default: ['idle', 'mixed', 'allocated'])
                         Common states: idle, mixed, allocated, down, drain, draining, drained
            sinfo_partitions: Optional whitelist of partition names to check (e.g., ['boost_usr_prod', 'dcgp_usr_prod'])
                             If None, all partitions are checked
        """
        if system_df is not None:
            self.df = system_df.copy()
        elif csv_path:
            self.df = load_leonardo_system_data(csv_path)
        else:
            map_file = Path(os.path.dirname(os.path.abspath(__file__))) / "leo_map.txt"
            self.df = load_leonardo_system_data(map_file)

        self._validate_dataframe()

        # Verify nodes with sinfo if requested
        self.unavailable_nodes = set()
        if verify_with_sinfo:
            if sinfo_states is None:
                sinfo_states = ["idle", "mixed", "allocated"]
            self._verify_nodes_with_sinfo(sinfo_states, sinfo_partitions)

        self._precompute_topology()

    def _verify_nodes_with_sinfo(
        self,
        acceptable_states: List[str],
        partition_whitelist: Optional[List[str]] = None,
    ):
        """
        Verify node availability using sinfo and mark unavailable nodes.

        Args:
            acceptable_states: List of node states considered available
            partition_whitelist: Optional list of partition names to filter (e.g., ['boost_usr_prod'])
                                If None, all partitions are considered
        """
        import subprocess

        partition_msg = (
            f" (partitions: {partition_whitelist})"
            if partition_whitelist
            else " (all partitions)"
        )
        print(
            f"Verifying node availability with sinfo{partition_msg} (acceptable states: {acceptable_states})..."
        )

        try:
            # Query sinfo for node states
            # Format: %N=nodelist, %T=state, %P=partition
            cmd = ["sinfo", "-N", "--noheader", "-o", "%N|%T|%P"]

            # Add partition filter if specified
            if partition_whitelist:
                cmd.extend(["-p", ",".join(partition_whitelist)])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Warning: sinfo command failed: {e}")
            print("Proceeding without node verification.")
            return
        except FileNotFoundError:
            print("Warning: sinfo command not found.")
            print("Proceeding without node verification.")
            return

        # Parse sinfo output
        available_nodes = set()
        unavailable_nodes = set()
        nodes_by_partition = {}

        for line in output.split("\n"):
            if not line.strip():
                continue

            parts = line.split("|")
            if len(parts) < 3:
                continue

            nodename = parts[0].strip()
            state = parts[1].strip().lower()
            partition = parts[2].strip()

            # Extract node number from nodename (e.g., "lrdn0001" -> 1)
            node_match = re.search(r"(\d+)", nodename)
            if not node_match:
                continue

            node_id = int(node_match.group(1))

            # Track partitions for informational purposes
            if partition not in nodes_by_partition:
                nodes_by_partition[partition] = {
                    "available": set(),
                    "unavailable": set(),
                }

            # Check if node is in acceptable state
            # Handle compound states like "idle*", "drain*", "mixed+drain"
            base_state = state.split("*")[0].split("+")[0].split("~")[0]

            if base_state in [s.lower() for s in acceptable_states]:
                available_nodes.add(node_id)
                nodes_by_partition[partition]["available"].add(node_id)
            else:
                unavailable_nodes.add(node_id)
                nodes_by_partition[partition]["unavailable"].add(node_id)

        # Filter out nodes from dataframe that are not available
        all_nodes = set(self.df["NODE"].values)

        # If partition whitelist is specified, also mark nodes not in those partitions as unavailable
        if partition_whitelist:
            nodes_in_whitelisted_partitions = set()
            for part in partition_whitelist:
                if part in nodes_by_partition:
                    nodes_in_whitelisted_partitions.update(
                        nodes_by_partition[part]["available"]
                    )
                    nodes_in_whitelisted_partitions.update(
                        nodes_by_partition[part]["unavailable"]
                    )

            # Nodes not in sinfo output (not in whitelisted partitions) are unavailable
            nodes_not_in_whitelist = all_nodes - nodes_in_whitelisted_partitions
            unavailable_nodes.update(nodes_not_in_whitelist)

            self.unavailable_nodes = (
                all_nodes - available_nodes
            ) | nodes_not_in_whitelist
        else:
            self.unavailable_nodes = all_nodes - available_nodes

        if self.unavailable_nodes:
            print(
                f"Found {len(self.unavailable_nodes)} unavailable nodes (not in states: {acceptable_states})"
            )
            print(f"Unavailable nodes will be excluded from nodelist generation.")

            # Print partition breakdown
            if partition_whitelist and len(nodes_by_partition) > 0:
                print(f"\nPartition breakdown:")
                for part, counts in nodes_by_partition.items():
                    print(
                        f"  {part}: {len(counts['available'])} available, {len(counts['unavailable'])} unavailable"
                    )

            # Optionally print some examples
            if len(self.unavailable_nodes) <= 20:
                print(f"Unavailable nodes: {sorted(list(self.unavailable_nodes))}")
            else:
                sample = sorted(list(self.unavailable_nodes))[:10]
                print(
                    f"Sample unavailable nodes: {sample} ... (and {len(self.unavailable_nodes) - 10} more)"
                )
        else:
            print(f"All {len(all_nodes)} nodes from topology file are available.")

    def _validate_dataframe(self):
        """Validate the input dataframe has required columns"""
        required_cols = ["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _precompute_topology(self):
        """Precompute topology mappings for efficient lookup"""
        # Filter out unavailable nodes
        if self.unavailable_nodes:
            df_available = self.df[~self.df["NODE"].isin(self.unavailable_nodes)]
        else:
            df_available = self.df

        # Group nodes by various topology levels
        self.nodes_by_cell = df_available.groupby("CELL")["NODE"].apply(list).to_dict()
        self.nodes_by_rack = df_available.groupby("RACK")["NODE"].apply(list).to_dict()
        self.nodes_by_switch = (
            df_available.groupby("SWITCH")["NODE"].apply(list).to_dict()
        )
        self.nodes_by_partition = (
            df_available.groupby("PARTITION")["NODE"].apply(list).to_dict()
        )

        # Create reverse mappings (use full df for lookups)
        self.node_to_cell = self.df.set_index("NODE")["CELL"].to_dict()
        self.node_to_rack = self.df.set_index("NODE")["RACK"].to_dict()
        self.node_to_switch = self.df.set_index("NODE")["SWITCH"].to_dict()
        self.node_to_partition = self.df.set_index("NODE")["PARTITION"].to_dict()

        # Map cells to racks and switches
        self.cell_to_racks = (
            df_available.groupby("CELL")["RACK"]
            .apply(lambda x: list(x.unique()))
            .to_dict()
        )
        self.cell_to_switches = (
            df_available.groupby("CELL")["SWITCH"]
            .apply(lambda x: list(x.unique()))
            .to_dict()
        )

        # Determine nodes per switch (for Booster vs DCGP)
        self.switch_capacities = (
            df_available.groupby("SWITCH")["NODE"].count().to_dict()
        )

    def generate_nodelists(
        self,
        resources: SlurmResources,
        constraint: TopologyConstraint = TopologyConstraint.NO_CONSTRAINT,
        max_nodelists: int = 10,
        min_nodelists: int = 5,
    ) -> List[List[int]]:
        """
        Generate nodelists that satisfy resource and topology constraints.

        Args:
            resources: SLURM resource requirements
            constraint: Network topology constraint
            max_nodelists: Maximum number of nodelists to generate
            min_nodelists: Minimum number of nodelists to generate

        Returns:
            List of nodelists, where each nodelist is a list of node IDs
        """
        # Filter by partition if specified
        available_nodes = self._filter_by_partition(resources.partition)

        # Generate candidate nodelists based on constraint
        if constraint == TopologyConstraint.SAME_CELL:
            candidates = self._generate_same_cell(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.SAME_RACK:
            candidates = self._generate_same_rack(available_nodes, resources.num_nodes)
        elif constraint == TopologyConstraint.SAME_SWITCH:
            candidates = self._generate_same_switch(
                available_nodes, resources.num_nodes
            )
        elif constraint == TopologyConstraint.DIFFERENT_CELLS:
            candidates = self._generate_different_cells(
                available_nodes, resources.num_nodes
            )
        elif constraint == TopologyConstraint.DIFFERENT_RACKS:
            candidates = self._generate_different_racks(
                available_nodes, resources.num_nodes
            )
        elif constraint == TopologyConstraint.MINIMIZE_L2_HOPS:
            candidates = self._generate_minimize_l2(
                available_nodes, resources.num_nodes
            )
        else:  # NO_CONSTRAINT
            candidates = self._generate_no_constraint(
                available_nodes, resources.num_nodes, max_nodelists
            )

        # Ensure we have enough nodelists
        nodelists = list(candidates)[:max_nodelists]

        if len(nodelists) < min_nodelists:
            print(
                f"Warning: Only {len(nodelists)} nodelists generated (requested min: {min_nodelists})"
            )

        return nodelists

    def _filter_by_partition(self, partition: Optional[int]) -> Set[int]:
        """Filter available nodes by partition"""
        if partition is None:
            return set(self.df["NODE"].values) - self.unavailable_nodes
        return set(self.nodes_by_partition.get(partition, []))

    def _generate_same_cell(
        self, available_nodes: Set[int], num_nodes: int
    ) -> List[List[int]]:
        """Generate nodelists with all nodes in the same cell"""
        nodelists = []

        for cell, nodes in self.nodes_by_cell.items():
            cell_nodes = [n for n in nodes if n in available_nodes]
            if len(cell_nodes) >= num_nodes:
                # Generate multiple combinations from this cell
                for i in range(min(3, len(cell_nodes) - num_nodes + 1)):
                    nodelist = sorted(cell_nodes[i : i + num_nodes])
                    nodelists.append(nodelist)

        return nodelists

    def _generate_same_rack(
        self, available_nodes: Set[int], num_nodes: int
    ) -> List[List[int]]:
        """Generate nodelists with all nodes in the same rack"""
        nodelists = []

        for rack, nodes in self.nodes_by_rack.items():
            rack_nodes = [n for n in nodes if n in available_nodes]
            if len(rack_nodes) >= num_nodes:
                nodelist = sorted(rack_nodes[:num_nodes])
                nodelists.append(nodelist)

        return nodelists

    def _generate_same_switch(
        self, available_nodes: Set[int], num_nodes: int
    ) -> List[List[int]]:
        """Generate nodelists with all nodes on the same L1 switch"""
        nodelists = []

        for switch, nodes in self.nodes_by_switch.items():
            switch_nodes = [n for n in nodes if n in available_nodes]
            if len(switch_nodes) >= num_nodes:
                nodelist = sorted(switch_nodes[:num_nodes])
                nodelists.append(nodelist)

        return nodelists

    def _generate_different_cells(
        self, available_nodes: Set[int], num_nodes: int
    ) -> List[List[int]]:
        """Generate nodelists with nodes spread across different cells"""
        nodelists = []

        # Get cells sorted by number of available nodes
        cells_with_nodes = []
        for cell, nodes in self.nodes_by_cell.items():
            cell_nodes = [n for n in nodes if n in available_nodes]
            if cell_nodes:
                cells_with_nodes.append((cell, cell_nodes))

        cells_with_nodes.sort(key=lambda x: len(x[1]), reverse=True)

        if len(cells_with_nodes) < 2:
            return []

        # Strategy: distribute nodes across cells as evenly as possible
        for start_idx in range(min(3, len(cells_with_nodes))):
            nodelist = []
            cells_to_use = cells_with_nodes[start_idx:]

            nodes_per_cell = num_nodes // len(cells_to_use)
            remainder = num_nodes % len(cells_to_use)

            for i, (cell, nodes) in enumerate(cells_to_use):
                n_from_cell = nodes_per_cell + (1 if i < remainder else 0)
                if len(nodes) >= n_from_cell:
                    nodelist.extend(nodes[:n_from_cell])
                else:
                    break

            if len(nodelist) == num_nodes:
                nodelists.append(sorted(nodelist))

        return nodelists

    def _generate_different_racks(
        self, available_nodes: Set[int], num_nodes: int
    ) -> List[List[int]]:
        """Generate nodelists with nodes spread across different racks"""
        nodelists = []

        # Get racks sorted by number of available nodes
        racks_with_nodes = []
        for rack, nodes in self.nodes_by_rack.items():
            rack_nodes = [n for n in nodes if n in available_nodes]
            if rack_nodes:
                racks_with_nodes.append((rack, rack_nodes))

        racks_with_nodes.sort(key=lambda x: len(x[1]), reverse=True)

        if len(racks_with_nodes) < 2:
            return []

        # Distribute nodes across racks
        for start_idx in range(min(3, len(racks_with_nodes))):
            nodelist = []
            racks_to_use = racks_with_nodes[start_idx:]

            nodes_per_rack = max(1, num_nodes // len(racks_to_use))

            for rack, nodes in racks_to_use:
                nodelist.extend(nodes[:nodes_per_rack])
                if len(nodelist) >= num_nodes:
                    break

            if len(nodelist) >= num_nodes:
                nodelists.append(sorted(nodelist[:num_nodes]))

        return nodelists

    def _generate_minimize_l2(
        self, available_nodes: Set[int], num_nodes: int
    ) -> List[List[int]]:
        """Generate nodelists minimizing inter-cell (L2) communication"""
        # Prioritize same-cell, then fall back to minimal cell usage
        same_cell = self._generate_same_cell(available_nodes, num_nodes)
        if same_cell:
            return same_cell

        # Find minimum number of cells needed
        nodelists = []
        cells_with_nodes = []
        for cell, nodes in self.nodes_by_cell.items():
            cell_nodes = [n for n in nodes if n in available_nodes]
            if cell_nodes:
                cells_with_nodes.append((cell, cell_nodes))

        cells_with_nodes.sort(key=lambda x: len(x[1]), reverse=True)

        # Try to use as few cells as possible
        for num_cells in range(2, min(5, len(cells_with_nodes) + 1)):
            for cell_combo in itertools.combinations(cells_with_nodes, num_cells):
                nodelist = []
                for cell, nodes in cell_combo:
                    nodelist.extend(nodes)
                    if len(nodelist) >= num_nodes:
                        break

                if len(nodelist) >= num_nodes:
                    nodelists.append(sorted(nodelist[:num_nodes]))
                    if len(nodelists) >= 5:
                        return nodelists

        return nodelists

    def _generate_no_constraint(
        self, available_nodes: Set[int], num_nodes: int, max_lists: int
    ) -> List[List[int]]:
        """Generate nodelists without topology constraints"""
        nodes_list = sorted(list(available_nodes))

        if len(nodes_list) < num_nodes:
            return []

        nodelists = []

        # Generate diverse nodelists by starting at different positions
        step = max(1, (len(nodes_list) - num_nodes) // max_lists)

        for i in range(0, min(len(nodes_list) - num_nodes + 1, max_lists * step), step):
            nodelist = nodes_list[i : i + num_nodes]
            nodelists.append(nodelist)

        return nodelists

    def format_nodelist_for_slurm(self, nodelist: List[int]) -> str:
        """
        Format nodelist for SLURM using compact range notation with 'lrdn' prefix
        and 4-digit zero-padded node IDs.

        Args:
            nodelist: List of node IDs

        Returns:
            SLURM-compatible nodelist string (e.g., "lrdn[0001-0010,0015,0020-0025]")
        """
        if not nodelist:
            return ""

        sorted_nodes = sorted(nodelist)
        ranges = []
        start = sorted_nodes[0]
        end = sorted_nodes[0]

        for node in sorted_nodes[1:]:
            if node == end + 1:
                end = node
            else:
                if start == end:
                    ranges.append(f"{start:04d}")
                else:
                    ranges.append(f"{start:04d}-{end:04d}")
                start = node
                end = node

        # Add final range
        if start == end:
            ranges.append(f"{start:04d}")
        else:
            ranges.append(f"{start:04d}-{end:04d}")

        return f"lrdn[{','.join(ranges)}]"

    def analyze_nodelist_topology(self, nodelist: List[int]) -> Dict:
        """
        Analyze the topology characteristics of a nodelist.

        Args:
            nodelist: List of node IDs

        Returns:
            Dictionary with topology statistics
        """
        cells = set(self.node_to_cell[n] for n in nodelist)
        racks = set(self.node_to_rack[n] for n in nodelist)
        switches = set(self.node_to_switch[n] for n in nodelist)
        partitions = set(self.node_to_partition[n] for n in nodelist)

        return {
            "num_nodes": len(nodelist),
            "num_cells": len(cells),
            "num_racks": len(racks),
            "num_switches": len(switches),
            "cells": sorted(list(cells)),
            "racks": sorted(list(racks)),
            "partitions": sorted(list(partitions)),
            "uses_l2": len(cells) > 1,
            "topology_type": self._classify_topology(
                len(cells), len(racks), len(switches)
            ),
        }

    def _classify_topology(
        self, num_cells: int, num_racks: int, num_switches: int
    ) -> str:
        """Classify the topology of a nodelist"""
        if num_switches == 1:
            return "same_switch"
        elif num_racks == 1:
            return "same_rack"
        elif num_cells == 1:
            return "same_cell"
        else:
            return "multi_cell"

    def rank_nodelists_by_availability(
        self,
        nodelists: List[List[int]],
        partition: Optional[str] = None,
        consider_pending: bool = True,
        time_weight: float = 0.3,
    ) -> List[Tuple[List[int], float, Dict]]:
        """
        Rank nodelists by their likelihood of being scheduled quickly based on current queue state.

        Args:
            nodelists: List of candidate nodelists to rank
            partition: Partition to query (None = all partitions)
            consider_pending: Whether to consider pending jobs in scoring
            time_weight: Weight for time-based scoring (0-1). Higher = prioritize jobs ending sooner

        Returns:
            List of tuples (nodelist, score, details) sorted by score (higher = better availability)
        """
        # Get queue state
        queue_state = self._query_slurm_queue(partition)

        # Parse and build node usage maps
        node_usage = self._build_node_usage_map(queue_state, consider_pending)

        # Score each nodelist
        scored_nodelists = []
        for nodelist in nodelists:
            score, details = self._score_nodelist_availability(
                nodelist, node_usage, time_weight
            )
            scored_nodelists.append((nodelist, score, details))

        # Sort by score (descending - higher is better)
        scored_nodelists.sort(key=lambda x: x[1], reverse=True)

        return scored_nodelists

    def _query_slurm_queue(self, partition: Optional[str] = None) -> List[Dict]:
        """
        Query SLURM queue and parse results.

        Args:
            partition: Partition to query (None = all partitions)

        Returns:
            List of job dictionaries with parsed information
        """
        import subprocess
        import shlex

        # Define format string with relevant fields
        # %D=nodes, %N=nodelist, %T=state, %L=time_left, %e=end_time, %S=start_time, %P=partition
        format_str = "%i|%D|%N|%T|%L|%e|%S|%P|%Q|%r"

        # Build squeue command
        cmd = ["squeue", "--noheader", "-o", format_str]
        if partition:
            cmd.extend(["-p", PARTITION_NAME_MAP[str(partition)]])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Warning: squeue command failed: {e}")
            return []
        except FileNotFoundError:
            print("Warning: squeue command not found. Returning empty queue state.")
            return []

        # Parse output
        jobs = []
        for line in output.split("\n"):
            if not line.strip():
                continue

            parts = line.split("|")
            if len(parts) < 10:
                continue

            job = {
                "job_id": parts[0].strip(),
                "num_nodes": self._parse_int_safe(parts[1]),
                "nodelist": parts[2].strip(),
                "state": parts[3].strip(),
                "time_left": parts[4].strip(),
                "end_time": parts[5].strip(),
                "start_time": parts[6].strip(),
                "partition": parts[7].strip(),
                "priority": self._parse_int_safe(parts[8]),
                "reason": parts[9].strip(),
            }

            # Parse nodelist into individual nodes
            job["nodes"] = self._expand_slurm_nodelist(job["nodelist"])

            # Parse time left into seconds
            job["time_left_seconds"] = self._parse_time_to_seconds(job["time_left"])

            jobs.append(job)

        return jobs

    def _parse_int_safe(self, value: str, default: int = 0) -> int:
        """Safely parse integer from string"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _expand_slurm_nodelist(self, nodelist_str: str) -> List[int]:
        """
        Expand SLURM nodelist notation into individual node IDs.

        Examples:
            "node[0001-0010]" -> [1, 2, 3, ..., 10]
            "node[0001,0005,0010-0012]" -> [1, 5, 10, 11, 12]
        """
        import re

        if not nodelist_str or nodelist_str in ["None", "N/A", ""]:
            return []

        nodes = []

        # Match patterns like node[0001-0010,0015,0020-0025]
        match = re.search(r"\[([^\]]+)\]", nodelist_str)
        if not match:
            # Try to extract single node number
            numbers = re.findall(r"\d+", nodelist_str)
            if numbers:
                return [int(numbers[-1])]  # Take last number as node ID
            return []

        ranges_str = match.group(1)

        for part in ranges_str.split(","):
            part = part.strip()
            if "-" in part:
                # Range: 0001-0010
                start, end = part.split("-")
                start_num = int(start)
                end_num = int(end)
                nodes.extend(range(start_num, end_num + 1))
            else:
                # Single node: 0001
                nodes.append(int(part))

        return nodes

    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        Parse SLURM time format to seconds.

        Formats: "days-hours:minutes:seconds", "hours:minutes:seconds", "minutes:seconds"
        Special: "UNLIMITED", "NOT_SET", "INVALID"
        """
        if not time_str or time_str in ["UNLIMITED", "NOT_SET", "INVALID", "N/A"]:
            return None

        try:
            # Handle days
            if "-" in time_str:
                days_str, time_str = time_str.split("-")
                days = int(days_str)
            else:
                days = 0

            # Split time components
            parts = time_str.split(":")

            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
            elif len(parts) == 2:
                hours = 0
                minutes, seconds = map(int, parts)
            elif len(parts) == 1:
                hours = 0
                minutes = 0
                seconds = int(parts[0])
            else:
                return None

            total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
            return float(total_seconds)

        except (ValueError, AttributeError):
            return None

    def _build_node_usage_map(
        self, jobs: List[Dict], consider_pending: bool
    ) -> Dict[int, Dict]:
        """
        Build a map of node usage from queue state.

        Args:
            jobs: List of parsed job dictionaries
            consider_pending: Whether to include pending jobs

        Returns:
            Dictionary mapping node_id -> usage information
        """
        node_usage = {}

        for job in jobs:
            state = job["state"]

            # Skip completed/failed jobs
            if state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]:
                continue

            # Skip pending jobs if not considering them
            if state == "PENDING" and not consider_pending:
                continue

            for node in job["nodes"]:
                if node not in node_usage:
                    node_usage[node] = {
                        "running_jobs": [],
                        "pending_jobs": [],
                        "earliest_free_time": None,
                        "total_priority_pending": 0,
                    }

                if state == "RUNNING":
                    node_usage[node]["running_jobs"].append(job)

                    # Update earliest free time
                    if job["time_left_seconds"] is not None:
                        free_time = job["time_left_seconds"]
                        if (
                            node_usage[node]["earliest_free_time"] is None
                            or free_time < node_usage[node]["earliest_free_time"]
                        ):
                            node_usage[node]["earliest_free_time"] = free_time

                elif state == "PENDING":
                    node_usage[node]["pending_jobs"].append(job)
                    node_usage[node]["total_priority_pending"] += job.get("priority", 0)

        return node_usage

    def _score_nodelist_availability(
        self, nodelist: List[int], node_usage: Dict[int, Dict], time_weight: float
    ) -> Tuple[float, Dict]:
        """
        Score a nodelist based on current node availability.

        Higher score = better availability (more likely to be scheduled soon)

        Args:
            nodelist: List of node IDs to score
            node_usage: Node usage map from _build_node_usage_map
            time_weight: Weight for time-based scoring (0-1)

        Returns:
            Tuple of (score, details_dict)
        """
        num_nodes = len(nodelist)

        # Initialize counters
        free_nodes = 0
        running_nodes = 0
        pending_nodes = 0
        total_time_to_free = 0
        max_time_to_free = 0
        total_pending_priority = 0

        for node in nodelist:
            if node not in node_usage:
                # Node is completely free
                free_nodes += 1
            else:
                usage = node_usage[node]

                if usage["running_jobs"]:
                    running_nodes += 1
                    if usage["earliest_free_time"] is not None:
                        total_time_to_free += usage["earliest_free_time"]
                        max_time_to_free = max(
                            max_time_to_free, usage["earliest_free_time"]
                        )
                    else:
                        # Unknown end time - penalize heavily
                        max_time_to_free = float("inf")
                        total_time_to_free += 1e9

                if usage["pending_jobs"]:
                    pending_nodes += 1
                    total_pending_priority += usage["total_priority_pending"]

        # Calculate component scores (normalized 0-1)

        # 1. Free nodes score (higher = more free nodes)
        free_score = free_nodes / num_nodes

        # 2. Running nodes penalty (lower = fewer running nodes)
        running_penalty = running_nodes / num_nodes

        # 3. Time-based score (exponential decay - jobs ending soon are better)
        if max_time_to_free == float("inf") or max_time_to_free > 86400 * 7:  # > 7 days
            time_score = 0.0
        elif max_time_to_free == 0:
            time_score = 1.0
        else:
            # Exponential decay: jobs ending in 1 hour = 0.9, 1 day = 0.5, 3 days = 0.1
            decay_rate = 1.0 / 86400  # 1 day half-life
            time_score = np.exp(-decay_rate * max_time_to_free)

        # 4. Pending jobs penalty (lower = fewer/lower priority pending jobs)
        if pending_nodes == 0:
            pending_score = 1.0
        else:
            # Normalize by assuming max priority per node is 1e6
            avg_priority = total_pending_priority / num_nodes
            pending_score = max(0, 1.0 - (avg_priority / 1e6))

        # 5. Contiguity bonus - nodes in same cell/rack are better for scheduling
        topology = self.analyze_nodelist_topology(nodelist)
        if topology["num_cells"] == 1:
            contiguity_bonus = 0.2
        elif topology["num_racks"] <= 3:
            contiguity_bonus = 0.1
        else:
            contiguity_bonus = 0.0

        # Weighted combination
        base_weight = 1.0 - time_weight
        score = (
            free_score * 0.4 * base_weight  # 40% weight on free nodes
            + (1 - running_penalty) * 0.3 * base_weight  # 30% weight on not running
            + time_score * time_weight  # Variable weight on time
            + pending_score * 0.2 * base_weight  # 20% weight on pending
            + contiguity_bonus * 0.1  # 10% bonus for topology
        )

        # Details for debugging/reporting
        details = {
            "free_nodes": free_nodes,
            "running_nodes": running_nodes,
            "pending_nodes": pending_nodes,
            "free_percentage": free_nodes / num_nodes * 100,
            "max_time_to_free_hours": (
                max_time_to_free / 3600 if max_time_to_free != float("inf") else None
            ),
            "avg_time_to_free_hours": (
                (total_time_to_free / num_nodes) / 3600
                if total_time_to_free < 1e8
                else None
            ),
            "total_pending_priority": total_pending_priority,
            "contiguity": topology["topology_type"],
            "score_components": {
                "free_score": free_score,
                "running_penalty": running_penalty,
                "time_score": time_score,
                "pending_score": pending_score,
                "contiguity_bonus": contiguity_bonus,
            },
        }

        return score, details

    def get_node_distance(self, node1: int, node2: int) -> int:
        """
        Calculate the network distance between two nodes.

        Distance levels:
        - 0: Same node
        - 1: Different nodes, same L1 switch
        - 2: Different L1 switches, same L2 switch (same cell)
        - 3: Different L2 switches (different cells)

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Distance level (0-3)

        Raises:
            ValueError: If either node ID is not found in the topology
        """
        # Check if nodes exist
        if node1 not in self.node_to_cell:
            raise ValueError(f"Node {node1} not found in topology")
        if node2 not in self.node_to_cell:
            raise ValueError(f"Node {node2} not found in topology")

        # Same node
        if node1 == node2:
            return 0

        # Get topology information for both nodes
        switch1 = self.node_to_switch[node1]
        switch2 = self.node_to_switch[node2]
        cell1 = self.node_to_cell[node1]
        cell2 = self.node_to_cell[node2]

        # Same L1 switch (different nodes on same switch)
        if switch1 == switch2:
            return 1

        # Same L2 switch / same cell (different L1 switches, same cell)
        if cell1 == cell2:
            return 2

        # Different L2 switches (different cells)
        return 3


def parse_topology_file(path: str) -> pd.DataFrame:
    pattern = re.compile(
        r"NODE\s+(\d+)\s+RACK\s+(\d+)\s+CELL\s+(\d+)\s+ROW\s+(\d+)\s+PARTITION\s+(\d+)\s+SWITCH\s+(\d+)"
    )

    records = []
    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                records.append(match.groups())

    df = pd.DataFrame(
        records, columns=["NODE", "RACK", "CELL", "ROW", "PARTITION", "SWITCH"]
    ).astype(int)

    return df


def example_usage():
    """Example usage of the Leonardo nodelist generator"""

    # Load your system dataframe
    df = parse_topology_file("leo_map.txt")

    # Initialize generator with sinfo verification
    print("=== Initializing Generator with sinfo Verification ===")
    generator = LeonardoNodelistGenerator(
        df,
        verify_with_sinfo=True,
        sinfo_states=["idle", "mixed", "allocated"],  # Only accept these states
        sinfo_partitions=[
            "boost_usr_prod"
        ],  # Only check nodes in boost_usr_prod partition
    )

    # Define resource requirements
    resources = SlurmResources(
        num_nodes=32, tasks_per_node=4, cpus_per_task=8, gpus_per_node=4, partition=1
    )

    # Generate nodelists with different constraints
    print("\n=== Generating Candidate Nodelists ===")
    nodelists = generator.generate_nodelists(
        resources, constraint=TopologyConstraint.SAME_CELL, max_nodelists=10
    )

    print(f"Generated {len(nodelists)} candidate nodelists")

    # Rank by availability based on current queue
    print("\n=== Ranking Nodelists by Availability ===")
    ranked = generator.rank_nodelists_by_availability(
        nodelists,
        partition="boost_usr_prod",  # Adjust to your partition name
        consider_pending=True,
        time_weight=0.3,
    )

    print(f"\nTop 5 nodelists by availability:")
    for i, (nodelist, score, details) in enumerate(ranked[:5]):
        slurm_format = generator.format_nodelist_for_slurm(nodelist)
        print(f"\n{i+1}. Score: {score:.3f}")
        print(f"   Nodelist: {slurm_format}")
        print(
            f"   Free: {details['free_nodes']}/{len(nodelist)} ({details['free_percentage']:.1f}%)"
        )
        print(
            f"   Running: {details['running_nodes']}, Pending: {details['pending_nodes']}"
        )
        if details["max_time_to_free_hours"] is not None:
            print(f"   Max time to free: {details['max_time_to_free_hours']:.1f} hours")
        print(f"   Topology: {details['contiguity']}")


if __name__ == "__main__":
    example_usage()


def rank_nodelists(
    generator: LeonardoNodelistGenerator,
    candidate_lists: List[List[int]],
    partition: str,
    _print=False,
):
    ranked = generator.rank_nodelists_by_availability(
        candidate_lists,
        partition=partition,
        consider_pending=True,
        time_weight=0.3,
    )

    if _print:
        print(
            f"\n{'Rank':<6} {'Score':<8} {'Free%':<8} {'Running':<10} {'Pending':<10} {'Topology':<15} {'Nodelist'}"
        )
        print("-" * 100)

        for i, (nodelist, score, details) in enumerate(ranked[:100], 1):
            slurm_format = generator.format_nodelist_for_slurm(nodelist)
            # Truncate long nodelists
            if len(slurm_format) > 50:
                slurm_format = slurm_format[:47] + "..."

            print(
                f"{i:<6} {score:<8.3f} {details['free_percentage']:<8.1f} "
                f"{details['running_nodes']:<10} {details['pending_nodes']:<10} "
                f"{details['contiguity']:<15} {slurm_format}"
            )

    return ranked


def get_nodelists_emulating_nanjing(
    generator: LeonardoNodelistGenerator,
    partition: str,
    nodes: int,
    do_rank_nodelists=True,
    min_nodelists=4,
    max_candidates=100,
    do_prints=False,
) -> str | None:
    """
    The generated nodelists will ensure a placement with the following properties:
    - The nodes will be splitted into two groups
    - Each group will be placed under the same L1 switch
    - The two groups will be placed under different L1 switches
    - Only one cell will be used (thus avoiding distance 3 communication)
    """
    if nodes % 2 != 0:
        warnings.warn(
            f'To produce a "nanjing" placement, The number of nodes must be divisible by 2. Ignoring {nodes=}'
        )
        return None

    nodes_per_l1 = int(nodes / 2)
    if do_prints:
        print(
            f"\n\n================== {nodes} NODES ({nodes_per_l1} per L1 switch) =================="
        )
    resources = SlurmResources(
        num_nodes=nodes_per_l1, partition=PARTITION_TO_ID_MAP[partition]
    )
    constraint = TopologyConstraint.SAME_SWITCH
    lists = generator.generate_nodelists(
        resources, constraint, max_nodelists=max_candidates, min_nodelists=min_nodelists
    )

    cells = {}
    for l in lists:
        cell = generator.node_to_cell[l[0]]
        if not cells.get(cell):
            cells[cell] = []
        cells[cell].append(l)

    candidate_lists = []

    for _, c in cells.items():
        for l1 in c:
            for l2 in c:
                if l1 != l2:
                    if set(l1) & set(l2) != set():
                        if do_prints:
                            print(f"Warning: overlapping lists {l1} -- {l2}")
                    else:
                        candidate_lists.append(list(set(l1) | set(l2)))

    if do_rank_nodelists:
        ranked = rank_nodelists(
            generator,
            candidate_lists[: min(max_candidates, len(candidate_lists))],
            partition,
            _print=do_prints,
        )
        if len(ranked) > 0:
            return generator.format_nodelist_for_slurm(ranked[0][0])

    if len(candidate_lists) > 0:
        return generator.format_nodelist_for_slurm(candidate_lists[0])

    return None


def get_nodelists_emulating_haicgu(
    generator: LeonardoNodelistGenerator,
    partition: str,
    nodes: int,
    do_rank_nodelists=True,
    min_nodelists=1,
    max_candidates=5,
    do_prints=False,
) -> str | None:
    """
    The generated nodelists will ensure a placement under the same L1 switch.
    """

    if do_prints:
        print(
            f"\n\n================== {nodes} NODES (All under the same L1 switch) =================="
        )
    resources = SlurmResources(
        num_nodes=nodes, partition=PARTITION_TO_ID_MAP[partition]
    )
    constraint = TopologyConstraint.SAME_SWITCH
    candidate_lists = generator.generate_nodelists(
        resources, constraint, max_nodelists=max_candidates, min_nodelists=min_nodelists
    )

    if do_prints:
        print("-- Candidate Lists ---")
        print(candidate_lists[:10])

    if do_rank_nodelists:
        ranked = rank_nodelists(generator, candidate_lists, partition, _print=do_prints)
        if len(ranked) > 0:
            return generator.format_nodelist_for_slurm(ranked[0][0])

    if len(candidate_lists) > 0:
        return generator.format_nodelist_for_slurm(candidate_lists[0])

    return None


def get_nodelists_different_distances(
    generator: LeonardoNodelistGenerator,
    partition: str,
    nodes: int,
    do_rank_nodelists=True,
    min_nodelists=4,
    max_candidates=200,
    do_prints=False,
) -> None | str:
    """
    The generated nodelists will ensure a placement that contains nodes at distances 1, 2 and 3.
    For instance, consider an 8-node placement. 2 cells will be selected, for each of them, 2 pairs of nodes will share the same L2 switch.
    Each pair will be placed under the same L1 switch.
    """

    if nodes < 8:
        warnings.warn(
            f'To produce a "different distances" placement, a minimum of 8 nodes is required. Ignoring {nodes=}'
        )
        return None
    if nodes % 4 != 0:
        warnings.warn(
            f'In "different distances", nodes must be a multiple of 4. Ignoring {nodes=}'
        )
        return None

    nodes_per_l1 = int(nodes / 4)
    if do_prints:
        print(
            f"\n\n================== {nodes} NODES ({nodes_per_l1} per L1 switch, 2 different L1, 2 different cells) =================="
        )

    resources = SlurmResources(
        num_nodes=nodes_per_l1, partition=PARTITION_TO_ID_MAP[partition]
    )
    constraint = TopologyConstraint.SAME_SWITCH
    lists = generator.generate_nodelists(
        resources, constraint, max_nodelists=max_candidates, min_nodelists=min_nodelists
    )

    # Step 1 — group pairs by cell
    cells = {}
    for l in lists:  # each l is a pair of 2 nodes
        cell = generator.node_to_cell[l[0]]
        cells.setdefault(cell, []).append(l)

    if do_prints:
        print("--- L1 lists grouped by cell ---")
        pprint.pprint(cells)

    candidate_lists = []

    # Step 2 — choose two different cells
    cell_ids = list(cells.keys())

    for cellA in cell_ids:
        for cellB in cell_ids:
            if cellA == cellB:
                continue

            listsA = cells[cellA]
            listsB = cells[cellB]

            # Need at least 2 valid pairs per cell
            if len(listsA) < 2 or len(listsB) < 2:
                continue

            # Step 3 — choose 2 disjoint lists from cell A
            for a1 in listsA:
                for a2 in listsA:
                    if set(a1) & set(a2):
                        continue  # overlap → reject
                    # Step 4 — choose 2 disjoint pairs from cell B
                    for b1 in listsB:
                        for b2 in listsB:
                            if set(b1) & set(b2):
                                continue

                            # Step 5 — ensure group A and group B do not overlap
                            combined = a1 + a2 + b1 + b2
                            if len(set(combined)) != nodes:
                                continue  # overlap across cells

                            candidate_lists.append(list(combined))

    if do_prints:
        print("-- Candidate Lists ---")
        print(candidate_lists[:10])

    if do_rank_nodelists:
        ranked = rank_nodelists(generator, candidate_lists, partition, _print=do_prints)
        if len(ranked) > 0:
            return generator.format_nodelist_for_slurm(ranked[0][0])

    if len(candidate_lists) > 0:
        return generator.format_nodelist_for_slurm(candidate_lists[0])

    return None
