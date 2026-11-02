# Tenstorrent benchmarks

This directory contains two SbatchMan campaigns and their shared parser and
plotter. C++ sources, CMake files, runtime launchers and power tools live in
[`../AI-accelerators-benchmark/tenstorrent`](../AI-accelerators-benchmark/tenstorrent).
The source repository includes both accelerator and CPU implementations;
these jobs schedule accelerator experiments.

| File | Purpose |
|---|---|
| `jobs.yaml` | MATRIX.md throughput campaign, without power measurement |
| `jobs_energy.yaml` | Two 16384-square GEMMs, with device and CPU power measurement |
| `config.yaml` | Local `disi-frankenstein` SbatchMan configuration |
| `compile.sh` | Build the sibling source repository |
| `parser.py` | Shared SbatchMan and stdout parser |
| `parse_results.py` | Export CSVs and coverage reports |
| `plots.py` | Generate combined throughput figures |

## Build and run

Initialize the benchmark submodule from the collection root if needed:

```bash
git submodule update --init AIAccelerators/AI-accelerators-benchmark
cd AIAccelerators/Tenstorrent
export TT_METAL_HOME=/path/to/compatible/tt-metal
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
python -m pip install -r ../../common/requirements_plots.txt PyYAML
bash compile.sh
sbatchman set-cluster-name disi-frankenstein
```

Use the server environment containing SbatchMan, Tracy and `tt-perf-report`.
The build output is `../AI-accelerators-benchmark/tenstorrent/build/bin/`.
Source changes must be published in the source repository before updating
the collection's submodule pin.

Run the throughput campaign:

```bash
sbatchman launch -f jobs.yaml
python parse_results.py
python plots.py
```

Run the energy campaign separately:

```bash
sbatchman launch -f jobs_energy.yaml
python parse_results.py --campaign energy
```

Both files specify sequential execution. Launch one campaign at a time so
processes from separate launches do not share the accelerator concurrently.

## Throughput campaign

`jobs.yaml` contains the 15 paired `(N, K)` shapes from
[MATRIX.md](../../MLBench/mlops/MATRIX.md), grouped as square, short/wide
and tall/narrow. It expands to 840 configurations, with five warmups and
ten measured iterations each. Every command uses `--no-power`.

| Operation | M | Settings |
|---|---|---|
| GEMM | 64, 512, 2048 | BFP8 LoFi/HiFi2; BF16 HiFi2/HiFi4; FP32 HiFi4 |
| GEMV | 1 | Same datatype/fidelity pairs as GEMM |
| GELU, last-dimension Softmax | 64, 512, 2048 | BFP8, BF16, FP32; interleaved and sharded layouts |

Matrix operations use `matmul`, interleaved memory and an 8×8 compute grid.
Nonlinear sharding uses an 8×8 grid with tile-aligned shard dimensions.
Large shapes may exceed device DRAM or L1 capacity; missing configurations
are listed in the coverage report.

Edit the top-level variables for token counts, repetitions and grids. Each
shape block sets a singleton `N` and `K`; keep them paired. The first shape
defines `&operation_jobs`, and the other shapes reuse `*operation_jobs`.
Operation overrides keep supported datatype/fidelity combinations together
and set GEMV to `M: [1]`. No generator is required.

The executable argument order is **M K N**, while MATRIX.md lists **M N K**.
Nonlinear executables receive **M N**. Preserve the tag format: the parser
reads logical dimensions and settings from expanded scheduled tags.

## Energy campaign

`jobs_energy.yaml` runs two sequential GEMMs with **M=N=K=16384**,
interleaved memory, an 8×8 compute grid and five warmups each:

| Datatype / fidelity | Measured iterations |
|---|---:|
| FP32 / HiFi4 | 100 |
| BFP8 / LoFi | 1,000 |

Both use `--cpu-power`, enabling device sampling through `tt-smi` and host
CPU-package energy through Linux `perf`. The requested device sampling
period is 0.333333 seconds, with five seconds of pre/post idle sampling.
BFP8 uses more iterations to lengthen its faster workload's sampling window.

The launcher reads `HICREST_MEASURED_INTERVAL` timestamps after the run and
writes `measured_energy_summary.json`. Both CPU and device energy use the
same measured-loop start and end: after warmups, through the final device
synchronization, before the final output transfer. Sampling can continue
outside the loop, but those periods are excluded from the bounded energy.

The CSV retains these fields:

| Field | Meaning |
|---|---|
| `measured_loop_duration_s` | Wall-clock duration of the synchronized GEMM loop |
| `device_measured_energy_j` | Device power integrated over that loop |
| `cpu_measured_energy_pkg_j` | CPU-package energy allocated to the same loop |
| `measured_total_energy_j` | Device plus CPU-package energy over the loop |
| `measured_energy_coverage_complete` | Both sampling streams cover the full interval |
| `device_measured_interior_samples` | Number of device readings strictly inside the loop |
| `device_measured_max_sample_gap_s` | Largest device sample spacing used in integration |

Device energy uses trapezoidal integration with linearly interpolated
boundary readings. CPU counters include high-resolution interval timestamps;
energy crossing a loop boundary is prorated assuming uniform power within
that sample. The perf stream is anchored to host realtime at process launch,
so timestamp alignment has process-startup uncertainty. Sysfs sampling uses
timestamped counter-read boundaries. The CPU monitor finishes a full sample
after workload exit so its samples can cover the loop end.

Missing device boundary coverage, CPU gaps or legacy CPU logs without
interval timestamps produce unavailable bounded energy, not whole-process
fallback values. Inspect coverage and sample spacing before interpreting
energy efficiency. Whole-monitor fields such as `cpu_energy_pkg_j` remain
as diagnostics; use the `*_measured_*` fields for loop energy. The combined
total covers device plus CPU packages, not wall-plug system energy.
These configurations still require validation on the hardware.

## Results and figures

Parsed files are written under `results/`:

```text
results/
  tenstorrent_matrix_<cluster>_data.csv
  tenstorrent_matrix_<cluster>_data.coverage.json
  tenstorrent_energy_<cluster>_data.csv
  tenstorrent_energy_<cluster>_data.coverage.json
```

`parse_results.py` collects completed active and archived jobs for the chosen
campaign. The coverage JSON lists missing configurations. Saved stdout can
also be parsed; filenames must be the exact job tag plus `.log`:

```bash
python parse_results.py --logs /path/tt_matrix_*.log --cluster disi-frankenstein
python parse_results.py --campaign energy --logs /path/tt_energy_*.log --cluster disi-frankenstein
```

Throughput uses logical dimensions and mean device time: `2*M*N*K / time`
for matrix operations, and `M*N / time` for nonlinear operations. GEMV work
excludes tile padding. Kernel timings exclude initialization and transfers.

`plots.py` automatically reads matrix campaign CSVs and writes three combined
figures under `plots/`: `tenstorrent_square`, `tenstorrent_short_wide`, and
`tenstorrent_tall_narrow`, each as PDF, PNG and an aggregated CSV. Datatype
rows and token-count columns compare matrix throughput in GFLOP/s with
nonlinear throughput in Gelem/s on separate logarithmic axes. Repeated jobs
are aggregated by median; missing measurements leave gaps. The plot layout
uses fixed M=64/512/2048 columns. Energy results are excluded by default.

Explicit CSV paths can be passed to `plots.py`; `--outdir` selects the output
directory. Multiple clusters are plotted separately under `plots/<cluster>/`.
`parse_results.py --output PATH` overrides the CSV path for a single cluster.

Raw Tracy and power files remain in the source repository at
`tenstorrent/generated/sbatchman/<result-id>/`. Parsed results, figures and
raw generated artifacts are gitignored. Validation so far covers campaign
expansion and synthetic parsing/plotting; compilation and execution still
need verification on the Tenstorrent host.
