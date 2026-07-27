# Tenstorrent benchmarks

This directory contains SbatchMan jobs for GEMM, single-core GEMM, and
nonlinearity benchmarks. All jobs run sequentially with the
`tenstorrent_local` configuration.

## Configuration

`config.yaml` defines the local `disi-frankenstein` target and exports
`TT_METAL_HOME` and `TT_METAL_RUNTIME_ROOT`. Select it once with:

```bash
sbatchman set-cluster-name disi-frankenstein
```

## Launch jobs

```bash
sbatchman launch -f jobs_gemm.yaml
sbatchman launch -f jobs_gemm_single_core.yaml
sbatchman launch -f jobs_nonlinearity.yaml
```

Single-core GEMM uses the regular GEMM executable with a `1 x 1` compute
grid. Keep the jobs sequential because concurrent processes must not share
the same Tenstorrent device.

## Parse results

Each completed benchmark prints one `HICREST_RESULT` JSON record. Generate
separate CSV files with:

```bash
python parse_results_gemm.py
python parse_results_gemm_single_core.py
python parse_results_nonlinearity.py
```

The CSV files are written under `results/`. `result_writer.py` contains the
shared SbatchMan collection, filtering, sorting, and CSV-writing logic.
