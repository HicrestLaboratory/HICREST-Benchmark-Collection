# Tenstorrent MATRIX.md campaign

The canonical shape list is [MATRIX.md](../../MLBench/mlops/MATRIX.md).
`matrix_suite.py` reads its 15 paired `(N, K)` shapes, including the two
rows with irregular Markdown table delimiters. GEMM uses M=64,512,2048;
GEMV uses M=1; GELU and last-dimension Softmax use the logical M×N output.
ReLU and the old square side-length sweeps are excluded.

The campaign retains the older local study's supported arithmetic pairs:
BFP8 LoFi/HiFi2, BF16 HiFi2/HiFi4, FP32 HiFi4. Matrix products use matmul
(without an added bias), interleaved memory and an 8×8 compute grid.
Nonlinear operations compare interleaved DRAM and sharded L1 layouts on
an 8×8 shard grid. There are 840 configurations, five warmups and ten
measured iterations each. Jobs run sequentially on one device.

## Build and run on the Tenstorrent host

```bash
git submodule update --init AIAccelerators/AI-accelerators-benchmark
cd AIAccelerators/Tenstorrent
export TT_METAL_HOME=/path/to/compatible/tt-metal
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
bash compile.sh
python matrix_suite.py
sbatchman set-cluster-name disi-frankenstein
sbatchman launch -f jobs.yaml
python -m pip install -r ../../common/requirements_plots.txt
python parse_results.py
python plots.py
```

C++ sources in `src/` are preserved from
`HICREST-Benchmark-Collection_old/AIAccelerators/Tenstorrent/src`.
The launcher and Tracy report tools in `scripts/` come from the existing
local `AI-accelerators-benchmark/tenstorrent/scripts` implementation.
The optional CPU power helpers are copied from that same local checkout.
Nonlinear shard dimensions are rounded up to 32×32 tile boundaries so
M=64 does not request sub-tile shards.
They are tracked here so this contribution does not depend on unpublished
changes to the source submodule. The submodule supplies argparse headers;
CMake also accepts `ARGPARSE_INCLUDE_DIR` for another installation.
Compilation and device compatibility must be checked on the accelerator host.

## One parser and one plotter

`parser.py` provides `parse(job, stdout=None)` for SbatchMan, plus the same
record parser for offline use. `parse_results.py` collects active and archived
completed jobs into one CSV and writes a `.coverage.json` with missing tags.
Only exact canonical campaign tags are accepted, preventing historical square
sweeps from contaminating the new results. Saved logs can also be parsed:

```bash
python parse_results.py --logs /path/tt_matrix_*.log
```

Saved log filenames must be the exact scheduled tag plus `.log`.
Result metadata, including energy fields emitted by the launcher, is retained.
Throughput is recomputed from logical dimensions and mean device time:
`2*M*N*K / time` for matrix products, `M*N / time` for nonlinear operations.
GEMV work excludes padded elements. These are kernel timings, excluding
initialization and transfers; they do not establish numerical accuracy.

`plots.py` writes exactly three combined figures (square, short_wide,
tall_narrow), each as PDF and PNG with its aggregated CSV. Each figure has
rows for BFP8/BF16/FP32 and columns for M=64/512/2048; GEMV M=1 is repeated
for comparison. As in the older BFP8 preview, matrix and nonlinear curves
use separate logarithmic axes with GFLOP/s and Gelem/s respectively.
N/K labels are categorical paired shapes, not a scalar size sweep.
Repeated measurements are aggregated by median; CSV `count` is the number
of completed jobs, not the number of kernel iterations.

Large shapes may exceed device DRAM or sharded L1 capacity. They remain
in the requested campaign; failures must be reported through coverage and
must not be replaced with smaller shapes. Missing points appear as gaps.
The new shapes require new hardware measurements: archived square results
cannot be relabeled as this campaign.

The older `jobs_gemm*.yaml`, `jobs_nonlinearity.yaml`, `vars_*.yaml`, and
specialized `parse_results_*.py` scripts are retained only as legacy files.
Use `jobs.yaml`, `parse_results.py`, and `plots.py` for this campaign.

## Local validation

```bash
python -m unittest discover -s tests -v
```

Campaign membership, generated argument order, logical GEMV throughput,
nonlinear units, invalid records and exclusion of old tags are checked.
All 840 cases were also passed through the parser with synthetic timings
and the three PDF/PNG figures rendered successfully. Synthetic figures are
validation artifacts only and are not benchmark measurements. No accelerator
compilation or hardware run has been performed on this Mac.
