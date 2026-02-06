# HICREST Benchmark Collection

This repository gathers and uniforms the usage of benchmarks from different domains.

## Benchmarks

### Distributed Memory

- [HICREST NetGraph500](./Graph500/) (submodule)
- [NVIDIA High-Performance LINPACK](./HPL/NVIDIA) (downloaded from NVIDIA)

<!-- ### Shared Memory -->

## Structure

Each benchmark has its dedicated sub-directory. Benchmarks may have different versions. For each version another sub-directory is created.

Each benchmark directory contains the following files:

- `compile.sh`
- `configs.yaml`
- `jobs.yaml`
- `parse_results.py`
- `plots.py`
- `README.md`

## Usage

First, refer to the README for specific details.  
In general, experiments are run and managed using [SbatchMan](https://sbatchman.readthedocs.io/en/latest/), please ensure you have it available. To install it, refer to the [documentation](https://sbatchman.readthedocs.io/en/latest/install/install/).

For each benchmark, if not specified otherwise, the pipeline is the following:

```bash
# If the benchmark is a submodule,
# get the code using git submodules
git submodule init
git submodule update <benchmark_name>

cd <benchmark_dir>

# Create a directory to store SbatchMan configs and results
sbatchman init

# Generate configs
sbatchman configure -f configs.yaml

# Launch jobs
sbatchman launch -f jobs.yaml

# Monitor
squeue --me # If using SLURM or PBS
sbatchman status

# Once all jobs have run
python parse_results.py

# After gathering all results files
python plots.py <results_file_1> <results_file_2> ...

# Checkout the results in the `plot` directory
```

> [!NOTE]  
> We suggest using a dedicated Python virtual environment.  
> You can find the commonly required packages in `common/requirements_*.txt` 

## Adding a new benchmark

1) Create the sub-directory(ies)
2) Add the submodule (if needed): `git submodule add [-b <branch>] --name <name> <repository_clone_url> <subdirectory_path>`
3) Populate it with the files listed in the `Structure` section
4) Document!
