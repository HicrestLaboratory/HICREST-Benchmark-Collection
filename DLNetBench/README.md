# HICREST DLNetBench

## Compile

Before compiling be sure to have the modules loaded. Look at the `configs.yaml` to check wich module to load for each system.

```bash
./compile <system_name>
```

Supported systems:
- `leonardo`
- `alps`
- `jupiter`
- `lumi`
- `baldo`
- `cpu` -> it is just a way to use the benchmarks using CPU-only MPI

## General Workflow

The experiment pipeline relies on a two-step process using Python scripts to first generate a workload configuration, and then expand that configuration into system-specific run files.

* **`experiments_generator.py`**: This script generates a JSON file containing the theoretical experiment layouts. You define the total scale (e.g., `-G` for total GPUs), the desired utilization boundaries (`--util-min`, `--util-max`), and the statistical nature of the workload (e.g., stochastic powerlaw patterns vs. uniform patterns). 
* **`expand_experiments.py`**: This script takes the generated JSON and translates it into physical placement maps and job directories tailored to a specific supercomputer. It applies system-specific constraints such as node reservations, GPU models, and the required communication library (like NCCL).
* **`run_baselines_placements.py`**: A secondary utility used to establish baseline metrics for the generated experiment setups, allowing for performance comparisons against the custom placements.

---

## Generic Command Sequence & Common Flags

Regardless of the target supercomputer, the execution order follows this pattern. First, you generate the theoretical job distributions, and second, you expand those into actual system placements. 

### Step 1: Generate the Workload JSON
This step creates the statistical distribution of jobs based on the scale and utilization you want to test.

**Common Generation Flags:**
* `-G`: The total number of GPUs for the experiment (e.g., nodes * GPUs per node).
* `--util-min` / `--util-max`: Defines the lower and upper bounds for system utilization (e.g., 0.8 to 1.0).
* `--n-stochastic-patterns`: Number of stochastic (e.g., powerlaw) workload patterns to generate. Set to 0 if only using uniform patterns.
* `--max-experiments`: The upper limit of experiment configurations to generate per bin.
* `--output-json`: The name of the resulting JSON file containing the generated layouts.

*Example generic structure:*
```bash
python experiments_generator.py -G <TOTAL_GPUS> --util-min <MIN> --util-max <MAX> --output-json <FILENAME.json>
```

### Step 2: Expand into System Placements
This step applies hardware and topology constraints to the theoretical JSON, creating the actual files needed to run the jobs.

**Common Expansion Flags:**
* `--placement-mode`: Defines how jobs are mapped to nodes (e.g., `hardcoded`, `linear`).
* `--system`: The target HPC environment (e.g., `leonardo`, `alps`).
* `--reserved-nodes`: A specific list or range of nodes reserved for this experiment.
* `--gpu-model` / `--gpus-per-node`: Hardware constraints for the target nodes.
* `--comm-lib`: The communication library to use, typically `nccl`.
* `--output-dir`: The directory where the expanded job scripts and placement files will be saved.
* `--use-placer-files`: JobPlacer will use txt file containing the output of sinfo and scontrol show topology.

*Example generic structure:*
```bash
python expand_experiments.py <FILENAME.json> --system <SYSTEM_NAME> --placement-mode <MODE> --output-dir <DIR_NAME>
```



## System-Specific Configurations

### 1. Leonardo (@ 340 Nodes)

The Leonardo system experiments with a total of 340-node setup using A100 GPUs (4 per node). 

#### Realistic Powerlaw Workload
This configuration uses stochastic patterns to simulate a realistic powerlaw distribution of jobs.

**1. Generate the experiment JSON:**
```bash
python experiments_generator.py -G $((510*4)) \
  --util-min 0.8 \
  --util-max 1.0 \
  --use-topology \
  --n-stochastic-patterns 5 \
  --max-experiments 12 \
  --n-samples-per-bin 1 \
  --n-placement-samples-per-bin 1 \
  --k-max 200 \
  --placement-bin-med-hi 2.55 \
  --output-json experiments_leonardo_3_groups_powerlaw.json
```

**2. Expand for Leonardo:**
```bash
python expand_experiments.py experiments_leonardo_3_groups_powerlaw.json \
  --placement-mode runtime \
  --system leonardo \
  --comm-lib nccl \
  --gpu-model A100 \
  --gpus-per-node 4 \
  --small-job-threshold $((128*4)) \
  --output-dir experiments_leonardo_3_groups_powerlaw
```

#### Uniform Workload
This configuration disables stochastic patterns in favor of uniformly distributed workload patterns.

**1. Generate the experiment JSON:**
```bash
python experiments_generator.py -G $((510*4)) \
  --util-min 0.8 \
  --util-max 1.0 \
  --util-steps 8 \
  --use-topology \
  --n-stochastic-patterns 0 \
  --include-uniform-patterns \
  --max-experiments 25 \
  --n-samples-per-bin 1 \
  --n-placement-samples-per-bin 2 \
  --k-max 300 \
  --placement-bin-med-hi 2.35 \
  --output-json experiments_leonardo_3_groups_uniform.json
```

**2. Expand for Leonardo:**
```bash
python expand_experiments.py experiments_leonardo_3_groups_uniform.json \
  --placement-mode runtime \
  --system leonardo \
  --comm-lib nccl \
  --gpu-model A100 \
  --gpus-per-node 4 \
  --output-dir experiments_leonardo_3_groups_uniform \
```

---

### 2. Alps (@ 64 Nodes)

The Alps experiments target a 64-node scale using H200 GPUs. 

#### Environment Setup
Before running the Python scripts on Alps, the correct software modules and environment must be loaded:

```bash
uenv start --view=modules prgenv-gnu-openmpi/25.12:v1
ml gcc/14.3.0 cuda/12.9.1 nccl/2.28.9-1 openmpi/5.0.9 python/3.14.0
```

#### Concurrent Setup
**1. Generate the experiment JSON:**
```bash
python experiments_generator.py -G $((64*4)) \
  --util-min 0.8 \
  --util-max 1.0 \
  --use-topology \
  --n-stochastic-patterns 5 \
  --max-experiments 4 \
  --n-samples-per-bin 1 \
  --n-placement-samples-per-bin 1 \
  --k-max 200 \
  --placement-bin-med-hi 2.55 \
  --topo-q2 28 \
  --topo-q3 440 \
  --output-json experiments_alps_test_64.json
```

**2. Expand for Alps:**
```bash
python expand_experiments.py experiments_alps_test_64.json \
  --placement-mode linear \
  --system alps \
  --comm-lib nccl \
  --gpu-model H200 \
  --gpus-per-node 4 \
  --output-dir experiments_alps__test_64 \
```

#### Baseline
Run the baseline placements using the generated JSON. 

```bash
python run_baselines_placements.py json experiments_alps_test_64.json \
  --system leonardo \
  --comm-lib nccl \
  --gpu-model H200 \
  --gpus-per-node 4 \
```

---

### 3. DGX A100

A smaller-scale generation specifically for a single DGX A100 node (8 GPUs).

**Generate the experiment JSON:**
```bash
python experiments_generator.py -G 8 \
  --util-min 0.8 \
  --util-max 1.0 \
  --n-stochastic-patterns 5 \
  --max-experiments 60 \
  --output-json experiments_dgxA100.json \
  --dgx DGX_A100
```