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


```bash
# Baselines
python experiments_generator.py -G $((352*4)) --util-min 0.8 --util-max 1.0 --util-steps 10 --use-topology --n-stochastic-patterns 6 --max-experiments 1 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 200 --placement-bin-med-hi 2.55 --output-json experiments_alps_baselines.json --baseline-extended

# Concurrent
# Powerlaw
python experiments_generator.py -G $((352*4)) --util-min 0.95 --util-max 1.0 --util-steps 10 --use-topology --n-stochastic-patterns 5 --max-experiments 15 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 200 --placement-bin-med-hi 2.55 --output-json experiments_alps.json
# Uniform
python experiments_generator.py -G $((352*4)) --util-min 0.94 --util-max 1.0 --util-steps 30 --n-stochastic-patterns 0 --include-uniform --max-experiments 20 --n-samples-per-bin 5 --k-max 300 --output-json experiments_alps_uniform.json

# Concurrent test @ 64 nodes
python experiments_generator.py -G $((64*4)) --util-min 0.95 --util-max 1.0 --util-steps 10 --use-topology --n-stochastic-patterns 5 --max-experiments 1 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 200 --placement-bin-med-hi 2.55 --output-json experiments_alps_test_64.json
python expand_experiments.py experiments_alps_test_64.json --placement-mode linear --system alps --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --output-dir experiments_alps_test_64

# python experiments_generator.py -G $((650*4)) --util-min 0.85 --util-max 1.0 --util-steps 40 --use-topology --n-stochastic-patterns 0 --include-uniform-patterns --max-experiments 10 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 400 --placement-bin-med-hi 2.55 --output-json experiments_jupiter_3_groups_uniform.json
```

```bash
# Baselines
py run_baselines_no_placement.py --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --cpus-per-task 72 --system alps experiments_alps_baselines.json --max-n-nodes 8 --no-serial

# Concurrent
python expand_experiments.py experiments_alps.json --placement-mode linear --system alps --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --output-dir experiments_alps

# python expand_experiments.py experiments_jupiter_3_groups_uniform.json --placement-mode runtime --system jupiter --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --output-dir experiments_jupiter_3_groups_uniform_nccl
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


### Jupiter

```bash
scontrol show reservation
sinfo -n "jpbo-016-[01-48],jpbo-017-[01-48],jpbo-018-[01-48],jpbo-019-[01-48],jpbo-020-[01-48],jpbo-046-[01-48],jpbo-047-[01-48],jpbo-048-[01-48],jpbo-049-[01-48],jpbo-050-[01-48],jpbo-096-[01-48],jpbo-097-[01-48],jpbo-098-[01-48],jpbo-099-[01-48],jpbo-100-[01-48]" -t idle,alloc,comp
```

```bash
# Baselines
python experiments_generator.py -G $((650*4)) --util-min 0.8 --util-max 1.0 --util-steps 10 --use-topology --n-stochastic-patterns 6 --max-experiments 1 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 200 --placement-bin-med-hi 2.55 --output-json experiments_jupiter_3_groups_baselines.json --baseline-extended

# Concurrent
python experiments_generator.py -G $((650*4)) --util-min 0.9 --util-max 1.0 --util-steps 5 --use-topology --n-stochastic-patterns 5 --max-experiments 15 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 200 --placement-bin-med-hi 2.55 --output-json experiments_jupiter_3_groups_powerlaw.json

python experiments_generator.py -G $((650*4)) --util-min 0.85 --util-max 1.0 --util-steps 40 --use-topology --n-stochastic-patterns 0 --include-uniform-patterns --max-experiments 10 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 400 --placement-bin-med-hi 2.55 --output-json experiments_jupiter_3_groups_uniform.json

# BACKUP 640 nodes
python experiments_generator.py -G $((640*4)) --util-min 0.92 --util-max 1.0 --util-steps 5 --use-topology --n-stochastic-patterns 5 --max-experiments 15 --n-samples-per-bin 1 --n-placement-samples-per-bin 1 --k-max 200 --placement-bin-med-hi 2.55 --output-json experiments_jupiter_3_groups_powerlaw.json
```

```bash
# Baselines
py run_baselines_placements.py --nodelist "jpbo-016-[01-27,29-46,48],jpbo-017-[01-33,35-36,38,40,43-48],jpbo-018-[01-15,17-20,23-30,32-33,35-42,44-45,48],jpbo-019-[01-06,08-22,25-28,30-40,43-44,47-48],jpbo-020-[01-30,32-35,37-48],jpbo-046-[01-09,11-24,27-34,36-48],jpbo-047-[01-08,10-20,23-30,32-36,39-48],jpbo-048-[01-08,10-20,22-35,37-43,45-46,48],jpbo-049-[02-09,11-16,19-28,30-48],jpbo-050-[01-48],jpbo-096-[01-05,07-33,35-48],jpbo-097-[01-17,19-28,32-48],jpbo-098-[01-20,23-25,27-48],jpbo-099-[01-46,48],jpbo-100-[01-06,09-20,23-30,32,34-48]" --system jupiter --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --cpus-per-task 72 experiments_jupiter_3_groups_baselines.json

py run_baselines_placements.py --nodelist "jpbo-004-[34,36,38-40,45],jpbo-005-[01-02,04-09,11,13-15,33,37-38,40],jpbo-007-[25-26,28,39-40],jpbo-008-[30-31],jpbo-011-[17-18,20-25,27,29,31-33,36-39,43-48],jpbo-012-[02-03,05-06,10-12,15-16,26,28,33-36,41-48],jpbo-014-[09-10,17-22,24,26-28,30-31],jpbo-015-[17,19-21,23-24,26-31,35,47-48],jpbo-018-[01-15,17-19,23-30,32-33,36,38-43,45,48],jpbo-019-[01-14,16-20,22,25-28,30-33,35-40,42-44,46,48],jpbo-031-[03,07,30-40,42-43,45-48],jpbo-032-[11,14,23-24,27-32],jpbo-033-[05-06,41-42],jpbo-034-[01-02,04-14,16,19-20,30,33-43,45-47],jpbo-035-[07,09,12,17,19-25,27-30,32],jpbo-036-[07-08,14-18,22-27],jpbo-037-[04,06-08,10,13,33-34,36-37,39-42,46],jpbo-040-[01-02,05-07,09-11,14-15,17-24,29,31],jpbo-045-[46-47],jpbo-046-[11-12,14,37-40],jpbo-047-[19-20],jpbo-048-[17,20,22,29-30],jpbo-050-[03,09,13-14],jpbo-051-[09-10,12-15],jpbo-053-[01-04,06-07,09-10,12,15],jpbo-054-31,jpbo-056-[03-04,07-16,23-26],jpbo-058-[17-18,25-27,29,33-36,38-45,48],jpbo-059-[01-07,09-10,12-14,29-31,33-38,42,44-46,48],jpbo-060-[12-13],jpbo-061-[18-19,21-22,30-31,33,40-43,45],jpbo-063-[02-03,09,13,34-40,43,45-48],jpbo-064-[33,35-37,39-45,47-48],jpbo-065-[11,13,17-34,38-44,47-48],jpbo-066-01,jpbo-070-[18-21,23,25,29],jpbo-071-[17-20,22-23,27-32,35-37,40-41,43-48],jpbo-072-[02,09-10,38-41,45-46],jpbo-073-[06,10,33-34,37-38,41,44-48],jpbo-074-[03-06,08,15-17,19-21,25-29,31-32],jpbo-075-[01-02,05,07-16,33-35,37,43,45-46,48],jpbo-076-[14-16],jpbo-079-[09,13,15],jpbo-080-[08,14-16,33-38,41,47-48],jpbo-091-[01-02,10-13,21-24,26],jpbo-092-[01,03-07,16,19-23,25-29,32-33,35-39,41-42,45-48],jpbo-093-[21-27],jpbo-094-[17-18,20-24,26-31],jpbo-095-[01,03,05-12,14-16,33-40,45,47-48],jpbo-096-[38-43],jpbo-097-[17,19-28,32],jpbo-098-[02-03,05,12,14-15],jpbo-099-[01-04,06-17,23-25,27],jpbo-100-[17-20,23-32,34-48],jpbo-112-[12,15-16,26-29],jpbo-114-[04-05,08,10-12],jpbo-115-[02,10,14,16],jpbo-116-[17-19,21-30],jpbo-117-[12-13],jpbo-119-[11-12],jpbo-121-[06,11-16,33-35],jpbo-122-[23-26,28,30-32,36-38,48],jpbo-123-[02-08,15-16]" --system jupiter --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --cpus-per-task 72 experiments_jupiter_baseline.json --max-n-nodes 256

# Concurrent
python expand_experiments.py experiments_jupiter_3_groups_powerlaw.json --placement-mode runtime --system jupiter --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --output-dir experiments_jupiter_3_groups_powerlaw_nccl

python expand_experiments.py experiments_jupiter_3_groups_uniform.json --placement-mode runtime --system jupiter --comm-lib nccl --gpu-model GH200 --gpus-per-node 4 --output-dir experiments_jupiter_3_groups_uniform_nccl
```


<!--
# python expand_experiments.py experiments_jupiter_3_groups_powerlaw.json   --placement-mode hardcoded --reserved-nodes "jpbo-016-[01-46,48],jpbo-017-[01-40,43-48],jpbo-018-[01-15,17-20,23-30,32-46,48],jpbo-019-[01-22,25-28,30-40,42-44,46-48],jpbo-020-[01-30,32-35,37-48],jpbo-046-[01-24,27-48],jpbo-047-[01-08,10-20,22-36,39-48],jpbo-048-[01-20,22-35,37-46,48],jpbo-049-[02-09,11-16,19-28,30-48],jpbo-050-[01-48],jpbo-096-[01-05,07-33,35-48],jpbo-097-[01-28,32-48],jpbo-098-[01-20,23-25,27-48],jpbo-099-[01-46,48],jpbo-100-[01-06,09-20,23-30,32,34-48]"   --system jupiter    --comm-lib nccl   --gpu-model GH200   --gpus-per-node 4  --output-dir experiments_jupiter_3_groups_powerlaw --use-placer-files

python expand_experiments.py experiments_jupiter_3_groups_powerlaw.json --placement-mode runtime --system jupiter --comm-lib mpi_gpu_cuda --gpu-model GH200 --gpus-per-node 4 --output-dir experiments_jupiter_3_groups_powerlaw_mpi_gpu_cuda
-->