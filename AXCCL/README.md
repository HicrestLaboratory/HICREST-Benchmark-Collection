# HICREST AXCCL

## Compile

```bash
./compile <system_name>
```

## Forcing Job Placement

To ensure the benchmark is run under controlled placements (emulating different topologies), there are two main strategies:
1) **Set an arbitrary SLURM nodelist in advance.** WIP...
2) **Get a (potentially bigger) random allocation and check if a subset of nodes satisfies placement constraints.** An example in [`jobs_2nodes_placement.yaml`](./jobs_2nodes_placement.yaml)

## Running

This directory contains various example SbatchMan YAML job files. Before running them, make sure you understand which experiments they will run.

## Plots

1) Bandwidth with increasing message sizes
```bash
python plots.py path/to/hicrest-axccl_my_system_data.parquet -std -minmax
```

2) Latency with increasing message sizes
```bash
python plots.py path/to/hicrest-axccl_my_system_data.parquet -std -minmax --metric latency --max-size 8KiB
```

## Expected Runtimes

| Prim.-Coll. / System | Leonardo*                              |
|----------------------|----------------------------------------|
| Peer-to-Peer (p2p)   | TrivialStaging ~30s,   CudaAware ~10s  |
| Ping-Pong    (pp)    | TrivialStaging ~30s,   CudaAware ~10s  |
| AlltoAll     (a2a)   | TrivialStaging ~4min,  CudaAware ~1min |
| AllReduce    (ar)    | TrivialStaging ~1min,  CudaAware ~10s  |

*Leonardo has 100 Gb/s theoretical max bandwidth for inter-node communication.