# HICREST AXCCL

## Compile

```bash
./compile <system_name>
```

## Forcing Job Placement

To ensure the benchmark is run under controlled placements (emulating different topologies), there are two main strategies:
1) Set an arbitrary SLURM nodelist
2) Get a (potentially bigger) random allocation and check if a subset of nodes satisfies placement constraints
