# HICREST DLNetBench

## Compile

```bash
./compile <system_name>
```

Supported systems:
- `leonardo`
- `lumi`
- `baldo`
- `cpu` -> it is just a way to use the benchmarks using CPU-only MPI

## Forcing Job Placement

To ensure the benchmark is run under controlled placements (emulating different topologies), there are two main strategies:
1) Set an arbitrary SLURM nodelist
2) Get a (potentially bigger) random allocation and check if a subset of nodes satisfies placement constraints

The `run_ensuring_placement.py` script uses strategy (2).  
Using SbatchMan it relaunches failed jobs until all are complete.