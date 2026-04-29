# PointerChasing

Utilities to measure read access times of caches, memory, and hardware prefetches for simple and fused operations.  
This package provides the following three utilities:
* _random-chase_: measure average read access times of all cache
  levels and main memory.
* _linear-chase_: measure read access times for a linear access
  pattern with a constant stride.
* _fused-linear-chase_: like _linear-chase_ but for an interleaved
  access pattern of multiple linear sequences, all with the same stride.
* _fused-random-chase_: like _random-chase_ but for an interleaved
  access pattern of multiple random sequences.

All of them work with memory buffers that are organized as an array
of pointers where:
* all pointers point into the very same buffer, and where
* beginning from any pointer all other pointers can be reached
  following the pointer chain, and where
* all locations are reached.

As configs, use:

```
../common/sbm_configs/shared_memory.yaml
```