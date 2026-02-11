METRICS_TO_EXTRACT = [
  # Overall Performance (FLOP/S)
  {
    "out_key": "gflops_opt",
    "section": "GFLOP/s Summary",
    "candidates": ["Total with convergence and optimization phase overhead"],
  },
  {
    "out_key": "gflops",
    "section": "GFLOP/s Summary",
    "candidates": ["Total with convergence overhead"],
  },
  {
    "out_key": "gflops_ddot",
    "section": "GFLOP/s Summary",
    "candidates": ["Raw DDOT"],
  },
  {
    "out_key": "gflops_waxpby",
    "section": "GFLOP/s Summary",
    "candidates": ["Raw WAXPBY"],
  },
  {
    "out_key": "gflops_spmv",
    "section": "GFLOP/s Summary",
    "candidates": ["Raw SpMV"],
  },
  {
    "out_key": "gflops_mg",
    "section": "GFLOP/s Summary",
    "candidates": ["Raw MG"],
  },
  
  # Overall Performance (Runtime)
  {
    "out_key": "time_ddot",
    "section": "Benchmark Time Summary",
    "candidates": ["DDOT"],
  },
  {
    "out_key": "time_mg",
    "section": "Benchmark Time Summary",
    "candidates": ["MG"],
  },
  {
    "out_key": "time_opt",
    "section": "Benchmark Time Summary",
    "candidates": ["Optimization phase"],
  },
  {
    "out_key": "time_spmv",
    "section": "Benchmark Time Summary",
    "candidates": ["SpMV"],
  },
  {
    "out_key": "time_waxpby",
    "section": "Benchmark Time Summary",
    "candidates": ["WAXPBY"],
  },
  {
    "out_key": "time_tot",
    "section": "Benchmark Time Summary",
    "candidates": ["Total"],
  },
  
  # Communication Breakdown
  {
    "out_key": "ddot_allreduce_min",
    "section": "DDOT Timing Variations",
    "candidates": ["Min DDOT MPI_Allreduce time"],
  },
  {
    "out_key": "ddot_allreduce_max",
    "section": "DDOT Timing Variations",
    "candidates": ["Max DDOT MPI_Allreduce time"],
  },
  {
    "out_key": "ddot_allreduce_avg",
    "section": "DDOT Timing Variations",
    "candidates": ["Avg DDOT MPI_Allreduce time"],
  },
  {
    "out_key": "halo_min",
    "section": "Sparse Operations Overheads",
    "candidates": ["Min Halo exchange time"],
  },
  {
    "out_key": "halo_max",
    "section": "Sparse Operations Overheads",
    "candidates": ["Max Halo exchange time"],
  },
  {
    "out_key": "halo_avg",
    "section": "Sparse Operations Overheads",
    "candidates": ["Avg Halo exchange time"],
  },
  
  # Reproducibility
  {
    "out_key": "num_equations",
    "section": "Linear System Information",
    "candidates": ["Number of Equations"],
  },
  {
    "out_key": "global_nx",
    "section": "Global Problem Dimensions",
    "candidates": ["Global nx", "Global Nx", "Global NX"],
  },
  {
    "out_key": "global_ny",
    "section": "Global Problem Dimensions",
    "candidates": ["Global ny", "Global Ny", "Global NY"],
  },
  {
    "out_key": "global_nz",
    "section": "Global Problem Dimensions",
    "candidates": ["Global nz", "Global Nz", "Global NZ"],
  },
  {
    "out_key": "mem",
    "section": "Memory Use Information",
    "candidates": ["Total memory used for data (Gbytes)"],
  },
]