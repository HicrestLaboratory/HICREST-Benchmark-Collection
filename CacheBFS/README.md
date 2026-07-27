# Optimizing Breadth-First Search on Modern Energy-Efficient Multicore CPUs

For more information about the BFS refer to the original repository [https://github.com/Sasso0101/thesis/tree/main](https://github.com/Sasso0101/thesis/tree/main).

Related publication [https://dl.acm.org/doi/10.1145/3711708.3723452](https://dl.acm.org/doi/10.1145/3711708.3723452).

```bibtex
@inproceedings{10.1145/3711708.3723452,
    author = {Andaloro, Salvatore Domenico and Pasquali, Thomas and Vella, Flavio},
    title = {Cache-optimized BFS on multi-core CPUs},
    year = {2025},
    isbn = {9798400714467},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3711708.3723452},
    doi = {10.1145/3711708.3723452},
    abstract = {Breadth-First Search (BFS) performance on shared-memory systems is often limited by irregular memory access and cache inefficiencies. This work presents two optimizations for BFS graph traversal: a bitmap-based algorithm designed for small-diameter graphs and MergedCSR, a graph storage format that improves cache locality for large-scale graphs. Experimental results on real-world datasets show an average 1.3\texttimes{} speedup over a state-of-the-art implementation, with MergedCSR reducing RAM accesses by approximately 15\%.},
    booktitle = {Proceedings of the 1st FastCode Programming Challenge},
    pages = {23–27},
    numpages = {5},
    keywords = {graph, algorithm, breadth-first search, parallel computing, multi-core CPU},
    location = {The Westin Las Vegas Hotel \& Spa, Las Vegas, NV, USA},
    series = {FCPC '25}
}
```

# Dependencies & Datasets Setup

```bash
# From `CacheBFS/`
./setup_deps.sh

# For datasets (inside a Python venv)
pip install pipx
pipx install mtxman

# Download graphs
mtxman sync graphs.yaml --binary-mtx
```