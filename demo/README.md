# Campaign DEMO

### 1. Setup the remote

Tell SbatchMan how to push (`sync`) apps and pull (`fetch`) results

From the TUI
```bash
sbatchman remotes-config
```

Add this config (replacing placeholders)
```toml
[[clusters]]
name = "clariden"
host = "cscs-clariden"
user = "<USER>"

[[clusters.fetch_dirs]]
remote = "~/HICREST-Benchmark-Collection/DLNetBench/DLNetBench/SbatchMan"
local = "path/to/HICREST-Benchmark-Collection/demos/SbatchMan"

[[clusters.fetch_dirs]]
remote = "~/HICREST-Benchmark-Collection/Graph500/SbatchMan"
local = "path/to/HICREST-Benchmark-Collection/demos/SbatchMan"

[[clusters.sync_dirs]]
local = "path/to/HICREST-Benchmark-Collection"
remote = "~/HICREST-Benchmark-Collection"
alias = "hicrest"
```

### 2. Sync apps
```bash
sbatchman sync -c clariden -a hicrest
```

### 3. Open a terminal on the remote 

```bash
ssh ...
```

### 4. Run the campaign

```bash
# From ~/HICREST-Benchmark-Collection
sbatchman campaign demos/dlnetbench_graph500_campaign_demo.yaml
```

### 5. Fetch results

```bash
sbatchman fetch -c clariden
```

### 6. Check jobs

```bash
cd demos
python ../common/check_jobs.py
```

### 7. Parse results

```bash
python show_results.py
```