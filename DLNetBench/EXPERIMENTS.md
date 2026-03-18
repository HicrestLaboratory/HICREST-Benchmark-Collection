# Experiment Plan

## Overview

This plan describes a two-phase experimental design for measuring distributed training performance on a Dragonfly network topology. The first phase establishes isolated baselines — how individual strategies perform at scale. The second phase introduces concurrency and network congestion, mimicking realistic multi-tenant workloads.

Both phases are parameterized by **strategy**, **scale** (GPU count), and **placement** (how nodes are mapped onto the network topology).

---

## Terminology

### Strategies

Each strategy represents a distinct parallelization approach and is benchmarked on one representative model. The feasible GPU counts (and corresponding node counts) for each are fixed by the model's memory and parallelism requirements:

| Strategy | Model | GPUs (nodes) |
|----------|-------|--------------|
| DP | ViT-H (Google) | 8 (2), 16 (4) |
| FSDP | Llama3-8B (Meta) | 16 (4), 32 (8) |
| DP+PP | Minerva-7B | 16 (4), 32 (8), 64 (16) |
| DP+PP+TP | Llama3-70B (Meta) | 224 (56), 256 (64), 512 (128) |
| DP+PP+Expert | Mixtral-8x7B (Mistral) | 512 (128), 1024 (256) |

---

### Pattern

A **pattern** specifies the GPU counts allocated to each concurrent training job. It defines how many jobs run simultaneously and how large each one is.

**Example:** `(16, 8)` → two concurrent jobs: one using 16 GPUs (4 nodes) and one using 8 GPUs (2 nodes).

> **Constraints:** GPU counts must be multiples of 4, with a minimum of 8 GPUs (2 nodes) per job.

---

### Entropy Bins (H-bin)

H-bins describe the diversity of parallelization strategies present in a given configuration:

| H-bin | Meaning |
|-------|---------|
| Low | Few distinct strategies |
| Medium | Moderate variety |
| High | Many or all strategies represented |

---

### Placement Classes

Placement classes define how a job's nodes are mapped onto the Dragonfly topology:

| Class | Description |
|-------|-------------|
| `intra-l1` | All nodes on a single switch |
| `intra-group` | Nodes spread across switches within one Dragonfly group |
| `inter-group` | Nodes span multiple Dragonfly groups |
| `intra-group-same-l1-2/4` | Nodes grouped in blocks of 2 or 4, each block on the same switch, all within one group |
| `inter-group-same-l1-2/4` | Same block structure, but blocks distributed across multiple groups |

> The blocked placements (`same-l1-2/4`) reflect topology constraints imposed by specific parallelization strategies.

---

### Placement Bins (P-bin)

P-bins summarize how spread out a job's nodes are across the topology, scored as follows:

| P-bin | Score | Example (4-node job) |
|-------|-------|----------------------|
| Low | 1 | All nodes on a single switch |
| Medium | 2 | Nodes span multiple switches within one group |
| High | 3 | Nodes span multiple Dragonfly groups |

> The score of a concurrent configuration is the average of its job scores. 

---

### Experiment Sets

An **Experiment Set** is a collection of configurations specifying *what* to run — strategy, GPU count, and H-bin — without yet specifying *where* (placement). Placement is assigned separately when constructing the Hierarchical Experiment Set.

---

## Experimental Design

### Phase 1 — Baseline Experiments

Baseline experiments are **isolated runs**: one job at a time, no concurrent traffic. They establish the performance envelope for each strategy under ideal (interference-free) conditions.

**Construction:** For each strategy, runs are sampled across all feasible GPU counts, using the full range of placement classes.

**In total we have 29 baselines that run sequentially shall take approximately 40 minutes.**

---

### Phase 2 — Concurrent Experiments

Concurrent experiments introduce multiple simultaneous jobs, generating realistic network contention.

**Construction:**
1. Sample feasible patterns that fully (non strict) utilize the target GPU count.
2. For each pattern, sample configurations from H-bins.
3. Assign placement vectors to each job, sampled from P-bins.

**In total we have 20 concurrent runs that shall take approximately 1:30 hours.**  
These runs span:
- 3 power-law realistic multi-tenant patterns
- 2 entropy bins: 3 medium, 1 high
- 3 placement score bins: 8 low, 8 medium, 4 high

---

## Experiment Objectives

### Baseline Objectives

Baselines are designed to answer:

- How does throughput, blocking communication time, and synchronization overhead in non-blocking communication each evolve as GPU count increases? How do different strategies compare?
- What is each strategy's ideal performance envelope, free from network interference?
- How sensitive is each strategy to placement, independent of congestion?
- For each strategy, what is a good or bad placement?
- How does computation/communication overlap change with scale?
- Are there topology-induced bottlenecks visible even in isolated runs?

---

### Concurrent Objectives

Concurrent experiments are designed to answer:

#### Congestion & Strategy Behavior
- How does network congestion affect each strategy's throughput and latency?
- Which strategies are more robust to interference from co-running jobs?
- How do mixed-strategy workloads interact — do certain combinations amplify or dampen congestion?

#### Fairness & Resource Contention
- Is performance degradation evenly distributed across co-running jobs, or do certain strategies disproportionately suffer (or cause) congestion?

#### Placement Effects
- How does P-bin (job spread across topology) influence slowdown under contention?
- Does spreading a job across groups (high P-bin) consistently worsen performance, or are there cases where it helps?
- Are some strategies more resilient to poor placement than others?

#### Predictability & System Efficiency
- Can per-job slowdown be predicted from baseline performance and P-bin score alone?
- How much variance is there in performance across repeated runs of the same configuration?
- What configurations maximize system-wide throughput, not just per-job performance?


## Other questions we could answer extending the experiment set

#### Fairness & Resource Contention
- Do certain strategies monopolize shared network resources?

#### Placement Effects
- Do hierarchical placements (same-l1 blocks) improve stability under congestion?

#### Pattern-Level & Scale Effects
- Do certain load patterns (e.g., many small jobs vs. few large jobs) produce worse congestion?
- How does total GPU occupancy affect system-wide efficiency?
- Are large or small jobs disproportionately penalized as contention increases?
- Does scaling efficiency degrade uniformly across strategies, or are there threshold effects where congestion sharply worsens?
