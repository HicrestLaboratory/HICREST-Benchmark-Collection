/*
 * numa_aware_stream.c
 *
 * STREAM memory bandwidth benchmark with explicit control over where the
 * threads run and where the data lives, so that NUMA effects can be measured.
 *
 * How it works
 * ------------
 * The three STREAM arrays are cut into one slice per NUMA node.  The threads
 * pinned to node N always work on slice N, so every configuration does the
 * same work with the same threads.  The only thing that changes from one
 * configuration to the next is which node's memory holds each slice:
 *
 *   aware       slice N is on node N (every node uses its own memory)
 *   remote+K    slice N is on node (N+K) mod nodes (every access is remote)
 *   all-nodeX   every slice is on node X (what a program gets when one
 *               thread allocates and initialises all of its data)
 *   interleave  pages alternate between the nodes (numactl --interleave=all)
 *
 * In "pairs" mode only the threads of one node run, and their data is bound
 * to each node in turn.  This gives the local bandwidth (node0->node0) and
 * the remote bandwidth (node0->node1, node0->node2, ...).
 *
 * The rate is computed as in STREAM: bytes moved divided by the best time
 * over the iterations, not counting the warm-up iterations.  All threads start
 * and stop a kernel together, so the time is that of the slowest thread.  When
 * some threads are local and others remote (all-nodeX), the remote ones set
 * the pace for everybody.
 *
 * A rate is printed only if the run can be trusted.  After every run the
 * program checks that
 *   - OpenMP started exactly the number of threads that was asked for,
 *   - every thread stayed on its CPU, and that CPU belongs to the right node,
 *   - the pages of a, b and c are on the node they were requested on,
 *   - the arrays hold the values STREAM expects.
 * If any of these does not hold the run is reported as FAILED.  A run that
 * cannot be done on this machine at all is reported as SKIPPED.
 *
 * Build
 * -----
 *   gcc -O3 -fopenmp numa_aware_stream.c -o numa_aware_stream -lnuma -lm
 *   (add -DSTREAM_TYPE_FLOAT for single precision)
 *
 * Usage
 * -----
 *   ./numa_aware_stream                  all configurations
 *   ./numa_aware_stream -m pairs -c 0    node 0 threads against every node
 *   ./numa_aware_stream -m system        all nodes at once
 *   ./numa_aware_stream -T 16            at most 16 threads per node
 *   ./numa_aware_stream -v               show where threads and pages are
 *   ./numa_aware_stream -h               all options
 *
 * Do not set OMP_PROC_BIND or OMP_PLACES, and do not start the program under
 * numactl: it pins its own threads and binds its own memory.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

/*
 * The array values grow by a factor of 15 per iteration, so the number of
 * iterations has to stay below the point where they overflow.
 */
#ifdef STREAM_TYPE_FLOAT
typedef float real_t;
#define REAL_NAME "float"
#define TOLERANCE 1e-6
#define MAX_ITERATIONS 30
#else
typedef double real_t;
#define REAL_NAME "double"
#define TOLERANCE 1e-13
#define MAX_ITERATIONS 250
#endif

#define DEFAULT_ELEMENTS 64000000ULL
#define DEFAULT_ITERATIONS 10
#define DEFAULT_WARMUP 1
#define MIN_SLICE_ELEMENTS 65536ULL
#define MAX_ELEMENTS (1ULL << 40)
#define PAGE_SAMPLES 64 /* pages looked up per array */
#define SCALAR 3.0

#define MIB (1024.0 * 1024.0)
#define GIB (1024.0 * 1024.0 * 1024.0)

enum kernel
{
    COPY,
    SCALE,
    ADD,
    TRIAD,
    NUM_KERNELS
};

static const char *kernel_name[NUM_KERNELS] = {"copy", "scale", "add", "triad"};

/* Arrays read or written per element.  STREAM does not count write-allocate. */
static const int kernel_streams[NUM_KERNELS] = {2, 2, 3, 3};

struct options
{
    size_t elements; /* per array, summed over all slices */
    int iterations;
    int warmup;           /* leading iterations left out of the result */
    int threads_per_node; /* 0 = every usable CPU */
    int source_node;      /* pairs mode: -1 = every node */
    bool run_system;
    bool run_pairs;
    bool verbose;
    bool csv;
};

struct node
{
    int *cpus;     /* one CPU per core first, SMT siblings after them */
    int num_cpus;  /* CPUs this program will use */
    int num_cores; /* physical cores among those CPUs */
    bool has_memory;
};

enum status
{
    NOT_RUN,
    OK,
    SKIPPED,
    FAILED
};

/* One placement of threads and data, and what came out of running it. */
struct config
{
    char name[32];
    bool *active;     /* [node] do this node's threads take part? */
    int *memory_node; /* [node] node whose memory holds its slice */
    bool interleave;

    enum status status;
    char reason[200];
    int num_threads;
    double rate[NUM_KERNELS]; /* GB/s, all threads together */
};

struct slice
{
    real_t *a, *b, *c;
};

struct worker
{
    int node;
    int cpu;
    size_t begin, end; /* this thread's part of the slice */
    real_t *a, *b, *c;

    bool pinned;
    int cpu_at_start;
    int cpu_at_end;
};

static struct options opt;
static struct node *nodes;
static int num_nodes;
static int num_memory_nodes;
static FILE *info; /* stdout, or stderr when printing CSV */

static int runs_ok, runs_skipped, runs_failed;

static void die(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size);

    if (!p)
        die("out of memory");
    return p;
}

static void set_status(struct config *cfg, enum status status, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(cfg->reason, sizeof(cfg->reason), fmt, ap);
    va_end(ap);
    cfg->status = status;
}

/*
 * Topology
 */

/*
 * Hardware threads of one core share the same thread_siblings_list, so its
 * first entry identifies the core.
 */
static int core_of_cpu(int cpu)
{
    char path[128];
    FILE *f;
    int first = cpu;

    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    f = fopen(path, "r");
    if (!f)
        return cpu;
    if (fscanf(f, "%d", &first) != 1)
        first = cpu;
    fclose(f);
    return first;
}

/*
 * Put one CPU of every core at the front of the list and the SMT siblings
 * behind them.  With -T this fills whole cores before it doubles up on one.
 */
static int order_cores_first(int *cpus, int count)
{
    int *core = xcalloc(count, sizeof(int));
    int *ordered = xcalloc(count, sizeof(int));
    bool *is_sibling = xcalloc(count, sizeof(bool));
    int num_cores = 0;
    int n = 0;

    for (int i = 0; i < count; i++)
    {
        core[i] = core_of_cpu(cpus[i]);
        for (int j = 0; j < i; j++)
        {
            if (core[j] == core[i])
                is_sibling[i] = true;
        }
        if (!is_sibling[i])
            ordered[n++] = cpus[i];
    }
    num_cores = n;

    for (int i = 0; i < count; i++)
    {
        if (is_sibling[i])
            ordered[n++] = cpus[i];
    }
    memcpy(cpus, ordered, count * sizeof(int));

    free(core);
    free(ordered);
    free(is_sibling);
    return num_cores;
}

static void discover_topology(void)
{
    struct bitmask *mask = numa_allocate_cpumask();
    cpu_set_t allowed;

    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        die("sched_getaffinity: %s", strerror(errno));

    num_nodes = numa_max_node() + 1;
    nodes = xcalloc(num_nodes, sizeof(*nodes));

    for (int n = 0; n < num_nodes; n++)
    {
        struct node *node = &nodes[n];

        node->cpus = xcalloc(mask->size, sizeof(int));
        node->has_memory = numa_node_size64(n, NULL) > 0;
        if (node->has_memory)
            num_memory_nodes++;

        if (numa_node_to_cpus(n, mask) != 0)
            continue;
        for (unsigned long cpu = 0; cpu < mask->size && cpu < CPU_SETSIZE; cpu++)
        {
            if (numa_bitmask_isbitset(mask, cpu) && CPU_ISSET(cpu, &allowed))
                node->cpus[node->num_cpus++] = (int)cpu;
        }
        node->num_cores = order_cores_first(node->cpus, node->num_cpus);

        if (opt.threads_per_node > 0 && node->num_cpus > opt.threads_per_node)
        {
            node->num_cpus = opt.threads_per_node;
            if (node->num_cores > node->num_cpus)
                node->num_cores = node->num_cpus;
        }
    }
    numa_free_cpumask(mask);
}

static void print_topology(void)
{
    fprintf(info, "NUMA nodes: %d\n", num_nodes);
    for (int n = 0; n < num_nodes; n++)
    {
        fprintf(info, "  node %d: %d threads on %d cores, %.1f GiB\n", n,
                nodes[n].num_cpus, nodes[n].num_cores, numa_node_size64(n, NULL) / GIB);
    }

    fprintf(info, "Distances reported by the firmware (10 = local):\n      ");
    for (int j = 0; j < num_nodes; j++)
        fprintf(info, "%4d", j);
    fprintf(info, "\n");
    for (int i = 0; i < num_nodes; i++)
    {
        fprintf(info, "  %2d: ", i);
        for (int j = 0; j < num_nodes; j++)
            fprintf(info, "%4d", numa_distance(i, j));
        fprintf(info, "\n");
    }
    if (num_nodes == 1)
        fprintf(info, "Only one NUMA node here, so only the local case can be measured.\n");
    fprintf(info, "\n");
}

/*
 * The benchmark kernels
 *
 * Not inlined on purpose: the timed loops and the computation of the
 * expected values below then run exactly the same machine code.
 *
 * GCC would also turn the copy loop into a call to memcpy().  For large
 * blocks memcpy uses non-temporal stores on x86, which skip the write-allocate
 * that the other three kernels pay, and copy would come out about 1.5x too
 * high compared with them and with a normally built stream.c.  Switching
 * that transformation off keeps all four kernels plain loops.
 */
#if defined(__GNUC__) && !defined(__clang__)
#define KERNEL_ATTRIBUTES \
    __attribute__((noinline, optimize("no-tree-loop-distribute-patterns")))
#else
#define KERNEL_ATTRIBUTES __attribute__((noinline))
#endif

static KERNEL_ATTRIBUTES void
run_kernel(enum kernel kernel, real_t *restrict a, real_t *restrict b,
           real_t *restrict c, size_t begin, size_t end)
{
    const real_t scalar = SCALAR;

    switch (kernel)
    {
    case COPY:
        for (size_t j = begin; j < end; j++)
            c[j] = a[j];
        break;
    case SCALE:
        for (size_t j = begin; j < end; j++)
            b[j] = scalar * c[j];
        break;
    case ADD:
        for (size_t j = begin; j < end; j++)
            c[j] = a[j] + b[j];
        break;
    case TRIAD:
        for (size_t j = begin; j < end; j++)
            a[j] = b[j] + scalar * c[j];
        break;
    default:
        break;
    }
}

static void initialise(real_t *a, real_t *b, real_t *c, size_t begin, size_t end)
{
    for (size_t j = begin; j < end; j++)
    {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }
    for (size_t j = begin; j < end; j++)
        a[j] = 2.0 * a[j];
}

static bool pin_to_cpu(int cpu)
{
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

/*
 * Memory
 */

/* MPOL_BIND does not fall back to another node, so check before allocating. */
static bool memory_is_available(struct config *cfg, size_t slice_bytes)
{
    double *needed = xcalloc(num_nodes, sizeof(double));
    bool ok = true;

    for (int n = 0; n < num_nodes && ok; n++)
    {
        double bytes = 3.0 * slice_bytes;

        if (!cfg->active[n])
            continue;

        if (cfg->interleave)
        {
            for (int m = 0; m < num_nodes; m++)
            {
                if (nodes[m].has_memory)
                    needed[m] += bytes / num_memory_nodes;
            }
        }
        else if (!nodes[cfg->memory_node[n]].has_memory)
        {
            set_status(cfg, SKIPPED, "node %d has no memory", cfg->memory_node[n]);
            ok = false;
        }
        else
        {
            needed[cfg->memory_node[n]] += bytes;
        }
    }

    for (int n = 0; n < num_nodes && ok; n++)
    {
        long long free_bytes = 0;

        if (needed[n] == 0)
            continue;
        numa_node_size64(n, &free_bytes);
        if (needed[n] > 0.9 * free_bytes)
        {
            set_status(cfg, SKIPPED,
                       "not enough free memory on node %d (need %.1f GiB, %.1f GiB free)",
                       n, needed[n] / GIB, free_bytes / GIB);
            ok = false;
        }
    }

    free(needed);
    return ok;
}

static void *allocate_array(const struct config *cfg, int node, size_t bytes)
{
    if (cfg->interleave)
        return numa_alloc_interleaved(bytes);
    return numa_alloc_onnode(bytes, cfg->memory_node[node]);
}

static bool allocate_slices(const struct config *cfg, struct slice *slices, size_t slice_bytes)
{
    for (int n = 0; n < num_nodes; n++)
    {
        if (!cfg->active[n])
            continue;
        slices[n].a = allocate_array(cfg, n, slice_bytes);
        slices[n].b = allocate_array(cfg, n, slice_bytes);
        slices[n].c = allocate_array(cfg, n, slice_bytes);
        if (!slices[n].a || !slices[n].b || !slices[n].c)
            return false;
    }
    return true;
}

static void free_slices(struct slice *slices, size_t slice_bytes)
{
    for (int n = 0; n < num_nodes; n++)
    {
        if (slices[n].a)
            numa_free(slices[n].a, slice_bytes);
        if (slices[n].b)
            numa_free(slices[n].b, slice_bytes);
        if (slices[n].c)
            numa_free(slices[n].c, slice_bytes);
    }
}

/*
 * Checks done after a run
 */

static void print_thread_map(const struct config *cfg, const struct worker *workers,
                             int num_threads)
{
    fprintf(info, "[%s] where the threads ran (sched_getcpu)\n", cfg->name);
    for (int t = 0; t < num_threads; t++)
    {
        const struct worker *w = &workers[t];
        int node_at_end = w->cpu_at_end >= 0 ? numa_node_of_cpu(w->cpu_at_end) : -1;
        bool good = w->cpu_at_start == w->cpu && w->cpu_at_end == w->cpu &&
                    node_at_end == w->node;

        fprintf(info, "  thread %2d: node %d  intended cpu %3d  running on cpu %3d (start) "
                      "%3d (end)  cpu belongs to node %d  %s\n",
                t, w->node, w->cpu, w->cpu_at_start, w->cpu_at_end, node_at_end,
                good ? "ok" : "MISMATCH");
    }
}

static bool threads_stayed_put(struct config *cfg, const struct worker *workers, int num_threads)
{
    int misplaced = 0;

    for (int t = 0; t < num_threads; t++)
    {
        const struct worker *w = &workers[t];

        if (!w->pinned || w->cpu_at_start != w->cpu || w->cpu_at_end != w->cpu ||
            numa_node_of_cpu(w->cpu_at_end) != w->node)
            misplaced++;
    }
    if (misplaced == 0)
        return true;

    set_status(cfg, FAILED, "%d of %d threads were not on their intended CPU",
               misplaced, num_threads);
    return false;
}

/* Node that holds the page at this address right now, or -1 if unknown. */
static int node_of_page(void *addr)
{
    int node = -1;

    if (get_mempolicy(&node, NULL, 0, addr, MPOL_F_NODE | MPOL_F_ADDR) != 0)
        return -1;
    return node;
}

/*
 * Look up PAGE_SAMPLES pages of one array and count them per node.  The pages
 * are picked at random: regularly spaced samples could all land on the same
 * node of an interleaved array.
 */
static int sample_pages(void *array, size_t bytes, int *pages_on_node)
{
    static uint64_t seed = 88172645463325252ULL;
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    size_t num_pages = bytes / page_size;
    int found = 0;

    if (num_pages == 0)
        return 0;

    for (int i = 0; i < PAGE_SAMPLES; i++)
    {
        int node;

        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        node = node_of_page((char *)array + (seed % num_pages) * page_size);
        if (node < 0 || node >= num_nodes)
            continue;
        pages_on_node[node]++;
        found++;
    }
    return found;
}

static bool pages_are_in_place(struct config *cfg, struct slice *slices, size_t slice_bytes)
{
    int *pages_on_node = xcalloc(num_nodes, sizeof(int));
    bool ok = true;

    for (int n = 0; n < num_nodes && ok; n++)
    {
        int wanted = cfg->memory_node[n];
        int found = 0;
        int nodes_used = 0;

        if (!cfg->active[n])
            continue;

        memset(pages_on_node, 0, num_nodes * sizeof(int));
        found += sample_pages(slices[n].a, slice_bytes, pages_on_node);
        found += sample_pages(slices[n].b, slice_bytes, pages_on_node);
        found += sample_pages(slices[n].c, slice_bytes, pages_on_node);
        for (int m = 0; m < num_nodes; m++)
        {
            if (pages_on_node[m] > 0)
                nodes_used++;
        }

        if (opt.verbose)
        {
            if (cfg->interleave)
                fprintf(info, "[%s] slice of node %d, interleaved: pages found on", cfg->name, n);
            else
                fprintf(info, "[%s] slice of node %d, requested on node %d: pages found on",
                        cfg->name, n, wanted);
            for (int m = 0; m < num_nodes; m++)
                fprintf(info, " node%d=%d", m, pages_on_node[m]);
            fprintf(info, "\n");
        }

        if (found != 3 * PAGE_SAMPLES)
        {
            set_status(cfg, FAILED, "could not find out where %d of %d sampled pages are",
                       3 * PAGE_SAMPLES - found, 3 * PAGE_SAMPLES);
            ok = false;
        }
        else if (cfg->interleave)
        {
            if (num_memory_nodes > 1 && nodes_used < 2)
            {
                set_status(cfg, FAILED, "the interleaved pages are all on one node");
                ok = false;
            }
        }
        else if (pages_on_node[wanted] != found)
        {
            set_status(cfg, FAILED, "%d of %d sampled pages of node %d's slice are not on node %d",
                       found - pages_on_node[wanted], found, n, wanted);
            ok = false;
        }
    }

    free(pages_on_node);
    return ok;
}

/* Average relative error against the expected value; NaN counts as wrong. */
static bool array_matches(const real_t *x, size_t count, double expected)
{
    double sum = 0.0;
    double error;

    for (size_t j = 0; j < count; j++)
        sum += fabs((double)x[j] - expected);

    error = sum / (double)count;
    if (expected != 0.0)
        error /= fabs(expected);
    return error <= TOLERANCE;
}

/*
 * Every element goes through the same sequence of operations, so running the
 * benchmark on a single element gives the value all of them must have.
 */
static bool results_are_correct(struct config *cfg, const struct slice *slices,
                                size_t slice_elements)
{
    real_t a, b, c;

    initialise(&a, &b, &c, 0, 1);
    for (int iter = 0; iter < opt.iterations; iter++)
    {
        for (int k = 0; k < NUM_KERNELS; k++)
            run_kernel(k, &a, &b, &c, 0, 1);
    }

    for (int n = 0; n < num_nodes; n++)
    {
        if (!cfg->active[n])
            continue;
        if (!array_matches(slices[n].a, slice_elements, a) ||
            !array_matches(slices[n].b, slice_elements, b) ||
            !array_matches(slices[n].c, slice_elements, c))
        {
            set_status(cfg, FAILED, "wrong values in the arrays of node %d's slice", n);
            return false;
        }
    }
    return true;
}

/*
 * Running one configuration
 */

/* Give each thread its CPU and its share of its node's slice. */
static void assign_workers(const struct config *cfg, const struct slice *slices,
                           size_t slice_elements, struct worker *workers)
{
    int t = 0;

    for (int n = 0; n < num_nodes; n++)
    {
        if (!cfg->active[n])
            continue;
        for (int i = 0; i < nodes[n].num_cpus; i++)
        {
            struct worker *w = &workers[t++];

            w->node = n;
            w->cpu = nodes[n].cpus[i];
            w->begin = slice_elements * (size_t)i / (size_t)nodes[n].num_cpus;
            w->end = slice_elements * (size_t)(i + 1) / (size_t)nodes[n].num_cpus;
            w->a = slices[n].a;
            w->b = slices[n].b;
            w->c = slices[n].c;
            w->cpu_at_start = -1;
            w->cpu_at_end = -1;
        }
    }
}

static void run_config(struct config *cfg)
{
    struct slice *slices;
    struct worker *workers;
    double *elapsed[NUM_KERNELS];
    size_t slice_elements, slice_bytes;
    int active_nodes = 0;
    int num_threads = 0;
    int team_size = 0;

    for (int n = 0; n < num_nodes; n++)
    {
        if (cfg->active[n])
        {
            active_nodes++;
            num_threads += nodes[n].num_cpus;
        }
    }
    cfg->num_threads = num_threads;
    if (num_threads == 0)
    {
        set_status(cfg, SKIPPED, "no usable CPUs");
        return;
    }

    slice_elements = opt.elements / (size_t)active_nodes;
    slice_bytes = slice_elements * sizeof(real_t);
    if (slice_elements < MIN_SLICE_ELEMENTS)
    {
        set_status(cfg, SKIPPED, "only %zu elements per node, use a larger -n", slice_elements);
        return;
    }
    if (!memory_is_available(cfg, slice_bytes))
        return;

    fprintf(stderr, "running %s ...\n", cfg->name);

    slices = xcalloc(num_nodes, sizeof(*slices));
    workers = xcalloc(num_threads, sizeof(*workers));
    for (int k = 0; k < NUM_KERNELS; k++)
        elapsed[k] = xcalloc(opt.iterations, sizeof(double));

    if (!allocate_slices(cfg, slices, slice_bytes))
    {
        set_status(cfg, FAILED, "could not allocate the arrays");
        goto out;
    }
    assign_workers(cfg, slices, slice_elements, workers);

#pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        struct worker *w = &workers[tid];

#pragma omp single
        team_size = omp_get_num_threads();

        /*
         * After the single construct every thread sees the same team_size,
         * so either the whole team does the work or nobody does.
         */
        if (team_size == num_threads)
        {
            w->pinned = pin_to_cpu(w->cpu);
            w->cpu_at_start = sched_getcpu();

            /* The thread that will use the data is also the one to touch it first. */
            initialise(w->a, w->b, w->c, w->begin, w->end);

            for (int iter = 0; iter < opt.iterations; iter++)
            {
                for (int k = 0; k < NUM_KERNELS; k++)
                {
                    double start;

#pragma omp barrier
                    start = omp_get_wtime();
                    run_kernel(k, w->a, w->b, w->c, w->begin, w->end);
#pragma omp barrier
                    if (tid == 0)
                        elapsed[k][iter] = omp_get_wtime() - start;
                }
            }
            w->cpu_at_end = sched_getcpu();
        }
    }

    if (team_size != num_threads)
    {
        set_status(cfg, FAILED, "OpenMP started %d threads instead of %d "
                                "(check OMP_THREAD_LIMIT and cgroup limits)",
                   team_size, num_threads);
        goto out;
    }
    if (opt.verbose)
        print_thread_map(cfg, workers, num_threads);

    if (!threads_stayed_put(cfg, workers, num_threads))
        goto out;
    if (!pages_are_in_place(cfg, slices, slice_bytes))
        goto out;
    if (!results_are_correct(cfg, slices, slice_elements))
        goto out;

    for (int k = 0; k < NUM_KERNELS; k++)
    {
        double bytes = (double)kernel_streams[k] * sizeof(real_t) *
                       (double)slice_elements * active_nodes;
        double best = elapsed[k][opt.warmup];

        for (int iter = opt.warmup + 1; iter < opt.iterations; iter++)
        {
            if (elapsed[k][iter] < best)
                best = elapsed[k][iter];
        }
        cfg->rate[k] = bytes / best / 1e9;
    }
    set_status(cfg, OK, "ok");

out:
    free_slices(slices, slice_bytes);
    free(slices);
    free(workers);
    for (int k = 0; k < NUM_KERNELS; k++)
        free(elapsed[k]);
}

/*
 * Configurations and their output
 */

static void init_config(struct config *cfg, const char *fmt, ...)
{
    va_list ap;

    memset(cfg, 0, sizeof(*cfg));
    va_start(ap, fmt);
    vsnprintf(cfg->name, sizeof(cfg->name), fmt, ap);
    va_end(ap);
    cfg->active = xcalloc(num_nodes, sizeof(bool));
    cfg->memory_node = xcalloc(num_nodes, sizeof(int));
}

static void free_config(struct config *cfg)
{
    free(cfg->active);
    free(cfg->memory_node);
}

static void count_run(const struct config *cfg)
{
    if (cfg->status == OK)
        runs_ok++;
    else if (cfg->status == SKIPPED)
        runs_skipped++;
    else
        runs_failed++;
}

static void print_header(const char *last_column)
{
    printf("%-14s %7s", "config", "threads");
    for (int k = 0; k < NUM_KERNELS; k++)
        printf(" %9s", kernel_name[k]);
    printf("  %8s\n", last_column);
}

/* One line of the table; the last column compares triad with a reference run. */
static void print_row(const struct config *cfg, const struct config *reference)
{
    if (cfg->status != OK)
    {
        printf("%-14s %s: %s\n", cfg->name,
               cfg->status == SKIPPED ? "SKIPPED" : "FAILED", cfg->reason);
        return;
    }

    printf("%-14s %7d", cfg->name, cfg->num_threads);
    for (int k = 0; k < NUM_KERNELS; k++)
        printf(" %9.2f", cfg->rate[k]);
    if (reference->status == OK)
        printf("   %6.1f%%", 100.0 * cfg->rate[TRIAD] / reference->rate[TRIAD]);
    printf("\n");
}

static void print_csv_rows(const char *group, const struct config *cfg)
{
    if (cfg->status != OK)
    {
        fprintf(stderr, "%s %s: %s\n", cfg->name,
                cfg->status == SKIPPED ? "SKIPPED" : "FAILED", cfg->reason);
        return;
    }
    for (int k = 0; k < NUM_KERNELS; k++)
        printf("%s,%s,%d,%s,%.3f\n", group, cfg->name, cfg->num_threads,
               kernel_name[k], cfg->rate[k]);
}

/* All nodes run at once; only the placement of the data changes. */
static void run_system_mode(void)
{
    int max_configs = num_nodes + 2;
    struct config *configs = xcalloc(max_configs, sizeof(*configs));
    struct config *cfg;
    int count = 0;
    int first_memory_node = 0;

    while (!nodes[first_memory_node].has_memory)
        first_memory_node++;

    cfg = &configs[count++];
    init_config(cfg, "aware");
    for (int n = 0; n < num_nodes; n++)
    {
        cfg->active[n] = nodes[n].num_cpus > 0;
        cfg->memory_node[n] = n;
    }

    for (int shift = 1; shift < num_nodes; shift++)
    {
        cfg = &configs[count++];
        init_config(cfg, "remote+%d", shift);
        for (int n = 0; n < num_nodes; n++)
        {
            cfg->active[n] = nodes[n].num_cpus > 0;
            cfg->memory_node[n] = (n + shift) % num_nodes;
        }
    }

    if (num_nodes > 1)
    {
        cfg = &configs[count++];
        init_config(cfg, "all-node%d", first_memory_node);
        for (int n = 0; n < num_nodes; n++)
        {
            cfg->active[n] = nodes[n].num_cpus > 0;
            cfg->memory_node[n] = first_memory_node;
        }

        cfg = &configs[count++];
        init_config(cfg, "interleave");
        cfg->interleave = true;
        for (int n = 0; n < num_nodes; n++)
            cfg->active[n] = nodes[n].num_cpus > 0;
    }

    for (int i = 0; i < count; i++)
    {
        run_config(&configs[i]);
        count_run(&configs[i]);
    }

    if (opt.csv)
    {
        for (int i = 0; i < count; i++)
            print_csv_rows("system", &configs[i]);
    }
    else
    {
        const struct config *aware = &configs[0];

        printf("All nodes active at once, best rate of all threads together in GB/s\n");
        printf("(last column: triad compared with the NUMA-aware run)\n");
        print_header("vs aware");
        for (int i = 0; i < count; i++)
            print_row(&configs[i], aware);

        if (aware->status == OK && num_nodes > 1)
        {
            double worst = aware->rate[TRIAD];

            for (int i = 1; i < count; i++)
            {
                if (configs[i].status == OK && configs[i].rate[TRIAD] < worst)
                    worst = configs[i].rate[TRIAD];
            }
            printf("\nNUMA-aware placement is %.2fx faster (triad) than the worst "
                   "placement above.\n",
                   aware->rate[TRIAD] / worst);
        }
        printf("\n");
    }

    for (int i = 0; i < count; i++)
        free_config(&configs[i]);
    free(configs);
}

/* Only the threads of one node run; their data is bound to each node in turn. */
static void run_pairs_mode(int source)
{
    struct config *configs = xcalloc(num_nodes, sizeof(*configs));

    for (int target = 0; target < num_nodes; target++)
    {
        struct config *cfg = &configs[target];

        init_config(cfg, "node%d->node%d", source, target);
        cfg->active[source] = true;
        cfg->memory_node[source] = target;
        run_config(cfg);
        count_run(cfg);
    }

    if (opt.csv)
    {
        for (int target = 0; target < num_nodes; target++)
            print_csv_rows("pair", &configs[target]);
    }
    else
    {
        printf("Threads on node %d only (%d threads), memory bound to node j, GB/s\n",
               source, nodes[source].num_cpus);
        print_header("vs local");
        for (int target = 0; target < num_nodes; target++)
            print_row(&configs[target], &configs[source]);
        printf("\n");
    }

    for (int target = 0; target < num_nodes; target++)
        free_config(&configs[target]);
    free(configs);
}

/*
 * Command line
 */

static void usage(const char *program)
{
    printf("Usage: %s [options]\n"
           "\n"
           "  -n N       elements per array, shared between the nodes (default %llu)\n"
           "  -t N       iterations of each kernel, at most %d (default %d)\n"
           "  -w N       leading iterations left out of the result (default %d)\n"
           "  -T N       threads per NUMA node; whole cores are used before SMT\n"
           "             siblings (default: every usable CPU)\n"
           "  -m MODE    system, pairs or both (default both)\n"
           "  -c NODE    pairs mode: only run the threads of this node\n"
           "  -v         show the CPU of every thread and where the pages are\n"
           "  --csv      print results as CSV: group,config,threads,kernel,GBps\n"
           "  -h         this help\n",
           program, DEFAULT_ELEMENTS, MAX_ITERATIONS, DEFAULT_ITERATIONS, DEFAULT_WARMUP);
}

static unsigned long long parse_number(const char *text, const char *option,
                                       unsigned long long min, unsigned long long max)
{
    unsigned long long value;
    char *end;

    /* strtoull would quietly accept "-5" and wrap it around */
    if (text[0] < '0' || text[0] > '9')
        die("%s: '%s' is not a number", option, text);

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value < min || value > max)
        die("%s: '%s' must be a whole number from %llu to %llu", option, text, min, max);
    return value;
}

static void parse_options(int argc, char **argv)
{
    static const struct option long_options[] = {
        {"csv", no_argument, NULL, 'C'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};
    int c;

    opt.elements = DEFAULT_ELEMENTS;
    opt.iterations = DEFAULT_ITERATIONS;
    opt.warmup = DEFAULT_WARMUP;
    opt.source_node = -1;
    opt.run_system = true;
    opt.run_pairs = true;

    while ((c = getopt_long(argc, argv, "n:t:w:T:m:c:vh", long_options, NULL)) != -1)
    {
        switch (c)
        {
        case 'n':
            opt.elements = parse_number(optarg, "-n", 1, MAX_ELEMENTS);
            break;
        case 't':
            opt.iterations = (int)parse_number(optarg, "-t", 1, MAX_ITERATIONS);
            break;
        case 'w':
            opt.warmup = (int)parse_number(optarg, "-w", 0, MAX_ITERATIONS);
            break;
        case 'T':
            opt.threads_per_node = (int)parse_number(optarg, "-T", 1, 100000);
            break;
        case 'c':
            opt.source_node = (int)parse_number(optarg, "-c", 0, 100000);
            break;
        case 'm':
            if (strcasecmp(optarg, "system") == 0)
                opt.run_pairs = false;
            else if (strcasecmp(optarg, "pairs") == 0)
                opt.run_system = false;
            else if (strcasecmp(optarg, "both") != 0)
                die("-m: '%s' is not system, pairs or both", optarg);
            break;
        case 'v':
            opt.verbose = true;
            break;
        case 'C':
            opt.csv = true;
            break;
        case 'h':
            usage(argv[0]);
            exit(0);
        default:
            fprintf(stderr, "Try '%s -h' for the options.\n", argv[0]);
            exit(1);
        }
    }

    if (optind < argc)
        die("unexpected argument '%s'", argv[optind]);
    if (opt.warmup >= opt.iterations)
        die("-w (%d) must be smaller than -t (%d)", opt.warmup, opt.iterations);
}

int main(int argc, char **argv)
{
    int cpu_nodes = 0;

    parse_options(argc, argv);
    info = opt.csv ? stderr : stdout;

    if (numa_available() < 0)
        die("this system has no NUMA support");
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES"))
        fprintf(stderr, "warning: OMP_PROC_BIND/OMP_PLACES are set; "
                        "unset them, the program pins its own threads\n");

    /* Bind strictly: a page that cannot go to the requested node is an error. */
    numa_set_bind_policy(1);
    omp_set_dynamic(0);

    discover_topology();
    for (int n = 0; n < num_nodes; n++)
    {
        if (nodes[n].num_cpus > 0)
            cpu_nodes++;
    }
    if (cpu_nodes == 0 || num_memory_nodes == 0)
        die("no usable CPUs or no memory found");
    if (opt.source_node >= num_nodes ||
        (opt.source_node >= 0 && nodes[opt.source_node].num_cpus == 0))
        die("-c %d: there is no such node with usable CPUs (nodes are 0 to %d)",
            opt.source_node, num_nodes - 1);

    fprintf(info, "NUMA-aware STREAM (%s), %zu elements per array (%.0f MiB), %d iterations\n",
            REAL_NAME, opt.elements, opt.elements * sizeof(real_t) / MIB, opt.iterations);
    print_topology();

    if (opt.csv)
        printf("group,config,threads,kernel,GBps\n");

    if (opt.run_system)
        run_system_mode();

    if (opt.run_pairs)
    {
        for (int n = 0; n < num_nodes; n++)
        {
            if (nodes[n].num_cpus == 0)
                continue;
            if (opt.source_node >= 0 && n != opt.source_node)
                continue;
            run_pairs_mode(n);
        }
    }

    fprintf(info, "%d run(s) validated, %d skipped, %d FAILED\n",
            runs_ok, runs_skipped, runs_failed);
    return (runs_failed > 0 || runs_ok == 0) ? 1 : 0;
}
