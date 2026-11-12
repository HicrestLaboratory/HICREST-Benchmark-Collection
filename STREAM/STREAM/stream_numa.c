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
 * the pace for everybody.  Rates are in GB/s, meaning 10^9 bytes per second.
 *
 * A machine with a single memory node, or a kernel without NUMA support, has
 * nothing to place.  There the program runs as a plain STREAM (config "local")
 * with the same thread pinning and the same checks.
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
 * Relation to STREAM
 * ------------------
 * This is a variant of the STREAM benchmark by John McCalpin, not the original
 * code, and results obtained with it should be published as such.  The four
 * kernels and the way bytes are counted are those of STREAM.  The differences
 * are that the arrays are cut into per-node slices whose memory is placed
 * explicitly, that every thread is pinned, and that the floating-point types
 * use a scalar of 0.4 instead of 3 (explained at the type definitions).
 *
 * As in STREAM, each array should be at least four times the size of the
 * cache.  The program prints a warning when a slice is smaller than that.
 *
 * Build
 * -----
 *   gcc -O3 -fopenmp numa_aware_stream.c -o numa_aware_stream -lnuma -lm
 *
 * The element type is double.  For another one add -DSTREAM_TYPE_FLOAT,
 * -DSTREAM_TYPE_INT32 or -DSTREAM_TYPE_UINT32.
 *
 * On a machine that does not have libnuma installed:
 *   gcc -O3 -fopenmp -DNO_LIBNUMA numa_aware_stream.c -o numa_aware_stream -lm
 *
 * Usage
 * -----
 *   ./numa_aware_stream                  all configurations, every iteration
 *   ./numa_aware_stream --summary        all configurations, best rates only
 *   ./numa_aware_stream -m pairs -c 0    node 0 threads against every node
 *   ./numa_aware_stream -m system        all nodes at once
 *   ./numa_aware_stream -T 16            at most 16 threads per node
 *   ./numa_aware_stream -v               full log (see below)
 *   ./numa_aware_stream -h               all options
 *
 * Output
 * ------
 * By default the program prints the rate of every iteration of every kernel,
 * one line per iteration, so that averages and the spread between iterations
 * can be worked out from the raw values.  The warm-up iterations (-w) are not
 * used for the best rate and are left out of this table; the CSV output keeps
 * them, marked "warmup".  With --summary the program prints one line per
 * configuration instead: the best rate of each kernel over the iterations
 * after the warm-up, as STREAM reports it.
 * The program itself computes no mean, variance or standard deviation.
 *
 * Without -v only these tables are printed, with the rates in GB/s.  A
 * run that failed or was skipped shows that in its table row, and the exit
 * status is non-zero if a run failed or none succeeded.  With -v the program
 * also prints the system description (nodes, distances, kernel settings, cache
 * and array sizes, with a warning if the arrays are too small), a title for
 * every table, the CPU each thread ran on, where the pages were found and a
 * summary line.  Keep a -v log with any results you publish.
 *
 * The program pins its own threads and binds its own memory.  Do not start it
 * under numactl or taskset unless you mean to restrict it, and do not set
 * OMP_PROC_BIND, OMP_PLACES, GOMP_CPU_AFFINITY or KMP_AFFINITY: the program
 * stops if one of them asks the OpenMP runtime to bind threads.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <math.h>
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
#include <sys/mman.h>
#include <unistd.h>

#ifndef NO_LIBNUMA
#include <numa.h>
#include <numaif.h>
#endif

/*
 * Element type, chosen when compiling.
 *
 * One iteration multiplies every value by SCALAR * (SCALAR + 2).  With the
 * classic STREAM scalar of 3 that is a factor of 15, which overflows a float
 * after about 30 iterations and a double after about 260.  The floating-point
 * types therefore use 0.4: the values shrink by 4% per iteration, slowly
 * enough to stay normal numbers for 1000 iterations, and they still change
 * on every pass, so the check at the end notices a pass that was missed.
 *
 * The integer types keep the scalar of 3 and simply wrap around.  arith_t is
 * the type the arithmetic is done in: for int32 that is uint32, because signed
 * overflow is undefined in C while unsigned arithmetic wraps.
 *
 * EPSILON is the rounding error of one operation (none for integers).
 */
#if defined(STREAM_TYPE_FLOAT)
typedef float elem_t;
typedef float arith_t;
#define ELEM_NAME   "float"
#define SCALAR      0.4f
#define EPSILON     FLT_EPSILON
#elif defined(STREAM_TYPE_INT32)
typedef int32_t elem_t;
typedef uint32_t arith_t;
#define ELEM_NAME   "int32"
#define SCALAR      3
#define EPSILON     0.0
#elif defined(STREAM_TYPE_UINT32)
typedef uint32_t elem_t;
typedef uint32_t arith_t;
#define ELEM_NAME   "uint32"
#define SCALAR      3
#define EPSILON     0.0
#else
typedef double elem_t;
typedef double arith_t;
#define ELEM_NAME   "double"
#define SCALAR      0.4
#define EPSILON     DBL_EPSILON
#endif

#define VERSION             "1.0"

#if defined(__clang__)
#define COMPILER            __VERSION__
#elif defined(__GNUC__)
#define COMPILER            "GCC " __VERSION__
#else
#define COMPILER            "an unknown compiler"
#endif

#define DEFAULT_ELEMENTS    64000000ULL
#define DEFAULT_ITERATIONS  10
#define DEFAULT_WARMUP      1
#define MIN_SLICE_ELEMENTS  65536ULL
#define MAX_ELEMENTS        (1ULL << 40)
#define MAX_ITERATIONS      1000
#define PAGE_SAMPLES        64      /* pages looked up per array */

#define MIB (1024.0 * 1024.0)
#define GIB (1024.0 * 1024.0 * 1024.0)

enum kernel { COPY, SCALE, ADD, TRIAD, NUM_KERNELS };

static const char *kernel_name[NUM_KERNELS] = { "copy", "scale", "add", "triad" };

/* Arrays read or written per element.  STREAM does not count write-allocate. */
static const int kernel_streams[NUM_KERNELS] = { 2, 2, 3, 3 };

struct options {
    size_t elements;            /* per array, summed over all slices */
    int    iterations;
    int    warmup;              /* leading iterations left out of the result */
    int    threads_per_node;    /* 0 = every usable CPU */
    int    source_node;         /* pairs mode: -1 = every node */
    bool   run_system;
    bool   run_pairs;
    bool   verbose;
    bool   csv;
    bool   summary;            /* best rates only, not every iteration */
};

struct node {
    int  *cpus;         /* one CPU per core first, SMT siblings after them */
    int   num_cpus;     /* CPUs this program will use */
    int   num_cores;    /* physical cores among those CPUs */
    bool  has_memory;
};

enum status { NOT_RUN, OK, SKIPPED, FAILED };

/* One placement of threads and data, and what came out of running it. */
struct config {
    char    name[32];
    bool   *active;         /* [node] do this node's threads take part? */
    int    *memory_node;    /* [node] node whose memory holds its slice */
    bool    interleave;

    enum status status;
    char    reason[200];
    int     num_threads;
    double  rate[NUM_KERNELS];      /* best rate in GB/s, all threads together */
    double  bytes[NUM_KERNELS];     /* bytes one iteration of the kernel moves */
    double *seconds[NUM_KERNELS];   /* [iteration] time each iteration took */
};

struct slice {
    elem_t *a, *b, *c;
};

struct worker {
    int     node;
    int     cpu;
    size_t  begin, end;     /* this thread's part of the slice */
    elem_t *a, *b, *c;

    bool    pinned;
    int     cpu_at_start;
    int     cpu_at_end;
};

static struct options opt;
static struct node *nodes;
static int num_nodes;
static int num_memory_nodes;
static bool numa_works;         /* false: treat the machine as one node */
static FILE *info;              /* stdout, or stderr when printing CSV */

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
 * NUMA layer
 *
 * Every call into libnuma is in this section.  If the kernel has no NUMA
 * support, or the program was built with -DNO_LIBNUMA, numa_works stays false
 * and the machine is treated as one node with all the CPUs and all the memory.
 */

static void numa_init(void)
{
#ifndef NO_LIBNUMA
    if (numa_available() >= 0) {
        numa_works = true;
        /* Bind strictly: a page that cannot go to the requested node is an error. */
        numa_set_bind_policy(1);
    }
#endif
}

static int count_nodes(void)
{
#ifndef NO_LIBNUMA
    if (numa_works)
        return numa_max_node() + 1;
#endif
    return 1;
}

/* Mark the CPUs of a node in *set.  Returns false if the node cannot be queried. */
static bool get_node_cpus(int node, cpu_set_t *set)
{
    CPU_ZERO(set);

#ifndef NO_LIBNUMA
    if (numa_works) {
        struct bitmask *mask = numa_allocate_cpumask();
        bool ok = numa_node_to_cpus(node, mask) == 0;

        for (unsigned int cpu = 0; ok && cpu < mask->size && cpu < CPU_SETSIZE; cpu++) {
            if (numa_bitmask_isbitset(mask, cpu))
                CPU_SET(cpu, set);
        }
        numa_free_cpumask(mask);
        return ok;
    }
#endif

    /* a single node owns every CPU */
    (void)node;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
        CPU_SET(cpu, set);
    return true;
}

static int node_of_cpu(int cpu)
{
#ifndef NO_LIBNUMA
    if (numa_works)
        return numa_node_of_cpu(cpu);
#endif
    (void)cpu;
    return 0;
}

static int node_distance(int from, int to)
{
#ifndef NO_LIBNUMA
    if (numa_works)
        return numa_distance(from, to);
#endif
    (void)from;
    (void)to;
    return 10;
}

/* MemAvailable from /proc/meminfo: free memory plus what the kernel can reclaim. */
static long long available_memory(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    char line[128];
    long long kib = -1;

    if (f) {
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %lld kB", &kib) == 1)
                break;
        }
        fclose(f);
    }
    if (kib < 0)
        return (long long)sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGESIZE);
    return kib * 1024;
}

/* Size of a node's memory in bytes.  If free_bytes is given it gets the free part. */
static long long node_memory(int node, long long *free_bytes)
{
#ifndef NO_LIBNUMA
    if (numa_works)
        return numa_node_size64(node, free_bytes);
#endif
    (void)node;
    if (free_bytes)
        *free_bytes = available_memory();
    return (long long)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
}

/* Page-aligned memory with no placement policy, as libnuma's allocators return it. */
static void *alloc_plain(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    return p == MAP_FAILED ? NULL : p;
}

static void *alloc_on_node(size_t bytes, int node)
{
#ifndef NO_LIBNUMA
    if (numa_works)
        return numa_alloc_onnode(bytes, node);
#endif
    (void)node;
    return alloc_plain(bytes);
}

static void *alloc_interleaved(size_t bytes)
{
#ifndef NO_LIBNUMA
    if (numa_works)
        return numa_alloc_interleaved(bytes);
#endif
    return alloc_plain(bytes);
}

static void free_array(void *array, size_t bytes)
{
#ifndef NO_LIBNUMA
    if (numa_works) {
        numa_free(array, bytes);
        return;
    }
#endif
    munmap(array, bytes);
}

/* Node that holds the page at this address right now, or -1 if unknown. */
static int node_of_page(void *addr)
{
#ifndef NO_LIBNUMA
    int node = -1;

    if (numa_works && get_mempolicy(&node, NULL, 0, addr, MPOL_F_NODE | MPOL_F_ADDR) == 0)
        return node;
#endif
    (void)addr;
    return -1;
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

    for (int i = 0; i < count; i++) {
        core[i] = core_of_cpu(cpus[i]);
        for (int j = 0; j < i; j++) {
            if (core[j] == core[i])
                is_sibling[i] = true;
        }
        if (!is_sibling[i])
            ordered[n++] = cpus[i];
    }
    num_cores = n;

    for (int i = 0; i < count; i++) {
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
    cpu_set_t allowed, on_node;

    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        die("sched_getaffinity: %s", strerror(errno));

    num_nodes = count_nodes();
    nodes = xcalloc(num_nodes, sizeof(*nodes));

    for (int n = 0; n < num_nodes; n++) {
        struct node *node = &nodes[n];

        node->cpus = xcalloc(CPU_SETSIZE, sizeof(int));
        node->has_memory = node_memory(n, NULL) > 0;
        if (node->has_memory)
            num_memory_nodes++;

        if (!get_node_cpus(n, &on_node))
            continue;
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (CPU_ISSET(cpu, &on_node) && CPU_ISSET(cpu, &allowed))
                node->cpus[node->num_cpus++] = cpu;
        }
        node->num_cores = order_cores_first(node->cpus, node->num_cpus);

        if (opt.threads_per_node > 0 && node->num_cpus > opt.threads_per_node) {
            node->num_cpus = opt.threads_per_node;
            if (node->num_cores > node->num_cpus)
                node->num_cores = node->num_cpus;
        }
    }
}

static void print_topology(void)
{
    int nodes_without_cpus = 0;

    if (!numa_works) {
#ifdef NO_LIBNUMA
        fprintf(info, "Built without libnuma: memory placement is not controlled or checked.\n");
        if (access("/sys/devices/system/node/node1", F_OK) == 0)
            fprintf(info, "WARNING: this machine has more than one NUMA node.  "
                    "Build with libnuma to measure them.\n");
#else
        fprintf(info, "No NUMA support in this kernel.\n");
#endif
        fprintf(info, "One memory node: %d threads on %d cores, %.1f GiB\n",
                nodes[0].num_cpus, nodes[0].num_cores, node_memory(0, NULL) / GIB);
        fprintf(info, "Running as a plain STREAM with local memory only.\n");
        return;
    }

    fprintf(info, "NUMA nodes: %d\n", num_nodes);
    for (int n = 0; n < num_nodes; n++) {
        fprintf(info, "  node %d: %d threads on %d cores, %.1f GiB\n", n,
                nodes[n].num_cpus, nodes[n].num_cores, node_memory(n, NULL) / GIB);
        if (nodes[n].num_cpus == 0)
            nodes_without_cpus++;
    }
    if (nodes_without_cpus > 0)
        fprintf(info, "Note: %d of the %d nodes have no CPUs this program may use "
                "(memory-only nodes, or a restricted CPU affinity).\n",
                nodes_without_cpus, num_nodes);

    fprintf(info, "Distances reported by the firmware (10 = local):\n      ");
    for (int j = 0; j < num_nodes; j++)
        fprintf(info, "%4d", j);
    fprintf(info, "\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(info, "  %2d: ", i);
        for (int j = 0; j < num_nodes; j++)
            fprintf(info, "%4d", node_distance(i, j));
        fprintf(info, "\n");
    }
    if (num_nodes == 1)
        fprintf(info, "Only one NUMA node here: running as a plain STREAM "
                "with local memory only.\n");
}


/*
 * Caches and system settings
 */

/* Read the first line of a small text file.  Returns false if it cannot be read. */
static bool read_line(const char *path, char *text, size_t size)
{
    FILE *f = fopen(path, "r");
    bool ok;

    if (!f)
        return false;
    ok = fgets(text, (int)size, f) != NULL;
    fclose(f);
    if (ok)
        text[strcspn(text, "\n")] = '\0';
    return ok;
}

/*
 * Size in bytes of the last-level data cache of a CPU, or 0 if the kernel does
 * not say.  *first_cpu gets the lowest-numbered CPU that shares the cache,
 * which tells one cache from another.
 */
static long long last_level_cache(int cpu, int *first_cpu)
{
    long long best_size = 0;
    int best_level = 0;

    *first_cpu = cpu;
    for (int index = 0; index < 16; index++) {
        char path[160], text[128];
        long long size = 0;
        char unit = 'K';
        int level, first = cpu;

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/level",
                 cpu, index);
        if (!read_line(path, text, sizeof(text)))
            break;
        level = atoi(text);

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/type",
                 cpu, index);
        if (read_line(path, text, sizeof(text)) && strcmp(text, "Instruction") == 0)
            continue;

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/size",
                 cpu, index);
        if (!read_line(path, text, sizeof(text)) || sscanf(text, "%lld%c", &size, &unit) < 1)
            continue;
        if (unit == 'M')
            size *= 1024 * 1024;
        else if (unit == 'G')
            size *= 1024 * 1024 * 1024LL;
        else
            size *= 1024;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", cpu, index);
        if (read_line(path, text, sizeof(text)))
            sscanf(text, "%d", &first);

        if (level > best_level) {
            best_level = level;
            best_size = size;
            *first_cpu = first;
        }
    }
    return best_size;
}

/* Last-level cache of the CPUs a node will use, counting a shared cache once. */
static long long node_cache_bytes(const struct node *node)
{
    int *counted = xcalloc(node->num_cpus, sizeof(int));
    int num_counted = 0;
    long long total = 0;

    for (int i = 0; i < node->num_cpus; i++) {
        int first;
        long long size = last_level_cache(node->cpus[i], &first);
        bool seen = false;

        for (int j = 0; j < num_counted; j++) {
            if (counted[j] == first)
                seen = true;
        }
        if (size > 0 && !seen) {
            counted[num_counted++] = first;
            total += size;
        }
    }
    free(counted);
    return total;
}

/*
 * STREAM asks for each array to be at least four times the size of the cache,
 * so that the kernels measure the memory and not the cache.  Warn if the
 * smallest slice this run will use does not meet that.
 */
static void check_array_size(void)
{
    long long cache = 0;
    long long slice_bytes;
    int cpu_nodes = 0;
    int sharing = 1;

    for (int n = 0; n < num_nodes; n++) {
        long long node_cache;

        if (nodes[n].num_cpus == 0)
            continue;
        cpu_nodes++;
        node_cache = node_cache_bytes(&nodes[n]);
        if (node_cache > cache)
            cache = node_cache;
    }
    /* In system mode the elements are shared between all the nodes with CPUs. */
    if (num_nodes > 1 && opt.run_system)
        sharing = cpu_nodes;
    slice_bytes = (long long)(opt.elements / (size_t)sharing * sizeof(elem_t));

    if (cache == 0) {
        fprintf(info, "Cache sizes are not known here: make sure each array (%.0f MiB per node) "
                "is at least 4 times the cache.\n", slice_bytes / MIB);
        return;
    }

    fprintf(info, "Last-level cache per node: %.0f MiB, smallest array per node: %.0f MiB\n",
            cache / MIB, slice_bytes / MIB);
    if (slice_bytes < 4 * cache) {
        long long needed = (4 * cache + (long long)sizeof(elem_t) - 1) / (long long)sizeof(elem_t);

        fprintf(info, "WARNING: the arrays are smaller than 4 times the cache, so the rates "
                "include cache effects.\n         Use -n %lld or more.\n", needed * sharing);
    }
}

/* Kernel settings that change the rates, printed so that they end up in the log. */
static void print_system_settings(void)
{
    char text[128];

    if (read_line("/sys/kernel/mm/transparent_hugepage/enabled", text, sizeof(text)))
        fprintf(info, "Transparent huge pages: %s\n", text);
    if (read_line("/proc/sys/kernel/numa_balancing", text, sizeof(text)))
        fprintf(info, "Automatic NUMA balancing: %s\n", strcmp(text, "0") == 0 ? "off" : "on");
    if (read_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", text, sizeof(text)))
        fprintf(info, "CPU frequency governor: %s\n", text);
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
#elif defined(__clang__)
#define KERNEL_ATTRIBUTES __attribute__((noinline, no_builtin("memcpy")))
#else
#define KERNEL_ATTRIBUTES __attribute__((noinline))
#endif

static KERNEL_ATTRIBUTES void
run_kernel(enum kernel kernel, elem_t *restrict a, elem_t *restrict b,
           elem_t *restrict c, size_t begin, size_t end)
{
    const arith_t scalar = SCALAR;

    switch (kernel) {
    case COPY:
        for (size_t j = begin; j < end; j++)
            c[j] = a[j];
        break;
    case SCALE:
        for (size_t j = begin; j < end; j++)
            b[j] = (elem_t)(scalar * (arith_t)c[j]);
        break;
    case ADD:
        for (size_t j = begin; j < end; j++)
            c[j] = (elem_t)((arith_t)a[j] + (arith_t)b[j]);
        break;
    case TRIAD:
        for (size_t j = begin; j < end; j++)
            a[j] = (elem_t)((arith_t)b[j] + scalar * (arith_t)c[j]);
        break;
    default:
        break;
    }
}

static void initialise(elem_t *a, elem_t *b, elem_t *c, size_t begin, size_t end)
{
    for (size_t j = begin; j < end; j++) {
        a[j] = 1;
        b[j] = 2;
        c[j] = 0;
    }
    for (size_t j = begin; j < end; j++)
        a[j] = (elem_t)(2 * (arith_t)a[j]);
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

    for (int n = 0; n < num_nodes && ok; n++) {
        double bytes = 3.0 * slice_bytes;

        if (!cfg->active[n])
            continue;

        if (cfg->interleave) {
            for (int m = 0; m < num_nodes; m++) {
                if (nodes[m].has_memory)
                    needed[m] += bytes / num_memory_nodes;
            }
        } else if (!nodes[cfg->memory_node[n]].has_memory) {
            set_status(cfg, SKIPPED, "node %d has no memory", cfg->memory_node[n]);
            ok = false;
        } else {
            needed[cfg->memory_node[n]] += bytes;
        }
    }

    for (int n = 0; n < num_nodes && ok; n++) {
        long long free_bytes = 0;

        if (needed[n] == 0)
            continue;
        node_memory(n, &free_bytes);
        if (needed[n] > 0.9 * free_bytes) {
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
        return alloc_interleaved(bytes);
    return alloc_on_node(bytes, cfg->memory_node[node]);
}

static bool allocate_slices(const struct config *cfg, struct slice *slices, size_t slice_bytes)
{
    for (int n = 0; n < num_nodes; n++) {
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
    for (int n = 0; n < num_nodes; n++) {
        if (slices[n].a)
            free_array(slices[n].a, slice_bytes);
        if (slices[n].b)
            free_array(slices[n].b, slice_bytes);
        if (slices[n].c)
            free_array(slices[n].c, slice_bytes);
    }
}


/*
 * Checks done after a run
 */

static void print_thread_map(const struct config *cfg, const struct worker *workers,
                             int num_threads)
{
    fprintf(info, "[%s] where the threads ran (sched_getcpu)\n", cfg->name);
    for (int t = 0; t < num_threads; t++) {
        const struct worker *w = &workers[t];
        int node_at_end = w->cpu_at_end >= 0 ? node_of_cpu(w->cpu_at_end) : -1;
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

    for (int t = 0; t < num_threads; t++) {
        const struct worker *w = &workers[t];

        if (!w->pinned || w->cpu_at_start != w->cpu || w->cpu_at_end != w->cpu ||
            node_of_cpu(w->cpu_at_end) != w->node)
            misplaced++;
    }
    if (misplaced == 0)
        return true;

    set_status(cfg, FAILED, "%d of %d threads were not on their intended CPU",
               misplaced, num_threads);
    return false;
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

    for (int i = 0; i < PAGE_SAMPLES; i++) {
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
    int *pages_on_node;
    bool ok = true;

    /* Without NUMA there is only one place the pages can be. */
    if (!numa_works)
        return true;

    pages_on_node = xcalloc(num_nodes, sizeof(int));
    for (int n = 0; n < num_nodes && ok; n++) {
        int wanted = cfg->memory_node[n];
        int found = 0;
        int nodes_used = 0;

        if (!cfg->active[n])
            continue;

        memset(pages_on_node, 0, num_nodes * sizeof(int));
        found += sample_pages(slices[n].a, slice_bytes, pages_on_node);
        found += sample_pages(slices[n].b, slice_bytes, pages_on_node);
        found += sample_pages(slices[n].c, slice_bytes, pages_on_node);
        for (int m = 0; m < num_nodes; m++) {
            if (pages_on_node[m] > 0)
                nodes_used++;
        }

        if (opt.verbose) {
            if (cfg->interleave)
                fprintf(info, "[%s] slice of node %d, interleaved: pages found on", cfg->name, n);
            else
                fprintf(info, "[%s] slice of node %d, requested on node %d: pages found on",
                        cfg->name, n, wanted);
            for (int m = 0; m < num_nodes; m++)
                fprintf(info, " node%d=%d", m, pages_on_node[m]);
            fprintf(info, "\n");
        }

        if (found != 3 * PAGE_SAMPLES) {
            set_status(cfg, FAILED, "could not find out where %d of %d sampled pages are",
                       3 * PAGE_SAMPLES - found, 3 * PAGE_SAMPLES);
            ok = false;
        } else if (cfg->interleave) {
            if (num_memory_nodes > 1 && nodes_used < 2) {
                set_status(cfg, FAILED, "the interleaved pages are all on one node");
                ok = false;
            }
        } else if (pages_on_node[wanted] != found) {
            set_status(cfg, FAILED, "%d of %d sampled pages of node %d's slice are not on node %d",
                       found - pages_on_node[wanted], found, n, wanted);
            ok = false;
        }
    }

    free(pages_on_node);
    return ok;
}

/* Number of elements further from the expected value than the tolerance allows. */
static size_t count_wrong(const elem_t *x, size_t count, double expected, double tolerance)
{
    double allowed = tolerance * fabs(expected);
    size_t wrong = 0;

    for (size_t j = 0; j < count; j++) {
        /* written so that NaN counts as wrong */
        if (!(fabs((double)x[j] - expected) <= allowed))
            wrong++;
    }
    return wrong;
}

/*
 * Every element goes through the same sequence of operations, so running the
 * benchmark on a single element gives the value all of them must have.
 *
 * Every element is compared, not an average as in STREAM, so a single wrong
 * element is enough to fail the run.  Integers must match exactly.
 * Floating-point values may differ by rounding if the compiler builds the
 * vector loop and the single-element case with different instructions (fused
 * multiply-add or not), so they get a few rounding errors per iteration as
 * tolerance.
 */
static bool results_are_correct(struct config *cfg, const struct slice *slices,
                                size_t slice_elements)
{
    double tolerance = 8.0 * EPSILON * opt.iterations;
    elem_t a, b, c;

    initialise(&a, &b, &c, 0, 1);
    for (int iter = 0; iter < opt.iterations; iter++) {
        for (int k = 0; k < NUM_KERNELS; k++)
            run_kernel(k, &a, &b, &c, 0, 1);
    }

    for (int n = 0; n < num_nodes; n++) {
        size_t wrong;

        if (!cfg->active[n])
            continue;
        wrong = count_wrong(slices[n].a, slice_elements, (double)a, tolerance) +
                count_wrong(slices[n].b, slice_elements, (double)b, tolerance) +
                count_wrong(slices[n].c, slice_elements, (double)c, tolerance);
        if (wrong > 0) {
            set_status(cfg, FAILED, "%zu wrong values in the arrays of node %d's slice",
                       wrong, n);
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

    for (int n = 0; n < num_nodes; n++) {
        if (!cfg->active[n])
            continue;
        for (int i = 0; i < nodes[n].num_cpus; i++) {
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
    double **elapsed = cfg->seconds;
    size_t slice_elements, slice_bytes;
    int active_nodes = 0;
    int num_threads = 0;
    int team_size = 0;

    for (int n = 0; n < num_nodes; n++) {
        if (cfg->active[n]) {
            active_nodes++;
            num_threads += nodes[n].num_cpus;
        }
    }
    cfg->num_threads = num_threads;
    if (num_threads == 0) {
        set_status(cfg, SKIPPED, "no usable CPUs");
        return;
    }

    slice_elements = opt.elements / (size_t)active_nodes;
    slice_bytes = slice_elements * sizeof(elem_t);
    if (slice_elements < MIN_SLICE_ELEMENTS) {
        set_status(cfg, SKIPPED, "only %zu elements per node, use a larger -n", slice_elements);
        return;
    }
    if (!memory_is_available(cfg, slice_bytes))
        return;

    /* Progress message, switched off so that only the results are printed. */
    /* fprintf(stderr, "running %s ...\n", cfg->name); */

    slices = xcalloc(num_nodes, sizeof(*slices));
    workers = xcalloc(num_threads, sizeof(*workers));

    if (!allocate_slices(cfg, slices, slice_bytes)) {
        set_status(cfg, FAILED, "could not allocate the arrays");
        goto out;
    }
    assign_workers(cfg, slices, slice_elements, workers);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int team = omp_get_num_threads();
        struct worker *w = &workers[tid];

        if (tid == 0)
            team_size = team;

        /*
         * Every thread gets the same answer from omp_get_num_threads(), so
         * either the whole team does the work or nobody does.
         */
        if (team == num_threads) {
            w->pinned = pin_to_cpu(w->cpu);
            w->cpu_at_start = sched_getcpu();

            /* The thread that will use the data is also the one to touch it first. */
            initialise(w->a, w->b, w->c, w->begin, w->end);

            for (int iter = 0; iter < opt.iterations; iter++) {
                for (int k = 0; k < NUM_KERNELS; k++) {
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

    if (team_size != num_threads) {
        set_status(cfg, FAILED, "OpenMP started %d threads instead of %d "
                   "(check OMP_THREAD_LIMIT and cgroup limits)", team_size, num_threads);
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

    for (int k = 0; k < NUM_KERNELS; k++) {
        double bytes = (double)kernel_streams[k] * sizeof(elem_t) *
                       (double)slice_elements * active_nodes;
        double best = elapsed[k][opt.warmup];

        cfg->bytes[k] = bytes;

        for (int iter = opt.warmup + 1; iter < opt.iterations; iter++) {
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
    for (int k = 0; k < NUM_KERNELS; k++)
        cfg->seconds[k] = xcalloc(opt.iterations, sizeof(double));
}

static void free_config(struct config *cfg)
{
    free(cfg->active);
    free(cfg->memory_node);
    for (int k = 0; k < NUM_KERNELS; k++)
        free(cfg->seconds[k]);
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
    if (last_column)
        printf("  %8s", last_column);
    printf("\n");
}

/* One line of the table.  With a reference run, the last column compares triad with it. */
static void print_row(const struct config *cfg, const struct config *reference)
{
    if (cfg->status != OK) {
        printf("%-14s %s: %s\n", cfg->name,
               cfg->status == SKIPPED ? "SKIPPED" : "FAILED", cfg->reason);
        return;
    }

    printf("%-14s %7d", cfg->name, cfg->num_threads);
    for (int k = 0; k < NUM_KERNELS; k++)
        printf(" %9.2f", cfg->rate[k]);
    if (reference && reference->status == OK)
        printf("   %6.1f%%", 100.0 * cfg->rate[TRIAD] / reference->rate[TRIAD]);
    printf("\n");
}

/* Rate of one iteration of one kernel in GB/s. */
static double iteration_rate(const struct config *cfg, int kernel, int iteration)
{
    return cfg->bytes[kernel] / cfg->seconds[kernel][iteration] / 1e9;
}

/*
 * Table with one line per iteration: the rate of every kernel in that
 * iteration.  The warm-up iterations are left out, as they are for the best rate.
 */
static void print_iterations(const struct config *configs, int count)
{
    if (opt.verbose)
        printf("Rate of every iteration in GB/s, all threads together "
               "(warm-up iterations are not shown)\n");
    printf("%-14s %7s %5s", "config", "threads", "iter");
    for (int k = 0; k < NUM_KERNELS; k++)
        printf(" %9s", kernel_name[k]);
    printf("\n");

    for (int i = 0; i < count; i++) {
        const struct config *cfg = &configs[i];

        if (cfg->status != OK) {
            printf("%-14s %s: %s\n", cfg->name,
                   cfg->status == SKIPPED ? "SKIPPED" : "FAILED", cfg->reason);
            continue;
        }
        for (int iter = opt.warmup; iter < opt.iterations; iter++) {
            printf("%-14s %7d %5d", cfg->name, cfg->num_threads, iter);
            for (int k = 0; k < NUM_KERNELS; k++)
                printf(" %9.3f", iteration_rate(cfg, k, iter));
            printf("\n");
        }
    }
    printf("\n");
}

static void print_csv_header(void)
{
    if (opt.summary)
        printf("group,config,threads,kernel,GBps\n");
    else
        printf("group,config,threads,kernel,stat,iteration,time_s,GBps\n");
}

/* One CSV line per kernel and iteration; stat is "warmup" or "run". */
static void print_csv_iterations(const char *group, const struct config *cfg)
{
    if (cfg->status != OK) {
        fprintf(stderr, "%s %s: %s\n", cfg->name,
                cfg->status == SKIPPED ? "SKIPPED" : "FAILED", cfg->reason);
        return;
    }
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int iter = 0; iter < opt.iterations; iter++)
            printf("%s,%s,%d,%s,%s,%d,%.9f,%.3f\n", group, cfg->name, cfg->num_threads,
                   kernel_name[k], iter < opt.warmup ? "warmup" : "run", iter,
                   cfg->seconds[k][iter], iteration_rate(cfg, k, iter));
    }
}

/* Summary: one CSV line per kernel with its best rate. */
static void print_csv_rows(const char *group, const struct config *cfg)
{
    if (cfg->status != OK) {
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
    for (int n = 0; n < num_nodes; n++) {
        cfg->active[n] = nodes[n].num_cpus > 0;
        cfg->memory_node[n] = n;
    }

    for (int shift = 1; shift < num_nodes; shift++) {
        cfg = &configs[count++];
        init_config(cfg, "remote+%d", shift);
        for (int n = 0; n < num_nodes; n++) {
            cfg->active[n] = nodes[n].num_cpus > 0;
            cfg->memory_node[n] = (n + shift) % num_nodes;
        }
    }

    if (num_nodes > 1) {
        cfg = &configs[count++];
        init_config(cfg, "all-node%d", first_memory_node);
        for (int n = 0; n < num_nodes; n++) {
            cfg->active[n] = nodes[n].num_cpus > 0;
            cfg->memory_node[n] = first_memory_node;
        }

        cfg = &configs[count++];
        init_config(cfg, "interleave");
        cfg->interleave = true;
        for (int n = 0; n < num_nodes; n++)
            cfg->active[n] = nodes[n].num_cpus > 0;
    }

    for (int i = 0; i < count; i++) {
        run_config(&configs[i]);
        count_run(&configs[i]);
    }

    if (opt.csv) {
        for (int i = 0; i < count; i++) {
            if (opt.summary)
                print_csv_rows("system", &configs[i]);
            else
                print_csv_iterations("system", &configs[i]);
        }
    } else if (!opt.summary) {
        print_iterations(configs, count);
    } else {
        const struct config *aware = &configs[0];

        if (opt.verbose) {
            printf("All nodes active at once, best rate of all threads together in GB/s\n");
            printf("(last column: triad compared with the NUMA-aware run)\n");
        }
        print_header("vs aware");
        for (int i = 0; i < count; i++)
            print_row(&configs[i], aware);

        if (opt.verbose && aware->status == OK && num_nodes > 1) {
            double worst = aware->rate[TRIAD];

            for (int i = 1; i < count; i++) {
                if (configs[i].status == OK && configs[i].rate[TRIAD] < worst)
                    worst = configs[i].rate[TRIAD];
            }
            if (worst < aware->rate[TRIAD])
                printf("\nNUMA-aware placement is %.2fx faster (triad) than the worst "
                       "placement above.\n", aware->rate[TRIAD] / worst);
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

    for (int target = 0; target < num_nodes; target++) {
        struct config *cfg = &configs[target];

        init_config(cfg, "node%d->node%d", source, target);
        cfg->active[source] = true;
        cfg->memory_node[source] = target;
        run_config(cfg);
        count_run(cfg);
    }

    if (opt.csv) {
        for (int target = 0; target < num_nodes; target++) {
            if (opt.summary)
                print_csv_rows("pair", &configs[target]);
            else
                print_csv_iterations("pair", &configs[target]);
        }
    } else if (!opt.summary) {
        print_iterations(configs, num_nodes);
    } else {
        if (opt.verbose)
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

/* One memory node: there is only the local case to measure. */
static void run_single_node(void)
{
    struct config cfg;

    init_config(&cfg, "local");
    cfg.active[0] = true;
    cfg.memory_node[0] = 0;
    run_config(&cfg);
    count_run(&cfg);

    if (opt.csv) {
        if (opt.summary)
            print_csv_rows("local", &cfg);
        else
            print_csv_iterations("local", &cfg);
    } else if (!opt.summary) {
        print_iterations(&cfg, 1);
    } else {
        if (opt.verbose)
            printf("Best rate of all threads together in GB/s\n");
        print_header(NULL);
        print_row(&cfg, NULL);
        printf("\n");
    }
    free_config(&cfg);
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
           "  -v         also print the system description, the warnings, the table\n"
           "             titles, the CPU of every thread and where the pages are\n"
           "  -s, --summary\n"
           "             print only the best rate of each kernel per configuration;\n"
           "             without it the rate of every iteration after the warm-up\n"
           "             is printed\n"
           "  --csv      print the results as CSV.  Every iteration:\n"
           "             group,config,threads,kernel,stat,iteration,time_s,GBps\n"
           "             (stat is warmup or run; the CSV keeps the warm-up\n"
           "             iterations).  With --summary:\n"
           "             group,config,threads,kernel,GBps\n"
           "  -h         this help\n"
           "\n"
           "-m and -c only matter on a machine with more than one NUMA node.\n"
           "Rates are printed in GB/s, meaning 10^9 bytes per second.\n",
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
        { "csv",  no_argument, NULL, 'C' },
        { "summary", no_argument, NULL, 's' },
        { "help", no_argument, NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };
    unsigned long long value;
    int c;

    opt.elements = DEFAULT_ELEMENTS;
    opt.iterations = DEFAULT_ITERATIONS;
    opt.warmup = DEFAULT_WARMUP;
    opt.source_node = -1;
    opt.run_system = true;
    opt.run_pairs = true;

    while ((c = getopt_long(argc, argv, "n:t:w:T:m:c:svh", long_options, NULL)) != -1) {
        switch (c) {
        case 'n':
            value = parse_number(optarg, "-n", 1, MAX_ELEMENTS);
            if (value > SIZE_MAX / (4 * sizeof(elem_t)))
                die("-n: %s elements do not fit in the address space of this machine", optarg);
            opt.elements = (size_t)value;
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
        case 's':
            opt.summary = true;
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

/*
 * With one of these variables set, the OpenMP runtime binds the main thread to
 * a single place before main() starts.  The program would then find only the
 * CPUs of that place and quietly run with far fewer threads than intended.
 */
static void refuse_openmp_binding(void)
{
    static const char *variables[] = {
        "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"
    };

    for (size_t i = 0; i < sizeof(variables) / sizeof(variables[0]); i++) {
        const char *value = getenv(variables[i]);

        if (!value || strcasecmp(value, "false") == 0 || strcasecmp(value, "none") == 0 ||
            strcasecmp(value, "disabled") == 0)
            continue;
        die("%s is set to '%s'.  Unset it and run again: the program pins its own\n"
            "threads, and this setting hides most CPUs from it.", variables[i], value);
    }
}

int main(int argc, char **argv)
{
    int cpu_nodes = 0;

    parse_options(argc, argv);
    info = opt.csv ? stderr : stdout;

    refuse_openmp_binding();
    numa_init();
    omp_set_dynamic(0);

    discover_topology();
    for (int n = 0; n < num_nodes; n++) {
        if (nodes[n].num_cpus > 0)
            cpu_nodes++;
    }
    if (cpu_nodes == 0 || num_memory_nodes == 0)
        die("no usable CPUs or no memory found");
    if (opt.source_node >= num_nodes ||
        (opt.source_node >= 0 && nodes[opt.source_node].num_cpus == 0))
        die("-c %d: there is no such node with usable CPUs (nodes are 0 to %d)",
            opt.source_node, num_nodes - 1);

    /* By default only the result tables are printed; -v adds the rest. */
    if (opt.verbose) {
        fprintf(info, "NUMA-aware STREAM (%s), %zu elements per array (%.0f MiB), %d iterations\n",
                ELEM_NAME, opt.elements, opt.elements * sizeof(elem_t) / MIB, opt.iterations);
        fprintf(info, "Version %s, compiled with %s\n", VERSION, COMPILER);
        print_topology();
        print_system_settings();
        check_array_size();
        fprintf(info, "\n");
    }

    if (opt.csv)
        print_csv_header();

    if (num_nodes == 1) {
        run_single_node();
    } else {
        if (opt.run_system)
            run_system_mode();

        for (int n = 0; n < num_nodes && opt.run_pairs; n++) {
            if (nodes[n].num_cpus == 0)
                continue;
            if (opt.source_node >= 0 && n != opt.source_node)
                continue;
            run_pairs_mode(n);
        }
    }

    if (opt.verbose)
        fprintf(info, "%d run(s) validated, %d skipped, %d FAILED\n",
                runs_ok, runs_skipped, runs_failed);
    return (runs_failed > 0 || runs_ok == 0) ? 1 : 0;
}