#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <inttypes.h>
#include <stdint.h>

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 10000000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#ifndef OFFSET
#define OFFSET 0
#endif

#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

/* ---- Select STREAM_TYPE ---- */
#if defined(STREAM_TYPE_DOUBLE)
typedef double STREAM_TYPE;
#define STREAM_TYPE_IS_DOUBLE

#elif defined(STREAM_TYPE_FLOAT)
typedef float STREAM_TYPE;
#define STREAM_TYPE_IS_FLOAT

#elif defined(STREAM_TYPE_UINT32)
typedef uint32_t STREAM_TYPE;
#define STREAM_TYPE_IS_UINT32

#elif defined(STREAM_TYPE_INT32)
typedef int32_t STREAM_TYPE;
#define STREAM_TYPE_IS_INT32

#else
typedef double STREAM_TYPE;
#define STREAM_TYPE_IS_DOUBLE
#endif


/* ---- Formatting / Conversion Helpers ---- */

#if defined(STREAM_TYPE_IS_DOUBLE)

#define ST_FMT "%e"
#define ST_PRINT_VAL(x) (x)

#elif defined(STREAM_TYPE_IS_FLOAT)

#define ST_FMT "%e"
#define ST_PRINT_VAL(x) ((double)(x))

#elif defined(STREAM_TYPE_IS_UINT32)

#define ST_FMT "%" PRIu32
#define ST_PRINT_VAL(x) (x)

#elif defined(STREAM_TYPE_IS_INT32)

#define ST_FMT "%" PRId32
#define ST_PRINT_VAL(x) (x)

#else
#error Unsupported STREAM_TYPE
#endif

#define FP_FMT "%e"
#define ST_TO_DOUBLE(x) ((double)(x))
#define ST_ABS(x) fabs(x)


static STREAM_TYPE a[STREAM_ARRAY_SIZE+OFFSET],
                   b[STREAM_ARRAY_SIZE+OFFSET],
                   c[STREAM_ARRAY_SIZE+OFFSET];

static double avgtime[4] = {0}, maxtime[4] = {0},
              mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char *label[4] = {
    "Copy:      ",
    "Scale:     ",
    "Add:       ",
    "Triad:     "
};

static double bytes[4] = {
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
};

double mysecond(void);
void checkSTREAMresults(void);
int checktick(void);

int main(void)
{
    int quantum;
    int BytesPerWord;
    int k;
    ssize_t j;
    STREAM_TYPE scalar;
    double t, times[4][NTIMES];

    printf(HLINE);
    printf("STREAM benchmark\n");
    printf(HLINE);

    BytesPerWord = sizeof(STREAM_TYPE);

    printf("This system uses %d bytes per array element.\n", BytesPerWord);
    printf("Array size = %llu (elements), Offset = %d\n",
           (unsigned long long)STREAM_ARRAY_SIZE, OFFSET);

    printf("Memory per array = %.1f MiB (= %.1f GiB)\n",
           BytesPerWord * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0),
           BytesPerWord * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0));

    printf("Total memory required = %.1f MiB (= %.1f GiB)\n",
           3.0 * BytesPerWord * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0),
           3.0 * BytesPerWord * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0));

    printf(HLINE);

#pragma omp parallel for
    for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    quantum = checktick();

    printf("Clock granularity = %d microseconds\n", quantum);

    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
        a[j] = 2.0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Estimated test time = %d microseconds\n", (int)t);

    scalar = 3.0;

    for (k = 0; k < NTIMES; k++) {

        times[0][k] = mysecond();
#pragma omp parallel for
        for (j = 0; j < STREAM_ARRAY_SIZE; j++)
            c[j] = a[j];
        times[0][k] = mysecond() - times[0][k];

        times[1][k] = mysecond();
#pragma omp parallel for
        for (j = 0; j < STREAM_ARRAY_SIZE; j++)
            b[j] = scalar * c[j];
        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();
#pragma omp parallel for
        for (j = 0; j < STREAM_ARRAY_SIZE; j++)
            c[j] = a[j] + b[j];
        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();
#pragma omp parallel for
        for (j = 0; j < STREAM_ARRAY_SIZE; j++)
            a[j] = b[j] + scalar * c[j];
        times[3][k] = mysecond() - times[3][k];
    }

    for (k = 1; k < NTIMES; k++) {
        for (j = 0; j < 4; j++) {
            avgtime[j] += times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    printf(HLINE);
    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");

    for (j = 0; j < 4; j++) {
        avgtime[j] /= (double)(NTIMES - 1);

        printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n",
               label[j],
               1.0E-06 * bytes[j] / mintime[j],
               avgtime[j],
               mintime[j],
               maxtime[j]);
    }

    printf(HLINE);

    checkSTREAMresults();

    return 0;
}

#define M 20

int checktick(void)
{
    int i, minDelta, Delta;
    double t1, t2, timesfound[M];

    for (i = 0; i < M; i++) {
        t1 = mysecond();
        while (((t2 = mysecond()) - t1) < 1.0E-6)
            ;
        timesfound[i] = t2;
    }

    minDelta = 1000000;

    for (i = 1; i < M; i++) {
        Delta = (int)(1.0E6 * (timesfound[i] - timesfound[i-1]));
        minDelta = MIN(minDelta, MAX(Delta, 0));
    }

    return minDelta;
}

double mysecond(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void checkSTREAMresults(void)
{
    STREAM_TYPE aj, bj, cj, scalar;
    STREAM_TYPE aSumErr, bSumErr, cSumErr;
    STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;
    double epsilon;
    ssize_t j;
    int k, ierr, err;

    aj = 1.0;
    bj = 2.0;
    cj = 0.0;

    aj = 2.0 * aj;
    scalar = 3.0;

    for (k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    aSumErr = bSumErr = cSumErr = 0.0;

    for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
        aSumErr += ST_ABS(ST_TO_DOUBLE(a[j]) - ST_TO_DOUBLE(aj));
        bSumErr += ST_ABS(ST_TO_DOUBLE(b[j]) - ST_TO_DOUBLE(bj));
        cSumErr += ST_ABS(ST_TO_DOUBLE(c[j]) - ST_TO_DOUBLE(cj));
    }

    aAvgErr = aSumErr / STREAM_ARRAY_SIZE;
    bAvgErr = bSumErr / STREAM_ARRAY_SIZE;
    cAvgErr = cSumErr / STREAM_ARRAY_SIZE;

    epsilon = (sizeof(STREAM_TYPE) == 8) ? 1.e-13 : 1.e-6;

    err = 0;

#define CHECK_ARRAY(NAME, ARR, EXP, AVGERR) \
    do { \
        double relErr = ST_ABS(ST_TO_DOUBLE(AVGERR) / ST_TO_DOUBLE(EXP)); \
        if (relErr > epsilon) { \
            err++; \
            printf("Failed Validation on array %s[], AvgRelAbsErr > epsilon (" FP_FMT ")\n", NAME, epsilon); \
            printf("     Expected Value: " ST_FMT ", AvgAbsErr: " ST_FMT ", AvgRelAbsErr: " FP_FMT "\n", \
                   ST_PRINT_VAL(EXP), \
                   ST_PRINT_VAL(AVGERR), \
                   relErr); \
            ierr = 0; \
            for (j = 0; j < STREAM_ARRAY_SIZE; j++) { \
                if (ST_ABS(ST_TO_DOUBLE(ARR[j]) / ST_TO_DOUBLE(EXP) - 1.0) > epsilon) \
                    ierr++; \
            } \
            printf("     For array %s[], %d errors were found.\n", NAME, ierr); \
        } \
    } while (0)

    CHECK_ARRAY("a", a, aj, aAvgErr);
    CHECK_ARRAY("b", b, bj, bAvgErr);
    CHECK_ARRAY("c", c, cj, cAvgErr);

#undef CHECK_ARRAY

    if (!err) {
        printf("Solution Validates: avg error less than " FP_FMT " on all arrays\n", epsilon);
    }
}