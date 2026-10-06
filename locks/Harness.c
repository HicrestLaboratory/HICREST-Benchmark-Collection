// The test harness creates N pthread worker-threads, and then blocks for a fixed period of time, T, after which a
// global stop-flag is set to indicate an experiment is over.  The N threads repeatedly attempt entry into a
// self-checking critical-section until the stop flag is set.  During the T seconds, each thread counts the number of
// times it enters the critical section. The higher the aggregate count, the better an algorithm, as it is able to
// process more requests for the critical section per unit time.  When the stop flag is set, a worker thread stops
// entering the critical section, and atomically adds it subtotal entry-counter to a global total entry-counter. When
// the driver unblocks after T seconds, it busy waits until all threads have noticed the stop flag and added their
// subtotal to the global counter, which is then stored.  Five identical experiments are performed, each lasting T
// seconds. The median value of the five results is printed.
//
// Portability: x86-64, i386, AArch64, ARMv7, RISC-V (rv32/rv64); gcc and clang.
//
// Changes relative to the previous portable revision:
//  - Restored Cas/Casm, Casw/Caswm, Casvw/Casvwm, Tas/Tasm, Clr/Clrm (used by HemLock and other algorithms).
//  - Restored C++ Fai as std::atomic fetch_add.
//  - Restored Allocator tail padding (size rounded up to whole cache lines) and abort on allocation failure.
//  - Each worker thread seeds its own thread-local PRNG state before running Worker (previously only the main
//    thread was seeded, so worker PRNGs returned 0 and the interference checksum could not detect violations).
//  - RANDOM critical-section delay uses CSTimes (previously NCSTimes, a division by zero when NCS_DELAY = 0).
//  - ARM default Pause() is "isb sy" as in the original harness; -DARMYIELD selects "yield".
//  - Restored BARRIER output mode, RANDOM tag in CFMT header, plg2 host alias, temporary pinning of the main
//    thread during worker creation.
//  - Added bpif3 host (Banana Pi BPI-F3, 8 cores, linear affinity) and a check that the target CPU exists.

#ifndef __cplusplus
#ifndef _GNU_SOURCE
#define _GNU_SOURCE										// See feature_test_macros(7)
#endif // _GNU_SOURCE
#define TYPEOF( T ) __typeof__( T )						// works in strict ISO C for gcc and clang
#else
#include <atomic>
#define _Atomic( T ) std::atomic<T>
#define TYPEOF( T ) decltype( +T )						// silly magic with the '+'
#endif // __cplusplus

#include <stdio.h>
#include <stdlib.h>										// abort, exit, atoi, rand, qsort, posix_memalign
#include <stdbool.h>									// true, false
#include <math.h>										// sqrt
#include <assert.h>
#include <pthread.h>
#include <errno.h>										// errno
#include <stdint.h>										// uintptr_t, uintmax_t, SIZE_MAX
#include <sys/types.h>									// ssize_t
#include <sys/time.h>
#include <time.h>										// clock_gettime
#include <poll.h>										// poll
#include <unistd.h>										// getpid, sleep, sysconf
#include <string.h>										// strcmp, memset
#include <limits.h>
#ifdef CFMT												// output comma format
#include <locale.h>
#endif // CFMT

// Weakly-ordered architectures: statements only needed there.
#if defined( __ARM_ARCH ) || defined( __riscv )
#define WO( stmt ) stmt
#else
#define WO( stmt ) do {} while (0)
#endif

#if defined( __clang__ ) || ( defined( __GNUC__ ) && __GNUC__ >= 7 ) // valid compiler diagnostic ?
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"	// Mute g++/clang
#endif

#define CACHE_ALIGN 128									// Intel recommendation
#define CALIGN __attribute__(( aligned(CACHE_ALIGN) ))

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifdef FAST
	// unlikely
	#define FASTPATH(x) __builtin_expect(!!(x), 0)
	#define SLOWPATH(x) __builtin_expect(!!(x), 1)
#else
	// likely
	#define FASTPATH(x) __builtin_expect(!!(x), 1)
	#define SLOWPATH(x) __builtin_expect(!!(x), 0)
#endif // FASTPATH

#define MIN( x, y ) ((x < y) ? (x) : (y))
#define MAX( x, y ) ((x > y) ? (x) : (y))

#define xstr(s) str(s)
#define str(s) #s

//------------------------------------------------------------------------------

typedef size_t TYPE;									// unsigned addressable unsigned word-size
typedef ssize_t STYPE;									// signed addressable signed word-size
typedef volatile TYPE VTYPE;							// volatile addressable unsigned word-size
typedef volatile STYPE VSTYPE;							// volatile addressable signed word-size
typedef uint32_t RTYPE;									// unsigned 32-bit integer

typedef uint8_t BYTESIZE;
typedef volatile uint8_t VBYTESIZE;
#if __SIZEOF_POINTER__ == 8								// portable (glibc and musl); __WORDSIZE is glibc only
typedef uint32_t HALFSIZE;
typedef volatile uint32_t VHALFSIZE;
typedef uint64_t WHOLESIZE;
typedef volatile uint64_t VWHOLESIZE;
#else
typedef uint16_t HALFSIZE;
typedef volatile uint16_t VHALFSIZE;
typedef uint32_t WHOLESIZE;
typedef volatile uint32_t VWHOLESIZE;
#endif // __SIZEOF_POINTER__ == 8

#ifdef ATOMIC
#define VTYPE _Atomic(TYPE)
#define VBYTESIZE _Atomic(BYTESIZE)
#define VHALFSIZE _Atomic(HALFSIZE)
#define VWHOLESIZE _Atomic(WHOLESIZE)
#endif // ATOMIC

//------------------------------------------------------------------------------

// Architectural ST-LD memory fences: In theory explicit memory fences should be obviated by the C11 automated atomic
// declarations, which is provided by specifying -DATOMIC.  Unfortunately, the current implementations of C11 atomic
// operators is still sub-optimal compared to hand fencing. Manually annotating every load/store with atomic macros is
// more error prone because there is more to get wrong than manually fencing.
//
// The "memory" clobber is required so the compiler cannot move loads/stores across the fence.

#ifdef ATOMIC
	#if defined( LPAUSE ) || defined( MPAUSE )
		#error Compilation options LPAUSE/MPAUSE are incompatible with ATOMIC.
	#endif
	#define Fence() do {} while (0)
#else
	#if defined( __x86_64__ )
		//#define Fence() __asm__ __volatile__ ( "mfence" ::: "memory" )
		#define Fence() __asm__ __volatile__ ( "lock; addq $0,128(%%rsp);" ::: "cc", "memory" )
	#elif defined( __i386__ )
		#define Fence() __asm__ __volatile__ ( "lock; addl $0,128(%%esp);" ::: "cc", "memory" )
	#elif defined( __aarch64__ ) || defined( __ARM_ARCH )
		#define Fence() __asm__ __volatile__ ( "dmb ish" ::: "memory" )
	#elif defined( __riscv )
		// RISC-V full memory barrier (read/write to read/write)
		#define Fence() __asm__ __volatile__ ( "fence rw, rw" ::: "memory" )
	#else
		#error unsupported architecture
	#endif
#endif // ATOMIC

// pause to prevent excess processor bus usage
#if defined( LPAUSE ) && defined( MPAUSE )
	#error Compilation options LPAUSE and MPAUSE cannot be used together.
#endif

#if defined( __i386__ ) || defined( __x86_64__ )

#if defined( LPAUSE )
	#define Pause() __asm__ __volatile__ ( "lfence" ::: "memory" )
#elif defined( MPAUSE )
	#if ! defined( __x86_64__ )
		#error MPAUSE (monitorx/mwaitx) is only supported on x86-64.
	#endif
	// Do not use VTYPE because -DATOMIC changes it.  Note, monitorx/mwaitx exist only on AMD processors.
	static inline TYPE MonitorLD( volatile TYPE * A ) {
		TYPE rv = 0;
		__asm__ __volatile__ (
			"xorq %%rcx,%%rcx; xorq %%rdx,%%rdx; monitorx; mov (%[RA]), %[RV];  "
			: [RV] "=r" (rv)
			: [RA] "a" (A)
			: "rcx", "rdx", "memory") ;
		return rv ;
	}
	static inline void MWait( int timo __attribute__(( unused )) ) { // newer mwait takes an argument
		__asm__ __volatile__ ( "xorq %%rcx,%%rcx; xorq %%rax,%%rax; mwaitx; " ::: "rax", "rcx", "memory" );
	}
	#define MPause( E, C ) { while ( (TYPEOF(E))MonitorLD( (volatile TYPE *)(&(E)) ) C ) { MWait( 0 ); } }
	#define MPauseS( S, E, C ) { while ( ( S (TYPEOF(E))MonitorLD( (volatile TYPE *)(&(E)) ) ) C ) { MWait( 0 ); } }
	#define Pause() __asm__ __volatile__ ( "pause" ::: "memory" )
#else
	#define Pause() __asm__ __volatile__ ( "pause" ::: "memory" )
#endif // LPAUSE

#elif defined( __ARM_ARCH ) || defined( __aarch64__ )

// Default spin pause is "isb sy", as in the original harness: on many ARM cores "yield" is effectively a no-op,
// whereas "isb" gives a real back-off delay.  -DARMYIELD selects "yield".  Results with different choices are not
// comparable.
#ifdef ARMYIELD
	#define ARM_SPIN_PAUSE "yield"
#else
	#define ARM_SPIN_PAUSE "isb sy"
#endif // ARMYIELD

#if defined( LPAUSE )
	#define Pause() __asm__ __volatile__ ( "dmb ish" ::: "memory" )
#elif defined( MPAUSE )
	#if ! defined( __aarch64__ )
		#error MPAUSE (wfe/ldaxr) is only supported on AArch64.
	#endif
	static inline TYPE MonitorLD( volatile TYPE * A ) {
		TYPE v = 0;
		// Polymorphic based on size of operand => automagically selects w or x register for %0.
		__asm__ __volatile__ ( "wfe; ldaxr %0,%1; " : "=r" (v) : "Q" (*A) : "memory" );
		return v;
	} // MonitorLD

	#define sevl() __asm__ __volatile__ ( "sevl" )
	// Polymorphic on operand using type erasure => use uintptr_t so both values and pointers work.
	#define MPause( E, C ) { sevl(); while ( (TYPEOF(E))MonitorLD( (volatile TYPE *)(&(E)) ) C ) {} }
	#define MPauseS( S, E, C ) { sevl(); while ( ( S (TYPEOF(E))MonitorLD( (volatile TYPE *)(&(E)) ) ) C ) {} }
	#define Pause() __asm__ __volatile__ ( "yield" ::: "memory" )
#else
	#define Pause() __asm__ __volatile__ ( ARM_SPIN_PAUSE ::: "memory" )
#endif // LPAUSE

#elif defined( __riscv )

#if defined( LPAUSE ) || defined( MPAUSE )
	#error LPAUSE/MPAUSE not implemented for RISC-V.
#else
	// Zihintpause "pause" is encoded as a FENCE hint (0x0100000F), so it executes as a harmless fence-with-empty-
	// successor-set on cores without the extension, and it assembles regardless of the -march string.
	#define Pause() __asm__ __volatile__ ( ".4byte 0x0100000F" ::: "memory" )
#endif

#else
	#error unsupported architecture
#endif

//------------------------------------------------------------------------------

// Atomic operations take the lock/change variable by name (not by address).
//
// gcc accepts the GNU __atomic_*_n builtins on _Atomic-qualified, volatile and plain objects. Clang rejects the
// builtins when the pointer is to an _Atomic-qualified object ("address argument to atomic operation must be a pointer
// to integer, pointer ..."), and its C11 <stdatomic.h> macros reject non-_Atomic objects. So for clang, ATOMICPTR
// strips the _Atomic qualifier from the pointer type (the value of a conditional expression is an unqualified rvalue
// type) and adds volatile, so one macro set works for _Atomic, volatile and plain objects. Both compilers then use the
// same __ATOMIC_* memory-order constants.  ATOMICSTRIP forces this path on gcc for testing.
//
// Note: these builtins do not work on C++ std::atomic objects (except Fai, which uses fetch_add in C++).

#if defined( __clang__ ) || defined( ATOMICSTRIP )
#define ATOMICPTR( x ) ( (volatile __typeof__( 0 ? (x) : (x) ) *)&(x) )
#else
#define ATOMICPTR( x ) ( &(x) )
#endif

#define Ldm( lock, memorder ) __atomic_load_n( ATOMICPTR( lock ), memorder )
#define Ld( lock ) Ldm( lock, __ATOMIC_ACQUIRE )
#define Strm( lock, value, memorder ) __atomic_store_n( ATOMICPTR( lock ), value, memorder )
#define Str( lock, value ) Strm( lock, value, __ATOMIC_RELEASE )
#define Clrm( lock, memorder ) __atomic_clear( ATOMICPTR( lock ), memorder )
#define Clr( lock ) Clrm( lock, __ATOMIC_RELEASE )
#define Tasm( lock, memorder ) __atomic_test_and_set( ATOMICPTR( lock ), memorder )
#define Tas( lock ) Tasm( lock, __ATOMIC_ACQUIRE )

// Value comparand: comp may be an rvalue; it is copied into a temporary, which is discarded.
#define Casm( change, comp, assn, smemorder, fmemorder ) \
	({ TYPEOF(comp) __temp = (comp); \
	   __atomic_compare_exchange_n( ATOMICPTR( change ), &__temp, (assn), false, smemorder, fmemorder ); })
#define Cas( change, comp, assn ) Casm( change, comp, assn, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST )
#define Caswm( change, comp, assn, smemorder, fmemorder ) \
	({ TYPEOF(comp) __temp = (comp); \
	   __atomic_compare_exchange_n( ATOMICPTR( change ), &__temp, (assn), true, smemorder, fmemorder ); })
#define Casw( change, comp, assn ) Caswm( change, comp, assn, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST )

// Variable comparand: on failure, comp is updated with the observed value.
#define Casvm( change, comp, assn, smemorder, fmemorder ) \
	__atomic_compare_exchange_n( ATOMICPTR( change ), &(comp), (assn), false, smemorder, fmemorder )
#define Casv( change, comp, assn ) Casvm( change, comp, assn, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST )
#define Casvwm( change, comp, assn, smemorder, fmemorder ) \
	__atomic_compare_exchange_n( ATOMICPTR( change ), &(comp), (assn), true, smemorder, fmemorder )
#define Casvw( change, comp, assn ) Casvwm( change, comp, assn, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST )

#define Fasm( change, assn, memorder ) __atomic_exchange_n( ATOMICPTR( change ), (assn), memorder )
#define Fas( change, assn ) Fasm( change, assn, __ATOMIC_SEQ_CST )
#define Faim( change, Inc, memorder ) __atomic_fetch_add( ATOMICPTR( change ), (Inc), memorder )
#ifndef __cplusplus
#define Fai( change, Inc ) Faim( change, Inc, __ATOMIC_SEQ_CST )
#else
#define Fai( change, Inc ) (change).fetch_add( Inc )	// std::atomic (stop, Arrived, sumOfThreadChecksums)
#endif // __cplusplus

#define await( E ) while ( ! (E) ) Pause()

//------------------------------------------------------------------------------

// Read a high-resolution timestamp. Returns 64 bits on all targets (a size_t is only 32 bits on rv32/armv7/i386).
static __attribute__(( unused )) inline uint64_t rdtscl( void ) {
	#if defined( __x86_64__ ) || defined( __i386__ )
	uint32_t lo, hi;
	__asm__ __volatile__ ( "rdtsc" : "=a"(lo), "=d"(hi) );
	return ( (uint64_t)hi << 32 ) | lo;

	#elif defined( __aarch64__ )
	uint64_t v;
	__asm__ __volatile__ ( "mrs %0, cntvct_el0" : "=r"(v) );
	return v;

	#elif defined( __riscv ) && __riscv_xlen == 64
	uint64_t v;
	// rdtime is the standard user-space timestamp counter for RISC-V
	__asm__ __volatile__ ( "rdtime %0" : "=r"(v) );
	return v;

	#elif defined( __riscv )							// rv32: read high/low consistently
	uint32_t hi, lo, tmp;
	__asm__ __volatile__ (
		"1: rdtimeh %0\n"
		"   rdtime  %1\n"
		"   rdtimeh %2\n"
		"   bne %0, %2, 1b"
		: "=&r"(hi), "=&r"(lo), "=&r"(tmp) );
	return ( (uint64_t)hi << 32 ) | lo;

	#else												// e.g., 32-bit ARM has no user-readable cycle counter
	struct timespec ts;
	clock_gettime( CLOCK_MONOTONIC, &ts );
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
	#endif
} // rdtscl

// Marsaglia shift-XOR PRNG with thread-local state
// Period is 4G-1
// 0 is absorbing and must be avoided (seed with "| 1")
// Low-order bits are not particularly random
// Each thread seeds its own state (see WorkerStart); the main thread seeds its own copy in main.
// Not reproducible run-to-run because of timestamp seeding and address space randomization
// Pipelined to allow OoO overlap with reduced dependencies
// Critically, return the current value, and compute and store the next value
// Optionally, sequester R on its own cache line to avoid false sharing
// but on linux __thread "initial-exec" TLS variables are already segregated.

static __thread RTYPE RandomState;
static __attribute__(( unused, noinline )) RTYPE PRNG( void ) { // must be called
	uint32_t ret = RandomState;
	RandomState ^= RandomState << 6;
	RandomState ^= RandomState >> 21;
	RandomState ^= RandomState << 7;
	return ret;
} // PRNG

static inline RTYPE seedPRNG( uintptr_t salt ) {		// never returns 0 (absorbing state)
	uint64_t s = rdtscl() ^ ( (uint64_t)salt * 0x9E3779B97F4A7C15ull ) ^ (uint64_t)(uintptr_t)&RandomState;
	return (RTYPE)( s ^ ( s >> 32 ) ) | 1;
} // seedPRNG

//------------------------------------------------------------------------------

static __attribute__(( unused )) inline TYPE cycleUp( TYPE v, TYPE n ) { return ( ((v) >= (n - 1)) ? 0 : (v + 1) ); }
static __attribute__(( unused )) inline TYPE cycleDown( TYPE v, TYPE n ) { return ( ((v) == 0) ? (n - 1) : (v - 1) ); }

#if defined( __GNUC__ )									// GNU gcc/clang compiler ?
// O(1) polymorphic integer log2, using clz, which returns the number of leading 0-bits, starting at the most
// significant bit (single instruction on x86). Operand must be 4 or 8 bytes.
#define Log2( n ) ( sizeof(n) * __CHAR_BIT__ - 1 - (			\
				  ( sizeof(n) == 4 ) ? __builtin_clz( n ) :		\
				  ( sizeof(n) == 8 ) ? __builtin_clzll( n ) :	\
				  -1 ) )
#else
static __attribute__(( unused )) int Log2( int n ) {	// fallback integer log2( n )
	return n > 1 ? 1 + Log2( n / 2 ) : n == 1 ? 0 : -1;
}
#endif // __GNUC__

static __attribute__(( unused )) inline int Clog2( int n ) { // integer ceil( log2( n ) )
	if ( n <= 0 ) { fprintf( stderr, "***ERROR*** Clog2 argument <= 0\n" ); abort(); }
	int ln = Log2( n );
	return ln + ( (n - (1 << ln)) != 0 );				// check for any 1 bits to the right of the most significant bit
}

//------------------------------------------------------------------------------

#ifdef CNT
struct CALIGN cnts {
	uint64_t cnts[CNT + 1];
};
static struct cnts ** counters CALIGN;
#endif // CNT

//------------------------------------------------------------------------------

// Do not use VTYPE because -DATOMIC changes it.
static _Atomic(TYPE) stop CALIGN = false;
static _Atomic(TYPE) Arrived CALIGN = 0;
static uintptr_t N CALIGN, Threads CALIGN, Time CALIGN;
static intptr_t Degree CALIGN = -1;

static int RUNS = 5;									// total experiment repetitions
static volatile int Run CALIGN = 0;						// current experiment repetition

//------------------------------------------------------------------------------

// Memory allocator to align storage.  The size is rounded up to a whole number of cache lines so the end of an
// allocation never shares a cache line with the next allocation (prevents false sharing).
static __attribute__(( unused )) inline void * Allocator( size_t size ) {
	void * p;
	size_t rsize = ( size + CACHE_ALIGN - 1 ) / CACHE_ALIGN * CACHE_ALIGN;
	if ( rsize == 0 ) rsize = CACHE_ALIGN;
	if ( posix_memalign( &p, CACHE_ALIGN, rsize ) != 0 ) {
		fprintf( stderr, "***ERROR*** Allocator failure for %zu bytes\n", size );
		abort();
	} // if
	return p;
} // Allocator

//------------------------------------------------------------------------------

#ifdef FAST
enum { MaxStartPoints = 64 };
static unsigned int NoStartPoints CALIGN;
static uint64_t * Startpoints CALIGN;

// To ensure the single thread exercises all aspects of an algorithm, it is assigned different start-points on each
// access to the critical section by randomly changing its thread id.  The randomness is accomplished using
// approximately 64 pseudo-random thread-ids, where 64 is divided by N to get R repetitions, e.g., for N = 5, R = 64 / 5
// = 12.  Each of the 12 repetition is filled with 5 random value in the range, 0..N-1, without replacement, e.g., 0 3 4
// 1 2.  There are no consecutive thread-ids within a repetition but there may be between repetition.  The thread cycles
// through this array of ids during an experiment.

static inline unsigned int startpoint( unsigned int pos ) {
	assert( pos < NoStartPoints );
	return Startpoints[pos];
//	return rand() % N;
} // startpoint
#endif // FAST

//------------------------------------------------------------------------------

#ifndef NCS_DELAY
#define NCS_DELAY 0
#endif // NCS_DELAY

#ifndef CS_DELAY
#define CS_DELAY 20
#endif // CS_DELAY

enum {
	NCSTimes = NCS_DELAY,								// time delay before attempting entry to CS
	CSTimes  = CS_DELAY,								// time spent in CS + random-number call
};

static TYPE HPAD1 CALIGN __attribute__(( unused ));		// protect further false sharing
static volatile RTYPE randomChecksum CALIGN = 0;
_Atomic(RTYPE) sumOfThreadChecksums CALIGN = 0;
// Do not use VTYPE because -DATOMIC changes it.
static volatile TYPE CurrTid CALIGN = 0;				// shared, current thread id in critical section
static TYPE HPAD2 CALIGN __attribute__(( unused ));		// protect further false sharing

#if NCS_DELAY != 0
	#define NCS_DECL
	// #define NCS if ( UNLIKELY( id == 0 && N > 1 ) ) NonCriticalSection( id )
	#define NCS if ( UNLIKELY( N > 1 ) ) NonCriticalSection()
	static inline void NonCriticalSection() {
		#ifdef RANDOM
		TYPE times = rdtscl() % NCSTimes;
		#else
		TYPE times = NCSTimes;
		#endif // RANDOM
		// Do not use VTYPE because -DATOMIC changes it.
		for ( volatile TYPE delay = 0; delay < times; delay += 1 ) {} // short fixed delay
	} // NonCriticalSection
#else
	#define NCS_DECL
	#define NCS
#endif // NCS_DELAY != 0

#ifdef CONVOY
static TYPE convoy[16][128][128];						// [RUNS][THREADS][THREADS]
#endif // CONVOY

#if CS_DELAY != 0
static inline RTYPE CS( const TYPE tid __attribute__(( unused )) ) { // parameter unused for CSTIME == 0
	#ifdef CONVOY
	if ( CurrTid < 128 ) convoy[Run][tid][CurrTid] += 1; // can be off by 1 for thread 0; CurrTid is reset between runs
	#endif // CONVOY

	CurrTid = tid;										// CurrTid is global

	// If the critical section is violated, the additions are corrupted because of the load/store race unless there is
	// perfect interleaving. Note, the load, delay, store to increase the chance of detecting a violation.
	RTYPE randomNumber = PRNG();						// belt and
	volatile RTYPE copy = randomChecksum;
	#ifdef RANDOM
	TYPE times = rdtscl() % CSTimes;
	#else
	TYPE times = CSTimes;
	#endif // RANDOM
	// Do not use VTYPE because -DATOMIC changes it.
	for ( volatile TYPE delay = 0; delay < times; delay += 1 ) {} // short delay
	randomChecksum = copy + randomNumber;

	// The assignment to CurrTid above can delay in a store buffer, that is, committed but not pushed into coherent
	// space. Hence, the load below fetchs the value from the store buffer via look aside instead of the coherent
	// version.  If this scenario occurs for multiple threads in the CS, these threads do not detect the violation
	// because their copy of CurrTid in the store buffer is unchanged.
	if ( CurrTid != tid ) {								// suspenders
		printf( "Interference Id:%zu\n", (size_t)tid );
		abort();
	} // if

	return randomNumber;
} // CS
#else
static inline RTYPE CS( const TYPE tid __attribute__(( unused )) ) { // parameter unused for CSTIME == 0
	#ifdef CONVOY
	if ( CurrTid < 128 ) convoy[Run][tid][CurrTid] += 1; // can be off by 1 for thread 0; CurrTid is reset between runs
	#endif // CONVOY

	CurrTid = tid;										// CurrTid is global
	return 0;
} // CS
#endif // CS_DELAY != 0

//------------------------------------------------------------------------------

#if defined( FAST ) || defined( NCS_DELAY )
static __attribute__(( unused )) void randPoints( uint64_t points[], unsigned int numPoints, unsigned int N ) {
	points[0] = N;
	for ( unsigned int i = 0; i < numPoints; i += N ) {
		for ( unsigned int j = i; j < i + N; j += 1 ) {
			unsigned int v;
		  L: v = rand() % N;
			size_t k;
			for ( k = i; k < j; k += 1 ) {
				if ( points[k] == v ) goto L;
			} // for
			// Unknown performance drop caused by following assert, use -DNDEBUG for experiments
			assert( k < numPoints );
			points[k] = v;
		} // for
	} // for
} // randPoints
#endif // FAST || NCS_DELAY

//------------------------------------------------------------------------------

// Vary concurrency level to help detect exclusion failure and progress-liveness bugs in lock algorithms and
// implementations.  In many cases lock bugs do not ever manifest in steady-state, so varying the concurrency level
// randomly every 10 msecs is usually able to perturb the system to "shake out" more lock bugs.
//
// All threads are explicitly and intentionally quiesced while changing concurrency levels to increase the frequency at
// which the lock shifts between contended and uncontended states.  Specifically, concurrency shifts from M to 0 to N
// instead of from M to N.
//
// We expect "Threads" to remain stable - Should be a stationary field.  When the barrier is operating BVO VStress != 0,
// Threads serves as a bound on concurrency.  The actual concurrency level at any given time will be in [1,Threads].
// Arrive() respects the Halt flag.

#ifdef STRESSINTERVAL
static int StressInterval CALIGN = STRESSINTERVAL;		// 500 is good
static volatile int BarHalt CALIGN = 0;

static int __attribute__((noinline)) PollBarrier() {
	if ( BarHalt == 0 ) return 0;
	// singleton barrier instance state
	static volatile int Ticket = 0;
	static volatile int Grant  = 0;
	static volatile int Gate   = 0;
	static volatile int nrun   = 0;
	static const int Verbose   = 1;

	static int ConcurrencyLevel = 0;

	// We have distinct non-overlapping arrival and draining/departure phases
	// Lead threads waits inside CS for quorum
	// Follower threads wait at entry to CS on ticket lock
	// Plain volatile ints: use the builtin directly (Fai is std::atomic-only in C++).
	int t = __atomic_fetch_add( ATOMICPTR( Ticket ), 1, __ATOMIC_SEQ_CST );
	while ( Ld( Grant ) != t ) Pause();					// acquire: pairs with release of Grant below

	if ( Gate == 0 ) {
		// Wait for full quorum
		while ( (uintptr_t)( Ld( Ticket ) - t ) != Threads ) Pause();
		// Compute new/next concurrency level - cohort
		if ( (rand() % 10) == 0 ) {
			Gate = 1;
		} else {
			Gate = (rand() % Threads) + 1;
		}
		ConcurrencyLevel = Gate;
		if ( Verbose ) printf ("L%d", Gate);
		BarHalt = 0;
		nrun = 0;
	} // if

	if ( Verbose ) {
		int k = __atomic_fetch_add( ATOMICPTR( nrun ), 1, __ATOMIC_SEQ_CST );
		if ( k == (ConcurrencyLevel-1) ) printf( "; " );
		if ( k >= ConcurrencyLevel ) printf( "?" );
	} // if

	Gate -= 1;
	// ST-ST ordering: the release store publishes Gate/nrun updates before the next ticket holder runs.
	// Release ticket lock
	Str( Grant, Grant + 1 );

	return 0;
} // PollBarrier
#endif // STRESSINTERVAL

//------------------------------------------------------------------------------

#if defined( __linux__ ) && defined( PIN )
static void setCPUmask( pthread_t pthreadid, int cpu ) { // -1 => turn off affinity
	cpu_set_t mask;
	CPU_ZERO( &mask );

	if ( cpu >= 0 ) {
		long ncpus = sysconf( _SC_NPROCESSORS_CONF );
		if ( ncpus > 0 && cpu >= ncpus ) {
			fprintf( stderr, "***ERROR*** affinity maps a thread to CPU %d but this machine has %ld CPUs; "
					 "reduce threads or define the host (e.g., -Dbpif3)\n", cpu, ncpus );
			abort();
		} // if
		CPU_SET( cpu, &mask );
	} else {
		memset( &mask, '\xff', sizeof(cpu_set_t) );		// must turn on all bits to reset
	} // if

	int rc = pthread_setaffinity_np( pthreadid, sizeof(cpu_set_t), &mask );
	if ( rc != 0 ) {
		errno = rc;
		char buf[64];
		snprintf( buf, 64, "***ERROR*** setaffinity failure for CPU %d", cpu );
		perror( buf );
		abort();
	} // if
} // setCPUmask
#endif // linux && PIN

static __attribute__(( unused )) void affinity( pthread_t pthreadid __attribute__(( unused )), unsigned int tid __attribute__(( unused )) ) {
// There are many ways to assign threads to processors: cores, chips, etc.
// On the AMD, we find starting at core 32 and sequential assignment is sufficient.
// Below are alternative approaches.

#if defined( __linux__ ) && defined( PIN )
#if defined( nasus )
	#if ! defined( HYPERAFF ) && ! defined( LINEARAFF )		// default affinity
	#define HYPERAFF
	#endif // HYPERAFF

	enum { OFFSETSOCK = 1 /* 0 origin */, SOCKETS = 2, CORES = 64, HYPER = 1 };
	#if defined( LINEARAFF )
	int cpu = tid + ((tid < CORES) ? OFFSETSOCK * CORES : HYPER < 2 ? OFFSETSOCK * CORES : CORES * SOCKETS);
	#endif // LINEARAFF
	#if defined( HYPERAFF )
	int cpu = OFFSETSOCK * CORES + (tid / 2) + ((tid % 2 == 0) ? 0 : CORES * SOCKETS);
	#endif // HYPERAFF
#elif defined( swift ) || defined( plg2 )
	#if ! defined( HYPERAFF ) && ! defined( LINEARAFF )		// default affinity
	#define HYPERAFF
	#endif // HYPERAFF

	enum { OFFSETSOCK = 1 /* 0 origin */, SOCKETS = 2, CORES = 128, HYPER = 1 };
	#if defined( LINEARAFF )
	int cpu = tid + ((tid < CORES) ? OFFSETSOCK * CORES : HYPER < 2 ? OFFSETSOCK * CORES : CORES * SOCKETS);
	#endif // LINEARAFF
	#if defined( HYPERAFF )
	int cpu = OFFSETSOCK * CORES + (tid / 2) + ((tid % 2 == 0) ? 0 : CORES * SOCKETS);
	#endif // HYPERAFF
#elif defined( pyke )
	#if ! defined( HYPERAFF ) && ! defined( LINEARAFF )		// default affinity
	#define HYPERAFF
	#endif // HYPERAFF

	enum { OFFSETSOCK = 0 /* 0 origin */, SOCKETS = 2, CORES = 24, HYPER = 1 /* wrap on socket */ };
	#if defined( LINEARAFF )
	int cpu = tid + ((tid < CORES) ? OFFSETSOCK * CORES : HYPER < 2 ? OFFSETSOCK * CORES : CORES * SOCKETS);
	#endif // LINEARAFF
	#if defined( HYPERAFF )
	int cpu = OFFSETSOCK * CORES + (tid / 2) + ((tid % 2 == 0) ? 0 : CORES * SOCKETS );
	#endif // HYPERAFF
#elif defined( java )
	#if ! defined( HYPERAFF ) && ! defined( LINEARAFF )		// default affinity
	#define HYPERAFF
	#endif // HYPERAFF

	enum { OFFSETSOCK = 0 /* 0 origin */, SOCKETS = 2, CORES = 32, HYPER = 1 /* wrap on socket */ };
	#if defined( LINEARAFF )
	int cpu = tid + ((tid < CORES) ? OFFSETSOCK * CORES : HYPER < 2 ? OFFSETSOCK * CORES : CORES * SOCKETS);
	#endif // LINEARAFF
	#if defined( HYPERAFF )
	int cpu = OFFSETSOCK * CORES + (tid / 2) + ((tid % 2 == 0) ? 0 : CORES * SOCKETS );
	#endif // HYPERAFF
#else
#if defined( HYPERAFF )
	#error HYPERAFF unsupported for this architecture.
#endif // HYPERAFF
#ifndef LINEARAFF
#define LINEARAFF
#endif // LINEARAFF
#if defined( algol )
	enum { OFFSETSOCK = 1 /* 0 origin */, SOCKETS = 2, CORES = 48, HYPER = 1 };
#elif defined( prolog )
	enum { OFFSETSOCK = 0 /* 0 origin */, SOCKETS = 2, CORES = 64, HYPER = 1 }; // pretend 2 sockets
#elif defined( jax )
	enum { OFFSETSOCK = 1 /* 0 origin */, SOCKETS = 4, CORES = 24, HYPER = 2 /* wrap on socket */ };
#elif defined( cfapi1 )
	enum { OFFSETSOCK = 0 /* 0 origin */, SOCKETS = 1, CORES = 4, HYPER = 1 };
#elif defined( bpif3 )										// Banana Pi BPI-F3 (SpacemiT K1), 8 cores, no SMT
	enum { OFFSETSOCK = 0 /* 0 origin */, SOCKETS = 1, CORES = 8, HYPER = 1 };
#else // default
	enum { OFFSETSOCK = 0 /* 0 origin */, SOCKETS = 2, CORES = 16, HYPER = 1 };
#endif // HOSTS
	int cpu = tid + ((tid < CORES) ? OFFSETSOCK * CORES : HYPER < 2 ? OFFSETSOCK * CORES : CORES * SOCKETS);
#endif // computer

#if 0
	// 4x8x2 : 4 sockets, 8 cores per socket, 2 hyperthreads per core
	int cpu = (tid & 0x30) | ((tid & 1) << 3) | ((tid & 0xE) >> 1) + 32;
#endif // 0

	setCPUmask( pthreadid, cpu );
#endif // linux && PIN
} // affinity

//------------------------------------------------------------------------------

static uint64_t ** entries CALIGN;						// holds CS entry results for each threads for all runs

#ifdef __cplusplus
#include xstr(Algorithm.cc)								// include algorithm for testing
#else
#include xstr(Algorithm.c)								// include algorithm for testing
#endif // __cplusplus

//------------------------------------------------------------------------------

// Thread start routine: seed this thread's PRNG state (thread-local, otherwise 0 = absorbing state) and then run the
// algorithm's Worker.  An algorithm that seeds RandomState itself simply overrides this seed.
static void * WorkerStart( void * arg ) {
	RandomState = seedPRNG( (uintptr_t)arg + 1 );
	return Worker( arg );
} // WorkerStart

//------------------------------------------------------------------------------

static __attribute__(( unused )) void shuffle( unsigned int set[], const int size ) {
	unsigned int p1, p2, temp;

	for ( size_t i = 0; i < 200; i += 1 ) {				// shuffle array S times
		p1 = rand() % size;
		p2 = rand() % size;
		temp = set[p1];
		set[p1] = set[p2];
		set[p2] = temp;
	} // for
} // shuffle

//------------------------------------------------------------------------------

#define median(a) ((RUNS & 1) == 0 ? (a[RUNS/2-1] + a[RUNS/2]) / 2 : a[RUNS/2] )
static int compare( const void * p1, const void * p2 ) {
	uint64_t i = *((const uint64_t *)p1);				// uint64_t, not size_t, to avoid truncation on 32-bit
	uint64_t j = *((const uint64_t *)p2);
	return i > j ? 1 : i < j ? -1 : 0;
} // compare

//------------------------------------------------------------------------------

static void statistics( size_t N, uint64_t values[/* C++ does handle N */], double * avg, double * std, double * rstd ) {
	double sum = 0.;
	for ( size_t r = 0; r < N; r += 1 ) {
		sum += values[r];
	} // for
	*avg = sum / N;										// average
	sum = 0.;
	for ( size_t r = 0; r < N; r += 1 ) {				// sum squared differences from average
		double diff = values[r] - *avg;
		sum += diff * diff;
	} // for
	*std = sqrt( sum / N );
	*rstd = *avg == 0.0 ? 0.0 : *std / *avg * 100;
} // statisitics

//------------------------------------------------------------------------------

int main( int argc, char * argv[] ) {
	N = 8;												// defaults
	Time = 10;											// seconds

	switch ( argc ) {
	  case 5:
		if ( strcmp( argv[4], "d" ) != 0 ) {			// default ?
			Degree = atoi( argv[4] );					// Zhang d-ary
			if ( Degree < 2 ) goto USAGE;
		} // if
	  case 4:
		if ( strcmp( argv[3], "d" ) != 0 ) {			// default ?
			RUNS = atoi( argv[3] );						// experiment repetitions
			if ( RUNS < 1 || (RUNS & 1) == 0 ) goto USAGE;
		} // if
	  case 3:
		if ( strcmp( argv[2], "d" ) != 0 ) {			// default ?
			Time = atoi( argv[2] );						// experiment duration
			if ( (intptr_t)Time < 1 ) goto USAGE;
		} // if
	  case 2:
		if ( strcmp( argv[1], "d" ) != 0 ) {			// default ?
			N = atoi( argv[1] );						// number of threads
			if ( (intptr_t)N < 1 ) goto USAGE;
		} // if
		break;
	  USAGE:
	  default:
		printf( "Usage: %s [ threads (> 0) | 'd' (default) %jd [ duration (> 0, seconds) | 'd' (default) %jd "
				"[ repetitions (> 0 & odd) | 'd' (default) %d [ Zhang D-ary (> 1) | 'd' (default) %jd ] ] ] ]\n",
				argv[0], (intmax_t)N, (intmax_t)Time, RUNS, (intmax_t)(Degree == -1 ? 0 : Degree) );
		exit( EXIT_FAILURE );
	} // switch

	RandomState = seedPRNG( 0 );						// seed main thread's PRNG, never 0 (absorbing state)

	#ifdef CFMT
	if ( N == 1 ) {										// title
		printf(
			"%s"
		#ifdef __cplusplus
			".cc"
		#else
			".c"
		#endif // __cplusplus
			, xstr(Algorithm)
		);
	} // if
	#endif // CFMT

	ctor();												// global algorithm constructor (may print)

	#ifdef CFMT
	#define QUOTE "'13"
	setlocale( LC_NUMERIC, "en_US.UTF-8" );

	if ( N == 1 ) {										// title
		printf(
			","
			#if defined( LINEARAFF )
			" LINEAR AFFINITY,"
			#endif // LINEARAFF
			#if defined( HYPERAFF )
			" HYPER AFFINITY,"
			#endif // HYPERAFF
			#ifdef FAST
			" FAST,"
			#endif // FAST
			#ifdef ATOMIC
			" ATOMIC,"
			#endif // ATOMIC
			#ifdef ATOMICINST
			" " xstr(ATOMICINST) ","
			#endif // ATOMICINST
			#ifdef THREADLOCAL
			" THREADLOCAL,"
			#endif // THREADLOCAL
			#if defined( NOEXPBACK ) && ! defined( MPAUSE )
			" NOEXPBACK,"
			#endif // NOEXPBACK
			#ifdef LPAUSE
			" LFENCE pause,"
			#endif // LPAUSE
			#ifdef MPAUSE
			" MONITOR pause,"
			#endif // MPAUSE
			#ifdef ARMYIELD
			" ARMYIELD,"
			#endif // ARMYIELD
			#ifdef FCFS
			" %s,"
			#endif // FCFS
			#ifdef FCFSTest
			" FCFSTest,"
			#endif // FCFSTest
			" %d NCS spins,"
			#ifdef RANDOM
			" RANDOM,"
			#endif // RANDOM
			#ifndef BARRIER
			" %d CS spins,"
			#endif // ! BARRIER
			" %d runs median",
			#ifdef FCFS
			xstr(FCFS),
			#endif // FCFS
			(int)NCSTimes,
			#ifndef BARRIER
			(int)CSTimes,
			#endif // ! BARRIER
			RUNS
		);
		if ( Degree != -1 ) printf( " %jd-ary", (intmax_t)Degree ); // Zhang only
		#ifndef BARRIER
		printf(
			"\n  N   T    CS Entries           AVG           STD   RSTD"
			#ifdef CNT
			"   CAVG"
			#endif // CNT
			"\n"
		);
		#else
		printf( "\n  N   T  Barrier Entries\n" );
		#endif // ! BARRIER
	} // if
	#else
	#define QUOTE ""
	#endif // CFMT

	printf( "%3ju %3jd ", (uintmax_t)N, (intmax_t)Time );

	#ifdef FAST
	assert( N <= MaxStartPoints );
	Threads = 1;										// fast test, Threads=1, N=1..32
	NoStartPoints = MaxStartPoints / N * N;				// floor( MaxStartPoints / N )
	Startpoints = (uint64_t *)Allocator( sizeof(TYPEOF(Startpoints[0])) * NoStartPoints );
	randPoints( Startpoints, NoStartPoints, N );
	#else
	Threads = N;										// allow testing of T < N
	#endif // FAST

	entries = (TYPEOF(entries[0]) *)malloc( sizeof(TYPEOF(entries[0])) * RUNS );
	#ifdef CNT
	counters = (TYPEOF(counters[0]) *)malloc( sizeof(TYPEOF(counters[0])) * RUNS );
	#endif // CNT
	for ( TYPEOF(RUNS) r = 0; r < RUNS; r += 1 ) {
		entries[r] = (TYPEOF(entries[0][0]) *)Allocator( sizeof(TYPEOF(entries[0][0])) * Threads );
#ifdef CNT
#ifdef FAST
		counters[r] = (TYPEOF(counters[0][0]) *)Allocator( sizeof(TYPEOF(counters[0][0])) * N );
#else
		counters[r] = (TYPEOF(counters[0][0]) *)Allocator( sizeof(TYPEOF(counters[0][0])) * Threads );
#endif // FAST
#endif // CNT
	} // for

#ifdef CNT
	// For FAST experiments, there is only thread but it changes its thread id to visit all the start points. Therefore,
	// all the counters for each id must be initialized and summed at the end.
	for ( TYPEOF(RUNS) r = 0; r < RUNS; r += 1 ) {
		for ( size_t id = 0; id < N; id += 1 ) {
			for ( size_t i = 0; i < CNT + 1; i += 1 ) { // reset for each run
				counters[r][id].cnts[i] = 0;
			} // for
		} // for
	} // for
#endif // CNT

	#if defined( CONVOY )
	assert( RUNS <= 16 && N <= 128 );
	for ( int r = 0; r < RUNS; r += 1 ) {
		for ( size_t tid1 = 0; tid1 < N; tid1 += 1 ) {
			for ( size_t tid2 = 0; tid2 < N; tid2 += 1 ) {
				convoy[r][tid1][tid2] = 0;
			} // for
		} // for
	} // for
	#endif // CONVOY

	unsigned int set[Threads];
	for ( size_t i = 0; i < Threads; i += 1 ) set[ i ] = i;
	// srand( getpid() );
	// shuffle( set, Threads );							// randomize thread ids
	#ifdef STATS
	fprintf( stderr, "\nthread set: " );
	for ( size_t i = 0; i < Threads; i += 1 ) fprintf( stderr, "%u ", set[ i ] );
	fprintf( stderr, "\n" );
	#endif // STATS

	pthread_t workers[Threads];

	#if defined( __linux__ ) && defined( PIN ) && ! defined( NOAFFINITY )
	affinity( pthread_self(), 0 );						// temporary pin to target CPU while creating workers
	#endif // linux && PIN && ! NOAFFINITY
	for ( size_t tid = 0; tid < Threads; tid += 1 ) {	// start workers
		int rc = pthread_create( &workers[tid], NULL, WorkerStart, (void *)(uintptr_t)set[tid] );
		if ( rc != 0 ) {
			errno = rc;
			perror( "***ERROR*** pthread create" );
			abort();
		} // if
		#ifndef NOAFFINITY
		affinity( workers[tid], tid );
		#endif // ! NOAFFINITY
	} // for
	#if defined( __linux__ ) && defined( PIN ) && ! defined( NOAFFINITY )
	setCPUmask( pthread_self(), -1 );					// unpin main thread
	#endif // linux && PIN && ! NOAFFINITY

	for ( ; Run < RUNS;  ) {							// global variable
		// threads start first experiment immediately
		sleep( Time );									// delay for experiment duration
		stop = true;									// stop threads (atomic)
		while ( Arrived != Threads ) Pause();			// all threads stopped ?

		if ( randomChecksum != sumOfThreadChecksums ) {
			printf( "Interference run %d randomChecksum %u sumOfThreadChecksums %u\n", Run,
					(unsigned int)randomChecksum, (unsigned int)sumOfThreadChecksums );
			abort();
		} // if
		randomChecksum = sumOfThreadChecksums = 0;
		CurrTid = SIZE_MAX;								// reset for next run (matches TYPE = size_t)

		Run += 1;
		stop = false;									// start threads (atomic)
		while ( Arrived != 0 ) Pause();					// all threads started ?
	} // for

	for ( size_t tid = 0; tid < Threads; tid += 1 ) {	// terminate workers
		int rc = pthread_join( workers[tid], NULL );
		if ( rc != 0 ) {
			errno = rc;
			perror( "***ERROR*** pthread join" );
			abort();
		} // if
	} // for

	dtor();												// global algorithm destructor

	double avg = 0.0, std = 0.0, rstd;

	#ifdef STATS
	fprintf( stderr, "\nthreads:\n" );
	#endif // STATS
	for ( TYPEOF(RUNS) r = 0; r < RUNS; r += 1 ) {
		#ifdef STATS
		for ( size_t tid = 0; tid < Threads; tid += 1 ) {
			fprintf( stderr, "%" QUOTE "ju ", (uintmax_t)entries[r][tid] );
		} // for
		#endif // STATS
		statistics( Threads, entries[r], &avg, &std, &rstd );
		#ifdef STATS
		fprintf( stderr, ": avg %'1.f  std %'1.f  rstd %'2.f%%\n", avg, std, rstd );
		#endif // STATS
	} // for

	uint64_t totalCols[RUNS];

	#ifdef STATS
	fprintf( stderr, "\nruns:\n" );
	for ( size_t tid = 0; tid < Threads; tid += 1 ) {
		for ( TYPEOF(RUNS) r = 0; r < RUNS; r += 1 ) {
			totalCols[r] = entries[r][tid];				// must copy, row major order
			fprintf( stderr, "%" QUOTE "ju ", (uintmax_t)entries[r][tid] );
		} // for
		statistics( RUNS, totalCols, &avg, &std, &rstd );
		fprintf( stderr, ": avg %'1.f  std %'1.f  rstd %'2.f%%\n", avg, std, rstd );
	} // for
	#endif // STATS

	uint64_t sort[RUNS];

	for ( TYPEOF(RUNS) r = 0; r < RUNS; r += 1 ) {
		totalCols[r] = 0;
		for ( size_t tid = 0; tid < Threads; tid += 1 ) {
			totalCols[r] += entries[r][tid];
		} // for
		sort[r] = totalCols[r];
	} // for
	statistics( RUNS, totalCols, &avg, &std, &rstd );
	const double percent = 10.0;
	if ( rstd > percent ) printf( "Warning relative standard deviation %.1f%% greater than %.0f%% over %d runs.\n", rstd, percent, RUNS );

	qsort( sort, RUNS, sizeof(TYPEOF(sort[0])), compare );
	uint64_t med = median( sort );
	TYPEOF(RUNS) posn;									// run with median result
	for ( posn = 0; posn < RUNS && totalCols[posn] != med; posn += 1 ); // assumes RUNS is odd

	#ifdef STATS
	fprintf( stderr, "\ntotals: " );
	for ( TYPEOF(RUNS) i = 0; i < RUNS; i += 1 ) {		// print values
		fprintf( stderr, "%" QUOTE "ju ", (uintmax_t)totalCols[i] );
	} // for
	fprintf( stderr, "\nsorted: " );
	for ( TYPEOF(RUNS) i = 0; i < RUNS; i += 1 ) {		// print values
		fprintf( stderr, "%" QUOTE "ju ", (uintmax_t)sort[i] );
	} // for
	fprintf( stderr, "\nmedian posn:%d\n\n", posn );
	#endif // STATS

	printf( "%" QUOTE "ju", (uintmax_t)med );			// median round
	#ifndef BARRIER
	statistics( Threads, entries[posn], &avg, &std, &rstd ); // median thread
	printf( " %" QUOTE ".1f %" QUOTE ".1f %5.1f%%", avg, std, rstd );
	#endif // ! BARRIER

	#ifdef CNT
	// posn is the run containing the median result. Other runs are ignored.
	uint64_t cntsum;
	for ( size_t i = 0; i < CNT + 1; i += 1 ) {
		cntsum = 0;
		#ifdef FAST
		for ( size_t tid = 0; tid < N; tid += 1 ) {
		#else
		for ( size_t tid = 0; tid < Threads; tid += 1 ) {
		#endif // FAST
			cntsum += counters[posn][tid].cnts[i];
		} // for
		printf( " %5.1f%%", (double)cntsum / (double)totalCols[posn] * 100.0 );
	} // for
	#endif // CNT

	#ifdef CONVOY
	printf( "\n\n" );
	for ( int r = 0; r < RUNS; r += 1 ) {
		for ( size_t tid1 = 0; tid1 < N; tid1 += 1 ) {
			for ( size_t tid2 = 0; tid2 < N; tid2 += 1 ) {
				#ifdef CFMT
				printf( "%" QUOTE "ju ", (uintmax_t)convoy[r][tid1][tid2] ); // posn for median
				#else
				printf( "%13ju ", (uintmax_t)convoy[r][tid1][tid2] ); // posn for median
				#endif // CFMT
			} // for
			printf( "\n" );
		} // for
		printf( "\n" );
	} // for
	#endif // CONVOY

	for ( int r = 0; r < RUNS; r += 1 ) {
		free( entries[r] );
		#ifdef CNT
		free( counters[r] );
		#endif // CNT
	} // for
	#ifdef CNT
	free( counters );
	#endif // CNT

	free( entries );

	#ifdef FAST
	free( Startpoints );
	#endif // FAST

	printf( "\n" );
	return EXIT_SUCCESS;
} // main

// Local Variables: //
// tab-width: 4 //
// compile-command: "gcc -Wall -Wextra -std=gnu11 -O3 -DNDEBUG -DPIN -Dbpif3 -DAlgorithm=HemLock Harness.c -lpthread -lm" //
// End: //
