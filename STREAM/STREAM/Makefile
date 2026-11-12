CC ?= gcc
CFLAGS ?= -O3 -fopenmp

FC ?= gfortran
FFLAGS ?= -O3 -fopenmp

CUSTOM_PREPROCESSOR_VARS ?= -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10
CUSTOM_PREPROCESSOR_VARS_NUMA ?= 

NONUMA ?= 0

all: stream_f stream_c stream_numa

stream_f: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FC) $(FFLAGS) -c stream.f
	$(FC) $(FFLAGS) stream.o mysecond.o -o stream_f

stream_c: stream.c
	$(CC) $(CFLAGS) $(CUSTOM_PREPROCESSOR_VARS) stream.c -o $@

stream_numa:
ifeq ($(NONUMA),0)
	$(CC) $(CFLAGS) $(CUSTOM_PREPROCESSOR_VARS_NUMA) stream_numa.c -lnuma -lm -o $@
else
	$(CC) $(CFLAGS) $(CUSTOM_PREPROCESSOR_VARS_NUMA) -DNO_LIBNUMA stream_numa.c -lm -o $@
endif


clean:
	rm -f stream_f stream_c stream_c stream_numa *.o