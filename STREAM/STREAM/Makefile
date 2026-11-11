CC ?= gcc
CFLAGS ?= -O3 -fopenmp

FC ?= gfortran
FFLAGS ?= -O3 -fopenmp

CUSTOM_PREPROCESSOR_VARS ?= -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10
CUSTOM_PREPROCESSOR_VARS_NUMA ?= 

all: stream_f stream_c stream_numa

stream_f: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FC) $(FFLAGS) -c stream.f
	$(FC) $(FFLAGS) stream.o mysecond.o -o stream_f

stream_c: stream.c
	$(CC) $(CFLAGS) $(CUSTOM_PREPROCESSOR_VARS) stream.c -o $@

stream_numa:
	$(CC) $(CFLAGS) $(CUSTOM_PREPROCESSOR_VARS_NUMA) stream_numa.c -lnuma -lm -o $@

clean:
	rm -f stream_f stream_c stream_c stream_numa *.o