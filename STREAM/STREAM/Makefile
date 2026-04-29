CC ?= gcc
CFLAGS ?= -O3 -fopenmp

FC ?= gfortran
FFLAGS ?= -O3 -fopenmp

CUSTOM_PREPROCESSOR_VARS =? -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=10

all: stream_f stream_c

stream_f: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FC) $(FFLAGS) -c stream.f
	$(FC) $(FFLAGS) stream.o mysecond.o -o stream_f

stream_c: stream.c
	$(CC) $(CFLAGS) stream.c -o stream_c

stream_c_custom: stream.c
	$(CC) $(CFLAGS) $(CUSTOM_PREPROCESSOR_VARS) stream.c -o $@

clean:
	rm -f stream_f stream_c *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -ffreestanding -qopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc
