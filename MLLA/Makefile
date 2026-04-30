# mlla Makefile
# ─────────────────────────────────────────────────────────────────
# Usage:
#   make                             # OpenBLAS, float32, OpenMP on
#   make BACKEND=mkl                 # Intel MKL
#   make BACKEND=eigen               # Eigen (header-only)
#   make DTYPE=double                # float64
#   make OMP=0                       # disable OpenMP (serial baseline)
#   make BACKEND=mkl DTYPE=double OMP=0
#   make run                         # build + run
#   make clean
# ─────────────────────────────────────────────────────────────────

BACKEND ?= openblas
DTYPE   ?= float
OMP     ?= 1

CXX      ?= g++
CXXFLAGS  = -O3 -march=native -std=c++17 -Wall -Wextra

BIN = mlla

# ── Data type ─────────────────────────────────────────────────────
ifeq ($(DTYPE), double)
    CXXFLAGS += -DUSE_DOUBLE
endif

# ── OpenMP ────────────────────────────────────────────────────────
ifeq ($(OMP), 1)
    CXXFLAGS += -fopenmp
    OMPLIB    = -fopenmp
else
    OMPLIB    =
endif

# ── Backend ───────────────────────────────────────────────────────
ifeq ($(BACKEND), openblas)
    CXXFLAGS += -DUSE_OPENBLAS
    # Adjust prefix if OpenBLAS is not in the default search path
    LIBS      = -lopenblas -lpthread -lm $(OMPLIB)

else ifeq ($(BACKEND), mkl)
    CXXFLAGS += -DUSE_MKL
    MKL_ROOT  ?= /opt/intel/oneapi/mkl/latest
    CXXFLAGS  += -I$(MKL_ROOT)/include
    # Use MKL's own OpenMP runtime to avoid conflicts with GOMP
    ifeq ($(OMP), 1)
        CXXFLAGS := $(filter-out -fopenmp,$(CXXFLAGS))
        CXXFLAGS += -fopenmp=libiomp5
        OMPLIB    = -fopenmp=libiomp5
    endif
    LIBS       = -L$(MKL_ROOT)/lib/intel64 \
                 -lmkl_rt -lpthread -ldl -lm $(OMPLIB)

else ifeq ($(BACKEND), eigen)
    CXXFLAGS += -DUSE_EIGEN
    EIGEN_ROOT ?= /usr/include/eigen3
    CXXFLAGS   += -I$(EIGEN_ROOT)
    LIBS        = -lm $(OMPLIB)

else
    $(error Unknown BACKEND=$(BACKEND). Choose: openblas | mkl | eigen)
endif

# ── Rules ─────────────────────────────────────────────────────────
.PHONY: all run clean

all: $(BIN)

$(BIN): mlla.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

run: all
	./$(BIN)

clean:
	rm -f $(BIN)