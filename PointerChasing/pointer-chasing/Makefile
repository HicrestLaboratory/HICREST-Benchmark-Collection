CPPSources := $(wildcard *.cpp)
Objects := $(patsubst %.cpp,%.o,$(CPPSources))

BINDIR ?= bin

fused-linear-chase-objects := fused-linear-chase.o chase-pointers.o linear-chain.o
fused-random-chase-objects := fused-random-chase.o chase-pointers.o
linear-chase-objects := linear-chase.o chase-pointers.o linear-chain.o
random-chase-objects := random-chase.o chase-pointers.o

CXX ?= g++
CPPFLAGS ?= -std=gnu++11 -Ifmt
EXTRA_FLAGS ?=

Targets := fused-linear-chase fused-random-chase linear-chase random-chase
BinTargets := $(addprefix $(BINDIR)/,$(Targets))

.PHONY: all clean realclean depend

all: $(Objects) $(BinTargets)

clean:
	rm -f $(Objects)

realclean: clean
	rm -f $(BinTargets)
	rmdir --ignore-fail-on-non-empty $(BINDIR)

$(BINDIR):
	mkdir -p $@

$(BINDIR)/fused-linear-chase: $(fused-linear-chase-objects) | $(BINDIR)
	$(CXX) $(CPPFLAGS) $(EXTRA_FLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/fused-random-chase: $(fused-random-chase-objects) | $(BINDIR)
	$(CXX) $(CPPFLAGS) $(EXTRA_FLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/linear-chase: $(linear-chase-objects) | $(BINDIR)
	$(CXX) $(CPPFLAGS) $(EXTRA_FLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/random-chase: $(random-chase-objects) | $(BINDIR)
	$(CXX) $(CPPFLAGS) $(EXTRA_FLAGS) $(LDFLAGS) -o $@ $^


random-chase.o: random-chase.cpp fmt/printf.hpp chase-pointers.hpp uniform-int-distribution.hpp
chase-pointers.o: chase-pointers.cpp fmt/printf.hpp chase-pointers.hpp walltime.hpp
fused-linear-chase.o: fused-linear-chase.cpp fmt/printf.hpp linear-chain.hpp walltime.hpp
fused-random-chase.o: fused-random-chase.cpp fmt/printf.hpp chase-pointers.hpp walltime.hpp
linear-chase.o: linear-chase.cpp fmt/printf.hpp chase-pointers.hpp linear-chain.hpp
linear-chain.o: linear-chain.cpp linear-chain.hpp