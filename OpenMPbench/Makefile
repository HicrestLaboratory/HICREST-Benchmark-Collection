ifeq ($(strip $(CC)),)
    CC = cc
endif

CFLAGS   ?= -fopenmp 
LDFLAGS  ?= -fopenmp  
CPPFLAGS ?=
LIBS     ?= -lm 

BIN_DIR ?= .

IDAS                = 1 3 9 27 81 243 729 2187 6561 19683 59049
ARRAY_BINS          = $(patsubst %,$(BIN_DIR)/arraybench_%,$(IDAS))
ARRAY_ALIASES       = $(patsubst %,arraybench_%,$(IDAS))
CLEAN_ARRAY_ALIASES = $(patsubst %,clean-arraybench_%,$(IDAS))

SYNCOBJS  = $(BIN_DIR)/syncbench.o $(BIN_DIR)/common.o
SCHEDOBJS = $(BIN_DIR)/schedbench.o $(BIN_DIR)/common.o
TASKOBJS  = $(BIN_DIR)/taskbench.o $(BIN_DIR)/common.o

.PHONY: all clean clean-syncbench clean-schedbench clean-taskbench clean-arraybench \
        syncbench schedbench taskbench arraybench $(ARRAY_ALIASES) $(CLEAN_ARRAY_ALIASES)

all: $(BIN_DIR)/syncbench $(BIN_DIR)/schedbench $(BIN_DIR)/taskbench $(ARRAY_BINS)

# Build aliases
syncbench: $(BIN_DIR)/syncbench
schedbench: $(BIN_DIR)/schedbench
taskbench: $(BIN_DIR)/taskbench
arraybench: $(ARRAY_BINS)

# Static pattern rule for arraybench IDA aliases (e.g., `make arraybench_81`)
$(ARRAY_ALIASES): arraybench_%: $(BIN_DIR)/arraybench_%

# Output directory creation
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Pattern rule for general object files
$(BIN_DIR)/%.o: %.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -I$(BIN_DIR) -c $< -o $@

# Pattern rule for arraybench objects using Make stem ($*) to define IDA
$(BIN_DIR)/arraybench_%.o: arraybench.c arraybench.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -I$(BIN_DIR) -DIDA=$* -c $< -o $@

# Executable targets
$(BIN_DIR)/arraybench_%: $(BIN_DIR)/arraybench_%.o $(BIN_DIR)/common.o | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LIBS) -o $@

$(BIN_DIR)/syncbench: $(SYNCOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LIBS) -o $@

$(BIN_DIR)/schedbench: $(SCHEDOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LIBS) -o $@

$(BIN_DIR)/taskbench: $(TASKOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LIBS) -o $@

# Granular clean targets
clean-syncbench:
	rm -f $(BIN_DIR)/syncbench $(BIN_DIR)/syncbench.o $(BIN_DIR)/common.o

clean-schedbench:
	rm -f $(BIN_DIR)/schedbench $(BIN_DIR)/schedbench.o $(BIN_DIR)/common.o

clean-taskbench:
	rm -f $(BIN_DIR)/taskbench $(BIN_DIR)/taskbench.o $(BIN_DIR)/common.o

clean-arraybench:
	rm -f $(ARRAY_BINS) $(BIN_DIR)/arraybench_*.o $(BIN_DIR)/common.o

# Clean specific IDA target (e.g., `make clean-arraybench_81`)
$(CLEAN_ARRAY_ALIASES): clean-arraybench_%:
	rm -f $(BIN_DIR)/arraybench_$* $(BIN_DIR)/arraybench_$*.o $(BIN_DIR)/common.o

# Full clean
clean:
	rm -f $(BIN_DIR)/syncbench $(BIN_DIR)/schedbench $(BIN_DIR)/taskbench \
	      $(BIN_DIR)/arraybench_* $(BIN_DIR)/*.o