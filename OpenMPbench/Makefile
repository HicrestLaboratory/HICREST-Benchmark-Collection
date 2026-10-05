include Makefile.defs

# Default directory for outputs if BIN_DIR is not specified
BIN_DIR ?= .

SYNCOBJS  = $(BIN_DIR)/syncbench.o $(BIN_DIR)/common.o
SCHEDOBJS = $(BIN_DIR)/schedbench.o $(BIN_DIR)/common.o
ARRAYOBJS = $(BIN_DIR)/arraybench_$(IDA).o $(BIN_DIR)/common.o
TASKOBJS  = $(BIN_DIR)/taskbench.o $(BIN_DIR)/common.o

.PHONY: all prog clean

all: $(BIN_DIR)/syncbench $(BIN_DIR)/schedbench $(BIN_DIR)/taskbench
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=1 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=3 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=9 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=27 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=81 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=243 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=729 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=2187 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=6561 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=19683 prog
	$(MAKE) BIN_DIR=$(BIN_DIR) IDA=59049 prog

prog: $(BIN_DIR)/arraybench_$(IDA)

# Create the output directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Pattern rule for object files (adds -I. to find common.h in current dir)
$(BIN_DIR)/%.o: %.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I. -I$(BIN_DIR) -c $< -o $@

# Multiple header files due to multiple array sizes
$(BIN_DIR)/arraybench_$(IDA).h: arraybench.h | $(BIN_DIR)
	$(CC) -DIDA=$(IDA) $(CPPFLAGS) -I. arraybench.h -o $@

# Multiple object files due to multiple array sizes
$(BIN_DIR)/arraybench_$(IDA).o: $(BIN_DIR)/arraybench_$(IDA).h arraybench.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I. -I$(BIN_DIR) -DIDA=$(IDA) arraybench.c -c -o $@

# Executables
$(BIN_DIR)/arraybench_$(IDA): $(ARRAYOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $(ARRAYOBJS) $(LIBS) -o $@

$(BIN_DIR)/syncbench: $(SYNCOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $(SYNCOBJS) $(LIBS) -o $@

$(BIN_DIR)/schedbench: $(SCHEDOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $(SCHEDOBJS) $(LIBS) -o $@

$(BIN_DIR)/taskbench: $(TASKOBJS) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $(TASKOBJS) $(LIBS) -o $@

clean:
	-rm -f $(BIN_DIR)/syncbench $(BIN_DIR)/schedbench $(BIN_DIR)/taskbench \
	       $(BIN_DIR)/arraybench_* $(BIN_DIR)/*.o $(BIN_DIR)/*.h