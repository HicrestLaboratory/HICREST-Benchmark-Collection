# Popcorn K-means Build System - Pure Makefile
# Replaces CMake entirely with explicit compilation rules
# Supports multiple systems, compilers, and variants
# 
# Invoke via: make [SYSTEM=...] [VARIANT=...] [CXX=...] etc.

# ============================================================================
# CONFIGURATION & DEFAULTS
# ============================================================================

# Build metadata
BUILD_DIR         ?= build
SYSTEM            ?= unknown
VARIANT           ?= default
COMPILER_NAME     ?= gcc
SOURCE_DIR        ?= $(PWD)

# Compiler settings
CXX               ?= g++
CC                ?= gcc
CXXFLAGS          ?= -O3 -std=c++17 -Wall -Wextra
CFLAGS            ?= -O3 -std=c99 -Wall -Wextra
LDFLAGS           ?=

# BLAS configuration
BLAS_VENDOR       ?=
BLAS_LIBS         ?=
BLAS_INCLUDE_DIRS ?=
BLAS_DIR          ?=

# Additional flags
EXTRA_LIBS        ?=
EXTRA_LINK_OPTIONS ?=

# Project structure
SRCDIR            := $(SOURCE_DIR)/src
INCDIR            := $(SOURCE_DIR)/include
OBJDIR            := $(BUILD_DIR)/obj
DEPDIR            := $(BUILD_DIR)/deps
BINDIR            := $(BUILD_DIR)/bin
LOGDIR            := $(BUILD_DIR)/logs

# Source files
SOURCES           := $(SRCDIR)/kmeans.cpp $(SRCDIR)/main.cpp
OBJECTS           := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))
DEPS              := $(patsubst $(SRCDIR)/%.cpp,$(DEPDIR)/%.d,$(SOURCES))

# Targets
TARGETS           := popcornkmeans_blas popcornkmeans_openmp
TARGET_PATHS      := $(addprefix $(BINDIR)/,$(TARGETS))

# Verbose output (set V=1 for detailed build info)
V ?= 0
ifeq ($(V),1)
  Q =
  ECHO = @true
else
  Q = @
  ECHO = @echo
endif

# ============================================================================
# BLAS VENDOR MAPPING
# ============================================================================

ifeq ($(BLAS_VENDOR),NVPL)
  BLA_VENDOR_INT := 1
else ifeq ($(BLAS_VENDOR),Intel10_64lp)
  BLA_VENDOR_INT := 2
else ifeq ($(BLAS_VENDOR),OpenBLAS)
  BLA_VENDOR_INT := 0
else
  BLA_VENDOR_INT := 0
endif

# ============================================================================
# PHONY TARGETS
# ============================================================================

.PHONY: all clean info help ccutils-check

# ============================================================================
# MAIN TARGETS
# ============================================================================

all: info ccutils-check $(TARGET_PATHS)
	$(ECHO) "✓ All builds completed successfully"

help:
	@echo "Popcorn K-means Build System (Makefile-based)"
	@echo ""
	@echo "Usage:"
	@echo "  make -f Makefile.build [options]"
	@echo ""
	@echo "Common options:"
	@echo "  CXX=<compiler>        C++ compiler (default: g++)"
	@echo "  CC=<compiler>         C compiler (default: gcc)"
	@echo "  SYSTEM=<system>       Target system (default: unknown)"
	@echo "  VARIANT=<variant>     Build variant (default: default)"
	@echo "  BLAS_VENDOR=<vendor>  BLAS vendor (NVPL/Intel10_64lp/OpenBLAS)"
	@echo "  BLAS_LIBS=<libs>      BLAS libraries (semicolon-separated)"
	@echo "  EXTRA_LIBS=<libs>     Additional libraries to link"
	@echo "  V=1                   Verbose output"
	@echo ""
	@echo "Examples:"
	@echo "  make -f Makefile.build CXX=icpc CC=icc BLAS_VENDOR=Intel10_64lp"
	@echo "  make -f Makefile.build CXX=nvc++ CC=nvc BLAS_VENDOR=NVPL V=1"

info:
	$(ECHO) "======================================================================"
	$(ECHO) "Build Configuration"
	$(ECHO) "======================================================================"
	$(ECHO) "System:              $(SYSTEM)"
	$(ECHO) "Variant:             $(VARIANT)"
	$(ECHO) "Compiler:            $(COMPILER_NAME)"
	$(ECHO) "CXX:                 $(CXX)"
	$(ECHO) "CC:                  $(CC)"
	$(ECHO) "CXXFLAGS:            $(CXXFLAGS)"
	$(ECHO) "BLAS Vendor:         $(BLAS_VENDOR)"
	$(ECHO) "BLAS Libraries:      $(BLAS_LIBS)"
	$(ECHO) "BLAS Include Dirs:   $(BLAS_INCLUDE_DIRS)"
	$(ECHO) "Build Directory:     $(BUILD_DIR)"
	$(ECHO) "Source Directory:    $(SOURCE_DIR)"
	$(ECHO) "======================================================================"

clean:
	$(ECHO) "Cleaning build directory: $(BUILD_DIR)"
	$(Q) rm -rf $(BUILD_DIR)

# ============================================================================
# CCUTILS VALIDATION
# ============================================================================

CCUTILS_SEARCH_PATHS := \
  ~/.local/ccutils/install/lib/cmake/ccutils/ \
  /usr/local/ccutils/lib/cmake/ccutils/ \
  /opt/ccutils/lib/cmake/ccutils/ \
  $(CCUTILS_DIR)

ccutils-check:
	@for path in $(CCUTILS_SEARCH_PATHS); do \
		if [ -d "$$path" ] && [ -f "$$path/ccutilsConfig.cmake" ]; then \
			echo "✓ Found ccutils at: $$path"; \
			exit 0; \
		fi; \
	done; \
	echo "⚠ ccutils not found in standard paths"; \
	echo "  Set CCUTILS_DIR to ccutils installation lib/cmake path"

# ============================================================================
# COMPILE RULES
# ============================================================================

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR) $(DEPDIR)
	$(ECHO) "Compiling: $<"
	$(Q) $(CXX) \
		$(CXXFLAGS) \
		-I$(INCDIR) \
		$(if $(BLAS_INCLUDE_DIRS),-I$(BLAS_INCLUDE_DIRS)) \
		-DBLA_VENDOR=$(BLA_VENDOR_INT) \
		-MMD -MP -MF $(DEPDIR)/$*.d \
		-c $< -o $@

# ============================================================================
# LINKING RULES
# ============================================================================

$(BINDIR)/popcornkmeans_blas: $(OBJECTS) | $(BINDIR)
	$(ECHO) "Linking: popcornkmeans_blas (with BLAS backend)"
	$(Q) $(CXX) $(CXXFLAGS) \
		$(OBJECTS) \
		-o $@ \
		$(LDFLAGS) \
		$(if $(BLAS_LIBS),$(BLAS_LIBS)) \
		$(EXTRA_LIBS) \
		$(EXTRA_LINK_OPTIONS) \
		-lm
	$(ECHO) "✓ Target: $@"

$(BINDIR)/popcornkmeans_openmp: $(OBJECTS) | $(BINDIR)
	$(ECHO) "Linking: popcornkmeans_openmp (with OpenMP backend)"
	$(Q) $(CXX) $(CXXFLAGS) \
		$(OBJECTS) \
		-o $@ \
		$(LDFLAGS) \
		$(EXTRA_LIBS) \
		$(EXTRA_LINK_OPTIONS) \
		-lm
	$(ECHO) "✓ Target: $@"

# ============================================================================
# DIRECTORY CREATION
# ============================================================================

$(OBJDIR) $(DEPDIR) $(BINDIR) $(LOGDIR):
	$(Q) mkdir -p $@

# ============================================================================
# DEPENDENCY INCLUSION
# ============================================================================

-include $(DEPS)

# ============================================================================
# SECONDARY TARGETS (keep intermediate files)
# ============================================================================

.SECONDARY: $(OBJECTS)

# ============================================================================
# DEBUG TARGET
# ============================================================================

debug: CXXFLAGS := -g -O0 -std=c++17 -Wall -Wextra -DDEBUG
debug: CFLAGS   := -g -O0 -std=c99 -Wall -Wextra -DDEBUG
debug: all
	$(ECHO) "Debug build completed"