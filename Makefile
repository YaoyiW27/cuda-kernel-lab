# Build all CUDA kernels into shared libraries loadable from Python via ctypes.
#
# Usage:
#   make            # build every kernel
#   make clean      # remove build artifacts
#
# Requires: CUDA Toolkit 12.x (nvcc on PATH).

NVCC      ?= nvcc
NVCCFLAGS ?= -O2 -Xcompiler -fPIC --shared
BUILD_DIR := build

# CUDA source files -> shared objects, e.g. kernels/02_matmul/matmul_tiled.cu -> build/matmul_tiled.so
SOURCES := $(shell find kernels -name '*.cu')
TARGETS := $(patsubst %.cu,$(BUILD_DIR)/%.so,$(notdir $(SOURCES)))

.PHONY: all clean
all: $(TARGETS)

# Static pattern rule: compile each .cu into build/<name>.so
$(BUILD_DIR)/%.so:
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(filter %/$*.cu,$(SOURCES)) -o $@

# Regenerate .so when its source changes.
$(foreach src,$(SOURCES),$(eval $(BUILD_DIR)/$(basename $(notdir $(src))).so: $(src)))

clean:
	rm -rf $(BUILD_DIR)
