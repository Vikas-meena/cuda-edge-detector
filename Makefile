# Build file for the CUDA Canny edge detector.
#
#   make            Build bin/edge_detector (needs the CUDA toolkit).
#   make run        Build, then process data/input with the default options.
#   make test       Build and run the CPU-only unit test (no GPU needed).
#   make clean      Remove build products.
#
# Override CUDA_PATH if the toolkit is not in /usr/local/cuda, e.g.
#   make CUDA_PATH=/opt/cuda
# ARGS passes extra command-line options to `make run`, e.g.
#   make run ARGS="--sigma=2.0 --skip_cpu"

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(if $(wildcard $(CUDA_PATH)/bin/nvcc),$(CUDA_PATH)/bin/nvcc,nvcc)
CXX       ?= g++

# Floating-point contraction (fused multiply-add) is disabled for both
# compilers so that the CPU reference and the GPU kernels round identically.
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -Isrc -Ithird_party
NVCCFLAGS := -std=c++17 -O2 -fmad=false -Wno-deprecated-gpu-targets -Isrc

BUILD_DIR := obj
BIN_DIR   := bin
TARGET    := $(BIN_DIR)/edge_detector
TEST      := $(BIN_DIR)/canny_cpu_test

HOST_SOURCES := src/main.cc src/options.cc src/image.cc src/canny_cpu.cc \
                src/canny_params.cc
HOST_OBJECTS := $(HOST_SOURCES:src/%.cc=$(BUILD_DIR)/%.o)
CUDA_OBJECTS := $(BUILD_DIR)/canny_gpu.o
HEADERS      := $(wildcard src/*.h)

TEST_OBJECTS := $(BUILD_DIR)/canny_cpu_test.o $(BUILD_DIR)/canny_cpu.o \
                $(BUILD_DIR)/canny_params.o $(BUILD_DIR)/options.o

.PHONY: all build run test clean

all: build

build: $(TARGET)

$(TARGET): $(HOST_OBJECTS) $(CUDA_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cc $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: tests/%.cc $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST): $(TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

run: build
	./$(TARGET) $(ARGS)

test: $(TEST)
	./$(TEST)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
