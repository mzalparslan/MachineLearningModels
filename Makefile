CXX ?= g++
AR ?= ar

BUILD ?= debug
BUILD_DIR := build/$(BUILD)
OBJECT_DIR := $(BUILD_DIR)/obj
LIBRARY_DIR := $(BUILD_DIR)/lib
BINARY_DIR := $(BUILD_DIR)/bin

LIBRARY_NAME := $(LIBRARY_DIR)/libMachineLearningModels.a
EXAMPLE_NAME := $(BINARY_DIR)/MachineLearningModels.Examples
TEST_NAME := $(BINARY_DIR)/MachineLearningModels.Tests
BENCHMARKS_NAME := $(BINARY_DIR)/MachineLearningModels.Benchmarks

INCLUDE_DIRS := \
	-IMachineLearningModels/include \
	-IMachineLearningModels/include/data \
	-IMachineLearningModels/include/metrics \
	-IMachineLearningModels/include/models \
	-IMachineLearningModels/include/optimization \
	-IMachineLearningModels/include/pipelines \
	-IMachineLearningModels/include/preprocessing \
	-IMachineLearningModels/include/utilities \
	-IMachineLearningModels.Tests \
	-IMachineLearningModels.Benchmarks

CPPFLAGS := $(INCLUDE_DIRS) -MMD -MP
COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic

ifeq ($(BUILD),release)
	CXXFLAGS := $(COMMON_FLAGS) -O3 -DNDEBUG
else
	CXXFLAGS := $(COMMON_FLAGS) -O0 -g
endif

LIBRARY_SOURCES := $(shell find MachineLearningModels/src -type f -name '*.cpp')
EXAMPLE_SOURCES := $(shell find MachineLearningModels.Examples/src -type f -name '*.cpp')
TEST_SOURCES := $(shell find MachineLearningModels.Tests -type f -name '*.cpp')
BENCHMARKS_SOURCES := $(shell find MachineLearningModels.Benchmarks -type f -name '*.cpp')
RESOURCE_FILES := $(wildcard resources/*.csv)

LIBRARY_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(LIBRARY_SOURCES))
EXAMPLE_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(EXAMPLE_SOURCES))
TEST_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(TEST_SOURCES))
BENCHMARKS_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(BENCHMARKS_SOURCES))

DEPENDENCY_FILES := \
	$(LIBRARY_OBJECTS:.o=.d) \
	$(EXAMPLE_OBJECTS:.o=.d) \
	$(TEST_OBJECTS:.o=.d) \
	$(BENCHMARKS_OBJECTS:.o=.d)

GTEST_LIBS ?= -lgtest_main -lgtest -pthread

# libstdc++ routes the parallel execution policies (std::execution::par,
# par_unseq) through Intel TBB, and every target below links against it,
# so a missing libtbb-dev fails the build at the link step (cannot find
# -ltbb) rather than degrading ExecutionMode::Parallel/ParallelVectorized
# to silent sequential execution at runtime. Install it via your package
# manager (e.g. `apt install libtbb-dev`) if it isn't already on your system.
PARALLEL_LIBS ?= -ltbb

.PHONY: all library examples tests test benchmarks clean help

all: library examples tests benchmarks

library: $(LIBRARY_NAME)

examples: $(EXAMPLE_NAME)
	@mkdir -p $(BINARY_DIR)
	@cp -f $(RESOURCE_FILES) $(BINARY_DIR)/

tests: $(TEST_NAME)

test: $(TEST_NAME)
	$(TEST_NAME)

benchmarks: $(BENCHMARKS_NAME)

$(LIBRARY_NAME): $(LIBRARY_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(EXAMPLE_NAME): $(EXAMPLE_OBJECTS) $(LIBRARY_NAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(PARALLEL_LIBS) -o $@

$(TEST_NAME): $(TEST_OBJECTS) $(LIBRARY_NAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(GTEST_LIBS) $(PARALLEL_LIBS) -o $@

$(BENCHMARKS_NAME): $(BENCHMARKS_OBJECTS) $(LIBRARY_NAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(PARALLEL_LIBS) -o $@

$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

help:
	@echo "make                 Build library, examples, tests, and benchmarks (debug)"
	@echo "make BUILD=release   Build optimized release binaries"
	@echo "make library         Build the static library"
	@echo "make examples        Build the example executable"
	@echo "make tests           Build the Google Test executable"
	@echo "make test            Build and run all tests"
	@echo "make benchmarks      Build the execution-strategy benchmarks"
	@echo "make clean           Remove Makefile build outputs"

-include $(DEPENDENCY_FILES)
