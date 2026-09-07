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

INCLUDE_DIRS := \
	-IMachineLearningModels/include \
	-IMachineLearningModels/include/data \
	-IMachineLearningModels/include/metrics \
	-IMachineLearningModels/include/models \
	-IMachineLearningModels/include/optimization \
	-IMachineLearningModels/include/pipelines \
	-IMachineLearningModels/include/preprocessing \
	-IMachineLearningModels/include/utilities \
	-IMachineLearningModels.Tests

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
RESOURCE_FILES := $(wildcard resources/*.csv)

LIBRARY_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(LIBRARY_SOURCES))
EXAMPLE_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(EXAMPLE_SOURCES))
TEST_OBJECTS := $(patsubst %.cpp,$(OBJECT_DIR)/%.o,$(TEST_SOURCES))

DEPENDENCY_FILES := \
	$(LIBRARY_OBJECTS:.o=.d) \
	$(EXAMPLE_OBJECTS:.o=.d) \
	$(TEST_OBJECTS:.o=.d)

GTEST_LIBS ?= -lgtest_main -lgtest -pthread

.PHONY: all library examples tests test clean help

all: library examples tests

library: $(LIBRARY_NAME)

examples: $(EXAMPLE_NAME)
	@mkdir -p $(BINARY_DIR)
	@cp -f $(RESOURCE_FILES) $(BINARY_DIR)/

tests: $(TEST_NAME)

test: $(TEST_NAME)
	$(TEST_NAME)

$(LIBRARY_NAME): $(LIBRARY_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(EXAMPLE_NAME): $(EXAMPLE_OBJECTS) $(LIBRARY_NAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_NAME): $(TEST_OBJECTS) $(LIBRARY_NAME)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(GTEST_LIBS) -o $@

$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

help:
	@echo "make                 Build library, examples, and tests (debug)"
	@echo "make BUILD=release   Build optimized release binaries"
	@echo "make library         Build the static library"
	@echo "make examples        Build the example executable"
	@echo "make tests           Build the Google Test executable"
	@echo "make test            Build and run all tests"
	@echo "make clean           Remove Makefile build outputs"

-include $(DEPENDENCY_FILES)
