#pragma once

#include "ExecutionStrategy.h"
#include "ModelParameters.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

using Clock = std::chrono::steady_clock;

template <typename T>
void benchmarkStrategy(
    std::string_view name,
    ExecutionMode mode,
    const std::vector<T>& features,
    const ModelParameters<T>& parameters,
    std::size_t iterations)
{
	if (0 == iterations) {
		std::cerr << "benchmarkStrategy: Iterations must be greater than zero!\n";
		return;
	}

    const ExecutionStrategy<T> strategy{ mode };
	
    const std::size_t minWarmupIterations = 5;
	const std::size_t warmupIterations =
		std::min<std::size_t>(minWarmupIterations, iterations);

    // Warm up caches and parallel execution infrastructure.
    for (std::size_t i = 0; i < warmupIterations; i++) {
        volatile T warmupResult = 
            strategy.calculateLinearOutput(features, parameters);

        (void)warmupResult;
    }

    volatile T result{};

    const auto start = Clock::now();

    for (std::size_t i = 0; i < iterations; i++) {
        result =
            strategy.calculateLinearOutput(
                features,
                parameters);
    }

    const auto finish = Clock::now();

    const auto total =
        std::chrono::duration<double, std::micro>(
            finish - start);

    const double averageMicroseconds =
        total.count() /
        static_cast<double>(iterations);

    std::cout
        << name << ','
        << features.size() << ','
        << iterations << ','
        << std::fixed << std::setprecision(4)
        << averageMicroseconds << '\n';

    // Keep the result observable to discourage optimization removal.
    (void)result;
}

template <typename T>
void runBenchmarks(std::size_t featureCount)
{
	if (0 == featureCount) {
		std::cerr << "runBenchmarks: Feature count must be greater than zero!\n";
		return;
	}

    std::mt19937 generator{ 42 };

    std::uniform_real_distribution<T> distribution{
        T(-1),
        T(1)
    };

    std::vector<T> features(featureCount);

    ModelParameters<T> parameters;
    parameters.weights.resize(featureCount);
    parameters.bias = T(0.25);

    for (std::size_t j = 0; j < featureCount; j++) {
        features[j] = distribution(generator);
        parameters.weights[j] = distribution(generator);
    }

    constexpr std::uint64_t targetElementOperations = 1'000'000'000ULL;
    const std::size_t minimumIterations = 20;
	const std::size_t calculatedIterations = 
        static_cast<std::size_t>(targetElementOperations / featureCount);

    // Process approximately one billion elements per strategy.
    const std::size_t iterations =
        std::max<std::size_t>(
            minimumIterations,
            calculatedIterations);

    benchmarkStrategy(
        "inner_product",
        ExecutionMode::InnerProduct,
        features,
        parameters,
        iterations);

    benchmarkStrategy(
        "seq",
        ExecutionMode::Sequential,
        features,
        parameters,
        iterations);

    benchmarkStrategy(
        "unseq",
        ExecutionMode::Vectorized,
        features,
        parameters,
        iterations);

    benchmarkStrategy(
        "par",
        ExecutionMode::Parallel,
        features,
        parameters,
        iterations);

    benchmarkStrategy(
        "par_unseq",
        ExecutionMode::ParallelVectorized,
        features,
        parameters,
        iterations);
}