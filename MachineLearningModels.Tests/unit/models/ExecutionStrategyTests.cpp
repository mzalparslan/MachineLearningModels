#include "pch.h"
#include "ExecutionStrategy.h"

#include <initializer_list>

TEST(ExecutionStrategyTest, AllModesAgreeWithInnerProduct) {
    ModelParameters<double> parameters;
    parameters.weights = { 0.5, -1.25, 2.0, 0.1, -0.75 };
    parameters.bias = 0.3;

    const std::vector<double> features = { 1.0, 2.0, -3.0, 4.0, -5.0 };

    const ExecutionStrategy<double> reference(ExecutionMode::InnerProduct);
    const double expected = reference.calculateLinearOutput(features, parameters);

    // Every execution mode must agree with the reference within a small
    // tolerance: exact bit-for-bit equality isn't guaranteed once a
    // parallel policy is genuinely backed by multiple threads, since
    // floating-point addition isn't associative and summation order can
    // then differ from the sequential inner product.
    for (const ExecutionMode mode : {
        ExecutionMode::InnerProduct,
        ExecutionMode::Sequential,
        ExecutionMode::Vectorized,
        ExecutionMode::Parallel,
        ExecutionMode::ParallelVectorized }) {
        const ExecutionStrategy<double> strategy(mode);
        const double actual = strategy.calculateLinearOutput(features, parameters);
        EXPECT_NEAR(actual, expected, 1e-9);
    }
}
