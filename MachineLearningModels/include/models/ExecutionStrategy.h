#pragma once

#include "ModelParameters.h"

#include <execution>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Specifies how a linear-output calculation is executed.
 *
 * Parallel and vectorized modes permit the standard-library implementation
 * to use the requested execution behavior, but do not guarantee that a
 * particular implementation will use multiple threads or SIMD instructions.
 */
enum class ExecutionMode {
    // Ordered dot product using std::inner_product.
    InnerProduct,

    // Sequenced reduction using std::transform_reduce.
    Sequential,

    // Permits unsequenced or vectorized execution.
    Vectorized,

    // Permits execution across multiple threads.
    Parallel,

    // Permits both parallel and unsequenced execution.
    ParallelVectorized
};

/**
 * @brief Calculates model linear outputs using a selected execution mode.
 *
 * The calculation performed is:
 *
 * @f[
 *     z = b + \sum_{j=1}^{n} w_j x_j
 * @f]
 *
 * The execution mode controls whether the dot product uses inner_product or
 * transform_reduce with a standard execution policy.
 *
 * @tparam T Floating-point type used for features and model parameters.
 */
template <typename T>
class ExecutionStrategy {
    static_assert(
        std::is_floating_point_v<T>,
        "ExecutionStrategy requires a floating-point type.");

public:
	/// @brief Defaults execution strategy to inner_product.
	ExecutionStrategy() = default;

    /**
     * @brief Constructs a strategy with the requested execution mode.
     *
     * @param mode_ Execution mode used for linear-output calculations.
     */
    explicit ExecutionStrategy(ExecutionMode mode_)
        : mode(mode_) {
    }

    /**
     * @brief Calculates the linear output for one feature vector.
     *
     * The feature vector and weight vector must have identical sizes.
     * Input validation is expected to be performed by the calling model.
     *
     * @param features Input feature vector.
     * @param modelParameters Weights and bias used in the calculation.
     *
     * @return Dot product of weights and features, plus bias.
     *
     * @throws std::logic_error If the configured execution mode is invalid.
     */
    [[nodiscard]]
    T calculateLinearOutput(
        const std::vector<T>& features,
        const ModelParameters<T>& modelParameters) const
    {
        switch (mode) {
        case ExecutionMode::InnerProduct:
            return std::inner_product(
                modelParameters.weights.begin(),
                modelParameters.weights.end(),
                features.begin(),
                modelParameters.bias);

        case ExecutionMode::Sequential:
            return transformReduce(
                std::execution::seq,
                features,
                modelParameters);

        case ExecutionMode::Vectorized:
            return transformReduce(
                std::execution::unseq,
                features,
                modelParameters);

        case ExecutionMode::Parallel:
            return transformReduce(
                std::execution::par,
                features,
                modelParameters);

        case ExecutionMode::ParallelVectorized:
            return transformReduce(
                std::execution::par_unseq,
                features,
                modelParameters);
        }

        throw std::logic_error(
            "ExecutionStrategy::calculateLinearOutput: "
            "Invalid execution mode!");
    }

private:
    // Execution mode selected when the strategy was constructed.
    ExecutionMode mode = ExecutionMode::InnerProduct;

    // Applies transform_reduce using the supplied standard execution policy.
    template <typename ExecutionPolicy>
    [[nodiscard]]
    T transformReduce(
        ExecutionPolicy executionPolicy,
        const std::vector<T>& features,
        const ModelParameters<T>& modelParameters) const
    {
        return std::transform_reduce(
            executionPolicy,
            modelParameters.weights.begin(),
            modelParameters.weights.end(),
            features.begin(),
            modelParameters.bias);
    }
};