#pragma once

#include "IRegressionModel.h"
#include "ModelParameters.h"
#include "ExecutionStrategy.h"
#include "options.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief Provides shared state and behavior for trainable regression models.
 *
 * Stores  optimizer, optimizer-specific configuration, learned model
 * parameters, and strategy used to calculate linear outputs.
 *
 * Derived models remain responsible for implementing fit() and predict().
 *
 * @tparam T Floating-point type used for features, targets, and calculations.
 * @tparam Optimizer Optimizer type used to learn model parameters.
 */
template <typename T, typename Optimizer>
class BaseRegressionModel : public IRegressionModel<T> {
protected:
    // Optimizer-specific configuration type.
    using Options = typename Optimizer::Options;

	const ModelParameters<T>& getModelParameters() const {
		return modelParameters;
	}

public:
    virtual ~BaseRegressionModel() = default;

protected:
    /**
     * @brief Constructs the base model with its training and execution policies.
     *
     * @param optimizer_ Optimizer used to learn model parameters.
     * @param options_ Configuration supplied to the optimizer.
     * @param executionStrategy_ Strategy used to calculate linear outputs
     * during training and prediction.
     */
    BaseRegressionModel(
        Optimizer optimizer_,
        Options options_ = {},
        ExecutionStrategy<T> executionStrategy_ = {})
        : optimizer(std::move(optimizer_)),
        options(std::move(options_)),
        executionStrategy(std::move(executionStrategy_)) {
    }

    /**
     * @brief Calculates the raw linear output for one feature vector.
     *
     * The selected execution strategy calculates:
     *
     * @f[
     *     z = b + \sum_{j=1}^{n} w_j x_j
     * @f]
     *
     * For linear regression, this value is the numeric prediction. For
     * logistic regression, it is the logit supplied to the sigmoid function.
     *
     * @param features Feature vector used in the calculation.
     * @param parameters Weights and bias used in the calculation.
     *
     * @return Raw linear output.
     *
     * @throws std::runtime_error If the calculated output is NaN or infinite.
     */
    [[nodiscard]]
    T linearOutput(
        const std::vector<T>& features,
        const ModelParameters<T>& parameters) const
    {
        const T output =
            executionStrategy.calculateLinearOutput(
                features,
                parameters);

        if (false == std::isfinite(output)) {
            throw std::runtime_error(
                "BaseRegressionModel::linearOutput: "
                "Output is NaN or Inf!");
        }

        return output;
    }

    /**
     * @brief Validates the structural requirements of a training dataset.
     *
     * Every data point must contain at least one feature, and all data points
     * must contain the same number of features.
     *
     * @param trainingSet Dataset to validate.
     *
     * @throws std::invalid_argument If the dataset is empty, contains no
     * features, or contains inconsistent feature counts.
     */
    static void validateTrainingSet(
        const std::vector<DataPoint<T>>& trainingSet)
    {
        if (true == trainingSet.empty()) {
            throw std::invalid_argument(
                "BaseRegressionModel::validateTrainingSet: "
                "Training set is empty!");
        }

        const std::size_t expectedFeatureCount =
            trainingSet.front().features.size();

        if (0 == expectedFeatureCount) {
            throw std::invalid_argument(
                "BaseRegressionModel::validateTrainingSet: "
                "Training set has no features!");
        }

        for (const auto& dataPoint : trainingSet) {
            if (dataPoint.features.size() !=
                expectedFeatureCount) {
                throw std::invalid_argument(
                    "BaseRegressionModel::validateTrainingSet: "
                    "Inconsistent feature count in training set!");
            }
        }
    }

    /**
     * @brief Validates whether a feature vector can be used for prediction.
     *
     * @param features Feature vector to validate.
     *
     * @throws std::logic_error If the model has not been fitted or its learned
     * parameters are inconsistent.
     * @throws std::invalid_argument If the supplied feature count differs from
     * the fitted feature count.
     */
    void validateFeatures(
        const std::vector<T>& features) const
    {
        if (false == isFitted) {
            throw std::logic_error(
                "BaseRegressionModel::validateFeatures: "
                "Call fit() before predict()!");
        }

        if (features.size() != featureCount) {
            throw std::invalid_argument(
                "BaseRegressionModel::validateFeatures: "
                "Feature count mismatch!");
        }

        if (modelParameters.weights.size() != featureCount) {
            throw std::logic_error(
                "BaseRegressionModel::validateFeatures: "
                "Inconsistent model parameters!");
        }
    }

protected:
    // Optimizer used to update model weights and bias.
    Optimizer optimizer;

    // Optimizer-specific training configuration.
    Options options;

    // Weights and bias learned during fitting.
    ModelParameters<T> modelParameters;

    // Strategy used to calculate linear outputs.
    ExecutionStrategy<T> executionStrategy;

    // Number of features expected by the fitted model.
    std::size_t featureCount = 0;

    // Indicates whether fitting completed successfully.
    bool isFitted = false;
};