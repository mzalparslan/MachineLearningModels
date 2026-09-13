#pragma once

#include "IRegressionModel.h"
#include "ModelParameters.h"
#include "ExecutionStrategy.h"
#include "OptimizationPolicy.h"
#include "options.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief Provides shared state and behavior for trainable linear models.
 *
 * Stores optimizer, optimizer-specific configuration, learned model
 * parameters, and strategy used to calculate linear outputs.
 *
 * What LinearRegression and LogisticBinaryClassifier actually share is
 * "a linear model with weights and a bias" -- not that both are
 * regression models (a classifier is not a regression model). This base
 * models that shared shape; IRegressionModel remains the capability
 * interface for models that predict a continuous target.
 *
 * Derived models remain responsible for implementing fit() and predict().
 *
 * @tparam T Floating-point type used for features, targets, and calculations.
 * @tparam Optimizer Optimizer type used to learn model parameters.
 */
template <typename T, OptimizationPolicy<T> Optimizer>
class BaseLinearModel : public IRegressionModel<T> {
protected:
    // Optimizer-specific configuration type.
    using Options = typename Optimizer::Options;

public:
    virtual ~BaseLinearModel() = default;

    /**
     * @brief Read-only access to the learned weights and bias.
     *
     * For a library whose purpose is learning how these algorithms work,
     * being able to inspect the fitted coefficients (and compare them to
     * a closed-form solution) is most of the value.
     *
     * @return The model's learned parameters. Only meaningful once fit()
     * has completed successfully.
     */
    [[nodiscard]]
    const ModelParameters<T>& parameters() const noexcept {
        return modelParameters;
    }

protected:
    /**
     * @brief Constructs the base model with its training and execution policies.
     *
     * @param optimizer_ Optimizer used to learn model parameters.
     * @param options_ Configuration supplied to the optimizer.
     * @param executionStrategy_ Strategy used to calculate linear outputs
     * during training and prediction.
     */
    BaseLinearModel(
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
                "BaseLinearModel::linearOutput: "
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
                "BaseLinearModel::validateTrainingSet: "
                "Training set is empty!");
        }

        const std::size_t expectedFeatureCount =
            trainingSet.front().features.size();

        if (0 == expectedFeatureCount) {
            throw std::invalid_argument(
                "BaseLinearModel::validateTrainingSet: "
                "Training set has no features!");
        }

        for (const auto& dataPoint : trainingSet) {
            if (dataPoint.features.size() !=
                expectedFeatureCount) {
                throw std::invalid_argument(
                    "BaseLinearModel::validateTrainingSet: "
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
     * the fitted feature count, or a feature is non-finite.
     */
    void validateFeatures(
        const std::vector<T>& features) const
    {
        if (false == isFitted) {
            throw std::logic_error(
                "BaseLinearModel::validateFeatures: "
                "Call fit() before predict()!");
        }

        if (features.size() != featureCount) {
            throw std::invalid_argument(
                "BaseLinearModel::validateFeatures: "
                "Feature count mismatch!");
        }

        if (modelParameters.weights.size() != featureCount) {
            throw std::logic_error(
                "BaseLinearModel::validateFeatures: "
                "Inconsistent model parameters!");
        }

        for (const T feature : features) {
            if (false == std::isfinite(feature)) {
                throw std::invalid_argument(
                    "BaseLinearModel::validateFeatures: "
                    "Feature value is NaN or Inf!");
            }
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
