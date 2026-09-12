#pragma once

#include "BaseRegressionModel.h"

/**
 * @brief Multiple linear regression model trained by an optimizer.
 *
 * Learns one weight for each input feature and a bias term. Prediction
 * is calculated using:
 *
 * @f[
 *     \hat{y} = b + \sum_{j=1}^{n} w_j x_j
 * @f]
 *
 * Calling fit() resets existing model parameters to zero before
 * optimization begins.
 *
 * @tparam T Floating point data types for features, targets, and
 * model calculations.
 * @tparam Optimizer Optimization policy used to update model parameters.
 */
template <typename T, typename Optimizer>
class LinearRegression final : public BaseRegressionModel<T, Optimizer> {

    // Options from Optimizer used to configure optimization process.
    using Options = typename Optimizer::Options;
    // Validation of training set from BaseRegressionModel.
    using BaseRegressionModel<T, Optimizer>::validateTrainingSet;

public:
    ~LinearRegression() override = default;

    /**
    * @brief Constructs a linear regression model.
    *
    * @param optimizer Optimizer used to learn weights and bias.
    * @param options Configuration supplied to the optimizer.
    */
    explicit LinearRegression(Optimizer optimizer_,
		Options options_ = {}, 
        ExecutionStrategy<T> executionStrategy_ = {})
        : BaseRegressionModel<T, Optimizer>(std::move(optimizer_), 
            std::move(options_), 
            std::move(executionStrategy_)) {
    }

    /**
     * @brief Fits model using a training dataset.
     *
     * Initializes all weights and bias to zero, then delegates
     * parameter optimization to the configured optimizer.
     *
     * Calling this method replaces previously learned parameters.
     *
     * @param trainingSet Dataset used to learn model parameters.
     *
     * @throws std::invalid_argument If dataset is empty, contains
     * no features, or has inconsistent feature counts.
     *
     * @note Exceptions produced by optimizer are propagated to
     * the caller.
     */
    void fit(const std::vector<DataPoint<T>>& trainingSet) override {
        this->isFitted = false;

        // Exceptions are propagated to the caller if validation fails.
        validateTrainingSet(trainingSet);

        // Initialize model parameters.
        this->featureCount = trainingSet.front().features.size();
        this->modelParameters.weights.assign(this->featureCount, T(0));
        this->modelParameters.bias = T(0);

        const auto hypothesis =
            [this](const std::vector<T>& features,
                const ModelParameters<T>& parameters) {
					return this->linearOutput(features, parameters);
            };

        this->optimizer.optimize(
            trainingSet,
            this->options,
            this->modelParameters,
            hypothesis);

        this->isFitted = true;
    }

    /**
     * @brief Predicts a target value for one feature vector.
     *
     * @param features Input features used for prediction.
     *
     * @return Predicted target value.
     *
     * @note Exceptions are propagated to the caller if validation fails.
     * @throws std::logic_error If fit() was not called yet, or if
     * parameters' size are inconsistent.
     * @throws std::invalid_argument If feature count is not
     * matching with fitted model.
     */
    [[nodiscard]]
    T predict(const std::vector<T>& features) const override {
        // Exceptions are propagated to the caller if validation fails.
        this->validateFeatures(features);

		return this->linearOutput(features, this->modelParameters);
    }
};