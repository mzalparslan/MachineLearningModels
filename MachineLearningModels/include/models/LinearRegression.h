#pragma once

#include "IRegressionModel.h"
#include "ModelParameters.h"
#include "options.h"

#include <numeric>
#include <utility>
#include <stdexcept>

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
 * @tparam T Floating-point type used for features, targets, and
 * model calculations.
 * @tparam Optimizer Optimization policy used to update model parameters.
 */
template <typename T, typename Optimizer>
class LinearRegression final : public IRegressionModel<T> {
public:
    /**
    * @brief Constructs a linear regression model.
    *
    * @param optimizer Optimizer used to learn weights and bias.
    * @param options Configuration supplied to the optimizer.
    */
    explicit LinearRegression(Optimizer optimizer_, 
        GradientDescentOptions<T> options_ = {})
        : optimizer(std::move(optimizer_)), options(options_) {
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
        isFitted = false;

		// Exceptions are propagated to the caller if validation fails.
		validateTrainingSet(trainingSet);
		
		// Initialize model parameters.
        featureCount = trainingSet.front().features.size();
		modelParameters.weights.assign(featureCount, T(0));
		modelParameters.bias = T(0);

        const auto hypothesis =
            [](const std::vector<T>& features,
                const ModelParameters<T>& parameters) {
                    return std::inner_product(
                        parameters.weights.begin(),
                        parameters.weights.end(),
                        features.begin(),
                        parameters.bias);
            };

        optimizer.optimize(
            trainingSet,
            options,
            modelParameters,
            hypothesis);

        isFitted = true;
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
		validateFeatures(features);

        return std::inner_product(
            modelParameters.weights.begin(),
            modelParameters.weights.end(),
            features.begin(),
            modelParameters.bias);
    }
    
private:
	// Validates input before fitting the model.
	static void validateTrainingSet(const std::vector<DataPoint<T>>& trainingSet) {
		if (true == trainingSet.empty()) {
			throw std::invalid_argument(
				"LinearRegression::fit: Training set is empty!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		if (0 == featureCount) {
			throw std::invalid_argument(
				"LinearRegression::fit: Training set has no features!");
		}

		for (const auto& sample : trainingSet) {
			if (sample.features.size() != featureCount) {
				throw std::invalid_argument(
					"LinearRegression::fit: "
					"Inconsistent feature count in training set!");
			}
		}
	}

	// Validates input before predicting a target value.
	void validateFeatures(const std::vector<T>& features) const {
        if (false == isFitted) {
            throw std::logic_error(
                "LinearRegression::predict: Call fit() before predict()!");
        }

        if (features.size() != featureCount) {
            throw std::invalid_argument(
                "LinearRegression::predict: Feature count mismatch!");
        }

        if (modelParameters.weights.size() != featureCount) {
            throw std::logic_error(
                "LinearRegression::predict: Inconsistent model parameters!");
        }
	}

private:
	// Optimizer used to update model parameters (weights and bias).
    Optimizer optimizer;
	// Options supplied to the optimizer for training configuration.
    GradientDescentOptions<T> options;
	// Learned model parameters (weights and bias) after fitting.
    ModelParameters<T> modelParameters;

	// Number of features in the training set used to fit the model.
    std::size_t featureCount = 0;
	// Flag indicating whether fit() was called for training set.
    bool isFitted = false;
};