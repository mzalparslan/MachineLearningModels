#pragma once

#include "DataPoint.h"
#include "ModelParameters.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Optimizes model parameters using Batch Gradient Descent.
 *
 * Calculates gradients over training dataset before each
 * parameter update. 
 * Supports optional L2 regularization for weights and bias.
 *
 * Supplied hypothesis must produce an output for which loss
 * gradient with respect to linear output is:
 *
 * @f[
 *     \frac{\partial L}{\partial z} = \hat{y} - y
 * @f]
 *
 * This applies to linear regression with half mean-squared error and
 * logistic regression with sigmoid binary cross-entropy.
 *
 * @tparam T Floating-point type used for data, parameters, and
 * optimization calculations.
 */
template <typename T>
struct BatchGradientDescent {

	static_assert(
		std::is_floating_point_v<T>,
		"BatchGradientDescent requires a floating-point type.");
	
	using Hypothesis = std::function<
		T(
			const std::vector<T>& features,
			const ModelParameters<T>& modelParameters
		)
	>;

	using Options = GradientDescentOptions<T>;

public:
	/**
	 * @brief Updates model parameters using full-batch gradient descent.
	 *
	 * During each epoch, gradients are accumulated over every training
	 * sample, averaged, optionally regularized, and applied to model
	 * weights and bias.
	 *
	 * @param trainingSet Dataset used to calculate gradients.
	 * @param options Learning rate, epoch count, and regularization options.
	 * @param modelParameters Model parameters modified during optimization.
	 * @param predict Function that calculates model output using the
	 *        supplied features and current model parameters.
	 *
	 * @throws std::invalid_argument If prediction function is empty,
	 * options are invalid, training set is malformed, or model parameters 
	 * are inconsistent.
	 * @throws std::runtime_error If optimization value is non-finite.
	 *
	 * @note If an exception occurs during optimization, modelParameters
	 * may contain updates completed before the failure.
	 */
	void optimize(const std::vector<DataPoint<T>>& trainingSet, const Options &options, 
		ModelParameters<T> &modelParameters, const Hypothesis &predict) {

		if (false == static_cast<bool>(predict)) {
			throw std::invalid_argument(
				"BatchGradientDescent: Hypothesis function is not valid!");
		}

		validateOptions(options);
		validateTrainingSet(trainingSet);

		const std::size_t featureCount = trainingSet.front().features.size();
		validateModelParameters(modelParameters, featureCount);
		
		const T exampleCount = static_cast<T>(trainingSet.size());
		if (false == std::isfinite(exampleCount) || exampleCount <= T(0)) {
			throw std::runtime_error(
				"BatchGradientDescent: Example count is NaN, Inf, or non-positive!");
		}

		std::vector<T> weightGradients(featureCount, T(0));
		for (std::size_t epoch = 0; epoch < options.epochs; epoch++) {
			// Reset accumulated gradients for the new epoch.
			std::fill(weightGradients.begin(), weightGradients.end(), T(0));
			T biasGradient = T(0);

			// GD: Calculate gradients over all examples.
			for (const auto& sample : trainingSet) {
				// Prediction for one example.
				// 1. FORWARD PASS: Calculate output of the "network" given inputs.
				T prediction = predict(sample.features, modelParameters);
				if (false == std::isfinite(prediction)) {
					throw std::runtime_error(
						"BatchGradientDescent: Prediction is NaN or Inf!");
				}

				// 2. Compute LOSS GRADIENT (Backpropagation start here)
				// 2.a) For Linear Regression Derivative of half Mean Squared Error (MSE) 
				// cost function;
				// 2.b) For Logistic Regression Derivative of Cross-Entropy (CE) cost function;
				// J with respect to prediction is (prediction - target).
				// This error term is the gradient of the loss layer.
				const T error = prediction - sample.target;
				if (false == std::isfinite(error)) {
					throw std::runtime_error(
						"BatchGradientDescent: Error is NaN or Inf!");
				}

				// Update bias gradient.
				// 3. BACKWARD PASS (Propagate gradients to parameters)
				// Calculate gradients for bias and weights.
				biasGradient += error;
				if (false == std::isfinite(biasGradient)) {
					throw std::runtime_error(
						"BatchGradientDescent: Bias gradient is NaN or Inf!");
				}

				// Update weight gradients.
				for (std::size_t j = 0; j < featureCount; j++) {
					// Chain Rule: dJ/dw[j] = dJ/dError * dError/dPred * dPred/dw[j]
					//                      = 1         * error        * features[j]
					// Accumulate dJ/dw[j] = error * feature[j].
					weightGradients[j] += (error * sample.features[j]);
					if (false == std::isfinite(weightGradients[j])) {
						throw std::runtime_error(
							"BatchGradientDescent: Weight gradient is NaN or Inf!");
					}
				}
			}
			
			biasGradient = biasGradient / exampleCount;
			if (true == options.regularizeBias && options.lambda > T(0)) {
				biasGradient += 
					(options.lambda / exampleCount) * modelParameters.bias;
			}

			modelParameters.bias -= options.learningRate * biasGradient;
			if (false == std::isfinite(modelParameters.bias)) {
				throw std::runtime_error(
					"BatchGradientDescent: Bias is NaN or Inf!");
			}

			// Average w gradients and update weights.
			for (std::size_t j = 0; j < featureCount; j++) {
				weightGradients[j] /= exampleCount;
				if (false == std::isfinite(weightGradients[j])) {
					throw std::runtime_error(
						"BatchGradientDescent: Average Weight gradient is NaN or Inf!");
				}

				if (options.lambda > T(0)) {
					weightGradients[j] += 
						(options.lambda / exampleCount) * modelParameters.weights[j];
					if (false == std::isfinite(weightGradients[j])) {
						throw std::runtime_error(
							"BatchGradientDescent: "
							"Weight gradient regularization is NaN or Inf!");
					}
				}

				modelParameters.weights[j] -= 
					options.learningRate * weightGradients[j];
				if (false == std::isfinite(modelParameters.weights[j])) {
					throw std::runtime_error(
						"BatchGradientDescent: Model weight is NaN or Inf!");
				}
			}
		}
	}

private:
	// Validates optimization options for consistency and correctness.
	static void validateOptions(const Options& options) {
		if (false == std::isfinite(options.learningRate)) {
			throw std::invalid_argument(
				"BatchGradientDescent: Learning rate is NaN or Inf!");
		}

		if (options.learningRate <= T(0)) {
			throw std::invalid_argument(
				"BatchGradientDescent: Learning rate must be positive!");
		}

		if (0 == options.epochs) {
			throw std::invalid_argument(
				"BatchGradientDescent: Number of epochs must be greater than zero!");
		}

		if (false == std::isfinite(static_cast<T>(options.epochs))) {
			throw std::invalid_argument(
				"BatchGradientDescent: Number of epochs is NaN or Inf!");
		}

		if (options.lambda < T(0)) {
			throw std::invalid_argument(
				"BatchGradientDescent: Regularization strength must be non-negative!");
		}
	}

	// Validates model parameters for consistency and correctness.
	static void validateModelParameters(const ModelParameters<T>& modelParameters, 
										std::size_t featureCount) {
		if (modelParameters.weights.size() != featureCount) {
			throw std::invalid_argument(
				"BatchGradientDescent: Model weights size does not match feature count!");
		}

		if (false == std::isfinite(modelParameters.bias)) {
			throw std::invalid_argument(
				"BatchGradientDescent: Model bias is NaN or Inf!");
		}

		for (const T weight : modelParameters.weights) {
			if (false == std::isfinite(weight)) {
				throw std::invalid_argument(
					"BatchGradientDescent: Model weight is NaN or Inf!");
			}
		}
	}
	
	// Validates training set for consistency and correctness.
	static void validateTrainingSet(const std::vector<DataPoint<T>>& trainingSet) {
		if (true == trainingSet.empty()) {
			throw std::invalid_argument(
				"BatchGradientDescent: Training set is empty!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		if (0 == featureCount) {
			throw std::invalid_argument(
				"BatchGradientDescent: Training set has no features!");
		}

		for (const auto& sample : trainingSet) {
			if (sample.features.size() != featureCount) {
				throw std::invalid_argument(
					"BatchGradientDescent: "
					"DataPoint features size does not match feature count!");
			}

			if (false == std::isfinite(sample.target)) {
				throw std::invalid_argument(
					"BatchGradientDescent: DataPoint target is NaN or Inf!");
			}

			for (const T value : sample.features) {
				if (false == std::isfinite(value)) {
					throw std::invalid_argument(
						"BatchGradientDescent: Feature value is NaN or Inf!");
				}
			}
		}
	}
};
