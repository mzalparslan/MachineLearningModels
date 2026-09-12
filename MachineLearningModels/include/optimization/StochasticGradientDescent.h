#pragma once

#include "DataPoint.h"
#include "ModelParameters.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Optimizes model parameters using Stochastic Gradient Descent.
 *
 * Processes one sample at a time and immediately updates model
 * parameters after calculating that sample's gradients. DataPoint indices
 * can be shuffled before each epoch without modifying dataset.
 *
 * Hypothesis method will produce output for which:
 *
 * @f[
 *     \frac{\partial L}{\partial z} = \hat{y} - y
 * @f]
 *
 * It applies to linear regression with half mean-squared error and
 * logistic regression with sigmoid binary cross-entropy.
 *
 * Optional L2 regularization can be applied to weights and bias.
 *
 * @tparam T Floating-point mode used for data, parameters, and
 * optimization calculations.
 */
template <typename T>
struct StochasticGradientDescent {
	// Ensure that the template mode T is floating-point mode.
	static_assert(std::is_floating_point_v<T>, 
		"StochasticGradientDescent requires floating-point data mode!");

	// Declaration for hypothesis method to be used 
	// to predict output based on features and model parameters.
	using Hypothesis = std::function<T(
		const std::vector<T>& features,
		const ModelParameters<T>& modelParameters)>;

	// Declaration for options to configure SGD optimization.
	using Options = StochasticGradientDescentOptions<T>;

public:
	/**
	 * @brief Updates model parameters using Stochastic Gradient Descent.
	 *
	 * During each epoch, samples are processed individually. Each sample
	 * produces an immediate update to model weights and bias.
	 *
	 * @param trainingSet Dataset used to calculate gradients.
	 * @param options Common options and SGD-specific options.
	 * @param modelParameters Model parameters modified with optimization.
	 * @param predict Function that calculates model output using 
	 * supplied features and current model parameters.
	 *
	 * @throws std::invalid_argument If prediction function is empty,
	 * options are invalid, training set is inconsistent, or model
	 * parameters are inconsistent.
	 * @throws std::runtime_error If Optimization generate invalid values.
	 *
	 * @note If an exception occurs, modelParameters may contain updates
	 * completed before failure.
	 */
	void optimize(const std::vector<DataPoint<T>>& trainingSet, 
				  const Options& options,
				  ModelParameters<T>& modelParameters,
			      const Hypothesis& predict) {
		const auto& gradientOptions = options.gradientOptions;

		// Control if prediction method is valid.
		if (false == static_cast<bool>(predict)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Hypothesis method is not valid!");
		}

		validateTrainingSet(trainingSet);
		validateOptions(options);

		const std::size_t featureCount = trainingSet.front().features.size();
		validateModelParameters(modelParameters, featureCount);

		const std::size_t sampleCount = trainingSet.size();
		const T sampleCountT = static_cast<T>(sampleCount);
		if (false == std::isfinite(sampleCountT) || sampleCountT <= T(0)) {
			throw std::runtime_error(
				"StochasticGradientDescent: DataPoint count is NaN, Inf, or non-positive!");
		}

		std::vector<std::size_t> sampleIndices(sampleCount);
		std::iota(sampleIndices.begin(), sampleIndices.end(), std::size_t{ 0 });

		std::mt19937 randomGenerator(options.randomSeed);

		for (std::size_t epoch = 0; epoch < gradientOptions.epochs; epoch++) {
			if (true == options.shuffle) {
				// Shuffle sample indices without modifying the dataset.
				std::shuffle(sampleIndices.begin(), sampleIndices.end(), randomGenerator);
			}

			for (const std::size_t sampleIndex : sampleIndices) {
				const auto& sample = trainingSet[sampleIndex];

				const T prediction = predict(sample.features, modelParameters);
				if (false == std::isfinite(prediction)) {
					throw std::runtime_error(
						"StochasticGradientDescent: Prediction is NaN or Inf!");
				}

				const T error = prediction - sample.target;
				if (false == std::isfinite(error)) {
					throw std::runtime_error(
						"StochasticGradientDescent: Error is NaN or Inf!");
				}

				// Update bias (Immediate Update)
				// b = b - alpha * error
				T biasGradient = error;
				if ((true == gradientOptions.regularizeBias) && 
					(gradientOptions.lambda > T(0))) {
					biasGradient = biasGradient +
						(gradientOptions.lambda / sampleCountT) * modelParameters.bias;
					if (false == std::isfinite(biasGradient)) {
						throw std::runtime_error(
							"StochasticGradientDescent: Bias gradient is NaN or Inf!");
					}
				}

				modelParameters.bias = modelParameters.bias - 
					gradientOptions.learningRate * biasGradient;
				if (false == std::isfinite(modelParameters.bias)) {
					throw std::runtime_error(
						"StochasticGradientDescent: Bias is NaN or Inf!");
				}
				
				// Update weights (Immediate Update)
				// w = w - alpha * error * x
				for (std::size_t j = 0; j < modelParameters.weights.size(); j++) {
					T weightGradient =  error * sample.features[j];
					if (gradientOptions.lambda > T(0)) {
						weightGradient = weightGradient + 
							(gradientOptions.lambda / sampleCountT) * modelParameters.weights[j];
					}
					if (false == std::isfinite(weightGradient)) {
						throw std::runtime_error(
							"StochasticGradientDescent: Weight gradient is NaN or Inf!");
					}

					modelParameters.weights[j] = modelParameters.weights[j] - 
						gradientOptions.learningRate * weightGradient;
					if (false == std::isfinite(modelParameters.weights[j])) {
						throw std::runtime_error(
							"StochasticGradientDescent: Weight is NaN or Inf!");
					}
				}
			}
		}
	}

private:
	// Validates training data structure and values.
	static void validateTrainingSet(const std::vector<DataPoint<T>>& trainingSet) {
		if (true == trainingSet.empty()) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Training set has no samples!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		if (0 == featureCount) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Training set has no features!");
		}

		for (const auto& sample : trainingSet) {
			if (sample.features.size() != featureCount) {
				throw std::invalid_argument(
					"StochasticGradientDescent: "
					"Inconsistent feature count in training set!");
			}

			if (false == std::isfinite(sample.target)) {
				throw std::invalid_argument(
					"StochasticGradientDescent: DataPoint target is NaN or Inf!");
			}

			for (const T value : sample.features) {
				if (false == std::isfinite(value)) {
					throw std::invalid_argument(
						"StochasticGradientDescent: Feature value is NaN or Inf!");
				}
			}
		}
	}

	// Validates optimization options for consistency and correctness.
	static void validateOptions(const Options& options) {
		const auto& gradientOptions = options.gradientOptions;

		if (gradientOptions.learningRate <= T(0)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Learning rate must be positive!");
		}

		if (false == std::isfinite(gradientOptions.learningRate)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Learning rate is NaN or Inf!");
		}

		if (0 == gradientOptions.epochs) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Epochs must be greater than zero!");
		}

		if (gradientOptions.lambda < T(0)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Regularization strength must be non-negative!");
		}

		if (false == std::isfinite(gradientOptions.lambda)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Regularization strength is NaN or Inf!");
		}
	}

	// Validates model parameters for consistency and correctness.
	static void validateModelParameters(const ModelParameters<T>& modelParameters,
		std::size_t featureCount) {
		if (modelParameters.weights.size() != featureCount) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Model weights size does not match feature count!");
		}

		if (false == std::isfinite(modelParameters.bias)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Model bias is NaN or Inf!");
		}

		for (const T weight : modelParameters.weights) {
			if (false == std::isfinite(weight)) {
				throw std::invalid_argument(
					"StochasticGradientDescent: Model weight is NaN or Inf!");
			}
		}
	}
};
