#pragma once

#include "DataPoint.h"
#include "ModelParameters.h"
#include "options.h"

#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>

/**
 * @brief Optimizes model parameters using Mini-Batch Gradient Descent.
 *
 * Divides training dataset into smaller batches and updates model
 * parameters after processing each batch. DataPoint indices can be
 * shuffled before each epoch without modifying original dataset.
 *
 * Hypothesis must produce an output for which loss gradient 
 * with respect to the linear output is:
 *
 * @f[
 *     \frac{\partial L}{\partial z} = \hat{y} - y
 * @f]
 *
 * This applies to linear regression with half mean-squared error and
 * logistic regression with sigmoid binary cross-entropy.
 *
 * Optional L2 regularization can be applied to weights and bias.
 *
 * @tparam T Floating-point type used for data, parameters, and
 * optimization calculations.
 */
template <typename T>
struct MiniBatchGradientDescent {
	static_assert(
		std::is_floating_point_v<T>,
		"MiniBatchGradientDescent requires a floating-point type.");

	using Hypothesis = std::function<T(
		const std::vector<T>& features,
		const ModelParameters<T>& modelParameters)>;

	using Options = MiniBatchGradientDescentOptions<T>;

public:
	/**
	 * @brief Updates model parameters using mini-batch gradient descent.
	 *
	 * For each epoch, dataset is divided into batches. Gradients are
	 * accumulated and averaged within each batch, then immediately applied
	 * to model parameters.
	 *
	 * @param trainingSet Dataset used to calculate gradients.
	 * @param options Common gradient-descent settings and mini-batch
	 * configuration.
	 * @param modelParameters Model parameters modified during optimization.
	 * @param predict Function that calculates model output using 
	 * supplied features and current model parameters.
	 *
	 * @throws std::invalid_argument If prediction function is empty,
	 * options are invalid, training set is invalid, or 
	 * model parameters are inconsistent.
	 * @throws std::runtime_error If optimizer calculates non-finite value.
	 *
	 * @note If an exception occurs, modelParameters may contain updates
	 * completed before failure.
	 */
	void optimize(const std::vector<DataPoint<T>>& trainingSet, 
		const Options& options,
		ModelParameters<T>& modelParameters, 
		const Hypothesis& predict) {

		if (false == static_cast<bool>(predict)) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Hypothesis function is not valid!");
		}

		validateOptions(options);
		validateTrainingSet(trainingSet);

		const std::size_t sampleCount = trainingSet.size();
		const T sampleCountT = static_cast<T>(sampleCount);
		if (false == std::isfinite(sampleCountT) || sampleCountT <= T(0)) {
			throw std::runtime_error(
				"MiniBatchGradientDescent: DataPoint count is NaN, Inf, or non-positive!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		validateModelParameters(modelParameters, featureCount);

		const auto& gradientOptions = options.gradientOptions;

		std::mt19937 randomGenerator(options.randomSeed);

		std::vector<std::size_t> sampleIndices(sampleCount);
		std::iota(sampleIndices.begin(), sampleIndices.end(), std::size_t{ 0 });

		// Allocate gradient buffers reused for each mini-batch.
		std::vector<T> weightGradients(featureCount, T(0));
		T biasGradient = T(0);

		const std::size_t batchSize = options.batchSize;
		for (std::size_t epoch = 0; epoch < gradientOptions.epochs; epoch++) {
			if (true == options.shuffle) {
				// Shuffle sample indices without modifying the dataset.
				std::shuffle(sampleIndices.begin(), sampleIndices.end(), randomGenerator);
			}

			for (std::size_t batchIdx = 0; batchIdx < sampleCount; batchIdx += batchSize) {
				// Final batch may contain fewer samples.
				const std::size_t currentBatchSize = 
					std::min(batchSize, sampleCount - batchIdx);
				const T batchSizeT = static_cast<T>(currentBatchSize);
				
				// Reset accumulated gradients for the new batch.
				weightGradients.assign(featureCount, T(0));
				biasGradient = T(0);

				// Accumulate gradients over the current mini-batch.
				for (std::size_t k = 0; k < currentBatchSize; k++) {
					const auto& sample = trainingSet[sampleIndices[batchIdx + k]];

					// Calculate model output using the current parameters.
					const T prediction = predict(sample.features, modelParameters);
					if (false == std::isfinite(prediction)) {
						throw std::runtime_error(
							"MiniBatchGradientDescent: Prediction is NaN or Inf!");
					}

					// Calculate dL/dz for a supported model-loss combination.
					const T error = prediction - sample.target;
					if (false == std::isfinite(error)) {
						throw std::runtime_error(
							"MiniBatchGradientDescent: Error is NaN or Inf!");
					}

					biasGradient += error;
					if (false == std::isfinite(biasGradient)) {
						throw std::runtime_error(
							"MiniBatchGradientDescent: Bias gradient is NaN or Inf!");
					}

					// Accumulate dJ/db for current sample.
					for (std::size_t j = 0; j < featureCount; j++) {
						// Accumulate dJ/dw[j] = error * feature[j].
						weightGradients[j] += error * sample.features[j];
						if (false == std::isfinite(weightGradients[j])) {
							throw std::runtime_error(
								"MiniBatchGradientDescent: Weight gradient is NaN or Inf!");
						}
					}
				}

				// Average and optionally regularize bias gradient.
				biasGradient = biasGradient / batchSizeT;
				if (false == std::isfinite(biasGradient)) {
					throw std::runtime_error(
						"MiniBatchGradientDescent: Average Bias gradient is NaN or Inf!");
				}

				// Add L2 regularization gradient for the bias.
				if (true == gradientOptions.regularizeBias && gradientOptions.lambda != T(0)) {
					biasGradient += (gradientOptions.lambda / sampleCountT) * modelParameters.bias;
					if (false == std::isfinite(biasGradient)) {
						throw std::runtime_error(
							"MiniBatchGradientDescent: Bias gradient regularization is NaN or Inf!");
					}
				}
				
				// Apply bias update.
				modelParameters.bias -= gradientOptions.learningRate * biasGradient;
				if (false == std::isfinite(modelParameters.bias)) {
					throw std::runtime_error(
						"MiniBatchGradientDescent: Bias is NaN or Inf!");
				}

				// Average, regularize, and apply each weight gradient.
				for (std::size_t j = 0; j < featureCount; j++) {
					weightGradients[j] = weightGradients[j] / batchSizeT;
					if (false == std::isfinite(weightGradients[j])) {
						throw std::runtime_error(
							"MiniBatchGradientDescent: Average Weight gradient is NaN or Inf!");
					}

					// add regularization: (lambda / m) * w_j
					// Weak Regularization approach is dividing lambda by m
					// while other approaches directly using lambda * alpha.
					// For mini-batch it is often common to use (lambda * alpha) directly.
					// or normalize by batch size instead of m.
					// As our cost function is using m, GD currenly using same deritative.
					if (gradientOptions.lambda != T(0)) {
						weightGradients[j] += (gradientOptions.lambda / sampleCountT) * modelParameters.weights[j];
						if (false == std::isfinite(weightGradients[j])) {
							throw std::runtime_error(
								"MiniBatchGradientDescent: "
								"Weight gradient regularization is NaN or Inf!");
						}
					}

					// Apply weight update.
					modelParameters.weights[j] -= gradientOptions.learningRate * weightGradients[j];
					if (false == std::isfinite(modelParameters.weights[j])) {
						throw std::runtime_error(
							"MiniBatchGradientDescent: Model weight is NaN or Inf!");
					}
				}
			}
		}
	}

private:
	// Validates optimization options for consistency and correctness.
	static void validateOptions(const Options& options) {
		if (false == std::isfinite(options.gradientOptions.learningRate)) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Learning rate is NaN or Inf!");
		}

		if (options.gradientOptions.learningRate <= T(0)) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Learning rate must be positive!");
		}

		if (0 == options.gradientOptions.epochs) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: "
				"Epochs must be greater than zero!");
		}

		if (options.gradientOptions.lambda < T(0)) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: "
				"Regularization strength must be non-negative!");
		}

		if (false == std::isfinite(options.gradientOptions.lambda)) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Regularization strength is NaN or Inf!");
		}

		if (0 == options.batchSize) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Batch size must be greater than zero!");
		}
	}

	// Validates model parameters for consistency and correctness.
	static void validateModelParameters(const ModelParameters<T>& modelParameters,
		std::size_t featureCount) {
		if (modelParameters.weights.size() != featureCount) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: "
				"Model weights size does not match feature count!");
		}

		if (false == std::isfinite(modelParameters.bias)) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Model bias is NaN or Inf!");
		}

		for (const T weight : modelParameters.weights) {
			if (false == std::isfinite(weight)) {
				throw std::invalid_argument(
					"MiniBatchGradientDescent: Model weight is NaN or Inf!");
			}
		}
	}

	// Validates training set for consistency and correctness.
	static void validateTrainingSet(const std::vector<DataPoint<T>>& trainingSet) {
		if (true == trainingSet.empty()) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Training set is empty!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		if (0 == featureCount) {
			throw std::invalid_argument(
				"MiniBatchGradientDescent: Training set has no features!");
		}

		for (const auto& sample : trainingSet) {
			if (sample.features.size() != featureCount) {
				throw std::invalid_argument(
					"MiniBatchGradientDescent: "
					"DataPoint features size does not match feature count!");
			}

			if (false == std::isfinite(sample.target)) {
				throw std::invalid_argument(
					"MiniBatchGradientDescent: DataPoint target is NaN or Inf!");
			}

			for (const T value : sample.features) {
				if (false == std::isfinite(value)) {
					throw std::invalid_argument(
						"MiniBatchGradientDescent: Feature value is NaN or Inf!");
				}
			}
		}
	}
};

