#pragma once

#include "DataPoint.h"
#include "GradientDescentValidation.h"
#include "ModelParameters.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <concepts>
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
 * @tparam T Floating-point mode used for data, parameters, and
 * optimization calculations.
 */
template <typename T>
struct BatchGradientDescent {

	static_assert(
		std::is_floating_point_v<T>,
		"BatchGradientDescent requires a floating-point mode.");

	// Kept as an alias for backward compatibility: optimize() itself now
	// accepts any invocable hypothesis (see the template parameter below),
	// so a plain lambda is passed directly and dispatched without the
	// indirect call std::function requires. Retained so callers may still
	// hold a hypothesis in a std::function when type erasure is wanted.
	using Hypothesis = std::function<T(const std::vector<T>& features,
		const ModelParameters<T>& modelParameters)>;

	using Options = GradientDescentOptions<T>;

	// Invoked at the end of each epoch with the epoch index and the
	// mean squared residual over that epoch, so callers can watch for
	// convergence, plateaus, or divergence. This is a generic residual
	// metric rather than the model's own loss (e.g. not cross-entropy
	// for a classifier), since the optimizer only ever sees dL/dz = error.
	using EpochCallback = std::function<void(std::size_t epoch, T cost)>;

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
	 * @param onEpochEnd Optional callback invoked after each epoch with
	 *        the epoch index and mean squared residual.
	 *
	 * @throws std::invalid_argument If prediction function is empty,
	 * options are invalid, training set is malformed, or model parameters
	 * are inconsistent.
	 * @throws std::runtime_error If optimization value is non-finite.
	 *
	 * @note If an exception occurs during optimization, modelParameters
	 * may contain updates completed before the failure.
	 */
	template <std::invocable<const std::vector<T>&, const ModelParameters<T>&> Hypothesis_>
	void optimize(const std::vector<DataPoint<T>>& trainingSet, const Options &options,
		ModelParameters<T> &modelParameters, const Hypothesis_ &predict,
		const EpochCallback& onEpochEnd = nullptr) {

		// std::function has an explicit bool conversion signalling an
		// empty target; a plain lambda or function object has no such
		// state, so this check only applies when the caller passed a
		// Hypothesis (rather than some other invocable, like a lambda).
		if constexpr (std::same_as<Hypothesis_, Hypothesis>) {
			if (false == static_cast<bool>(predict)) {
				throw std::invalid_argument(
					"BatchGradientDescent: Hypothesis function is not valid!");
			}
		}

		using Validation = detail::GradientDescentValidation<T>;
		Validation::validateOptions(options, "BatchGradientDescent");
		Validation::validateTrainingSet(trainingSet, "BatchGradientDescent");

		const std::size_t featureCount = trainingSet.front().features.size();
		Validation::validateModelParameters(modelParameters, featureCount, "BatchGradientDescent");

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
			T sumSquaredError = T(0);

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

				sumSquaredError += error * error;

				// Update bias gradient.
				// 3. BACKWARD PASS (Propagate gradients to parameters)
				// Calculate gradients for bias and weights.
				biasGradient += error;

				// Update weight gradients. Not checked for finiteness here:
				// a non-finite partial sum can only arise from non-finite
				// inputs, which validateTrainingSet already rejected up
				// front, so one check after the full accumulation (below)
				// catches it just as reliably without paying a branch per
				// feature per sample.
				for (std::size_t j = 0; j < featureCount; j++) {
					// Chain Rule: dJ/dw[j] = dJ/dError * dError/dPred * dPred/dw[j]
					//                      = 1         * error        * features[j]
					// Accumulate dJ/dw[j] = error * feature[j].
					weightGradients[j] += (error * sample.features[j]);
				}
			}

			if (false == std::isfinite(biasGradient)) {
				throw std::runtime_error(
					"BatchGradientDescent: Bias gradient is NaN or Inf!");
			}

			for (std::size_t j = 0; j < featureCount; j++) {
				if (false == std::isfinite(weightGradients[j])) {
					throw std::runtime_error(
						"BatchGradientDescent: Weight gradient is NaN or Inf!");
				}
			}

			biasGradient = biasGradient / exampleCount;
			if (true == options.regularizeBias && options.lambda > T(0)) {
				biasGradient = Validation::regularize(
					biasGradient, options.lambda, modelParameters.bias, exampleCount);
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
					weightGradients[j] = Validation::regularize(
						weightGradients[j], options.lambda,
						modelParameters.weights[j], exampleCount);
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

			if (static_cast<bool>(onEpochEnd)) {
				onEpochEnd(epoch, sumSquaredError / exampleCount);
			}
		}
	}
};
