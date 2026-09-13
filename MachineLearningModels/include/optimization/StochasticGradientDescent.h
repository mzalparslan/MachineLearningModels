#pragma once

#include "DataPoint.h"
#include "GradientDescentValidation.h"
#include "ModelParameters.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <concepts>
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

	// Kept as an alias for backward compatibility: optimize() itself now
	// accepts any invocable hypothesis (see the template parameter below),
	// so a plain lambda is passed directly and dispatched without the
	// indirect call std::function requires. Retained so callers may still
	// hold a hypothesis in a std::function when type erasure is wanted.
	using Hypothesis = std::function<T(
		const std::vector<T>& features,
		const ModelParameters<T>& modelParameters)>;

	// Declaration for options to configure SGD optimization.
	using Options = StochasticGradientDescentOptions<T>;

	// Invoked at the end of each epoch with the epoch index and the mean
	// squared residual over that epoch (see BatchGradientDescent::EpochCallback
	// for why this is a generic residual metric rather than the model's
	// own loss).
	using EpochCallback = std::function<void(std::size_t epoch, T cost)>;

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
	 * @param onEpochEnd Optional callback invoked after each epoch with
	 * the epoch index and mean squared residual.
	 *
	 * @throws std::invalid_argument If prediction function is empty,
	 * options are invalid, training set is inconsistent, or model
	 * parameters are inconsistent.
	 * @throws std::runtime_error If Optimization generate invalid values.
	 *
	 * @note If an exception occurs, modelParameters may contain updates
	 * completed before failure.
	 */
	template <std::invocable<const std::vector<T>&, const ModelParameters<T>&> Hypothesis_>
	void optimize(const std::vector<DataPoint<T>>& trainingSet,
				  const Options& options,
				  ModelParameters<T>& modelParameters,
			      const Hypothesis_& predict,
				  const EpochCallback& onEpochEnd = nullptr) {
		const auto& gradientOptions = options.gradientOptions;

		// std::function has an explicit bool conversion signalling an
		// empty target; a plain lambda or function object has no such
		// state, so this check only applies when the caller passed a
		// Hypothesis (rather than some other invocable, like a lambda).
		if constexpr (std::same_as<Hypothesis_, Hypothesis>) {
			if (false == static_cast<bool>(predict)) {
				throw std::invalid_argument(
					"StochasticGradientDescent: Hypothesis method is not valid!");
			}
		}

		using Validation = detail::GradientDescentValidation<T>;
		Validation::validateOptions(gradientOptions, "StochasticGradientDescent");
		Validation::validateTrainingSet(trainingSet, "StochasticGradientDescent");

		if (false == std::isfinite(options.decay) || options.decay < T(0)) {
			throw std::invalid_argument(
				"StochasticGradientDescent: Decay must be non-negative and finite!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		Validation::validateModelParameters(
			modelParameters, featureCount, "StochasticGradientDescent");

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

			// Learning-rate decay: later epochs take smaller, more precise
			// steps instead of oscillating around the optimum forever at a
			// fixed step size. options.decay == 0 (default) disables this.
			const T epochLearningRate = gradientOptions.learningRate /
				(T(1) + options.decay * static_cast<T>(epoch));

			T sumSquaredError = T(0);

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

				sumSquaredError += error * error;

				// Update bias (Immediate Update)
				// b = b - alpha * error
				T biasGradient = error;
				if ((true == gradientOptions.regularizeBias) &&
					(gradientOptions.lambda > T(0))) {
					biasGradient = Validation::regularize(
						biasGradient, gradientOptions.lambda,
						modelParameters.bias, sampleCountT);
					if (false == std::isfinite(biasGradient)) {
						throw std::runtime_error(
							"StochasticGradientDescent: Bias gradient is NaN or Inf!");
					}
				}

				modelParameters.bias = modelParameters.bias -
					epochLearningRate * biasGradient;
				if (false == std::isfinite(modelParameters.bias)) {
					throw std::runtime_error(
						"StochasticGradientDescent: Bias is NaN or Inf!");
				}

				// Update weights (Immediate Update)
				// w = w - alpha * error * x
				for (std::size_t j = 0; j < modelParameters.weights.size(); j++) {
					T weightGradient =  error * sample.features[j];
					if (gradientOptions.lambda > T(0)) {
						weightGradient = Validation::regularize(
							weightGradient, gradientOptions.lambda,
							modelParameters.weights[j], sampleCountT);
					}
					if (false == std::isfinite(weightGradient)) {
						throw std::runtime_error(
							"StochasticGradientDescent: Weight gradient is NaN or Inf!");
					}

					modelParameters.weights[j] = modelParameters.weights[j] -
						epochLearningRate * weightGradient;
					if (false == std::isfinite(modelParameters.weights[j])) {
						throw std::runtime_error(
							"StochasticGradientDescent: Weight is NaN or Inf!");
					}
				}
			}

			if (static_cast<bool>(onEpochEnd)) {
				onEpochEnd(epoch, sumSquaredError / sampleCountT);
			}
		}
	}
};
