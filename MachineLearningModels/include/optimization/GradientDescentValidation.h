#pragma once

#include "DataPoint.h"
#include "ModelParameters.h"
#include "options.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace detail {

	/**
	 * @brief Validation shared by the gradient-descent optimizers.
	 *
	 * BatchGradientDescent, StochasticGradientDescent, and
	 * MiniBatchGradientDescent all accept the same shape of training set
	 * and model parameters, and share the same core learning-rate/epoch/
	 * regularization options -- previously reimplemented three times,
	 * with the copies quietly drifting apart (mismatched lambda-guard
	 * operators, differently ordered checks).
	 *
	 * @tparam T Floating-point mode used for data, parameters, and options.
	 */
	template <typename T>
	struct GradientDescentValidation {

		/**
		 * @brief Validates the shared core of gradient-descent options:
		 * learning rate, epoch count, and L2 regularization strength.
		 *
		 * @param callerName Name of the calling optimizer, used to prefix
		 * exception messages.
		 */
		static void validateOptions(
			const GradientDescentOptions<T>& options, std::string_view callerName) {
			if (false == std::isfinite(options.learningRate)) {
				throw std::invalid_argument(
					std::string(callerName) + ": Learning rate is NaN or Inf!");
			}

			if (options.learningRate <= T(0)) {
				throw std::invalid_argument(
					std::string(callerName) + ": Learning rate must be positive!");
			}

			if (0 == options.epochs) {
				throw std::invalid_argument(
					std::string(callerName) +
					": Number of epochs must be greater than zero!");
			}

			if (options.lambda < T(0)) {
				throw std::invalid_argument(
					std::string(callerName) +
					": Regularization strength must be non-negative!");
			}

			if (false == std::isfinite(options.lambda)) {
				throw std::invalid_argument(
					std::string(callerName) +
					": Regularization strength is NaN or Inf!");
			}
		}

		// Validates training set for consistency and correctness.
		static void validateTrainingSet(
			const std::vector<DataPoint<T>>& trainingSet, std::string_view callerName) {
			if (true == trainingSet.empty()) {
				throw std::invalid_argument(
					std::string(callerName) + ": Training set is empty!");
			}

			const std::size_t featureCount = trainingSet.front().features.size();
			if (0 == featureCount) {
				throw std::invalid_argument(
					std::string(callerName) + ": Training set has no features!");
			}

			for (const auto& sample : trainingSet) {
				if (sample.features.size() != featureCount) {
					throw std::invalid_argument(
						std::string(callerName) +
						": Inconsistent feature count in training set!");
				}

				if (false == std::isfinite(sample.target)) {
					throw std::invalid_argument(
						std::string(callerName) + ": DataPoint target is NaN or Inf!");
				}

				for (const T value : sample.features) {
					if (false == std::isfinite(value)) {
						throw std::invalid_argument(
							std::string(callerName) + ": Feature value is NaN or Inf!");
					}
				}
			}
		}

		// Validates model parameters for consistency and correctness.
		static void validateModelParameters(
			const ModelParameters<T>& modelParameters,
			std::size_t featureCount,
			std::string_view callerName) {
			if (modelParameters.weights.size() != featureCount) {
				throw std::invalid_argument(
					std::string(callerName) +
					": Model weights size does not match feature count!");
			}

			if (false == std::isfinite(modelParameters.bias)) {
				throw std::invalid_argument(
					std::string(callerName) + ": Model bias is NaN or Inf!");
			}

			for (const T weight : modelParameters.weights) {
				if (false == std::isfinite(weight)) {
					throw std::invalid_argument(
						std::string(callerName) + ": Model weight is NaN or Inf!");
				}
			}
		}

		/**
		 * @brief Applies L2 regularization to an averaged gradient.
		 *
		 * @param gradient Averaged (unregularized) gradient.
		 * @param lambda Regularization strength.
		 * @param parameter Current weight or bias being regularized.
		 * @param normalizer Sample count the cost function's
		 * regularization term is scaled by.
		 *
		 * @return gradient + (lambda / normalizer) * parameter.
		 */
		[[nodiscard]] static T regularize(T gradient, T lambda, T parameter, T normalizer) {
			return gradient + (lambda / normalizer) * parameter;
		}
	};

} // namespace detail
