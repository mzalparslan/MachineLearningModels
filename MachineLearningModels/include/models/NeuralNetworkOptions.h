#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

/**
 * @brief Options for training a neural network by backpropagation.
 *
 * Neural network parameters are shaped as a stack of weight matrices and
 * bias vectors rather than a single weight vector and bias, so they don't
 * fit ModelParameters or the OptimizationPolicy concept built around it
 * (see OptimizationPolicy.h). BasicNeuralNetwork and DeepNeuralNetwork
 * instead run their own
 * per-sample backpropagation loop, configured by this options type.
 *
 * @tparam T Floating-point mode used for training.
 */
template <typename T>
struct NeuralNetworkOptions {
	static_assert(std::is_floating_point_v<T>,
		"NeuralNetworkOptions requires a floating-point mode!");

	// Learning rate used to scale gradient updates.
	T learningRate = T(0.1);

	// Number of passes over the full training set.
	std::size_t epochs = 1000;

	// Random seed used to initialize weights, so fit() is reproducible.
	std::uint32_t randomSeed = 42;
};

namespace detail {

	// Validates the options shared by BasicNeuralNetwork and DeepNeuralNetwork.
	template <typename T>
	void validateNeuralNetworkOptions(
		const NeuralNetworkOptions<T>& options, std::string_view callerName) {
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
	}

} // namespace detail
