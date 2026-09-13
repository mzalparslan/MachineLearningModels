#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Options for Gradient Descent optimization.
 *
 * @tparam T Floating data mode for optimization parameters.
 */
template <typename T>
struct GradientDescentOptions {
	// Learning rate for the optimization algorithm.
	T learningRate = T(0.01);
	// Number of epochs to run the optimization.
	std::size_t epochs = 100;
	// Strength of L2 regularization applied to model weights.
	T lambda = T(0);
	// Option to apply L2 regularization to bias.
	bool regularizeBias = false;
};

/**
 * @brief Options for Mini-Batch Gradient Descent optimization.
 *
 * @tparam T Floating data mode for optimization parameters.
 */
template <typename T>
struct MiniBatchGradientDescentOptions {
	// Options for the underlying Gradient Descent optimization.
	GradientDescentOptions<T> gradientOptions;
	// Requested number of samples in each mini-batch.
	// Final batch may contain fewer samples.
	std::size_t batchSize = 32;
	// Whether samples are shuffled before each epoch.
	bool shuffle = true;
	// Random seed to be used by Shuffle.
	std::uint32_t randomSeed = 42;
};

/**
 * @brief Options for Stochastic Gradient Descent optimization.
 *
 * @tparam T Floating data mode for optimization parameters.
 */
template <typename T>
struct StochasticGradientDescentOptions {
	// Options for the underlying Gradient Descent optimization.
	GradientDescentOptions<T> gradientOptions;
	// Whether samples are shuffled before each epoch.
	bool shuffle = true;
	// Random seed to be used by Shuffle.
	std::uint32_t randomSeed = 42;
	// Learning-rate decay applied per epoch: effectiveRate =
	// learningRate / (1 + decay * epoch). Zero (default) disables decay,
	// so SGD uses a fixed learning rate as before. Without decay, SGD's
	// fixed step size makes it oscillate around the optimum rather than
	// settling into it; a small positive decay lets later epochs take
	// smaller, more precise steps.
	T decay = T(0);
};