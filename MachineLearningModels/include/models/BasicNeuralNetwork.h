#pragma once

#include "ActivationFunctions.h"
#include "DataPoint.h"
#include "GradientDescentValidation.h"
#include "IRegressionModel.h"
#include "NeuralNetworkOptions.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Two-layer neural network (one hidden layer) with sigmoid
 * activations.
 *
 * Architecture: Input -> Hidden layer (sigmoid) -> Output unit (sigmoid).
 *
 * @f[
 *     Z_1 = X W_1 + b_1, \quad A_1 = \sigma(Z_1)
 * @f]
 * @f[
 *     Z_2 = A_1 W_2 + b_2, \quad \hat{y} = \sigma(Z_2)
 * @f]
 *
 * Weight matrices are flattened to 1D vectors: W[i * n_out + j] is the
 * weight from input unit i to neuron j, keeping row-major access
 * contiguous during both the forward and backward pass.
 *
 * Trained with per-sample stochastic gradient descent using
 * backpropagation and binary cross-entropy loss; NeuralNetworkOptions
 * configures the learning rate, epoch count, and initialization seed.
 * Targets are expected to be zero or one, as for
 * LogisticBinaryClassifier, though this is not enforced.
 *
 * @tparam T Floating-point mode used for features, targets, weights,
 * and calculations.
 */
template <typename T>
class BasicNeuralNetwork final : public IRegressionModel<T> {
	static_assert(std::is_floating_point_v<T>,
		"BasicNeuralNetwork requires a floating-point mode!");

	// Weights (flattened n_in * n_out) and bias (n_out) for one layer.
	struct LayerParameters {
		std::vector<T> weights;
		std::vector<T> bias;
	};

public:
	/**
	 * @brief Constructs a two-layer network with the given hidden size.
	 *
	 * @param hiddenSize Number of neurons in the hidden layer.
	 * @param options Learning rate, epoch count, and random seed used by fit().
	 *
	 * @throws std::invalid_argument If hiddenSize is zero, or options are
	 * invalid (see NeuralNetworkOptions).
	 */
	explicit BasicNeuralNetwork(std::size_t hiddenSize, NeuralNetworkOptions<T> options = {})
		: hiddenSize(hiddenSize), options(options) {
		if (0 == hiddenSize) {
			throw std::invalid_argument(
				"BasicNeuralNetwork::BasicNeuralNetwork: Hidden layer size must be greater than zero!");
		}

		detail::validateNeuralNetworkOptions(this->options, "BasicNeuralNetwork::BasicNeuralNetwork");
	}

	~BasicNeuralNetwork() override = default;

	/**
	 * @brief Fits the network using backpropagation.
	 *
	 * Re-initializes weights (Xavier/Glorot-uniform) and zeroes biases
	 * before training, so calling this method replaces any previously
	 * learned parameters.
	 *
	 * @param trainingSet Dataset used to learn model parameters.
	 *
	 * @throws std::invalid_argument If the dataset is empty, contains no
	 * features, has inconsistent feature counts, or contains non-finite
	 * values.
	 */
	void fit(const std::vector<DataPoint<T>>& trainingSet) override {
		isFitted = false;

		detail::GradientDescentValidation<T>::validateTrainingSet(
			trainingSet, "BasicNeuralNetwork::fit");

		featureCount = trainingSet.front().features.size();
		initializeParameters();

		for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
			for (const DataPoint<T>& sample : trainingSet) {
				trainOnSample(sample);
			}
		}

		isFitted = true;
	}

	/**
	 * @brief Predicts the sigmoid output for one feature vector.
	 *
	 * @param features Input features.
	 *
	 * @return Predicted value in the open interval (0, 1).
	 *
	 * @throws std::logic_error If fit() was not called before prediction.
	 * @throws std::invalid_argument If feature count is incorrect or a
	 * feature value is non-finite.
	 */
	[[nodiscard]]
	T predict(const std::vector<T>& features) const override {
		validateFeatures(features);
		return forward(features);
	}

private:
	void validateFeatures(const std::vector<T>& features) const {
		if (false == isFitted) {
			throw std::logic_error(
				"BasicNeuralNetwork::validateFeatures: Call fit() before predict()!");
		}

		if (features.size() != featureCount) {
			throw std::invalid_argument(
				"BasicNeuralNetwork::validateFeatures: Feature count mismatch!");
		}

		for (const T feature : features) {
			if (false == std::isfinite(feature)) {
				throw std::invalid_argument(
					"BasicNeuralNetwork::validateFeatures: Feature value is NaN or Inf!");
			}
		}
	}

	// Forward pass using contiguous, pointer-based matrix-vector products.
	// If hiddenActivations is provided, stores A1 for use by backprop.
	[[nodiscard]]
	T forward(const std::vector<T>& x, std::vector<T>* hiddenActivations = nullptr) const {
		std::vector<T> Z1 = layer1.bias;
		const T* wPtr = layer1.weights.data();

		for (std::size_t i = 0; i < featureCount; ++i) {
			const T inputVal = x[i];
			const T* rowW = wPtr + (i * hiddenSize);

			for (std::size_t j = 0; j < hiddenSize; ++j) {
				Z1[j] += inputVal * rowW[j];
			}
		}

		std::vector<T> A1(hiddenSize);
		for (std::size_t j = 0; j < hiddenSize; ++j) {
			A1[j] = Sigmoid(Z1[j]);
		}

		T Z2 = layer2.bias[0];
		const T* w2Ptr = layer2.weights.data();
		for (std::size_t j = 0; j < hiddenSize; ++j) {
			Z2 += A1[j] * w2Ptr[j];
		}

		if (nullptr != hiddenActivations) {
			*hiddenActivations = std::move(A1);
		}

		return Sigmoid(Z2);
	}

	// One SGD step (forward + backpropagation + parameter update) for one sample.
	void trainOnSample(const DataPoint<T>& sample) {
		std::vector<T> A1;
		const T A2 = forward(sample.features, &A1);

		// dL/dZ2 for a sigmoid output with binary cross-entropy loss: A2 - y.
		const T dZ2 = A2 - sample.target;

		// dZ1[j] = (dZ2 * W2[j]) * SigmoidDerivative(A1[j])
		std::vector<T> dZ1(hiddenSize);
		for (std::size_t j = 0; j < hiddenSize; ++j) {
			dZ1[j] = (dZ2 * layer2.weights[j]) * SigmoidDerivative(A1[j]);
		}

		// Update output layer. Uses A1 computed before this update, not
		// weights, so ordering relative to the hidden-layer update below
		// does not matter.
		for (std::size_t j = 0; j < hiddenSize; ++j) {
			layer2.weights[j] -= options.learningRate * dZ2 * A1[j];
		}
		layer2.bias[0] -= options.learningRate * dZ2;

		// Update hidden layer.
		T* w1Mutable = layer1.weights.data();
		const T* xPtr = sample.features.data();

		for (std::size_t i = 0; i < featureCount; ++i) {
			const T lr_x = options.learningRate * xPtr[i];
			T* rowW = w1Mutable + (i * hiddenSize);

			for (std::size_t j = 0; j < hiddenSize; ++j) {
				rowW[j] -= lr_x * dZ1[j];
			}
		}

		for (std::size_t j = 0; j < hiddenSize; ++j) {
			layer1.bias[j] -= options.learningRate * dZ1[j];
		}
	}

	// Initializes both layers with Xavier/Glorot-uniform weights and zero biases.
	void initializeParameters() {
		std::mt19937 gen(options.randomSeed);

		initializeLayer(layer1, featureCount, hiddenSize, gen);
		initializeLayer(layer2, hiddenSize, 1, gen);
	}

	static void initializeLayer(
		LayerParameters& layer, std::size_t fanIn, std::size_t fanOut, std::mt19937& gen) {
		const T limit = std::sqrt(T(6) / T(fanIn + fanOut));
		std::uniform_real_distribution<T> distribution(-limit, limit);

		layer.weights.resize(fanIn * fanOut);
		for (T& weight : layer.weights) {
			weight = distribution(gen);
		}

		layer.bias.assign(fanOut, T(0));
	}

	std::size_t hiddenSize;
	NeuralNetworkOptions<T> options;

	std::size_t featureCount = 0;
	LayerParameters layer1; // Input -> Hidden
	LayerParameters layer2; // Hidden -> Output
	bool isFitted = false;
};
