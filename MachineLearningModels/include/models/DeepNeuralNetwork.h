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
#include <utility>
#include <vector>

/**
 * @brief Fully-connected feedforward neural network with an arbitrary
 * number of sigmoid layers.
 *
 * Topology is given as {inputSize, hidden_1, ..., hidden_k, outputSize}.
 * Layer l connects topology[l-1] units to topology[l] units:
 *
 * @f[
 *     Z_l = A_{l-1} W_l + b_l, \quad A_l = \sigma(Z_l), \quad A_0 = X
 * @f]
 *
 * The output layer (topology.back()) must have exactly one unit, whose
 * activation A_L is returned as the model's prediction.
 *
 * Weight matrices are flattened to 1D vectors: W[i * n_out + j] is the
 * weight from unit i in the previous layer to unit j in this layer,
 * keeping row-major access contiguous during both the forward and
 * backward pass.
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
class DeepNeuralNetwork final : public IRegressionModel<T> {
	static_assert(std::is_floating_point_v<T>,
		"DeepNeuralNetwork requires a floating-point mode!");

	struct LayerParameters {
		std::vector<T> weights; // Flattened weights matrix (n_in * n_out).
		std::vector<T> bias;    // Bias vector (n_out).
		std::size_t n_in;
		std::size_t n_out;
	};

public:
	/**
	 * @brief Constructs a network with the given layer topology.
	 *
	 * @param topology Unit count per layer: {inputSize, hidden..., outputSize}.
	 * @param options Learning rate, epoch count, and random seed used by fit().
	 *
	 * @throws std::invalid_argument If topology has fewer than two
	 * entries, any layer has zero units, the output layer does not have
	 * exactly one unit, or options are invalid (see NeuralNetworkOptions).
	 */
	explicit DeepNeuralNetwork(std::vector<std::size_t> topology, NeuralNetworkOptions<T> options = {})
		: topology(std::move(topology)), options(options) {
		if (this->topology.size() < 2) {
			throw std::invalid_argument(
				"DeepNeuralNetwork::DeepNeuralNetwork: Topology must have at least an input and an output layer!");
		}

		for (const std::size_t layerSize : this->topology) {
			if (0 == layerSize) {
				throw std::invalid_argument(
					"DeepNeuralNetwork::DeepNeuralNetwork: Every layer must have at least one unit!");
			}
		}

		if (1 != this->topology.back()) {
			throw std::invalid_argument(
				"DeepNeuralNetwork::DeepNeuralNetwork: Output layer must have exactly one unit!");
		}

		detail::validateNeuralNetworkOptions(this->options, "DeepNeuralNetwork::DeepNeuralNetwork");
	}

	~DeepNeuralNetwork() override = default;

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
	 * features, has inconsistent feature counts, contains non-finite
	 * values, or its feature count does not match the input layer size.
	 */
	void fit(const std::vector<DataPoint<T>>& trainingSet) override {
		isFitted = false;

		detail::GradientDescentValidation<T>::validateTrainingSet(
			trainingSet, "DeepNeuralNetwork::fit");

		const std::size_t trainingFeatureCount = trainingSet.front().features.size();
		if (trainingFeatureCount != topology.front()) {
			throw std::invalid_argument(
				"DeepNeuralNetwork::fit: Training set feature count does not match input layer size!");
		}

		featureCount = trainingFeatureCount;
		initializeParameters();

		for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
			for (const DataPoint<T>& sample : trainingSet) {
				trainOnSample(sample);
			}
		}

		isFitted = true;
	}

	/**
	 * @brief Predicts the network's output for one feature vector.
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
		return forwardProp(features);
	}

private:
	void validateFeatures(const std::vector<T>& features) const {
		if (false == isFitted) {
			throw std::logic_error(
				"DeepNeuralNetwork::validateFeatures: Call fit() before predict()!");
		}

		if (features.size() != featureCount) {
			throw std::invalid_argument(
				"DeepNeuralNetwork::validateFeatures: Feature count mismatch!");
		}

		for (const T feature : features) {
			if (false == std::isfinite(feature)) {
				throw std::invalid_argument(
					"DeepNeuralNetwork::validateFeatures: Feature value is NaN or Inf!");
			}
		}
	}

	// Forward pass across all layers. If cache_A/cache_Z are provided,
	// stores every layer's activations/pre-activations for backprop.
	// cache_A[0] = input features, cache_A[l] = output of layer l.
	[[nodiscard]]
	T forwardProp(
		const std::vector<T>& features,
		std::vector<std::vector<T>>* cache_A = nullptr,
		std::vector<std::vector<T>>* cache_Z = nullptr) const {

		std::vector<T> A_prev = features;
		if (nullptr != cache_A) {
			cache_A->push_back(A_prev);
		}

		for (const LayerParameters& layer : layers) {
			std::vector<T> Z = layer.bias;
			const T* wPtr = layer.weights.data();
			const T* aPrevPtr = A_prev.data();

			for (std::size_t i = 0; i < layer.n_in; ++i) {
				const T aVal = aPrevPtr[i];
				const T* rowW = wPtr + (i * layer.n_out);

				for (std::size_t j = 0; j < layer.n_out; ++j) {
					Z[j] += aVal * rowW[j];
				}
			}

			if (nullptr != cache_Z) {
				cache_Z->push_back(Z);
			}

			std::vector<T> A(layer.n_out);
			for (std::size_t j = 0; j < layer.n_out; ++j) {
				A[j] = Sigmoid(Z[j]);
			}

			if (nullptr != cache_A) {
				cache_A->push_back(A);
			}

			A_prev = std::move(A);
		}

		return A_prev[0];
	}

	// One SGD step (forward + backpropagation + parameter update) for one sample.
	void trainOnSample(const DataPoint<T>& sample) {
		std::vector<std::vector<T>> A_cache;
		std::vector<std::vector<T>> Z_cache;
		A_cache.reserve(layers.size() + 1);
		Z_cache.reserve(layers.size());

		const T prediction = forwardProp(sample.features, &A_cache, &Z_cache);

		// Output layer has exactly one unit; for a sigmoid output with
		// binary cross-entropy loss, dL/dZ_L = A_L - y.
		std::vector<T> dZ{ prediction - sample.target };

		for (std::size_t idx = layers.size(); idx-- > 0; ) {
			LayerParameters& layer = layers[idx];
			const std::vector<T>& A_prev = A_cache[idx]; // A[l - 1]

			// dZ_prev[i] = Sum_j (dZ[j] * W[i][j]) * SigmoidDerivative(A_prev[i])
			std::vector<T> dZ_prev;
			if (idx > 0) {
				dZ_prev.assign(layer.n_in, T(0));
				const T* wPtr = layer.weights.data();

				for (std::size_t i = 0; i < layer.n_in; ++i) {
					T sum = T(0);
					const T* rowW = wPtr + (i * layer.n_out);

					for (std::size_t j = 0; j < layer.n_out; ++j) {
						sum += dZ[j] * rowW[j];
					}

					dZ_prev[i] = sum * SigmoidDerivative(A_prev[i]);
				}
			}

			// Update weights and bias for this layer using A_prev
			// (computed before this update), so ordering relative to
			// dZ_prev above does not matter.
			T* wMutable = layer.weights.data();
			for (std::size_t i = 0; i < layer.n_in; ++i) {
				const T lr_a = options.learningRate * A_prev[i];
				T* rowW = wMutable + (i * layer.n_out);

				for (std::size_t j = 0; j < layer.n_out; ++j) {
					rowW[j] -= lr_a * dZ[j];
				}
			}

			for (std::size_t j = 0; j < layer.n_out; ++j) {
				layer.bias[j] -= options.learningRate * dZ[j];
			}

			if (idx > 0) {
				dZ = std::move(dZ_prev);
			}
		}
	}

	// Initializes every layer with Xavier/Glorot-uniform weights and zero biases.
	void initializeParameters() {
		std::mt19937 gen(options.randomSeed);

		layers.clear();
		layers.reserve(topology.size() - 1);

		for (std::size_t l = 1; l < topology.size(); ++l) {
			const std::size_t fanIn = topology[l - 1];
			const std::size_t fanOut = topology[l];

			LayerParameters layer;
			layer.n_in = fanIn;
			layer.n_out = fanOut;

			const T limit = std::sqrt(T(6) / T(fanIn + fanOut));
			std::uniform_real_distribution<T> distribution(-limit, limit);

			layer.weights.resize(fanIn * fanOut);
			for (T& weight : layer.weights) {
				weight = distribution(gen);
			}

			layer.bias.assign(fanOut, T(0));

			layers.push_back(std::move(layer));
		}
	}

	std::vector<std::size_t> topology;
	NeuralNetworkOptions<T> options;

	std::size_t featureCount = 0;
	std::vector<LayerParameters> layers;
	bool isFitted = false;
};
