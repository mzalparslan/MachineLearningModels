#pragma once

#include "DataPoint.h"
#include "DeepNeuralNetwork.h"
#include "Logger.h"
#include "NeuralNetworkOptions.h"
#include "ScopedBenchmarkTimer.h"

#include <cstddef>
#include <type_traits>
#include <vector>

/**
 * @brief Trains a small DeepNeuralNetwork on XOR and reports its predictions.
 *
 * XOR is not linearly separable, so it cannot be learned by
 * LinearRegression or LogisticBinaryClassifier -- it exists to
 * demonstrate what a hidden layer buys over a plain linear model.
 *
 * @tparam T Floating-point mode used for model calculations.
 */
template <typename T>
void runNeuralNetwork(Logger& logger) {
	static_assert(std::is_floating_point_v<T>,
		"runNeuralNetwork requires floating point data mode!");

	ScopedBenchmarkTimer benchmark(logger, "runNeuralNetwork");

	logger.info() << "Training DeepNeuralNetwork on XOR.";

	NeuralNetworkOptions<T> options;
	options.learningRate = T(0.5);
	options.epochs = 20000;

	DeepNeuralNetwork<T> model(std::vector<std::size_t>{ 2, 4, 1 }, options);

	const std::vector<DataPoint<T>> trainingSet = {
		DataPoint<T>{ { T(0), T(0) }, T(0) },
		DataPoint<T>{ { T(0), T(1) }, T(1) },
		DataPoint<T>{ { T(1), T(0) }, T(1) },
		DataPoint<T>{ { T(1), T(1) }, T(0) }
	};

	model.fit(trainingSet);

	for (const auto& sample : trainingSet) {
		const T prediction = model.predict(sample.features);
		logger.info() << sample.features[0] << " XOR " << sample.features[1]
			<< " = " << prediction << " (expected " << sample.target << ")";
	}
}
