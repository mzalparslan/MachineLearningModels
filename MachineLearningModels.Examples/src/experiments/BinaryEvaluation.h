#pragma once

#include "DataPoint.h"
#include "metrics.h"

#include <vector>

/**
 * @brief Scores a fitted binary classifier on a test set.
 *
 * Classifies each sample at a 0.5 probability threshold and computes
 * accuracy, precision, recall, F1, and binary cross-entropy.
 *
 * @tparam T Floating-point mode used for model calculations.
 * @tparam Model Any model whose predict() returns a positive-class probability.
 */
template <typename T, typename Model>
[[nodiscard]]
BinaryClassificationMetrics scoreBinaryModel(
	const Model& model,
	const std::vector<DataPoint<T>>& testSet) {

	std::vector<T> probabilities;
	std::vector<T> targets;
	std::vector<bool> predictedClasses;
	std::vector<bool> expectedClasses;

	for (const auto& sample : testSet) {
		const T probability = model.predict(sample.features);

		probabilities.push_back(probability);
		targets.push_back(sample.target);
		predictedClasses.push_back(probability >= T(0.5));
		expectedClasses.push_back(sample.target >= T(0.5));
	}

	BinaryClassificationMetrics result =
		metrics::classificationRates(predictedClasses, expectedClasses);
	result.binaryCrossEntropy = metrics::binaryCrossEntropy(probabilities, targets);

	return result;
}
