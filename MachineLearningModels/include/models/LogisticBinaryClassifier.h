#pragma once

#include "IBinaryClassifier.h"
#include "DataPoint.h"
#include "ModelParameters.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <functional>
#include <utility>

/**
 * @brief Logistic Regression for binary classification.
 *
 * Learns weight per feature and bias by using configured Optimizer.
 * Model calculates a linear output: logit.
 *
 * @f[
 *     z = b + \sum_{j=1}^{n} w_j x_j
 * @f]
 *
 * Sigmoid converts logit into probability that 
 * sample belongs to the positive class:
 *
 * @f[
 *     P(y = 1 \mid x) = \frac{1}{1 + e^{-z}}
 * @f]
 *
 * Training targets must be either zero or one.
 *
 * @tparam T Floating-point data type used for features, targets,
 * probabilities, and model calculations.
 * 
 * @tparam Optimizer Optimization policy used to learn model parameters.
 */
template <typename T, typename Optimizer>
class LogisticBinaryClassifier final : public IBinaryClassifier<T> {
	// Only floating-point types are supported for model calculations.
	static_assert(
		std::is_floating_point_v<T>,
		"LogisticBinaryClassifier requires a floating-point type.");

public:
	// Options from the optimizer are used to configure the optimization process.
	using Options = typename Optimizer::Options;

	// Default predict called from interface instead of the overridden version.
	using IBinaryClassifier<T>::predict;

	/**
	 * @brief Logistic binary classifier constructor.
	 *
	 * @param optimizer Optimizer used to learn weights and bias.
	 * @param options Configuration supplied to the optimizer.
	 */
	explicit LogisticBinaryClassifier(Optimizer optimizer_, Options options_ = {})
		: optimizer(std::move(optimizer_)), options(options_) {
	}

	/**
	 * @brief Fits classifier using binary-labeled training data.
	 *
	 * Initializes one weight per feature and a bias to zero, then
	 * delegates parameter updates to configured Optimizer.
	 *
	 * Calling this method replaces previously learned parameters.
	 *
	 * @param trainingSet Dataset whose targets are zero or one.
	 *
	 * @throws std::invalid_argument If dataset is empty, contains no
	 * features, has inconsistent feature counts, 
	 * contains non-finite values, or contains non-binary targets.
	 *
	 * @note Exceptions are propagated so caller should handle them.
	 */
	void fit(const std::vector<DataPoint<T>>& trainingSet) override {
		isFitted = false;

		validateTrainingSet(trainingSet); 

		// Initialize model parameters (weights and bias) based on the feature count
		featureCount = trainingSet.front().features.size();
		modelParameters.weights.assign(featureCount, T(0));
		modelParameters.bias = T(0);

		const auto hypothesis =
			[](const std::vector<T>& features,
				const ModelParameters<T>& parameters) -> T {
					return sigmoid(linearOutput(features, parameters));
			};

		// Optimizer updates model parameters (weights and bias).
		optimizer.optimize(
			trainingSet,
			options,
			modelParameters,
			hypothesis);

		isFitted = true;
	}

	/**
	 * @brief Predicts the probability of the positive class.
	 *
	 * @param features Input feature vector.
	 *
	 * @return Probability that the sample belongs to class one.
	 *
	 * @throws std::logic_error If fit() was not called before prediction.
	 * @throws std::invalid_argument If feature count is incorrect
	 * or an input feature is non-finite.
	 * @throws std::runtime_error If calculated logit is non-finite value.
	 */
	[[nodiscard]]
	T predictProbability(const std::vector<T>& features) const override {
		// Validate feature structure, numerical values, and binary targets.
		validateFeatures(features);

		// Convert a logit into a probability using sigmoid.
		return sigmoid(linearOutput(features, modelParameters));
	}

	/**
	 * @brief Predicts a binary class using a probability threshold.
	 *
	 * @param features Input feature vector.
	 * @param threshold Probability threshold for selecting class 1.
	 *
	 * @return true if positive-class probability is greater than or
	 * equal to the threshold; otherwise false.
	 *
	 * @throws std::logic_error Caller didnot call fit() before prediction.
	 * @throws std::invalid_argument If threshold is non-finite or
	 * outside the open interval (0, 1), feature count is incorrect, 
	 * or an input feature is non-finite.
	 * @throws std::runtime_error If calculated logit is non-finite value.
	 */
	[[nodiscard]]
	bool predict(const std::vector<T>& features, T threshold) const override {
		if (false == std::isfinite(threshold) ||
			threshold <= T(0) ||
			threshold >= T(1)) {
			throw std::invalid_argument(
				"LogisticBinaryClassifier::predict: "
				"Threshold must be between zero and one!");
		}

		return predictProbability(features) >= threshold;
	}

private:
	// Calculate raw linear output, also called logit.
	[[nodiscard]]
	static T linearOutput(const std::vector<T>& features, 
		const ModelParameters<T>& parameters) {

		const T logit = std::inner_product(
			parameters.weights.begin(),
			parameters.weights.end(),
			features.begin(),
			parameters.bias);

		if (false == std::isfinite(logit)) {
			throw std::runtime_error(
				"LogisticBinaryClassifier::linearOutput: "
				"Logit is NaN or Inf!");
		}

		return logit;
	}

	// Sigmoid: z = 1 / (1 + exp(-x)) where x >= 0; 
	// otherwise z = exp(x) / (1 + exp(x)) to avoid overflow.
	[[nodiscard]]
	static T sigmoid(T value) {
		if (value >= T(0)) {
			return T(1) /
				(T(1) + std::exp(-value));
		}

		const T exponential = std::exp(value);

		return exponential / (T(1) + exponential);
	}

	// Validate feature structure, numerical values, and binary targets.
	void validateFeatures(const std::vector<T>& features) const {
		if (false == isFitted) {
			throw std::logic_error(
				"LogisticBinaryClassifier::predict: "
				"Call fit() before prediction!");
		}

		if (features.size() != featureCount) {
			throw std::invalid_argument(
				"LogisticBinaryClassifier::predict: "
				"Feature count mismatch!");
		}

		for (const T value : features) {
			if (false == std::isfinite(value)) {
				throw std::invalid_argument(
					"LogisticBinaryClassifier::predict: "
					"Feature is NaN or Inf!");
			}
		}
	}

	// Validate training set structure, numerical values, and binary targets.
	static void validateTrainingSet(const std::vector<DataPoint<T>>& trainingSet) {
		if (true == trainingSet.empty()) {
			throw std::invalid_argument(
				"LogisticBinaryClassifier::fit: "
				"Training set is empty!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();

		if (0 == featureCount) {
			throw std::invalid_argument(
				"LogisticBinaryClassifier::fit: "
				"Training set has no features!");
		}

		for (const auto& sample : trainingSet) {
			if (sample.features.size() != featureCount) {
				throw std::invalid_argument(
					"LogisticBinaryClassifier::fit: "
					"Inconsistent feature count!");
			}

			for (const T value : sample.features) {
				if (false == std::isfinite(value)) {
					throw std::invalid_argument(
						"LogisticBinaryClassifier::fit: "
						"Feature is NaN or Inf!");
				}
			}

			if (false == std::isfinite(sample.target) ||
				(sample.target != T(0) &&
					sample.target != T(1))) {
				throw std::invalid_argument(
					"LogisticBinaryClassifier::fit: "
					"Targets must be zero or one!");
			}
		}
	}

private:
	// Configurable Optimizer used to learn model parameters (weights and bias).
	Optimizer optimizer;
	// Options supplied to the Optimizer for training configuration.
	Options options;
	// Learned model parameters (weights and bias) after fitting.
	ModelParameters<T> modelParameters;

	// Number of features in the training set, used for validation during prediction.
	std::size_t featureCount = 0;
	// Flag indicating whether the model has been fitted with training data.
	bool isFitted = false;
};