#pragma once

#include "BaseLinearModel.h"
#include "OptimizationPolicy.h"

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
 * @tparam T Floating-point data mode used for features, targets,
 * probabilities, and model calculations.
 * 
 * @tparam Optimizer Optimization policy used to learn model parameters.
 */
template <typename T, OptimizationPolicy<T> Optimizer>
class LogisticBinaryClassifier final : public BaseLinearModel<T, Optimizer> {

	// Options from Optimizer used to configure optimization process.
	using Options = typename Optimizer::Options;
	// Linear output (logit) calculation from BaseLinearModel.
	using BaseLinearModel<T, Optimizer>::linearOutput;
	// Validation of training set from BaseLinearModel.
	using BaseLinearModel<T, Optimizer>::validateTrainingSet;

public:
	/**
	 * @brief Logistic binary classifier constructor.
	 *
	 * @param optimizer Optimizer used to learn weights and bias.
	 * @param options Configuration supplied to the optimizer.
	 * @param executionStrategy Strategy used to calculate linear outputs
	 */
	explicit LogisticBinaryClassifier(Optimizer optimizer_, 
		Options options_ = {}, 
		ExecutionStrategy<T> executionStrategy_ = {})
		: BaseLinearModel<T, Optimizer>(std::move(optimizer_),
			std::move(options_), 
			std::move(executionStrategy_)) {
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
		this->isFitted = false;

		validateTrainingSet(trainingSet); 

		for (const auto& sample : trainingSet) {
			if (sample.target != T(0) && sample.target != T(1)) {
				throw std::invalid_argument(
					"LogisticBinaryClassifier::fit: "
					"Training target must be zero or one!");
			}
		}

		// Initialize model parameters (weights and bias) based on the feature count
		this->featureCount = trainingSet.front().features.size();
		this->modelParameters.weights.assign(this->featureCount, T(0));
		this->modelParameters.bias = T(0);

		const auto hypothesis =
			[this](const std::vector<T>& features,
				const ModelParameters<T>& parameters) -> T {
					return sigmoid(this->linearOutput(features, parameters));
			};

		// Optimizer updates model parameters (weights and bias).
		this->optimizer.optimize(
			trainingSet,
			this->options,
			this->modelParameters,
			hypothesis);

		this->isFitted = true;
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
	T predict(const std::vector<T>& features) const override {
		// Validate feature structure, numerical values, and binary targets.
		this->validateFeatures(features);

		// Convert a logit into a probability using sigmoid.
		return sigmoid(this->linearOutput(features, this->modelParameters));
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
	bool predictClass(const std::vector<T>& features, T threshold = T(0.5)) const {
		return classify(predict(features), threshold);
	}

	/**
	 * @brief Classifies an already-computed probability against a threshold.
	 *
	 * This is the decision rule predictClass() applies after calling
	 * predict() -- exposed separately so a caller that already has the
	 * probability in hand (e.g. to also report it, as
	 * BinaryClassificationWriter and BinaryClassificationPipeline::evaluate
	 * do) can classify it without a second, redundant scale-and-predict
	 * pass, while still going through the same rule as predictClass()
	 * rather than re-implementing it.
	 *
	 * @param probability Positive-class probability, normally from predict().
	 * @param threshold Probability threshold for selecting class 1.
	 *
	 * @return true if probability is greater than or equal to threshold;
	 * otherwise false.
	 *
	 * @throws std::invalid_argument If threshold is non-finite or outside
	 * the open interval (0, 1).
	 */
	[[nodiscard]]
	static bool classify(T probability, T threshold = T(0.5)) {
		if (false == std::isfinite(threshold) ||
			threshold <= T(0) ||
			threshold >= T(1)) {
			throw std::invalid_argument(
				"LogisticBinaryClassifier::classify: "
				"Threshold must be between zero and one!");
		}

		return probability >= threshold;
	}

private:
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

};