#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @brief Metrics to evaluate regression model performance.
 */
struct RegressionMetrics {
	// Mean Squared Error (MSE) of model predictions.
	double mse = 0;
	// Root Mean Squared Error (RMSE) of model predictions.
	double rmse = 0;
	// Mean Absolute Error (MAE) of model predictions.
	double mae = 0;
	// R-squared (coefficient of determination) of model predictions.
	double rSquared = 0;
	// Fraction of predictions within a tolerance of their target.
	//
	// This is not MAPE: Mean Absolute Percentage Error is undefined
	// whenever a target is zero, which this tolerance-hit-rate variant
	// avoids by testing |prediction - target| <= tolerance * |target|
	// only for non-zero targets and |prediction - target| <= tolerance
	// for zero targets. See metrics::withinToleranceRatio.
	double withinToleranceRatio = 0;
};

/**
 * @brief Metrics to evaluate binary classification model performance.
 */
struct BinaryClassificationMetrics {
	// BCE = -1/N * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
	double binaryCrossEntropy = 0;
	// Accuracy = (TP + TN) / (TP + TN + FP + FN) where
	// TP = True Positives, TN = True Negatives,
	// FP = False Positives, FN = False Negatives
	double accuracy = 0;
	// Precision = TP / (TP + FP) where
	// TP = True Positives, FP = False Positives
	double precision = 0;
	// Recall = TP / (TP + FN) where
	// TP = True Positives, FN = False Negatives
	double recall = 0;
	// f1Score = 2 * (precision * recall) / (precision + recall)
	double f1Score = 0;
};

/**
 * @brief Metrics to evaluate optimization performance.
 */
struct OptimizationMetrics {
	// Final value of objective function after optimization.
	double objectiveCost = 0;
	// Norm of gradient at final iteration of optimization.
	double gradientNorm = 0;
	// Number of epochs completed during optimization.
	std::size_t completedEpochs = 0;
	// Total time taken for optimization process in milliseconds.
	double elapsedMilliseconds = 0;
};

/**
 * @brief Free functions computing RegressionMetrics, BinaryClassificationMetrics,
 * and related quantities from raw predictions and targets.
 *
 * Kept as free functions rather than members of the metric structs above:
 * the structs are plain data, and these computations are what fill them in.
 */
namespace metrics {

	/**
	 * @brief Calculates the numerically stable binary cross-entropy loss
	 * for a given logit and target, without first converting the logit
	 * to a probability.
	 *
	 * @f[
	 *     \max(z, 0) - z y + \log(1 + e^{-|z|})
	 * @f]
	 *
	 * @param logit Logit value (pre-sigmoid linear output).
	 * @param target Target value (0 or 1).
	 *
	 * @return Binary cross-entropy loss.
	 */
	template <typename T>
	[[nodiscard]]
	T binaryCrossEntropyFromLogit(T logit, T target) {
		return std::max(logit, T(0))
			- logit * target
			+ std::log1p(std::exp(-std::abs(logit)));
	}

	/**
	 * @brief Calculates mean squared error between predictions and targets.
	 *
	 * @throws std::invalid_argument If the vectors are empty or differ in size.
	 */
	template <typename T>
	[[nodiscard]]
	double meanSquaredError(const std::vector<T>& predictions, 
							const std::vector<T>& targets) {
		if (predictions.size() != targets.size() || predictions.empty()) {
			throw std::invalid_argument(
				"metrics::meanSquaredError: "
				"Predictions and targets must be non-empty and equal in size!");
		}

		double sumSquaredError = 0.0;
		for (std::size_t i = 0; i < predictions.size(); ++i) {
			const double error = 
				static_cast<double>(predictions[i]) - static_cast<double>(targets[i]);
			sumSquaredError += error * error;
		}

		return sumSquaredError / static_cast<double>(predictions.size());
	}

	/**
	 * @brief Calculates root mean squared error between predictions and targets.
	 *
	 * @throws std::invalid_argument If the vectors are empty or differ in size.
	 */
	template <typename T>
	[[nodiscard]]
	double rootMeanSquaredError(const std::vector<T>& predictions,
								const std::vector<T>& targets) {
		return std::sqrt(meanSquaredError(predictions, targets));
	}

	/**
	 * @brief Calculates mean absolute error between predictions and targets.
	 *
	 * @throws std::invalid_argument If the vectors are empty or differ in size.
	 */
	template <typename T>
	[[nodiscard]]
	double meanAbsoluteError(const std::vector<T>& predictions,
							 const std::vector<T>& targets) {
		if (predictions.size() != targets.size() || predictions.empty()) {
			throw std::invalid_argument(
				"metrics::meanAbsoluteError: "
				"Predictions and targets must be non-empty and equal in size!");
		}

		double sumAbsoluteError = 0.0;
		for (std::size_t i = 0; i < predictions.size(); ++i) {
			sumAbsoluteError += std::abs(
				static_cast<double>(predictions[i]) - static_cast<double>(targets[i]));
		}

		return sumAbsoluteError / static_cast<double>(predictions.size());
	}

	/**
	 * @brief Calculates R-squared (coefficient of determination).
	 *
	 * Returns 0.0 when every target is identical, since the total sum of
	 * squares is then zero and R-squared is conventionally undefined.
	 *
	 * @throws std::invalid_argument If the vectors are empty or differ in size.
	 */
	template <typename T>
	[[nodiscard]]
	double rSquared(const std::vector<T>& predictions,
					const std::vector<T>& targets) {
		if (predictions.size() != targets.size() || predictions.empty()) {
			throw std::invalid_argument(
				"metrics::rSquared: "
				"Predictions and targets must be non-empty and equal in size!");
		}

		double targetSum = 0.0;
		for (const T target : targets) {
			targetSum += static_cast<double>(target);
		}
		const double targetMean = targetSum / static_cast<double>(targets.size());

		double totalSumOfSquares = 0.0;
		double residualSumOfSquares = 0.0;
		for (std::size_t i = 0; i < predictions.size(); ++i) {
			const double target = static_cast<double>(targets[i]);
			const double residual = target - static_cast<double>(predictions[i]);
			residualSumOfSquares += residual * residual;

			const double deviation = target - targetMean;
			totalSumOfSquares += deviation * deviation;
		}

		if (0.0 == totalSumOfSquares) {
			return 0.0;
		}

		return 1.0 - (residualSumOfSquares / totalSumOfSquares);
	}

	/**
	 * @brief Calculates the fraction of predictions within a tolerance of
	 * their target: |prediction - target| <= tolerance * max(|target|, 1).
	 *
	 * This is a tolerance-hit-rate, not MAPE: MAPE is undefined when a
	 * target is zero, which this metric sidesteps by falling back to an
	 * absolute tolerance for zero targets instead of a relative one.
	 *
	 * @param tolerance Relative tolerance, e.g. 0.1 for "within 10%".
	 *
	 * @throws std::invalid_argument If the vectors are empty, differ in
	 * size, or tolerance is negative.
	 */
	template <typename T>
	[[nodiscard]]
	double withinToleranceRatio(const std::vector<T>& predictions,
								const std::vector<T>& targets,
								T tolerance = T(0.1)) {
		if (predictions.size() != targets.size() || predictions.empty()) {
			throw std::invalid_argument(
				"metrics::withinToleranceRatio: "
				"Predictions and targets must be non-empty and equal in size!");
		}

		if (tolerance < T(0)) {
			throw std::invalid_argument(
				"metrics::withinToleranceRatio: Tolerance must be non-negative!");
		}

		std::size_t withinTolerance = 0;
		for (std::size_t i = 0; i < predictions.size(); ++i) {
			const double target = static_cast<double>(targets[i]);
			const double allowedError =
				static_cast<double>(tolerance) * std::max(std::abs(target), 1.0);
			const double error =
				std::abs(static_cast<double>(predictions[i]) - target);

			if (error <= allowedError) {
				++withinTolerance;
			}
		}

		return static_cast<double>(withinTolerance) /
			static_cast<double>(predictions.size());
	}

	/**
	 * @brief Calculates mean binary cross-entropy from predicted probabilities.
	 *
	 * @param probabilities Predicted probability of the positive class.
	 * @param targets Target class (0 or 1).
	 *
	 * @throws std::invalid_argument If the vectors are empty or differ in size.
	 */
	template <typename T>
	[[nodiscard]]
	double binaryCrossEntropy(const std::vector<T>& probabilities, 
							  const std::vector<T>& targets) {
		if (probabilities.size() != targets.size() || probabilities.empty()) {
			throw std::invalid_argument(
				"metrics::binaryCrossEntropy: "
				"Probabilities and targets must be non-empty and equal in size!");
		}

		// Clamp away from 0 and 1 so log() never receives a zero argument.
		constexpr double epsilon = 1e-12;

		double sumLoss = 0.0;
		for (std::size_t i = 0; i < probabilities.size(); ++i) {
			const double probability = std::clamp(
				static_cast<double>(probabilities[i]), epsilon, 1.0 - epsilon);
			const double target = static_cast<double>(targets[i]);

			sumLoss -=
				(target * std::log(probability)) +
				((1.0 - target) * std::log(1.0 - probability));
		}

		return sumLoss / static_cast<double>(probabilities.size());
	}

	/**
	 * @brief Calculates accuracy, precision, recall, and F1 score from
	 * predicted and expected binary classes.
	 *
	 * Precision and recall are defined as 0.0 when their denominator
	 * (no predicted positives, or no actual positives) is zero, and F1 is
	 * defined as 0.0 when precision and recall are both zero.
	 *
	 * @throws std::invalid_argument If the vectors are empty or differ in size.
	 */
	[[nodiscard]]
	inline BinaryClassificationMetrics classificationRates(
						const std::vector<bool>& predictedClasses,
						const std::vector<bool>& expectedClasses) {
		if (predictedClasses.size() != expectedClasses.size() ||
			predictedClasses.empty()) {
			throw std::invalid_argument(
				"metrics::classificationRates: "
				"Predicted and expected classes must be non-empty and equal in size!");
		}

		std::size_t truePositives = 0;
		std::size_t trueNegatives = 0;
		std::size_t falsePositives = 0;
		std::size_t falseNegatives = 0;

		for (std::size_t i = 0; i < predictedClasses.size(); ++i) {
			if (true == predictedClasses[i] && true == expectedClasses[i]) {
				++truePositives;
			}
			else if (false == predictedClasses[i] && false == expectedClasses[i]) {
				++trueNegatives;
			}
			else if (true == predictedClasses[i] && false == expectedClasses[i]) {
				++falsePositives;
			}
			else {
				++falseNegatives;
			}
		}

		BinaryClassificationMetrics result;

		result.accuracy = static_cast<double>(truePositives + trueNegatives) /
			static_cast<double>(predictedClasses.size());

		const std::size_t predictedPositives = truePositives + falsePositives;
		result.precision = (0 == predictedPositives) ? 0.0 :
			static_cast<double>(truePositives) / static_cast<double>(predictedPositives);

		const std::size_t actualPositives = truePositives + falseNegatives;
		result.recall = (0 == actualPositives) ? 0.0 :
			static_cast<double>(truePositives) / static_cast<double>(actualPositives);

		result.f1Score = (0.0 == result.precision && 0.0 == result.recall) ? 0.0 :
			(2.0 * result.precision * result.recall) / (result.precision + result.recall);

		return result;
	}

} // namespace metrics
