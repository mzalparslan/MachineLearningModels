#pragma once

#include <cstddef>

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
	// Mean Absolute Percentage Error (MAPE) of model predictions.
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

	/**
	 * @brief Calculates the binary cross-entropy loss for a given logit and target.
	 * @param logit Logit value.
	 * @param target Target value (0 or 1).
	 * @return Binary cross-entropy loss.
	 */
	template <typename T>
	static T binaryCrossEntropyFromLogit(T logit, T target) {
		return std::max(logit, T(0))
			- logit * target
			+ std::log1p(std::exp(-std::abs(logit)));
	}
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