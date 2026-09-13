#pragma once

#include "DataPoint.h"

#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>

/**
 * @brief Scales features by using Z-score scaling.
 *
 * Learns mean and standard deviation of each feature
 * from training set.
 *
 * Formula:
 * @f[
 *     z = \frac{x - \mu}{\sigma}
 * @f]
 *
 * Constant features' standard deviation is set to 1 to avoid division by zero.
 * Their observed constant value is mapped to 0 after scaling.
 *
 * @tparam T Floating-point mode used for all values and calculations.
 *
 */
template <typename T>
class ZScoreScaler {
	static_assert(std::is_floating_point_v<T>,
		"ZScoreScaler requires floating data mode T!");

public:
	/**
	* @brief Learns feature means and standard deviations.
	*
	* @param trainingSet Training dataset used to calculate statistics.
	*
	* @throws std::invalid_argument If dataset is empty, contains
	* no features, has inconsistent feature counts, or contains non-finite values.
	* @throws std::runtime_error If calculated mean, variance 
	* or standard deviation is non-finite.
	*/
	void fit(const std::vector<DataPoint<T>>& trainingSet) {
		// Reset flag.
		isFitted = false;

		// Number of training examples (rows) in training set.
		const std::size_t exampleCount = trainingSet.size();
		if (0 == exampleCount) {
			throw std::invalid_argument(
				"ZScoreScaler::fit: Training set is empty!");
		}

		// Number of features (cols) in one training example.
		const std::size_t featureCount = trainingSet.front().features.size();
		if (0 == featureCount) {
			throw std::invalid_argument(
				"ZScoreScaler::fit: Training set has no features!");
		}

		// Allocate required space and set 0 as initial value.
		featureMeans.assign(featureCount, T(0));
		featureSigma.assign(featureCount, T(0));

		// Accumulate in long double rather than T: with T = float and a large
		// example count, a running sum in T grows until each new addend
		// falls below the accumulator's ULP and is silently dropped, biasing
		// the mean. Only narrow back to T once the statistic is final.
		std::vector<long double> means(featureCount, 0.0L);

		// Accumulate feature values for mean calculation.
		for (const auto& example : trainingSet) {
			if (example.features.size() != featureCount) {
				throw std::invalid_argument(
					"ZScoreScaler::fit: Invalid feature count in training set!");
			}

			for (std::size_t j = 0; j < featureCount; j++) {
				if (false == std::isfinite(example.features[j])) {
					throw std::invalid_argument(
						"ZScoreScaler::fit: Feature value is NaN or Inf!");
				}
				// Accumulate sum for mean calculation.
				means[j] += static_cast<long double>(example.features[j]);

				if (false == std::isfinite(means[j])) {
					throw std::runtime_error(
						"ZScoreScaler::fit: Accumulated feature sum is NaN or Inf!");
				}
			}
		}

		// Each feature's average is based on total count of training examples.
		const long double exampleCountLD = static_cast<long double>(exampleCount);
		for (std::size_t j = 0; j < featureCount; j++) {
			means[j] /= exampleCountLD;

			if (false == std::isfinite(means[j])) {
				throw std::runtime_error(
					"ZScoreScaler::fit: Calculated mean is NaN or Inf!");
			}
		}

		// Accumulate squared differences for variance calculation.
		std::vector<long double> sumSquaredDiffs(featureCount, 0.0L);
		for (const auto& example : trainingSet) {
			for (std::size_t j = 0; j < featureCount; j++) {
				// Difference between the feature value and its mean.
				const long double diff =
					static_cast<long double>(example.features[j]) - means[j];
				// Add squared difference to feature accumulator.
				sumSquaredDiffs[j] += (diff * diff);

				if (false == std::isfinite(sumSquaredDiffs[j])) {
					throw std::runtime_error(
						"ZScoreScaler::fit: "
						"Accumulated squared difference is NaN or Inf!");
				}
			}
		}

		// Calculate population variance and standard deviation for each feature.
		for (std::size_t j = 0; j < featureCount; j++) {
			const long double variance = sumSquaredDiffs[j] / exampleCountLD;
			if (false == std::isfinite(variance)) {
				throw std::runtime_error(
					"ZScoreScaler::fit: Calculated variance is NaN or Inf!");
			}

			const long double sigma = std::sqrt(variance);
			if (false == std::isfinite(sigma)) {
				throw std::runtime_error(
					"ZScoreScaler::fit: Calculated sigma is NaN or Inf!");
			}

			featureMeans[j] = static_cast<T>(means[j]);

			// Replace zero standard deviation with one to avoid division by zero.
			// The constant value observed during fitting will map to zero.
			featureSigma[j] = (0.0L == sigma) ? T(1) : static_cast<T>(sigma);
		}

		// Set fitted flag to true after successful fitting.
		isFitted = true;
	}

	/**
	* @brief Apply Z-Score normalization to set of features.
	*
	* @param features Features to be modified with normalization.
	*
	* @throws std::logic_error If scaler has not been fitted.
	* @throws std::invalid_argument If feature count is incorrect
	* or an input feature is non-finite.
	* @throws std::runtime_error If scaling produces non-finite value.
	*/
	void transform(std::vector<T>& features) const {
		if (false == isFitted) {
			throw std::logic_error(
				"ZScoreScaler::transform: "
				"Scaling parameters have not been fitted. Call fit() before transform().");
		}

		if (features.size() != featureMeans.size()) {
			throw std::invalid_argument(
				"ZScoreScaler::transform: "
				"Feature count not matching scaling parameters!");
		}

		for (std::size_t j = 0; j < features.size(); j++) {
			// Avoid non-finite values to be used.
			if (false == std::isfinite(features[j])) {
				throw std::invalid_argument(
					"ZScoreScaler::transform: Feature value is NaN or Inf!");
			}

			// Apply Z-score normalization.
			features[j] = (features[j] - featureMeans[j]) / featureSigma[j];

			// Avoid non-finite values after scaling.
			if (false == std::isfinite(features[j])) {
				throw std::runtime_error(
					"ZScoreScaler::transform: Scaled value is NaN or Inf!");
			}
		}
	}

private:
	// Flag to indicate if scaling parameters have been fitted.
	bool isFitted = false;
	// Mean for each feature in training set.
	std::vector<T> featureMeans;
	// Standard deviation for each feature in training set.
	std::vector<T> featureSigma;
};

