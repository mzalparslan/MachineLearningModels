#pragma once

#include "IScalingPolicy.h"
#include "DataPoint.h"
#include "ModelParameters.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Scales features using mean normalization.
 *
 * Learns the mean and range of each feature from the training set.
 *
 * Formula:
 * @f[
 *     x' = \frac{x - \mu}{x_{\max} - x_{\min}}
 * @f]
 *
 * Constant feature values observed during fitting are mapped to zero.
 *
 * @tparam T Floating-point type used for values and calculations.
 *
 */
template <typename T>
class MeanNormScaler final : public IScalingPolicy<T> {
    static_assert(std::is_floating_point_v<T>,
        "MeanNormScaler requires floating data type T!");

public:
    /**
     * @brief Learns feature means and ranges from training data.
     *
     * @param trainingSet Training dataset used to calculate statistics.
     *
     * @throws std::invalid_argument If dataset is empty, has no feature, 
     * contains inconsistent feature counts, or contains non-finite values.
     * @throws std::runtime_error If accumulating or calculating a statistic
	 * produces non-finite value.
     */
    void fit(const std::vector<DataPoint<T>>& trainingSet) override {
		// Reset flag.
        isFitted = false;

        const std::size_t exampleCount = trainingSet.size();
        if (0 == exampleCount) {
            throw std::invalid_argument(
                "MeanNormScaler::fit: Training set is empty!");
        }
        
        const std::size_t featureCount = trainingSet.front().features.size();
        if (0 == featureCount) {
            throw std::invalid_argument(
                "MeanNormScaler::fit: Training set has no features!");
        }

		// Initialize mean and range vectors with zeros.
        featureMeans.assign(featureCount, T(0));
        featureRanges.assign(featureCount, T(0));

		// Initialize min and max vectors with extreme values.
        std::vector<T> featureMinimums(featureCount, std::numeric_limits<T>::max());
        std::vector<T> featureMaximums(featureCount, std::numeric_limits<T>::lowest());

        for (const auto& example : trainingSet) {
            if (example.features.size() != featureCount) {
                throw std::invalid_argument(
                    "MeanNormScaler::fit: Inconsistent feature count!");
            }

            for (std::size_t j = 0; j < featureCount; ++j) {
                const T value = example.features[j];

                if (false == std::isfinite(value)) {
                    throw std::invalid_argument(
                        "MeanNormScaler::fit: Feature is NaN or Inf!");
                }

                featureMeans[j] += value;

                if (false == std::isfinite(featureMeans[j])) {
                    throw std::runtime_error(
                        "MeanNormScaler::fit: Feature sum is NaN or Inf!");
                }

				// Update min and max for each feature.
                featureMinimums[j] = std::min(featureMinimums[j], value);
                featureMaximums[j] = std::max(featureMaximums[j], value);
            }
        }

        const T exampleCountT = static_cast<T>(exampleCount);

        for (std::size_t j = 0; j < featureCount; ++j) {
			// Calculate mean and range for each feature.
            featureMeans[j] /= exampleCountT;
            featureRanges[j] = featureMaximums[j] - featureMinimums[j];

            if (false == std::isfinite(featureMeans[j]) || 
                false == std::isfinite(featureRanges[j])) {
                throw std::runtime_error(
                    "MeanNormScaler::fit: Mean or range is NaN or Inf!");
            }
        }

        isFitted = true;
    }

    /**
     * @brief Applies mean normalization to a feature vector.
     *
	 * @param features Features modified during transformation.
     *
	 * @throws std::logic_error If scaler did not call fit() before transform().
     * @throws std::invalid_argument If feature count is incorrect
     * or an input value is non-finite.
     * @throws std::runtime_error If scaling produces non-finite value.
     */
    void transform(std::vector<T>& features) const override {
        if (false == isFitted) {
            throw std::logic_error(
                "MeanNormScaler::transform: Call fit() before transform()!");
        }

        if (features.size() != featureMeans.size()) {
            throw std::invalid_argument(
                "MeanNormScaler::transform: Inconsistent feature count!");
        }

        for (std::size_t j = 0; j < features.size(); ++j) {
            if (false == std::isfinite(features[j])) {
                throw std::invalid_argument(
                    "MeanNormScaler::transform: Feature is NaN or Inf!");
            }
			
            // Set constant features to zero to avoid division by zero.
            if (T(0) == featureRanges[j]) {
                features[j] = T(0);
                continue;
            }

			// Apply mean normalization formula: x' = (x - mean) / (max - min)
            features[j] = (features[j] - featureMeans[j]) / featureRanges[j];

            if (false == std::isfinite(features[j])) {
                throw std::runtime_error(
                    "MeanNormScaler::transform: Scaled value is NaN or Inf!");
            }
        }
    }

private:
    // Flag to indicate if scaling parameters have been fitted.
    bool isFitted = false;
    // Mean for each feature in training set.
    std::vector<T> featureMeans;
    // Range (max - min) for each feature in training set.
    std::vector<T> featureRanges;
};