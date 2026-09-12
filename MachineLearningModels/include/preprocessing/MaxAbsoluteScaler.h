#pragma once

#include "IScalingPolicy.h"
#include "DataPoint.h"
#include "ModelParameters.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Scales each feature by its maximum absolute value.
 *
 * Formula:
 * @f[
 *     x' = \frac{x}{\max(|x|)}
 * @f]
 *
 * Values observed during fitting are mapped to [-1, 1].
 * Values outside range may be mapped outside this interval.
 * Zero values remain zero, preserving sparsity.
 *
 * Features containing only zeros during fitting use a scale of one
 * to prevent division by zero.
 *
 * @tparam T Floating-point data mode used for values and calculations.
 */
template <typename T>
class MaxAbsoluteScaler final : public IScalingPolicy<T> {
    static_assert(std::is_floating_point_v<T>,
        "MaxAbsoluteScaler requires floating data mode T!");

public:
    /**
     * @brief Learns maximum absolute value of each feature.
     *
     * @param trainingSet Training dataset used to calculate statistics.
     *
     * @throws std::invalid_argument If dataset is empty, has no feature, 
     * contains inconsistent feature counts, or contains non-finite values.
     */
    void fit(const std::vector<DataPoint<T>>& trainingSet) override {
		// Reset fitted flag.
        isFitted = false;

        if (true == trainingSet.empty()) {
            throw std::invalid_argument(
                "MaxAbsoluteScaler::fit: Training set is empty!");
        }

        const std::size_t featureCount = trainingSet.front().features.size();
        if (0 == featureCount) {
            throw std::invalid_argument(
                "MaxAbsoluteScaler::fit: Training set has no features!");
        }
		// Initialize maxAbsValues to zero for each feature.
        maxAbsValues.assign(featureCount, T(0));

        for (const auto& example : trainingSet) {
            if (example.features.size() != featureCount) {
                throw std::invalid_argument(
                    "MaxAbsoluteScaler::fit: Inconsistent feature count!");
            }

            for (std::size_t j = 0; j < featureCount; j++) {
                const T value = example.features[j];

                if (false == std::isfinite(value)) {
                    throw std::invalid_argument(
                        "MaxAbsoluteScaler::fit: Feature is NaN or Inf!");
                }
				// Update maximum absolute value for each feature.
                maxAbsValues[j] = std::max(maxAbsValues[j], std::abs(value));
            }
        }

        for (std::size_t j = 0; j < featureCount; j++) {
			// Avoid division by zero for features with all zero values.
            if (T(0) == maxAbsValues[j]) {
                maxAbsValues[j] = T(1);
            }
        }

        isFitted = true;
    }

    /**
     * @brief Applies Max Absolute Scaling to features.
     *
     * @param features Feature vector updated by scaling.
     *
     * @throws std::logic_error Scaler did not call fit() for training set.
	 * @throws std::invalid_argument Feature count is inconsistent with training set
	 * or feature value is NaN or Inf value.
	 * @throws std::runtime_error Scaled value is NaN or Inf.
     */
    void transform(std::vector<T>& features) const override {
        if (false == isFitted) {
            throw std::logic_error(
                "MaxAbsoluteScaler::transform: Call fit() before transform()!");
        }

        if (features.size() != maxAbsValues.size()) {
            throw std::invalid_argument(
                "MaxAbsoluteScaler::transform: Invalid feature count!");
        }

        for (std::size_t j = 0; j < features.size(); j++) {
            if (false == std::isfinite(features[j])) {
                throw std::invalid_argument(
                    "MaxAbsoluteScaler::transform: Feature is NaN or Inf!");
            }

			// Scale feature by its maximum absolute value.
            features[j] /= maxAbsValues[j];

            if (false == std::isfinite(features[j])) {
                throw std::runtime_error(
                    "MaxAbsoluteScaler::transform: Scaled value is NaN or Inf!");
            }
        }
    }

private:
    // Flag to indicate if scaling parameters have been fitted.
    bool isFitted = false;
    // Maximum absolute value for each feature in training set.
    std::vector<T> maxAbsValues;
};