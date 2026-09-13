#pragma once

#include "DataPoint.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Centers features around their medians and scales them by IQR.
 *
 * Formula:
 * @f[
 *     x' = \frac{x - \text{median}}{Q_3 - Q_1}
 * @f]
 *
 * Robust scaling is less sensitive to outliers than scaling methods
 * based on means, standard deviations, minima, or maxima.
 *
 * Features with zero IQR use a scale of one to prevent division
 * by zero.
 *
 * @tparam T Floating-point mode used for values and calculations.
 */
template <typename T>
class RobustScaler {
    static_assert(std::is_floating_point_v<T>,
        "RobustScaler requires floating data mode T!");

public:
    /**
     * @brief Learns feature medians and interquartile ranges.
     *
     * @param trainingSet Training dataset used to calculate statistics.
     *
     * @throws std::invalid_argument If dataset is empty, has no feature,
     * contains inconsistent feature counts, or contains non-finite values.
	 * @throws std::runtime_error If calculating median or IQR 
     * produces non-finite value.
     */
    void fit(const std::vector<DataPoint<T>>& trainingSet) {
        // Reset fitted flag.
        isFitted = false;

		const std::size_t exampleCount = trainingSet.size();
        if (0 == exampleCount) {
            throw std::invalid_argument(
                "RobustScaler::fit: Training set is empty!");
        }

        const std::size_t featureCount = trainingSet.front().features.size();
        if (0 == featureCount) {
            throw std::invalid_argument(
                "RobustScaler::fit: Training set has no features!");
        }

		// Ensure that all examples have the same number of features.
		for (const auto& example : trainingSet) {
			if (example.features.size() != featureCount) {
				throw std::invalid_argument(
					"RobustScaler::fit: Inconsistent feature count in training set!");
			}
		}

        medians.assign(featureCount, T(0));
        interQuartileRanges.assign(featureCount, T(0));
        std::vector<T> columnData;
        columnData.reserve(exampleCount);

        for (std::size_t j = 0; j < featureCount; ++j) {
            columnData.clear();

            for (const auto& example : trainingSet) {
                const T value = example.features[j];

                if (false == std::isfinite(value)) {
                    throw std::invalid_argument(
                        "RobustScaler::fit: Feature is NaN or Inf!");
                }

                columnData.push_back(value);
            }

            const T q1 = calculatePercentile(columnData, T(0.25));

            medians[j] = calculatePercentile(columnData, T(0.50));
			if (false == std::isfinite(medians[j])) {
				throw std::runtime_error(
					"RobustScaler::fit: Calculated median is NaN or Inf!");
			}

            const T q3 = calculatePercentile(columnData, T(0.75));

            interQuartileRanges[j] = q3 - q1;

            if (false == std::isfinite(interQuartileRanges[j])) {
                throw std::runtime_error(
                    "RobustScaler::fit: Calculated IQR is NaN or Inf!");
            }

            // Avoid division by zero when the middle 50% has no variation.
            if (T(0) == interQuartileRanges[j]) {
                interQuartileRanges[j] = T(1);
            }
        }

        isFitted = true;
    }

    /**
	 * @brief Applies robust scaling to feature vector.
     *
     * @param features Feature vector modified in place.
     *
     * @throws std::logic_error If the scaler has not been fitted.
     * @throws std::invalid_argument If the feature count is incorrect
     *         or an input feature is non-finite.
     * @throws std::runtime_error If scaling produces a non-finite value.
     */
    void transform(std::vector<T>& features) const {
        if (false == isFitted) {
            throw std::logic_error(
                "RobustScaler::transform: Call fit() before transform!");
        }

        if (features.size() != medians.size()) {
            throw std::invalid_argument(
                "RobustScaler::transform: Invalid feature count!");
        }

        for (std::size_t j = 0; j < features.size(); ++j) {
            if (false == std::isfinite(features[j])) {
                throw std::invalid_argument(
                    "RobustScaler::transform: Feature is NaN or Inf!");
            }

            features[j] = (features[j] - medians[j]) / interQuartileRanges[j];

            if (false == std::isfinite(features[j])) {
                throw std::runtime_error(
                    "RobustScaler::transform: Scaled value is NaN or Inf!");
            }
        }
    }

private:
		/**
		 * @brief Calculates the percentile value from a column of values.
		 *
		 * Uses std::nth_element rather than a full sort: computing one
		 * percentile only needs the value(s) at its rank, not a total
		 * order over the whole column, so this is O(n) instead of
		 * O(n log n) per feature.
		 *
		 * @param columnData Column values; reordered in place.
		 * @param quantile Percentile to calculate (0.0 to 1.0).
		 *
		 * @return The calculated percentile value.
		 */
        static T calculatePercentile(std::vector<T>& columnData, T quantile)
        {
            const T position = quantile * static_cast<T>(columnData.size() - 1);

            const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(position));
            const std::size_t upperIndex = std::min(lowerIndex + 1, columnData.size() - 1);

            std::nth_element(columnData.begin(), columnData.begin() + upperIndex, columnData.end());
            std::nth_element(columnData.begin(), columnData.begin() + lowerIndex, columnData.begin() + upperIndex);

            const T weight = position - static_cast<T>(lowerIndex);

            return columnData[lowerIndex] + (columnData[upperIndex] - columnData[lowerIndex]) * weight;
        }

    // Flag to indicate if scaling parameters have been fitted.
    bool isFitted = false;
    // Median for each feature in training set.
    std::vector<T> medians;
    // Interquartile range (IQR) for each feature in training set.
    std::vector<T> interQuartileRanges;
};