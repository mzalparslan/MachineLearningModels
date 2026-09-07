#pragma once

#include "IScalingPolicy.h"
#include "DataPoint.h"
#include "ModelParameters.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <stdexcept>

/**
* @brief Scales features to [targetMin, targetMax] using Min-Max normalization;
* where [0, 1] is default target range.
* 
* Learns min and max of each feature from training set and scales each feature 
* to target range.
* 
* Constant features are mapped to midpoint of target range to avoid division by zero. 
* Values outside fitted range will be scaled outside target range.
* 
* Formula: 
* if constant feature: 
* @f[x' = targetMin + (targetMax - targetMin) / 2]@f
*; otherwise:
* @f[x' = targetMin + ((x - min(xj)) * (targetMax - targetMin) / (max(xj) - min(xj)))]@f
* 
* @tparam T Floating-point type used for all values and calculations.
* 
*/
template <typename T>
class MinMaxScaler final : public IScalingPolicy<T> {
	static_assert(std::is_floating_point_v<T>, 
		"MinMaxScaler requires floating data type T!");

public:
	/**
	* @brief Constructs a Min-Max scaler with specified target range.
	* 
	* @param targetMin_ Lower bound of target range.
	* @param targetMax_ Upper bound of target range.
	* 
	* @throws std::invalid_argument if targetMax_ is not greater than targetMin_ or 
	* targetMin_ is not finite or targetMax_ is not finite or targetScale is not finite.
	*/
	explicit MinMaxScaler(T targetMin_ = T(0), T targetMax_ = T(1)) 
		: targetMin(targetMin_), targetMax(targetMax_), targetScale(targetMax_ - targetMin_) {
		
		// Avoid invalid target range.
		if (targetMax_ <= targetMin_) {
			throw std::invalid_argument(
				"MinMaxScaler: targetMax must be greater than targetMin.");
		}

		if (false == std::isfinite(targetMin_)) {
			throw std::invalid_argument(
				"MinMaxScaler: targetMin must be finite number.");
		}

		if (false == std::isfinite(targetMax_)) {
			throw std::invalid_argument(
				"MinMaxScaler: targetMax must be finite number.");
		}

		if (false == std::isfinite(targetScale)) {
			throw std::invalid_argument(
				"MinMaxScaler: targetScale must be finite number.");
		}
	}

	/**
	 * @brief Transforms features to the target range.
	 * 
	 * @param features Vector of features to transform.
	 * 
	 * @throws std::logic_error if scaling parameters have not been fitted.
	 * @throws std::invalid_argument if feature size does not match scaling parameters or 
	 * feature value is not valid (NaN or Inf).
	 * @throws std::runtime_error if scaled value is not valid (NaN or Inf).
	 */
	void transform(std::vector<T>& features) const override {
		if (false == isFitted) {
			throw std::logic_error(
				"MinMaxScaler::transform: "
				"Parameters needs to be fitted. Call fit() first!");
		}

		if (features.size() != featureRanges.size()) {
			throw std::invalid_argument(
				"MinMaxScaler::transform: "
				"Feature size does not match scaling parameters.");
		}

		for (std::size_t j = 0; j < features.size(); j++) {
			// Avoid NaN or Inf values to be used during transformation.
			if (false == std::isfinite(features[j])) {
				throw std::invalid_argument(
					"MinMaxScaler::transform: Feature value is NaN or Inf!");
			}

			// Avoid division by zero when all values of feature[j] are the same.
			if (T(0) == featureRanges[j]) {
				// Set to midpoint of target range.
				features[j] = targetMin + targetScale / T(2);
				continue;
			}

			// x' = minRange + ((x - min(xj))*(maxRange - minRange)/((max(xj) - min(xj)))
			features[j] = targetMin + ((features[j] - featureMins[j]) * targetScale) / featureRanges[j];

			// Avoid NaN or Inf values after transformation.
			if (false == std::isfinite(features[j])) {
				throw std::runtime_error(
					"MinMaxScaler::transform: Scaled value is NaN or Inf!");
			}
		}
	}

	/**
	 * @brief Learn scaling parameters from training set.
	 * Calling this method will reset any previously fitted parameters.
	 * 
	 * @param trainingSet Dataset of training data to fit the scaling parameters.
	 * 
	 * @throws std::invalid_argument if training data is empty or 
	 * has inconsistent feature sizes or contains non-finite values.
	 * @throws std::runtime_error if calculated feature range is not finite.
	 */
	void fit(const std::vector<DataPoint<T>>& trainingSet) override {
		isFitted = false;

		if (true == trainingSet.empty()) {
			throw std::invalid_argument(
				"MinMaxScaler::fit: Training set is empty!");
		}

		const std::size_t featureCount = trainingSet.front().features.size();
		if (0 == featureCount) {
			throw std::invalid_argument(
				"MinMaxScaler::fit: Training set have no features!");
		}	

		// Features have changed. Reset the min, max and range vectors.
		featureMins.assign(featureCount, std::numeric_limits<T>::max());
		featureMaxs.assign(featureCount, std::numeric_limits<T>::lowest());
		featureRanges.assign(featureCount, T(0));

		// Min/Max per feature for each example in training set. 
		for (const auto& example : trainingSet) {
			// Ensure all training examples have same number of features.
			if (example.features.size() != featureCount) {
				throw std::invalid_argument(
					"MinMaxScaler::fit: Inconsistent feature size in training set!");
			}

			for (std::size_t j = 0; j < featureCount; j++) {
				// Avoid NaN or Inf values in training set.
				if (false == std::isfinite(example.features[j])) {
					throw std::invalid_argument(
						"MinMaxScaler::fit: Feature value is NaN or Inf!");
				}
				// Update min and max for each feature.
				featureMins[j] = std::min(featureMins[j], example.features[j]);
				featureMaxs[j] = std::max(featureMaxs[j], example.features[j]);
			}
		}

		// Cache each feature range to avoid recalculating max - min.
		for (std::size_t j = 0; j < featureCount; j++) {
			featureRanges[j] = featureMaxs[j] - featureMins[j];

			if (false == std::isfinite(featureRanges[j])) {
				throw std::runtime_error(
					"MinMaxScaler::fit: Feature range is NaN or Inf!");
			}
		}

		isFitted = true;
	}

private:
	// Flag to indicate if the scaling parameters have been fitted.
	bool isFitted = false;

	// Minimum for each feature in training set.
	std::vector<T> featureMins;
	// Maximum for each feature in training set.
	std::vector<T> featureMaxs;
	// Cache (max - min) to avoid recalculation.
	std::vector<T> featureRanges;

	// Default scale is between [0, 1] after normalization.
	const T targetMin = T(0);
	const T targetMax = T(1);
	const T targetScale = targetMax - targetMin;
};