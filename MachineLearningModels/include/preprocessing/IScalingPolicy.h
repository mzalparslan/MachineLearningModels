#pragma once

#include "DataPoint.h"
#include <vector>

/**
 * @brief Interface for feature scaling policies.
 *
 * A scaling policy learns feature statistics from training data and
 * applies learnt transformation to other features.
 *
 * Policy must be fitted using training data before transformation.
 * Same fitted parameters should then be used for training, validation, 
 * test, and inference data.
 *
 * Feature scaling can improve numerical stability and prevent overflows.
 *
 * @tparam T Floating-point type.
 */
template <typename T>
class IScalingPolicy {
public:
	/**
	 * @brief Learns scaling parameters from a training dataset.
	 *
	 * @param trainingSet Training data used to calculate scaling
	 * statistics.
	 */
	virtual void fit(const std::vector<DataPoint<T>>& trainingSet) = 0;

	/**
	 * @brief Transforms features to target range.
	 *
	 * @param features Vector of features to transform.
	 */
	virtual void transform(std::vector<T>& features) const = 0;

	virtual ~IScalingPolicy() = default;
};