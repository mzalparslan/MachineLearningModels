#pragma once

#include "DataPoint.h"

#include <concepts>
#include <type_traits>
#include <vector>

/**
 * @brief Concept a scaling policy must satisfy following requirements:
 * - fit() learns parameters from a training set, 
 * - transform() applies them in place.
 */
template <typename S, typename T>
concept ScalingPolicy = std::is_floating_point_v<T> &&
	requires(S & scaler, const S & constScaler,
		const std::vector<DataPoint<T>>& trainingSet,
		std::vector<T>& features) {
		{ scaler.fit(trainingSet) } -> std::same_as<void>;
		{ constScaler.transform(features) } -> std::same_as<void>;
};
