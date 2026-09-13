#pragma once

#include "DataPoint.h"

#include <concepts>
#include <type_traits>
#include <vector>

/**
 * @brief Contract a scaling policy must satisfy: 
 * - fit() learns parameters from a training set, 
 * - transform() applies them in place.
 *
 * RegressionPipeline and BinaryClassificationPipeline are templates, so
 * the scaler they use is dispatched statically -- constraining their
 * Scaler parameter with this concept turns a mismatched scaler into a
 * readable concept error at the call site, with no runtime cost, instead
 * of a wall of template-instantiation diagnostics or a vtable neither
 * pipeline ever needs.
 *
 */
template <typename S, typename T>
concept ScalingPolicy = std::is_floating_point_v<T> &&
	requires(S & scaler, const S & constScaler,
		const std::vector<DataPoint<T>>& trainingSet,
		std::vector<T>& features) {
		{ scaler.fit(trainingSet) } -> std::same_as<void>;
		{ constScaler.transform(features) } -> std::same_as<void>;
};
