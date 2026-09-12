#pragma once

#include <vector>

/**
 * @brief Represents one labeled data point in any dataset.
 *
 * @tparam T Floating data mode for features and target.
 */
template <typename T>
struct DataPoint {
	std::vector<T> features;
	T target = T(0);
};

