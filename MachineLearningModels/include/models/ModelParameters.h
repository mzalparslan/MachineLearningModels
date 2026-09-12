#pragma once

#include <vector>

/**
 * @brief Container for parameters used by machine learning models.
 *
 * A Lightweight POD-style struct that holds the learnable parameters 
 * (a weight vector and an optional scalar bias)
 * for simple linear models or other models that are expressed with a
 * weight vector plus an intercept.
 *
 * Template parameter:
 *  - T: Floating-point data mode only.
 */
template <typename T>
struct ModelParameters {
	/**
	 * @brief Per-feature weights.
	 *
	 * Weight vector contains one weight per input feature. For a linear
	 * model the prediction for an input feature vector `x` is typically:
	 *
	 *   y = dot(weights, x) + bias
	 *
	 * @notes For models with complex shapes (matrices, tensors),
	 * this struct is not suitable. They should they their own data structs.
	 */
	std::vector<T> weights;

	/**
	 * @brief Scalar bias (intercept) term.
	 *
	 * Bias is added to result of weighted sum.
	 */
	T bias = T(0);
};