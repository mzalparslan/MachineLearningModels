#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

/**
 * @brief Activation functions and their derivatives used by neural
 * network models (BasicNeuralNetwork, DeepNeuralNetwork).
 *
 * Every derivative is expressed in terms of the activation's own output
 * (the value g(z) already computed during the forward pass) rather than
 * its raw input z, since that is what backpropagation has on hand and
 * avoids recomputing g(z). ReLU-family derivatives are the exception --
 * they are not differentiable at their own output, so they take z.
 *
 * @tparam T Floating-point mode used for activations.
 */

 // Logistic sigmoid: g(z) = 1 / (1 + e^-z). Uses the numerically stable
 // form that evaluates e^z instead of e^-z for negative z, avoiding
 // overflow for large negative inputs.
template <typename T>
[[nodiscard]] inline T Sigmoid(T z) {
	static_assert(std::is_floating_point_v<T>, "Sigmoid requires a floating-point type!");

	if (z >= T(0)) {
		return T(1) / (T(1) + std::exp(-z));
	}

	const T exponential = std::exp(z);
	return exponential / (T(1) + exponential);
}

// Derivative of Sigmoid, in terms of its own output a = Sigmoid(z).
template <typename T>
[[nodiscard]] inline T SigmoidDerivative(T a) {
	return a * (T(1) - a);
}

// Hyperbolic tangent activation: g(z) = tanh(z).
template <typename T>
[[nodiscard]] inline T TanH(T z) {
	static_assert(std::is_floating_point_v<T>, "TanH requires a floating-point type!");
	return std::tanh(z);
}

// Derivative of TanH, in terms of its own output a = TanH(z).
template <typename T>
[[nodiscard]] inline T TanHDerivative(T a) {
	return T(1) - (a * a);
}

// Rectified Linear Unit: g(z) = max(0, z).
template <typename T>
[[nodiscard]] inline T ReLU(T z) {
	static_assert(std::is_floating_point_v<T>, "ReLU requires a floating-point type!");
	return std::max(T(0), z);
}

// Derivative of ReLU, evaluated at the pre-activation input z.
template <typename T>
[[nodiscard]] inline T ReLUDerivative(T z) {
	return z > T(0) ? T(1) : T(0);
}

// Leaky Rectified Linear Unit: g(z) = z for z > 0, otherwise negativeSlope * z.
template <typename T>
[[nodiscard]] inline T LReLU(T z, T negativeSlope = T(0.01)) {
	static_assert(std::is_floating_point_v<T>, "LReLU requires a floating-point type!");
	return z > T(0) ? z : negativeSlope * z;
}

// Derivative of LReLU, evaluated at the pre-activation input z.
template <typename T>
[[nodiscard]] inline T LReLUDerivative(T z, T negativeSlope = T(0.01)) {
	return z > T(0) ? T(1) : negativeSlope;
}

// Parametric Rectified Linear Unit (He et al., 2015): g(z) = z for z > 0,
// otherwise alpha * z. Same shape as LReLU, but alpha is meant to be a
// learned parameter rather than a fixed constant.
template <typename T>
[[nodiscard]] inline T PReLU(T z, T alpha) {
	static_assert(std::is_floating_point_v<T>, "PReLU requires a floating-point type!");
	return z > T(0) ? z : alpha * z;
}

// Derivative of PReLU, evaluated at the pre-activation input z.
template <typename T>
[[nodiscard]] inline T PReLUDerivative(T z, T alpha) {
	return z > T(0) ? T(1) : alpha;
}

// Exponential Linear Unit (Clevert et al., 2016): g(z) = z for z > 0,
// otherwise alpha * (e^z - 1).
template <typename T>
[[nodiscard]] inline T ELU(T z, T alpha = T(1)) {
	static_assert(std::is_floating_point_v<T>, "ELU requires a floating-point type!");
	return z > T(0) ? z : alpha * (std::exp(z) - T(1));
}

// Derivative of ELU, in terms of the pre-activation input z and its own
// output a = ELU(z). For z <= 0, a = alpha * (e^z - 1), so the derivative
// alpha * e^z simplifies to a + alpha.
template <typename T>
[[nodiscard]] inline T ELUDerivative(T z, T a, T alpha = T(1)) {
	return z > T(0) ? T(1) : a + alpha;
}
