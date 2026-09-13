#pragma once

#include "DataPoint.h"
#include "ModelParameters.h"

#include <concepts>
#include <type_traits>
#include <vector>

/**
 * @brief Structural contract an optimizer must satisfy: an Options type
 * used to configure it, and an optimize() that updates ModelParameters
 * from a training set and a hypothesis callable.
 *
 * LinearRegression, LogisticBinaryClassifier, and both pipelines are
 * templates, so the optimizer they use is dispatched statically --
 * constraining their Optimizer parameter with this concept turns a
 * mismatched optimizer into a readable concept error at the call site,
 * with no runtime cost, instead of a wall of template-instantiation
 * diagnostics surfacing deep inside fit().
 *
 * Fictional hypothesis parameter below (a plain function pointer)
 * only needs to be *some* type satisfying optimize()'s own
 * std::invocable constraint on its hypothesis argument -- it doesn't
 * need to be the exact callable a real model would pass in, since this
 * concept is checking optimize()'s shape, not invoking it.
 */
template <typename O, typename T>
concept OptimizationPolicy = std::is_floating_point_v<T> &&
	requires { typename O::Options; } &&
	requires(O & optimizer,
		const std::vector<DataPoint<T>>& trainingSet,
		const typename O::Options& options,
		ModelParameters<T>& modelParameters,
		T(*hypothesis)(const std::vector<T>&, const ModelParameters<T>&)) {
		{ optimizer.optimize(trainingSet, options, modelParameters, hypothesis) } -> std::same_as<void>;
};
