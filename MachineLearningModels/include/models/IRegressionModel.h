#pragma once

#include "DataPoint.h"
#include <vector>

/**
 * @brief Common interface for regression models.
 *
 * @tparam T Floating-point type used for features, targets,
 * predictions, and model calculations.
 */
template <typename T>
class IRegressionModel {
public:
    virtual ~IRegressionModel() = default;

    /**
     * @brief Fits the model using a training dataset.
     *
	 * Calling fit() will update model's internal parameters 
     * based on the provided training data.
     *
     * @param trainingSet Dataset used to learn model parameters.
     */
    virtual void fit(const std::vector<DataPoint<T>>& trainingSet) = 0;

    /**
     * @brief Predicts a target value for one feature vector.
     *
     * @param features Input features.
     *
     * @return Predicted target value.
     */
    [[nodiscard]]
    virtual T predict(const std::vector<T>& features) const = 0;
};