#pragma once

#include "DataPoint.h"

#include <vector>

/**
 * @brief Common interface for binary-classification models.
 *
 * Implementations learn from labeled samples and can return either 
 * estimated probability of positive class or a binary class prediction.
 *
 * DataPoint targets are expected to represent negative and positive classes,
 * normally using 0 and 1.
 *
 * @tparam T Floating-point type used for features, targets, and probabilities.
 */
template <typename T>
class IBinaryClassifier {
public:
    virtual ~IBinaryClassifier() = default;

    /**
     * @brief Fits classifier using a training dataset.
     *
     * @param trainingSet Labeled samples used to learn model parameters.
     */
    virtual void fit(const std::vector<DataPoint<T>>& trainingSet) = 0;

    /**
     * @brief Estimates probability for a sample if it belongs to positive
     * class.
     *
     * @param features Input features of the sample.
     *
     * @return Estimated positive-class probability between [0, 1].
     */
    [[nodiscard]]
    virtual T predictProbability(const std::vector<T>& features) const = 0;

    /**
     * @brief Predicts binary class using a specified probability threshold.
     *
     * @param features Input features of one sample.
     * @param threshold Minimum positive-class probability required to return
     * `true`.
     *
     * @return `true` for the positive class; otherwise `false`.
     */
    [[nodiscard]]
    virtual bool predict(const std::vector<T>& features, T threshold) const = 0;

    /**
     * @brief Predicts binary class using the default threshold of 0.5.
     *
     * @param features Input features of one sample.
     *
     * @return `true` for positive class; otherwise `false`.
     */
    [[nodiscard]]
    bool predict(const std::vector<T>& features) const {
        return predict(features, T(0.5));
    }
};