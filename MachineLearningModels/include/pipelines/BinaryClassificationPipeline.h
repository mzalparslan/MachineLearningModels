#pragma once

#include "LogisticBinaryClassifier.h"
#include "Logger.h"

#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief Coordinates feature scaling and binary-classification training.
 *
 * The pipeline learns scaling parameters from the training dataset, transforms
 * its features, and fits a logistic binary classifier using the selected
 * optimizer.
 *
 * Prediction inputs are automatically transformed using the scaling parameters
 * learned exclusively from the training dataset.
 *
 * @tparam T Floating-point mode used for features and calculations.
 * @tparam Scaler Feature-scaling strategy.
 * @tparam Optimizer Optimization strategy used to train the classifier.
 */
template <
    typename T,
    typename Scaler,
    typename Optimizer>
class BinaryClassificationPipeline {

public:
    /**
     * @brief Constructs a binary-classification pipeline.
     *
     * @param scaler Scaling strategy used to preprocess features.
     * @param optimizer Optimization strategy used to train the classifier.
     * @param options Configuration options supplied to the optimizer.
     * @param logger Generic logger to log details and issues.
     */
    BinaryClassificationPipeline(
        Scaler scaler_,
        Optimizer optimizer,
        Logger& logger_,
        GradientDescentOptions<T> options = {},
        ExecutionStrategy<T> execStrategy = {})
        : scaler(std::move(scaler_)),
        model(std::move(optimizer), options, execStrategy),
        logger(logger_) {

        logger.debug() 
            << "Binary Classification configuration: "
            << "Learning Rate=" << options.learningRate
            << ", Epochs=" << options.epochs
            << ", Regulate bias flag=" << options.regularizeBias 
            << ", Lambda: " << options.lambda;
    }

    /**
     * @brief Fits the scaler and classifier using training data.
     *
     * This operation modifies the feature values in `trainingData` by applying
     * the fitted scaler. All sample targets must represent valid binary labels.
     *
     * @param trainingData Samples used to fit the scaler and classifier.
     *
     * @throws std::exception If scaling, validation, or training fails.
     */
    void fit(std::vector<DataPoint<T>>& trainingData) {
        logger.info()
            << "Binary classification training started. Samples: "
            << trainingData.size();

        // Prevent prediction if any stage of refitting fails.
        isFitted = false;

        // Learn preprocessing parameters exclusively from training data.
        scaler.fit(trainingData);

        logger.debug() << "Scaling parameters fitted.";

        // Apply the learned scaling parameters to all training samples.
        for (auto& sample : trainingData) {
            scaler.transform(sample.features);
        }

        logger.debug() << "Training features transformed.";

        // Learn classifier parameters from the transformed training data.
        model.fit(trainingData);
        isFitted = true;

        logger.info() << "Binary classification training completed.";
    }

    /**
     * @brief Calculates the positive-class probability for one sample.
     *
     * The input vector is accepted by value so it can be transformed without
     * modifying the caller's original features.
     *
     * @param features Unscaled input features.
     *
     * @return Estimated probability of the positive class, normally in [0, 1].
     *
     * @throws std::logic_error If the pipeline has not been fitted.
     * @throws std::invalid_argument If the feature vector is invalid.
     */
    [[nodiscard]]
    T predict(std::vector<T> features) const {
        validateFitted();

        // Apply the parameters learned from the training dataset.
        scaler.transform(features);

        return model.predict(features);
    }

    /**
     * @brief Predicts the binary class of one sample.
     *
     * @param features Unscaled input features.
     * @param threshold Minimum positive-class probability required to return
     * `true`. The default value is 0.5.
     *
     * @return `true` for the positive class; otherwise `false`.
     *
     * @throws std::logic_error If the pipeline has not been fitted.
     * @throws std::invalid_argument If the features or threshold are invalid.
     */
    [[nodiscard]]
    bool predictClass(
        std::vector<T> features,
        T threshold = T(0.5)) const {
        validateFitted();

        // Apply the parameters learned from the training dataset.
        scaler.transform(features);

        return model.predictClass(features, threshold);
    }

private:
    // Ensure that scaling and model parameters are available.
    void validateFitted() const {
        if (false == isFitted) {
            throw std::logic_error(
                "BinaryClassificationPipeline: Call fit() first!");
        }
    }

    Scaler scaler;
    LogisticBinaryClassifier<T, Optimizer> model;
    bool isFitted = false;
    Logger& logger;
};