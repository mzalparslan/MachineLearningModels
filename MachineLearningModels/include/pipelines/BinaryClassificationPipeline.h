#pragma once

#include "ScalingPolicy.h"
#include "LogisticBinaryClassifier.h"
#include "Logger.h"
#include "OptimizationPolicy.h"
#include "ScopedBenchmarkTimer.h"
#include "metrics.h"

#include <cmath>
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
    ScalingPolicy<T> Scaler,
    OptimizationPolicy<T> Optimizer>
class BinaryClassificationPipeline {

public:
    /**
     * @brief Constructs a binary-classification pipeline.
     *
     * @param scaler Scaling strategy used to preprocess features.
     * @param optimizer Optimization strategy used to train the classifier.
     * @param options Configuration options supplied to the optimizer.
     * @param execStrategy Strategy used to calculate linear outputs.
     * @param logger Generic logger to log details and issues. Defaults to
     * a shared logger at LogLevel::Info if the caller doesn't supply one.
     */
    BinaryClassificationPipeline(
        Scaler scaler_,
        Optimizer optimizer,
        GradientDescentOptions<T> options = {},
        ExecutionStrategy<T> execStrategy = {},
        Logger& logger_ = defaultLogger())
        : scaler(std::move(scaler_)),
        // execStrategy isn't needed after construction, so it is moved in;
        // options is still read below for the debug log, so it is copied.
        model(std::move(optimizer), options, std::move(execStrategy)),
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
      * @param trainingData Samples used to fit the scaler and model. 
      * Taken by value: pass an lvalue to keep your own copy unscaled, 
      * or std::move() it to hand over ownership and avoid copy.
     *
     * @throws std::exception If scaling, validation, or training fails.
     */
    void fit(std::vector<DataPoint<T>> trainingData) {
        ScopedBenchmarkTimer benchmark(logger, "BinaryClassificationPipeline::fit");
        // Prevent prediction if any stage of refitting fails.
        isFitted = false;

        logger.info()
            << "Binary classification training started. Samples: "
            << trainingData.size();

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

    /**
     * @brief Classifies an already-computed probability against a threshold.
     *
     * Exposes the model's decision rule (the same one predictClass()
     * applies) to callers that already have a probability in hand -- such
     * as evaluate() below and BinaryClassificationWriter, which both also
     * report the probability itself and would otherwise need to predict
     * twice (once for the probability, once more inside predictClass())
     * to reuse it. Going through this method instead of comparing
     * `probability >= threshold` directly means the decision rule only
     * has one definition to keep in sync: LogisticBinaryClassifier::classify.
     *
     * @throws std::invalid_argument If threshold is non-finite or outside
     * the open interval (0, 1).
     */
    [[nodiscard]]
    static bool classify(T probability, T threshold = T(0.5)) {
        return LogisticBinaryClassifier<T, Optimizer>::classify(probability, threshold);
    }

    /**
     * @brief Evaluates the fitted pipeline against a held-out test set.
     *
     * @param testData Test samples with unscaled features.
     * @param threshold Minimum positive-class probability required to
     * count a prediction as positive. The default value is 0.5.
     *
     * @return Binary-classification metrics computed from the pipeline's
     * predictions.
     *
     * @throws std::logic_error If the pipeline has not been fitted.
     * @throws std::invalid_argument If testData is empty or threshold is invalid.
     */
    [[nodiscard]]
    BinaryClassificationMetrics evaluate(
        const std::vector<DataPoint<T>>& testData,
        T threshold = T(0.5)) const {
        validateFitted();

        if (true == testData.empty()) {
            throw std::invalid_argument(
                "BinaryClassificationPipeline::evaluate: Test data is empty.");
        }

        std::vector<T> probabilities;
        std::vector<T> targets;
        std::vector<bool> predictedClasses;
        std::vector<bool> expectedClasses;
        probabilities.reserve(testData.size());
        targets.reserve(testData.size());
        predictedClasses.reserve(testData.size());
        expectedClasses.reserve(testData.size());

        for (const auto& dataPoint : testData) {
            const T probability = predict(dataPoint.features);

            probabilities.push_back(probability);
            targets.push_back(dataPoint.target);
            predictedClasses.push_back(classify(probability, threshold));
            expectedClasses.push_back(dataPoint.target >= T(0.5));
        }

        BinaryClassificationMetrics result =
            metrics::classificationRates(predictedClasses, expectedClasses);
        result.binaryCrossEntropy = metrics::binaryCrossEntropy(probabilities, targets);

        return result;
    }

private:
    // Ensure that scaling and model parameters are available.
    void validateFitted() const {
        if (false == isFitted) {
            throw std::logic_error(
                "BinaryClassificationPipeline: Call fit() first!");
        }
    }

    // Shared logger used when the caller doesn't supply one.
    static Logger& defaultLogger() {
        static Logger instance;
        return instance;
    }

    Scaler scaler;
    LogisticBinaryClassifier<T, Optimizer> model;
    bool isFitted = false;
    Logger& logger;
};