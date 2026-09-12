#pragma once

#include "LinearRegression.h"
#include "Logger.h"
#include "ScopedBenchmarkTimer.h"

#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief Coordinates feature scaling and linear-regression training.
 *
 * Pipeline learns scaling parameters from training dataset, transforms
 * features, and then fits a linear-regression model using selected optimizer.
 *
 * Predictions are automatically transformed using the scaling parameters
 * learned from the training dataset.
 *
 * @tparam T Floating-point mode used for features and model calculations.
 * @tparam Scaler Feature-scaling strategy.
 * @tparam Optimizer Optimization strategy used by the regression model.
 */
template <
    typename T,
    typename Scaler,
    typename Optimizer>
class RegressionPipeline {
public:
    /**
     * @brief Constructs a regression pipeline.
     *
     * @param scaler Scaling strategy used to preprocess features.
     * @param optimizer Optimization strategy used to train the model.
     * @param options Configuration options supplied to the optimizer.
     * @param logger Generic logger to log details and issues.
     */
    RegressionPipeline(
        Scaler scaler_,
        Optimizer optimizer,
        Logger & logger_,
		GradientDescentOptions<T> options = {},
        ExecutionStrategy<T> execStrategy = {})
        : scaler(std::move(scaler_)),
        model(std::move(optimizer), options, execStrategy),
        logger(logger_) {

        logger.debug()
            << "Linear Regression configuration: "
            << "Learning Rate=" << options.learningRate
            << ", Epochs=" << options.epochs
            << ", Regulate bias flag=" << options.regularizeBias
            << ", Lambda: " << options.lambda;
    }

    /**
     * @brief Fits the scaler and regression model using training data.
     *
     * This operation modifies the feature values in `trainingData` by applying
     * the fitted scaler. Scaling parameters are learned exclusively from this
     * training dataset.
     *
     * @param trainingData Samples used to fit the scaler and model.
     *
     * @throws std::exception If fitting or transformation fails.
     */
    void fit(std::vector<DataPoint<T>>& trainingData) {
        ScopedBenchmarkTimer benchmark(logger, "RegressionPipeline::fit");
        // Prevent prediction if any stage of refitting fails.
        isFitted = false;

        logger.info()
            << "Linear regression training started. Samples: "
            << trainingData.size();

        // Learn preprocessing parameters exclusively from training data.
        scaler.fit(trainingData);

        logger.debug() << "Scaling parameters fitted.";

        // Apply the learned scaling parameters to every training sample.
        for (auto& sample : trainingData) {
            scaler.transform(sample.features);
        }

        logger.debug() << "Training features transformed.";

        // Learn model parameters from the transformed training data.
        model.fit(trainingData);

        logger.info() << "Linear regression training completed.";
        isFitted = true;
    }

    /**
     * @brief Predicts the target value for one unscaled feature vector.
     *
     * The feature vector is accepted by value because the pipeline must
     * transform it before prediction. The caller's original vector is
     * therefore not modified.
     *
     * @param features Unscaled input features.
     *
     * @return Predicted target value.
     *
     * @throws std::logic_error If the pipeline has not been fitted.
     * @throws std::invalid_argument If the feature count is invalid.
     */
    [[nodiscard]]
    T predict(std::vector<T> features) const {
        if (false == isFitted) {
            throw std::logic_error(
                "RegressionPipeline::predict: Call fit() first.");
        }

        // Use the parameters learned from training data.
        scaler.transform(features);

        return model.predict(features);
    }

private:
    Scaler scaler;
    LinearRegression<T, Optimizer> model;
    bool isFitted = false;
    Logger& logger;
};