#pragma once

#include "ScalingPolicy.h"
#include "LinearRegression.h"
#include "Logger.h"
#include "OptimizationPolicy.h"
#include "ScopedBenchmarkTimer.h"
#include "metrics.h"

#include <cmath>
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
    ScalingPolicy<T> Scaler,
    OptimizationPolicy<T> Optimizer>
class RegressionPipeline {
public:
    /**
     * @brief Constructs a regression pipeline.
     *
     * @param scaler Scaling strategy used to preprocess features.
     * @param optimizer Optimization strategy used to train the model.
     * @param options Configuration options supplied to the optimizer.
     * @param execStrategy Strategy used to calculate linear outputs.
     * @param logger Generic logger to log details and issues. Defaults to
     * a shared logger at LogLevel::Info if the caller doesn't supply one.
     */
    RegressionPipeline(
        Scaler scaler_,
        Optimizer optimizer,
        GradientDescentOptions<T> options = {},
        ExecutionStrategy<T> execStrategy = {},
        Logger& logger_ = Logger::instance())
        : scaler(std::move(scaler_)),
        // execStrategy isn't needed after construction, so it is moved in;
        // options is still read below for the debug log, so it is copied.
        model(std::move(optimizer), options, std::move(execStrategy)),
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
     * @param trainingData Samples used to fit the scaler and model. 
     * Taken by value: pass an lvalue to keep your own copy unscaled, 
     * or std::move() it to hand over ownership and avoid copy.
     *
     * @throws std::exception If fitting or transformation fails.
     */
    void fit(std::vector<DataPoint<T>> trainingData) {
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
        validateFitted();

        // Use the parameters learned from training data.
        scaler.transform(features);

        return model.predict(features);
    }

    /**
     * @brief Evaluates the fitted pipeline against a held-out test set.
     *
     * @param testData Test samples with unscaled features.
     *
     * @return Regression metrics computed from the pipeline's predictions.
     *
     * @throws std::logic_error If the pipeline has not been fitted.
     * @throws std::invalid_argument If testData is empty.
     */
    [[nodiscard]]
    RegressionMetrics evaluate(const std::vector<DataPoint<T>>& testData) const {
        validateFitted();

        if (true == testData.empty()) {
            throw std::invalid_argument(
                "RegressionPipeline::evaluate: Test data is empty.");
        }

        std::vector<T> predictions;
        std::vector<T> targets;
        predictions.reserve(testData.size());
        targets.reserve(testData.size());

        for (const auto& dataPoint : testData) {
            predictions.push_back(predict(dataPoint.features));
            targets.push_back(dataPoint.target);
        }

        RegressionMetrics result;
        result.mse = metrics::meanSquaredError(predictions, targets);
        result.rmse = std::sqrt(result.mse);
        result.mae = metrics::meanAbsoluteError(predictions, targets);
        result.rSquared = metrics::rSquared(predictions, targets);
        result.withinToleranceRatio =
            metrics::withinToleranceRatio(predictions, targets);

        return result;
    }

private:
    // Ensure that scaling and model parameters are available.
    void validateFitted() const {
        if (false == isFitted) {
            throw std::logic_error(
                "RegressionPipeline::predict: Call fit() first.");
        }
    }

    Scaler scaler;
    LinearRegression<T, Optimizer> model;
    bool isFitted = false;
    Logger& logger;
};