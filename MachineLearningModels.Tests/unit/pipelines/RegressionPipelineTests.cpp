#include "pch.h"
#include "RegressionPipeline.h"
#include "NoScaling.h"
#include "BatchGradientDescent.h"
#include "ZScoreScaler.h"

#include <cmath>

TEST(RegressionPipelineTest, EndToEndPipeline) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 1500;

    RegressionPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, {}, logger
    );

    std::vector<DataPoint<double>> trainingData = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 }
    };

    pipeline.fit(trainingData);

    double pred = pipeline.predict({ 4.0 });
    EXPECT_NEAR(pred, 9.0, 1e-1);
}

TEST(RegressionPipelineTest, PredictUnfittedThrows) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;

    RegressionPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, {}, logger
    );

    EXPECT_THROW(pipeline.predict({ 1.0 }), std::logic_error);
}

TEST(RegressionPipelineTest, EvaluateComputesRegressionMetrics) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 1500;

    RegressionPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, {}, logger
    );

    std::vector<DataPoint<double>> trainingData = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };
    pipeline.fit(trainingData);

    // Evaluate against the same well-fit linear data: predictions should
    // be close to the targets, so error metrics are small and R-squared
    // close to 1.
    const RegressionMetrics result = pipeline.evaluate(trainingData);

    EXPECT_LT(result.mse, 0.1);
    EXPECT_NEAR(result.rmse, std::sqrt(result.mse), 1e-9);
    EXPECT_GT(result.rSquared, 0.95);
}

TEST(RegressionPipelineTest, EvaluateUnfittedThrows) {
    Logger logger(LogLevel::Error);
    RegressionPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        NoScaling<double>{}, BatchGradientDescent<double>{}, {}, {}, logger
    );

    std::vector<DataPoint<double>> testData{ { {1.0}, 3.0 } };
    EXPECT_THROW(pipeline.evaluate(testData), std::logic_error);
}

TEST(RegressionPipelineTest, FitDoesNotModifyCallersData) {
    Logger logger(LogLevel::Error);
    RegressionPipeline<double, ZScoreScaler<double>, BatchGradientDescent<double>>
        pipeline(ZScoreScaler<double>{}, BatchGradientDescent<double>{}, {}, {}, logger);

    std::vector<DataPoint<double>> data{ {{1.0}, 3.0}, {{2.0}, 5.0}, {{3.0}, 7.0} };
    pipeline.fit(data);

    EXPECT_DOUBLE_EQ(data[0].features[0], 1.0);
    EXPECT_DOUBLE_EQ(data[1].features[0], 2.0);
    EXPECT_DOUBLE_EQ(data[2].features[0], 3.0);
}