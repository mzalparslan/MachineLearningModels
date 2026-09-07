#include "pch.h"
#include "RegressionPipeline.h"
#include "NoScaling.h"
#include "BatchGradientDescent.h"

TEST(RegressionPipelineTest, EndToEndPipeline) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 1500;

    RegressionPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, logger
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
        scaler, optimizer, options, logger
    );

    EXPECT_THROW(pipeline.predict({ 1.0 }), std::logic_error);
}
