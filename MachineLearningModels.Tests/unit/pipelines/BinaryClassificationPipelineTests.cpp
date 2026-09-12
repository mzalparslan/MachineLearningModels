#include "pch.h"
#include "BinaryClassificationPipeline.h"
#include "NoScaling.h"
#include "BatchGradientDescent.h"

TEST(BinaryClassificationPipelineTest, EndToEndPipeline) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.1;
    options.epochs = 1500;

    BinaryClassificationPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        scaler, optimizer, logger, options
    );

    std::vector<DataPoint<double>> trainingData = {
        DataPoint<double>{ { 0.1 }, 0.0 },
        DataPoint<double>{ { 0.9 }, 1.0 }
    };

    pipeline.fit(trainingData);

    EXPECT_FALSE(pipeline.predictClass({ 0.1 }));
    EXPECT_TRUE(pipeline.predictClass({ 0.9 }));
}

TEST(BinaryClassificationPipelineTest, PredictUnfittedThrows) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;

    BinaryClassificationPipeline<double, NoScaling<double>, BatchGradientDescent<double>> pipeline(
        scaler, optimizer, logger, options
    );

    EXPECT_THROW(pipeline.predictClass({ 1.0 }), std::logic_error);
    EXPECT_THROW(pipeline.predict({ 1.0 }), std::logic_error);
}
