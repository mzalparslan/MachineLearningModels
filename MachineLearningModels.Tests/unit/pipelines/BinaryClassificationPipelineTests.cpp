#include "pch.h"
#include "BinaryClassificationPipeline.h"
#include "NoScaling.h"
#include "BatchGradientDescent.h"
#include "ZScoreScaler.h"

TEST(BinaryClassificationPipelineTest, EndToEndPipeline) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.1;
    options.epochs = 1500;

    BinaryClassificationPipeline<double,
        NoScaling<double>,
        BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, {}, logger
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

    BinaryClassificationPipeline<double,
        NoScaling<double>,
        BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, {}, logger
    );

    EXPECT_THROW(pipeline.predictClass({ 1.0 }), std::logic_error);
    EXPECT_THROW(pipeline.predict({ 1.0 }), std::logic_error);
}

TEST(BinaryClassificationPipelineTest, EvaluateComputesClassificationMetrics) {
    Logger logger(LogLevel::Error);
    NoScaling<double> scaler;
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.1;
    options.epochs = 1500;

    BinaryClassificationPipeline<double,
        NoScaling<double>,
        BatchGradientDescent<double>> pipeline(
        scaler, optimizer, options, {}, logger
    );

    std::vector<DataPoint<double>> trainingData = {
        DataPoint<double>{ { 0.1 }, 0.0 },
        DataPoint<double>{ { 0.9 }, 1.0 }
    };
    pipeline.fit(trainingData);

    const BinaryClassificationMetrics result = pipeline.evaluate(trainingData);

    EXPECT_DOUBLE_EQ(result.accuracy, 1.0);
    EXPECT_DOUBLE_EQ(result.precision, 1.0);
    EXPECT_DOUBLE_EQ(result.recall, 1.0);
    EXPECT_DOUBLE_EQ(result.f1Score, 1.0);
    EXPECT_GT(result.binaryCrossEntropy, 0.0);
}

TEST(BinaryClassificationPipelineTest, EvaluateUnfittedThrows) {
    Logger logger(LogLevel::Error);
    BinaryClassificationPipeline<double,
        NoScaling<double>,
        BatchGradientDescent<double>> pipeline(
        NoScaling<double>{}, BatchGradientDescent<double>{}, {}, {}, logger
    );

    std::vector<DataPoint<double>> testData{ { {1.0}, 0.0 } };
    EXPECT_THROW(pipeline.evaluate(testData), std::logic_error);
}

TEST(BinaryClassificationPipelineTest, FitDoesNotModifyCallersData) {
    Logger logger(LogLevel::Error);
    BinaryClassificationPipeline<double,
        ZScoreScaler<double>,
        BatchGradientDescent<double>> pipeline(
        ZScoreScaler<double>{},
        BatchGradientDescent<double>{},
        {},
        {},
        logger
    );

    std::vector<DataPoint<double>> data{
        { {1.0}, 0.0 },
        { {2.0}, 0.0 },
        { {3.0}, 1.0 },
        { {4.0}, 1.0 }
    };

    pipeline.fit(data);

    EXPECT_DOUBLE_EQ(data[0].features[0], 1.0);
    EXPECT_DOUBLE_EQ(data[1].features[0], 2.0);
    EXPECT_DOUBLE_EQ(data[2].features[0], 3.0);
    EXPECT_DOUBLE_EQ(data[3].features[0], 4.0);
}