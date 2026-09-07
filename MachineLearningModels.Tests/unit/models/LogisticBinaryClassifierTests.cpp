#include "pch.h"
#include "LogisticBinaryClassifier.h"
#include "BatchGradientDescent.h"

TEST(LogisticBinaryClassifierTest, FitAndPredict) {
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.1;
    options.epochs = 2000;

    LogisticBinaryClassifier<double, BatchGradientDescent<double>> model(optimizer, options);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 0.1 }, 0.0 },
        DataPoint<double>{ { 0.2 }, 0.0 },
        DataPoint<double>{ { 0.8 }, 1.0 },
        DataPoint<double>{ { 0.9 }, 1.0 }
    };

    model.fit(trainingSet);

    double probLow = model.predictProbability({ 0.15 });
    double probHigh = model.predictProbability({ 0.85 });

    EXPECT_LT(probLow, 0.5);
    EXPECT_GT(probHigh, 0.5);

    EXPECT_FALSE(model.predict({ 0.15 }));
    EXPECT_TRUE(model.predict({ 0.85 }));
}

TEST(LogisticBinaryClassifierTest, NonBinaryTargetsThrows) {
    BatchGradientDescent<double> optimizer;
    LogisticBinaryClassifier<double, BatchGradientDescent<double>> model(optimizer);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 2.5 } // Target must be 0 or 1
    };

    EXPECT_THROW(model.fit(trainingSet), std::invalid_argument);
}

TEST(LogisticBinaryClassifierTest, PredictBeforeFitThrows) {
    BatchGradientDescent<double> optimizer;
    LogisticBinaryClassifier<double, BatchGradientDescent<double>> model(optimizer);

    EXPECT_THROW(model.predictProbability({ 1.0 }), std::logic_error);
    EXPECT_THROW(model.predict({ 1.0 }), std::logic_error);
}

TEST(LogisticBinaryClassifierTest, ThresholdValidation) {
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.epochs = 10;
    LogisticBinaryClassifier<double, BatchGradientDescent<double>> model(optimizer, options);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 1.0 }
    };
    model.fit(trainingSet);

    EXPECT_THROW(model.predict({ 1.0 }, 0.0), std::invalid_argument);
    EXPECT_THROW(model.predict({ 1.0 }, 1.0), std::invalid_argument);
    EXPECT_THROW(model.predict({ 1.0 }, -0.1), std::invalid_argument);
    EXPECT_THROW(model.predict({ 1.0 }, 1.1), std::invalid_argument);
}
