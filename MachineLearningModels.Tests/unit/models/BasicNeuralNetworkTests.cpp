#include "pch.h"
#include "BasicNeuralNetwork.h"

TEST(BasicNeuralNetworkTest, ConstructorRejectsZeroHiddenSize) {
    EXPECT_THROW((BasicNeuralNetwork<double>(0)), std::invalid_argument);
}

TEST(BasicNeuralNetworkTest, ConstructorRejectsInvalidOptions) {
    NeuralNetworkOptions<double> options;
    options.learningRate = 0.0;

    EXPECT_THROW((BasicNeuralNetwork<double>(4, options)), std::invalid_argument);
}

TEST(BasicNeuralNetworkTest, PredictBeforeFitThrows) {
    BasicNeuralNetwork<double> model(4);

    EXPECT_THROW(model.predict({ 0.0 }), std::logic_error);
}

TEST(BasicNeuralNetworkTest, FitRejectsEmptyTrainingSet) {
    BasicNeuralNetwork<double> model(4);

    EXPECT_THROW(model.fit({}), std::invalid_argument);
}

TEST(BasicNeuralNetworkTest, PredictRejectsFeatureCountMismatch) {
    BasicNeuralNetwork<double> model(4);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 0.0 }, 0.0 },
        DataPoint<double>{ { 1.0 }, 1.0 }
    };
    model.fit(trainingSet);

    EXPECT_THROW(model.predict({ 0.0, 1.0 }), std::invalid_argument);
}

TEST(BasicNeuralNetworkTest, LearnsSeparableThreshold) {
    NeuralNetworkOptions<double> options;
    options.learningRate = 0.5;
    options.epochs = 5000;
    options.randomSeed = 7;

    BasicNeuralNetwork<double> model(4, options);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 0.0 }, 0.0 },
        DataPoint<double>{ { 0.2 }, 0.0 },
        DataPoint<double>{ { 0.8 }, 1.0 },
        DataPoint<double>{ { 1.0 }, 1.0 }
    };
    model.fit(trainingSet);

    const double low = model.predict({ 0.0 });
    const double high = model.predict({ 1.0 });

    EXPECT_LT(low, 0.5);
    EXPECT_GT(high, 0.5);
    EXPECT_LT(low, high);
}
