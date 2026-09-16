#include "pch.h"
#include "DeepNeuralNetwork.h"

TEST(DeepNeuralNetworkTest, ConstructorRejectsShortTopology) {
    EXPECT_THROW((DeepNeuralNetwork<double>(std::vector<std::size_t>{ 3 })), std::invalid_argument);
}

TEST(DeepNeuralNetworkTest, ConstructorRejectsZeroSizedLayer) {
    EXPECT_THROW((DeepNeuralNetwork<double>(std::vector<std::size_t>{ 2, 0, 1 })), std::invalid_argument);
}

TEST(DeepNeuralNetworkTest, ConstructorRejectsMultiUnitOutput) {
    EXPECT_THROW((DeepNeuralNetwork<double>(std::vector<std::size_t>{ 2, 4, 2 })), std::invalid_argument);
}

TEST(DeepNeuralNetworkTest, ConstructorRejectsInvalidOptions) {
    NeuralNetworkOptions<double> options;
    options.epochs = 0;

    EXPECT_THROW(
        (DeepNeuralNetwork<double>(std::vector<std::size_t>{ 2, 4, 1 }, options)),
        std::invalid_argument);
}

TEST(DeepNeuralNetworkTest, PredictBeforeFitThrows) {
    DeepNeuralNetwork<double> model(std::vector<std::size_t>{ 2, 4, 1 });

    EXPECT_THROW(model.predict({ 0.0, 0.0 }), std::logic_error);
}

TEST(DeepNeuralNetworkTest, FitRejectsFeatureCountMismatch) {
    DeepNeuralNetwork<double> model(std::vector<std::size_t>{ 2, 4, 1 });

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 0.0 }, 0.0 }
    };

    EXPECT_THROW(model.fit(trainingSet), std::invalid_argument);
}

TEST(DeepNeuralNetworkTest, LearnsXor) {
    NeuralNetworkOptions<double> options;
    options.learningRate = 0.5;
    options.epochs = 20000;
    options.randomSeed = 3;

    DeepNeuralNetwork<double> model(std::vector<std::size_t>{ 2, 4, 1 }, options);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 0.0, 0.0 }, 0.0 },
        DataPoint<double>{ { 0.0, 1.0 }, 1.0 },
        DataPoint<double>{ { 1.0, 0.0 }, 1.0 },
        DataPoint<double>{ { 1.0, 1.0 }, 0.0 }
    };
    model.fit(trainingSet);

    EXPECT_LT(model.predict({ 0.0, 0.0 }), 0.5);
    EXPECT_GT(model.predict({ 0.0, 1.0 }), 0.5);
    EXPECT_GT(model.predict({ 1.0, 0.0 }), 0.5);
    EXPECT_LT(model.predict({ 1.0, 1.0 }), 0.5);
}
