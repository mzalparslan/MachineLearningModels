#include "pch.h"
#include "StochasticGradientDescent.h"
#include <numeric>

TEST(StochasticGradientDescentTest, OptimizeLinearHypothesis) {
    StochasticGradientDescent<double> sgd;

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    StochasticGradientDescentOptions<double> options;
    options.gradientOptions.learningRate = 0.01;
    options.gradientOptions.epochs = 1500;
    options.shuffle = true;

    ModelParameters<double> params;
    params.weights = { 0.0 };
    params.bias = 0.0;

    auto hypothesis = [](const std::vector<double>& features, const ModelParameters<double>& p) {
        return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
    };

    EXPECT_NO_THROW(sgd.optimize(trainingSet, options, params, hypothesis));

    EXPECT_NEAR(params.weights[0], 2.0, 1e-1);
    EXPECT_NEAR(params.bias, 1.0, 1e-1);
}

TEST(StochasticGradientDescentTest, InvalidHypothesisThrows) {
    StochasticGradientDescent<double> sgd;
    std::vector<DataPoint<double>> trainingSet = { DataPoint<double>{ { 1.0 }, 2.0 } };
    StochasticGradientDescentOptions<double> options;
    ModelParameters<double> params{ { 0.0 }, 0.0 };

    StochasticGradientDescent<double>::Hypothesis emptyHypothesis;
    EXPECT_THROW(sgd.optimize(trainingSet, options, params, emptyHypothesis), std::invalid_argument);
}

TEST(StochasticGradientDescentTest, NegativeDecayThrows) {
    StochasticGradientDescent<double> sgd;
    std::vector<DataPoint<double>> trainingSet = { DataPoint<double>{ { 1.0 }, 2.0 } };
    StochasticGradientDescentOptions<double> options;
    options.decay = -0.1;
    ModelParameters<double> params{ { 0.0 }, 0.0 };

    auto hypothesis = [](const std::vector<double>& features, const ModelParameters<double>& p) {
        return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
    };

    EXPECT_THROW(sgd.optimize(trainingSet, options, params, hypothesis), std::invalid_argument);
}

TEST(StochasticGradientDescentTest, LearningRateDecayStillConverges) {
    StochasticGradientDescent<double> sgd;

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    StochasticGradientDescentOptions<double> options;
    options.gradientOptions.learningRate = 0.05;
    options.gradientOptions.epochs = 500;
    options.decay = 0.01;

    ModelParameters<double> params{ { 0.0 }, 0.0 };

    auto hypothesis = [](const std::vector<double>& features, const ModelParameters<double>& p) {
        return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
    };

    EXPECT_NO_THROW(sgd.optimize(trainingSet, options, params, hypothesis));
    EXPECT_NEAR(params.weights[0], 2.0, 0.2);
    EXPECT_NEAR(params.bias, 1.0, 0.2);
}
