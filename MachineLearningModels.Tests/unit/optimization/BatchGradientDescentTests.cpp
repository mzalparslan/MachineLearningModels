#include "pch.h"
#include "BatchGradientDescent.h"
#include <numeric>

TEST(BatchGradientDescentTest, OptimizeLinearHypothesis) {
    BatchGradientDescent<double> bgd;
    
    // Dataset for y = 2x + 1
    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 2000;
    options.lambda = 0.0;

    ModelParameters<double> params;
    params.weights = { 0.0 };
    params.bias = 0.0;

    auto hypothesis = [](const std::vector<double>& features, const ModelParameters<double>& p) {
        return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
    };

    EXPECT_NO_THROW(bgd.optimize(trainingSet, options, params, hypothesis));

    EXPECT_NEAR(params.weights[0], 2.0, 1e-2);
    EXPECT_NEAR(params.bias, 1.0, 1e-2);
}

TEST(BatchGradientDescentTest, InvalidHypothesisThrows) {
    BatchGradientDescent<double> bgd;
    std::vector<DataPoint<double>> trainingSet = { DataPoint<double>{ { 1.0 }, 2.0 } };
    GradientDescentOptions<double> options;
    ModelParameters<double> params{ { 0.0 }, 0.0 };

    BatchGradientDescent<double>::Hypothesis emptyHypothesis;
    EXPECT_THROW(bgd.optimize(trainingSet, options, params, emptyHypothesis), std::invalid_argument);
}

TEST(BatchGradientDescentTest, EpochCallbackReceivesDecreasingCost) {
    BatchGradientDescent<double> bgd;

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 200;

    ModelParameters<double> params{ { 0.0 }, 0.0 };

    auto hypothesis = [](const std::vector<double>& features, const ModelParameters<double>& p) {
        return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
    };

    std::vector<double> costPerEpoch;
    bgd.optimize(trainingSet, options, params, hypothesis,
        [&costPerEpoch](std::size_t epoch, double cost) {
            EXPECT_EQ(costPerEpoch.size(), epoch);
            costPerEpoch.push_back(cost);
        });

    ASSERT_EQ(costPerEpoch.size(), options.epochs);
    // Training on a perfectly linear dataset should drive the mean
    // squared residual down substantially from its initial value.
    EXPECT_LT(costPerEpoch.back(), costPerEpoch.front() * 0.5);
}

TEST(BatchGradientDescentTest, InvalidOptionsThrows) {
    BatchGradientDescent<double> bgd;
    std::vector<DataPoint<double>> trainingSet = { DataPoint<double>{ { 1.0 }, 2.0 } };
    ModelParameters<double> params{ { 0.0 }, 0.0 };
    auto hypothesis = [](const std::vector<double>&, const ModelParameters<double>& p) { return p.bias; };

    GradientDescentOptions<double> badLr;
    badLr.learningRate = -0.01;
    EXPECT_THROW(bgd.optimize(trainingSet, badLr, params, hypothesis), std::invalid_argument);

    GradientDescentOptions<double> zeroEpochs;
    zeroEpochs.epochs = 0;
    EXPECT_THROW(bgd.optimize(trainingSet, zeroEpochs, params, hypothesis), std::invalid_argument);
}
