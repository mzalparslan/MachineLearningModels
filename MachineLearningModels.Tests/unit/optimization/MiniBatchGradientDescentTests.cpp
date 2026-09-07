#include "pch.h"
#include "MiniBatchGradientDescent.h"
#include <numeric>

TEST(MiniBatchGradientDescentTest, OptimizeLinearHypothesis) {
    MiniBatchGradientDescent<double> mbgd;

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    MiniBatchGradientDescentOptions<double> options;
    options.gradientOptions.learningRate = 0.02;
    options.gradientOptions.epochs = 1500;
    options.batchSize = 2;
    options.shuffle = true;

    ModelParameters<double> params;
    params.weights = { 0.0 };
    params.bias = 0.0;

    auto hypothesis = [](const std::vector<double>& features, const ModelParameters<double>& p) {
        return std::inner_product(p.weights.begin(), p.weights.end(), features.begin(), p.bias);
    };

    EXPECT_NO_THROW(mbgd.optimize(trainingSet, options, params, hypothesis));

    EXPECT_NEAR(params.weights[0], 2.0, 1e-1);
    EXPECT_NEAR(params.bias, 1.0, 1e-1);
}

TEST(MiniBatchGradientDescentTest, ZeroBatchSizeThrows) {
    MiniBatchGradientDescent<double> mbgd;
    std::vector<DataPoint<double>> trainingSet = { DataPoint<double>{ { 1.0 }, 2.0 } };
    MiniBatchGradientDescentOptions<double> options;
    options.batchSize = 0;
    ModelParameters<double> params{ { 0.0 }, 0.0 };

    auto hypothesis = [](const std::vector<double>& f, const ModelParameters<double>& p) { return p.bias; };
    EXPECT_THROW(mbgd.optimize(trainingSet, options, params, hypothesis), std::invalid_argument);
}
