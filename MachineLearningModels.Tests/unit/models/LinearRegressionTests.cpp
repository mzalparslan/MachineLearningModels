#include "pch.h"
#include "LinearRegression.h"
#include "BatchGradientDescent.h"

TEST(LinearRegressionTest, FitAndPredict) {
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 2000;

    LinearRegression<double, BatchGradientDescent<double>> model(optimizer, options);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    model.fit(trainingSet);

    EXPECT_NEAR(model.predict({ 5.0 }), 11.0, 1e-1);
}

TEST(LinearRegressionTest, PredictBeforeFitThrows) {
    BatchGradientDescent<double> optimizer;
    LinearRegression<double, BatchGradientDescent<double>> model(optimizer);

    EXPECT_THROW(model.predict({ 1.0 }), std::logic_error);
}

TEST(LinearRegressionTest, PredictFeatureMismatchThrows) {
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.epochs = 10;

    LinearRegression<double, BatchGradientDescent<double>> model(optimizer, options);

    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0, 2.0 }, 3.0 }
    };

    model.fit(trainingSet);

    // Fitted on 2 features, predicting with 1 feature
    EXPECT_THROW(model.predict({ 1.0 }), std::invalid_argument);
}

TEST(LinearRegressionTest, FitEmptyDatasetThrows) {
    BatchGradientDescent<double> optimizer;
    LinearRegression<double, BatchGradientDescent<double>> model(optimizer);
    std::vector<DataPoint<double>> emptySet;

    EXPECT_THROW(model.fit(emptySet), std::invalid_argument);
}

TEST(LinearRegressionTest, ParametersExposeLearnedWeightsAndBias) {
    BatchGradientDescent<double> optimizer;
    GradientDescentOptions<double> options;
    options.learningRate = 0.05;
    options.epochs = 2000;

    LinearRegression<double, BatchGradientDescent<double>> model(optimizer, options);

    // y = 2x + 1
    std::vector<DataPoint<double>> trainingSet = {
        DataPoint<double>{ { 1.0 }, 3.0 },
        DataPoint<double>{ { 2.0 }, 5.0 },
        DataPoint<double>{ { 3.0 }, 7.0 },
        DataPoint<double>{ { 4.0 }, 9.0 }
    };

    model.fit(trainingSet);

    const ModelParameters<double>& parameters = model.parameters();
    ASSERT_EQ(parameters.weights.size(), 1u);
    EXPECT_NEAR(parameters.weights[0], 2.0, 1e-1);
    EXPECT_NEAR(parameters.bias, 1.0, 1e-1);
}
