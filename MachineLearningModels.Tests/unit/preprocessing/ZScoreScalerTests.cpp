#include "pch.h"
#include "ZScoreScaler.h"
#include <cmath>

TEST(ZScoreScalerTest, FitAndTransformNormalData) {
    ZScoreScaler<double> scaler;

    // Mean = 2.0, StdDev = std::sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = std::sqrt(2/3) ~ 0.81649658
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 1.0 }, 1.0 },
        DataPoint<double>{ { 2.0 }, 2.0 },
        DataPoint<double>{ { 3.0 }, 3.0 }
    };

    EXPECT_NO_THROW(scaler.fit(dataset));

    std::vector<double> features = { 2.0 };
    scaler.transform(features);

    // Mean feature should transform to 0.0
    EXPECT_NEAR(features[0], 0.0, 1e-9);
}

TEST(ZScoreScalerTest, ConstantFeatureHandling) {
    ZScoreScaler<double> scaler;

    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 5.0 }, 1.0 },
        DataPoint<double>{ { 5.0 }, 2.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 5.0 };
    scaler.transform(features);

    // Constant feature with sigma set to 1.0: (5 - 5) / 1 = 0
    EXPECT_DOUBLE_EQ(features[0], 0.0);
}

TEST(ZScoreScalerTest, UnfittedTransformThrows) {
    ZScoreScaler<double> scaler;
    std::vector<double> features = { 1.0 };
    EXPECT_THROW(scaler.transform(features), std::logic_error);
}

TEST(ZScoreScalerTest, NaNFeatureInFitThrows) {
    ZScoreScaler<double> scaler;
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { std::numeric_limits<double>::quiet_NaN() }, 1.0 }
    };
    EXPECT_THROW(scaler.fit(dataset), std::invalid_argument);
}

TEST(ZScoreScalerTest, TransformReusesTrainingStatisticsForOutOfDistributionRow) {
    ZScoreScaler<double> scaler;

    // Training set: mean = 10, sigma = sqrt(((9-10)^2+(10-10)^2+(11-10)^2)/3)
    //             = sqrt(2/3).
    std::vector<DataPoint<double>> trainingData = {
        DataPoint<double>{ { 9.0 }, 0.0 },
        DataPoint<double>{ { 10.0 }, 0.0 },
        DataPoint<double>{ { 11.0 }, 0.0 }
    };
    scaler.fit(trainingData);

    // A row far outside the training distribution. If transform() leaked
    // test data (re-derived statistics from this row instead of reusing
    // the fitted training mean/sigma), a single value has zero spread and
    // would collapse to 0; reusing the training statistics instead
    // produces a large, precisely predictable value.
    std::vector<double> testFeatures = { 1000.0 };
    scaler.transform(testFeatures);

    const double trainingSigma = std::sqrt(2.0 / 3.0);
    const double expected = (1000.0 - 10.0) / trainingSigma;

    EXPECT_NEAR(testFeatures[0], expected, 1e-6);
}
