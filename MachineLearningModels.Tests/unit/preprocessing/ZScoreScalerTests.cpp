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
