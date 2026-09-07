#include "pch.h"
#include "MaxAbsoluteScaler.h"

TEST(MaxAbsoluteScalerTest, FitAndTransform) {
    MaxAbsoluteScaler<double> scaler;

    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { -10.0, 5.0 }, 1.0 },
        DataPoint<double>{ { 2.0, -20.0 }, 2.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { -10.0, 10.0 };
    scaler.transform(features);

    // Max absolute for f0 is 10, for f1 is 20
    EXPECT_DOUBLE_EQ(features[0], -1.0);
    EXPECT_DOUBLE_EQ(features[1], 0.5);
}

TEST(MaxAbsoluteScalerTest, AllZerosFeaturePreserved) {
    MaxAbsoluteScaler<double> scaler;

    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 0.0 }, 1.0 },
        DataPoint<double>{ { 0.0 }, 2.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 0.0 };
    scaler.transform(features);

    // Scale defaults to 1 for all zero features
    EXPECT_DOUBLE_EQ(features[0], 0.0);
}

TEST(MaxAbsoluteScalerTest, UnfittedTransformThrows) {
    MaxAbsoluteScaler<double> scaler;
    std::vector<double> features = { 1.0 };
    EXPECT_THROW(scaler.transform(features), std::logic_error);
}
