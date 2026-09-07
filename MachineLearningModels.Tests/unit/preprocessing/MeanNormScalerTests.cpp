#include "pch.h"
#include "MeanNormScaler.h"

TEST(MeanNormScalerTest, FitAndTransform) {
    MeanNormScaler<double> scaler;

    // f0: min = 10, max = 30, mean = 20, range = 20
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 10.0 }, 1.0 },
        DataPoint<double>{ { 20.0 }, 2.0 },
        DataPoint<double>{ { 30.0 }, 3.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 20.0 };
    scaler.transform(features);

    // (20 - 20) / 20 = 0
    EXPECT_DOUBLE_EQ(features[0], 0.0);

    std::vector<double> minFeatures = { 10.0 };
    scaler.transform(minFeatures);
    // (10 - 20) / 20 = -0.5
    EXPECT_DOUBLE_EQ(minFeatures[0], -0.5);

    std::vector<double> maxFeatures = { 30.0 };
    scaler.transform(maxFeatures);
    // (30 - 20) / 20 = 0.5
    EXPECT_DOUBLE_EQ(maxFeatures[0], 0.5);
}

TEST(MeanNormScalerTest, ConstantFeatureMappedToZero) {
    MeanNormScaler<double> scaler;

    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 5.0 }, 1.0 },
        DataPoint<double>{ { 5.0 }, 2.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 5.0 };
    scaler.transform(features);

    EXPECT_DOUBLE_EQ(features[0], 0.0);
}

TEST(MeanNormScalerTest, UnfittedTransformThrows) {
    MeanNormScaler<double> scaler;
    std::vector<double> features = { 1.0 };
    EXPECT_THROW(scaler.transform(features), std::logic_error);
}
