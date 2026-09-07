#include "pch.h"
#include "NoScaling.h"

TEST(NoScalingTest, NoOpFitAndTransform) {
    NoScaling<double> scaler;

    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 10.0, 20.0 }, 1.0 }
    };

    EXPECT_NO_THROW(scaler.fit(dataset));

    std::vector<double> features = { 10.0, 20.0 };
    EXPECT_NO_THROW(scaler.transform(features));

    // Values remain unmodified
    EXPECT_DOUBLE_EQ(features[0], 10.0);
    EXPECT_DOUBLE_EQ(features[1], 20.0);
}

TEST(NoScalingTest, TransformWithoutFitDoesNotThrow) {
    NoScaling<double> scaler;
    std::vector<double> features = { 5.5, 6.5 };
    EXPECT_NO_THROW(scaler.transform(features));
    EXPECT_DOUBLE_EQ(features[0], 5.5);
    EXPECT_DOUBLE_EQ(features[1], 6.5);
}
