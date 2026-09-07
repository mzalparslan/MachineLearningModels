#include "pch.h"
#include "RobustScaler.h"

TEST(RobustScalerTest, FitAndTransform) {
    RobustScaler<double> scaler;

    // Sorted values: 1, 2, 3, 4, 5
    // median (q2) = 3
    // q1 = 2, q3 = 4 -> IQR = 2
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 1.0 }, 1.0 },
        DataPoint<double>{ { 2.0 }, 2.0 },
        DataPoint<double>{ { 3.0 }, 3.0 },
        DataPoint<double>{ { 4.0 }, 4.0 },
        DataPoint<double>{ { 5.0 }, 5.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 3.0 };
    scaler.transform(features);

    // (3 - 3) / 2 = 0
    EXPECT_DOUBLE_EQ(features[0], 0.0);

    std::vector<double> q3Features = { 4.0 };
    scaler.transform(q3Features);
    // (4 - 3) / 2 = 0.5
    EXPECT_DOUBLE_EQ(q3Features[0], 0.5);
}

TEST(RobustScalerTest, ZeroIQRScaleDefaultsToOne) {
    RobustScaler<double> scaler;

    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 5.0 }, 1.0 },
        DataPoint<double>{ { 5.0 }, 2.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 5.0 };
    scaler.transform(features);

    // (5 - 5) / 1 = 0
    EXPECT_DOUBLE_EQ(features[0], 0.0);
}

TEST(RobustScalerTest, UnfittedTransformThrows) {
    RobustScaler<double> scaler;
    std::vector<double> features = { 1.0 };
    EXPECT_THROW(scaler.transform(features), std::logic_error);
}
