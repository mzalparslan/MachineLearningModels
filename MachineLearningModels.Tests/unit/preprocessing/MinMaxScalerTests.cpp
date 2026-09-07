#include "pch.h"
#include "MinMaxScaler.h"
#include <limits>

TEST(MinMaxScalerTest, ConstructorValidation) {
    EXPECT_NO_THROW(MinMaxScaler<double>(0.0, 1.0));
    EXPECT_NO_THROW(MinMaxScaler<double>(-1.0, 1.0));
    
    // targetMax <= targetMin
    EXPECT_THROW(MinMaxScaler<double>(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(MinMaxScaler<double>(1.0, 1.0), std::invalid_argument);

    // Non-finite parameters
    EXPECT_THROW(MinMaxScaler<double>(std::numeric_limits<double>::quiet_NaN(), 1.0), std::invalid_argument);
    EXPECT_THROW(MinMaxScaler<double>(0.0, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(MinMaxScalerTest, FitAndTransformNormalData) {
    MinMaxScaler<double> scaler(0.0, 1.0);
    
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 10.0, 200.0 }, 1.0 },
        DataPoint<double>{ { 20.0, 400.0 }, 2.0 },
        DataPoint<double>{ { 30.0, 600.0 }, 3.0 }
    };

    EXPECT_NO_THROW(scaler.fit(dataset));

    std::vector<double> features = { 20.0, 400.0 };
    scaler.transform(features);

    EXPECT_DOUBLE_EQ(features[0], 0.5);
    EXPECT_DOUBLE_EQ(features[1], 0.5);

    std::vector<double> minFeatures = { 10.0, 200.0 };
    scaler.transform(minFeatures);
    EXPECT_DOUBLE_EQ(minFeatures[0], 0.0);
    EXPECT_DOUBLE_EQ(minFeatures[1], 0.0);

    std::vector<double> maxFeatures = { 30.0, 600.0 };
    scaler.transform(maxFeatures);
    EXPECT_DOUBLE_EQ(maxFeatures[0], 1.0);
    EXPECT_DOUBLE_EQ(maxFeatures[1], 1.0);
}

TEST(MinMaxScalerTest, ConstantFeatureMappedToMidpoint) {
    MinMaxScaler<double> scaler(0.0, 1.0);
    
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 5.0, 10.0 }, 1.0 },
        DataPoint<double>{ { 5.0, 20.0 }, 2.0 }
    };

    scaler.fit(dataset);

    std::vector<double> features = { 5.0, 15.0 };
    scaler.transform(features);

    // Feature 0 is constant, should be mapped to midpoint 0.5
    EXPECT_DOUBLE_EQ(features[0], 0.5);
    EXPECT_DOUBLE_EQ(features[1], 0.5);
}

TEST(MinMaxScalerTest, TransformBeforeFitThrows) {
    MinMaxScaler<double> scaler;
    std::vector<double> features = { 1.0, 2.0 };
    EXPECT_THROW(scaler.transform(features), std::logic_error);
}

TEST(MinMaxScalerTest, FitEmptyDatasetThrows) {
    MinMaxScaler<double> scaler;
    std::vector<DataPoint<double>> emptyDataset;
    EXPECT_THROW(scaler.fit(emptyDataset), std::invalid_argument);
}

TEST(MinMaxScalerTest, TransformInconsistentFeatureSizeThrows) {
    MinMaxScaler<double> scaler;
    std::vector<DataPoint<double>> dataset = {
        DataPoint<double>{ { 1.0, 2.0 }, 1.0 }
    };
    scaler.fit(dataset);

    std::vector<double> invalidFeatures = { 1.0 };
    EXPECT_THROW(scaler.transform(invalidFeatures), std::invalid_argument);
}
