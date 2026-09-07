#include "pch.h"
#include "DataSplitter.h"
#include <cmath>

TEST(DataSplitterTest, ValidTrainTestSplit) {
    std::vector<DataPoint<double>> samples;
    for (int i = 0; i < 10; ++i) {
        samples.push_back(DataPoint<double>{ { static_cast<double>(i) }, static_cast<double>(i * 2) });
    }

    auto dataset = DataSplitter::trainTestSplit(samples, 0.8, 42);

    EXPECT_EQ(dataset.trainingData.size(), 8u);
    EXPECT_EQ(dataset.testData.size(), 2u);
}

TEST(DataSplitterTest, ReproducibleShuffleWithSeed) {
    std::vector<DataPoint<double>> samples;
    for (int i = 0; i < 20; ++i) {
        samples.push_back(DataPoint<double>{ { static_cast<double>(i) }, static_cast<double>(i) });
    }

    auto dataset1 = DataSplitter::trainTestSplit(samples, 0.5, 123);
    auto dataset2 = DataSplitter::trainTestSplit(samples, 0.5, 123);

    ASSERT_EQ(dataset1.trainingData.size(), dataset2.trainingData.size());
    for (size_t i = 0; i < dataset1.trainingData.size(); ++i) {
        EXPECT_DOUBLE_EQ(dataset1.trainingData[i].features[0], dataset2.trainingData[i].features[0]);
    }
}

TEST(DataSplitterTest, EmptySamplesThrows) {
    std::vector<DataPoint<double>> emptySamples;
    EXPECT_THROW(DataSplitter::trainTestSplit(emptySamples, 0.8), std::invalid_argument);
}

TEST(DataSplitterTest, InvalidRatioThrows) {
    std::vector<DataPoint<double>> samples = { DataPoint<double>{ {1.0}, 2.0 } };

    EXPECT_THROW(DataSplitter::trainTestSplit(samples, 0.0), std::invalid_argument);
    EXPECT_THROW(DataSplitter::trainTestSplit(samples, 1.0), std::invalid_argument);
    EXPECT_THROW(DataSplitter::trainTestSplit(samples, -0.5), std::invalid_argument);
    EXPECT_THROW(DataSplitter::trainTestSplit(samples, 1.5), std::invalid_argument);
    EXPECT_THROW(DataSplitter::trainTestSplit(samples, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}
