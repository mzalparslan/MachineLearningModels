#include "pch.h"
#include "DataSplitter.h"
#include <algorithm>
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

TEST(DataSplitterTest, StratifiedSplitPreservesClassBalance) {
    // Heavily imbalanced: 90 majority-class samples, 10 minority-class.
    // A plain uniform split at this ratio can easily hand the test fold
    // zero minority samples; stratified split must not.
    std::vector<DataPoint<double>> samples;
    for (int i = 0; i < 90; ++i) {
        samples.push_back(DataPoint<double>{ { static_cast<double>(i) }, 0.0 });
    }
    for (int i = 0; i < 10; ++i) {
        samples.push_back(DataPoint<double>{ { static_cast<double>(90 + i) }, 1.0 });
    }

    auto dataset = DataSplitter::stratifiedSplit(samples, 0.8, 42);

    const auto countClass = [](const std::vector<DataPoint<double>>& data, double target) {
        return std::count_if(data.begin(), data.end(),
            [target](const DataPoint<double>& point) { return point.target == target; });
    };

    // 80% of 90 majority samples = 72 training / 18 test;
    // 80% of 10 minority samples = 8 training / 2 test.
    EXPECT_EQ(countClass(dataset.trainingData, 0.0), 72);
    EXPECT_EQ(countClass(dataset.testData, 0.0), 18);
    EXPECT_EQ(countClass(dataset.trainingData, 1.0), 8);
    EXPECT_EQ(countClass(dataset.testData, 1.0), 2);
}

TEST(DataSplitterTest, StratifiedSplitThrowsWhenAStratumIsTooSmall) {
    // A single minority sample can't be split 80/20 without landing
    // entirely in one side.
    std::vector<DataPoint<double>> samples;
    for (int i = 0; i < 10; ++i) {
        samples.push_back(DataPoint<double>{ { static_cast<double>(i) }, 0.0 });
    }
    samples.push_back(DataPoint<double>{ { 100.0 }, 1.0 });

    EXPECT_THROW(DataSplitter::stratifiedSplit(samples, 0.8), std::invalid_argument);
}
