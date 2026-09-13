#include "pch.h"
#include "BinaryClassificationWriter.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

class FakeBinaryClassificationPipeline {
public:
    [[nodiscard]]
    double predict(
        const std::vector<double>& features) const {
        return features.front();
    }

    // Mirrors BinaryClassificationPipeline::classify's signature so the
    // writer can be tested against a fake without a real fitted model.
    [[nodiscard]]
    static bool classify(double probability, double threshold) {
        return probability >= threshold;
    }
};

TEST(BinaryClassificationWriterTests, WritesExpectedProbabilityPredictionAndCorrectness)
{
    const std::filesystem::path outputPath =
        std::filesystem::temp_directory_path() /
        "binary-classification-writer-test.csv";

    const std::vector<DataPoint<double>> testData{
        {{0.75}, 1.0},
        {{0.25}, 0.0}
    };

    const FakeBinaryClassificationPipeline pipeline;

    BinaryClassificationWriter::writeCsv(
        outputPath,
        testData,
        pipeline,
        0.5);

    std::ifstream input{ outputPath };

    ASSERT_TRUE(input.is_open());

    std::string header;
    std::string firstRow;
    std::string secondRow;

    ASSERT_TRUE(static_cast<bool>(std::getline(input, header)));

    ASSERT_TRUE(static_cast<bool>(std::getline(input, firstRow)));

    ASSERT_TRUE(static_cast<bool>(std::getline(input, secondRow)));

    EXPECT_EQ(header, "expected,probability,predicted,correct");

    EXPECT_EQ(firstRow, "1,0.75,1,1");

    EXPECT_EQ(secondRow, "0,0.25,0,1");

    input.close();

    std::filesystem::remove(outputPath);
}

TEST(BinaryClassificationWriterTests, UsesSpecifiedClassificationThreshold)
{
    const std::filesystem::path outputPath =
        std::filesystem::temp_directory_path() /
        "binary-classification-threshold-test.csv";

    const std::vector<DataPoint<double>> testData{
        {{0.75}, 0.0}
    };

    const FakeBinaryClassificationPipeline pipeline;

    BinaryClassificationWriter::writeCsv(
        outputPath,
        testData,
        pipeline,
        0.8);

    std::ifstream input{ outputPath };

    ASSERT_TRUE(input.is_open());

    std::string line;

    std::getline(input, line); // Header
    ASSERT_TRUE(static_cast<bool>(
        std::getline(input, line)));

    // Probability 0.75 is below the supplied threshold of 0.8.
    EXPECT_EQ(line, "0,0.75,0,1");

    input.close();
    std::filesystem::remove(outputPath);
}

TEST(BinaryClassificationWriterTests, RejectsInvalidThreshold)
{
    const FakeBinaryClassificationPipeline pipeline;

    const std::vector<DataPoint<double>> testData{
        {{0.5}, 1.0}
    };

    EXPECT_THROW(
        BinaryClassificationWriter::writeCsv(
            "unused.csv",
            testData,
            pipeline,
            -0.1),
        std::invalid_argument);

    EXPECT_THROW(
        BinaryClassificationWriter::writeCsv(
            "unused.csv",
            testData,
            pipeline,
            1.1),
        std::invalid_argument);
}

TEST(BinaryClassificationWriterTests, RejectsBoundaryThresholds)
{
    // 0.0 and 1.0 can never fire (probability >= 0.0 is always true,
    // probability >= 1.0 is never true for a value in [0, 1)), so the
    // threshold bound is the open interval (0, 1) -- matching
    // LogisticBinaryClassifier::predictClass rather than accepting the
    // closed endpoints.
    const FakeBinaryClassificationPipeline pipeline;

    const std::vector<DataPoint<double>> testData{
        {{0.5}, 1.0}
    };

    EXPECT_THROW(
        BinaryClassificationWriter::writeCsv(
            "unused.csv",
            testData,
            pipeline,
            0.0),
        std::invalid_argument);

    EXPECT_THROW(
        BinaryClassificationWriter::writeCsv(
            "unused.csv",
            testData,
            pipeline,
            1.0),
        std::invalid_argument);
}

TEST(BinaryClassificationWriterTests, RejectsEmptyTestData)
{
    const FakeBinaryClassificationPipeline pipeline;
    const std::vector<DataPoint<double>> emptyTestData;

    EXPECT_THROW(
        BinaryClassificationWriter::writeCsv(
            "unused.csv",
            emptyTestData,
            pipeline),
        std::invalid_argument);
}

