#include "pch.h"
#include "RegressionOutputWriter.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace {

    class FakeRegressionPipeline {
    public:
        [[nodiscard]]
        double predict(
            const std::vector<double>& features) const {
            return 2.0 * features.at(0) + 1.0;
        }
    };

    class NonFiniteRegressionPipeline {
    public:
        [[nodiscard]]
        double predict(
            const std::vector<double>&) const {
            return std::numeric_limits<double>::infinity();
        }
    };

    class MaximumRegressionPipeline {
    public:
        [[nodiscard]]
        double predict(
            const std::vector<double>&) const {
            return std::numeric_limits<double>::max();
        }
    };

    class RegressionOutputWriterTests
        : public ::testing::Test {
    protected:
        void SetUp() override {
            const std::string testName =
                ::testing::UnitTest::GetInstance()
                ->current_test_info()
                ->name();

            outputPath_ =
                std::filesystem::temp_directory_path() /
                ("regression-writer-" + testName + ".csv");
        }

        void TearDown() override {
            // Test cleanup should not cause an otherwise successful test to fail.
            std::error_code error;
            std::filesystem::remove(outputPath_, error);
        }

        std::filesystem::path outputPath_;
    };

} // namespace

TEST_F(
    RegressionOutputWriterTests,
    WritesExpectedPredictionAndResidualRows)
{
    const std::vector<DataPoint<double>> testData{
        {{2.0}, 4.0}, // Prediction 5, residual 1
        {{3.0}, 8.0}  // Prediction 7, residual -1
    };

    const FakeRegressionPipeline pipeline;

    RegressionOutputWriter::writeCsv(
        outputPath_,
        testData,
        pipeline);

    std::ifstream input{ outputPath_ };

    ASSERT_TRUE(input.is_open());

    std::string header;
    std::string firstRow;
    std::string secondRow;

    ASSERT_TRUE(static_cast<bool>(
        std::getline(input, header)));

    ASSERT_TRUE(static_cast<bool>(
        std::getline(input, firstRow)));

    ASSERT_TRUE(static_cast<bool>(
        std::getline(input, secondRow)));

    EXPECT_EQ(
        header,
        "expected,predicted,error");

    EXPECT_EQ(
        firstRow,
        "4,5,1");

    EXPECT_EQ(
        secondRow,
        "8,7,-1");

    std::string unexpectedRow;

    EXPECT_FALSE(static_cast<bool>(
        std::getline(input, unexpectedRow)));
}

TEST_F(
    RegressionOutputWriterTests,
    EmptyTestDataThrows)
{
    const std::vector<DataPoint<double>> testData;
    const FakeRegressionPipeline pipeline;

    EXPECT_THROW(
        RegressionOutputWriter::writeCsv(
            outputPath_,
            testData,
            pipeline),
        std::invalid_argument);

    // Validation happens before the output file is created.
    EXPECT_FALSE(
        std::filesystem::exists(outputPath_));
}

TEST_F(
    RegressionOutputWriterTests,
    InvalidOutputPathThrows)
{
    const std::vector<DataPoint<double>> testData{
        {{2.0}, 4.0}
    };

    const FakeRegressionPipeline pipeline;

    EXPECT_THROW(
        RegressionOutputWriter::writeCsv(
            std::filesystem::path{},
            testData,
            pipeline),
        std::runtime_error);
}

TEST_F(
    RegressionOutputWriterTests,
    NonFinitePredictionThrows)
{
    const std::vector<DataPoint<double>> testData{
        {{2.0}, 4.0}
    };

    const NonFiniteRegressionPipeline pipeline;

    EXPECT_THROW(
        RegressionOutputWriter::writeCsv(
            outputPath_,
            testData,
            pipeline),
        std::runtime_error);
}

TEST_F(
    RegressionOutputWriterTests,
    NonFiniteResidualThrows)
{
    const std::vector<DataPoint<double>> testData{
        {
            {1.0},
            std::numeric_limits<double>::lowest()
        }
    };

    const MaximumRegressionPipeline pipeline;

    EXPECT_THROW(
        RegressionOutputWriter::writeCsv(
            outputPath_,
            testData,
            pipeline),
        std::runtime_error);
}