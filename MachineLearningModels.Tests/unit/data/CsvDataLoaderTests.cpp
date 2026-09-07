#include "pch.h"
#include "CsvDataLoader.h"
#include <fstream>
#include <filesystem>

class CsvDataLoaderTest : public ::testing::Test {
protected:
    std::filesystem::path tempFilePath;

    void TearDown() override {
        if (std::filesystem::exists(tempFilePath)) {
            std::filesystem::remove(tempFilePath);
        }
    }

    void createTempCsv(const std::string& content) {
        tempFilePath = std::filesystem::temp_directory_path() / "test_data.csv";
        std::ofstream out(tempFilePath);
        out << content;
        out.close();
    }
};

TEST_F(CsvDataLoaderTest, LoadValidCsvWithHeader) {
    createTempCsv("feature1,feature2,target\n1.0,2.0,5.0\n3.0,4.0,10.0\n");
    
    auto samples = CsvDataLoader::load<double>(tempFilePath, ',', true);
    ASSERT_EQ(samples.size(), 2u);
    
    EXPECT_EQ(samples[0].features.size(), 2u);
    EXPECT_DOUBLE_EQ(samples[0].features[0], 1.0);
    EXPECT_DOUBLE_EQ(samples[0].features[1], 2.0);
    EXPECT_DOUBLE_EQ(samples[0].target, 5.0);

    EXPECT_DOUBLE_EQ(samples[1].features[0], 3.0);
    EXPECT_DOUBLE_EQ(samples[1].features[1], 4.0);
    EXPECT_DOUBLE_EQ(samples[1].target, 10.0);
}

TEST_F(CsvDataLoaderTest, LoadValidCsvWithoutHeader) {
    createTempCsv("1.5,2.5,0.0\n3.5,4.5,1.0\n");

    auto samples = CsvDataLoader::load<float>(tempFilePath, ',', false);
    ASSERT_EQ(samples.size(), 2u);

    EXPECT_FLOAT_EQ(samples[0].features[0], 1.5f);
    EXPECT_FLOAT_EQ(samples[0].target, 0.0f);
}

TEST_F(CsvDataLoaderTest, LoadCustomDelimiter) {
    createTempCsv("f1;f2;target\n10.0;20.0;1.0\n");

    auto samples = CsvDataLoader::load<double>(tempFilePath, ';', true);
    ASSERT_EQ(samples.size(), 1u);
    EXPECT_DOUBLE_EQ(samples[0].features[0], 10.0);
    EXPECT_DOUBLE_EQ(samples[0].features[1], 20.0);
    EXPECT_DOUBLE_EQ(samples[0].target, 1.0);
}

TEST_F(CsvDataLoaderTest, NonExistentFileThrows) {
    std::filesystem::path nonExistent = "non_existent_file_12345.csv";
    EXPECT_THROW(CsvDataLoader::load<double>(nonExistent), std::runtime_error);
}

TEST_F(CsvDataLoaderTest, EmptyFileThrows) {
    createTempCsv("");
    EXPECT_THROW(CsvDataLoader::load<double>(tempFilePath, ',', true), std::runtime_error);
}

TEST_F(CsvDataLoaderTest, InvalidTokenThrows) {
    createTempCsv("f1,f2,target\n1.0,abc,5.0\n");
    EXPECT_THROW(CsvDataLoader::load<double>(tempFilePath, ',', true), std::runtime_error);
}

TEST_F(CsvDataLoaderTest, InsufficientColumnsThrows) {
    createTempCsv("target\n5.0\n");
    EXPECT_THROW(CsvDataLoader::load<double>(tempFilePath, ',', true), std::runtime_error);
}

TEST_F(CsvDataLoaderTest, TrailingDelimiterThrows) {
    createTempCsv("f1,f2,target\n1.0,2.0,5.0,\n");
    EXPECT_THROW(CsvDataLoader::load<double>(tempFilePath, ',', true), std::runtime_error);
}
