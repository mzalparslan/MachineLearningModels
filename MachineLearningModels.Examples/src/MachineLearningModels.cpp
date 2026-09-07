
#include "CsvDataLoader.h"
#include "DataSplitter.h"
#include "ZScoreScaler.h"
#include "BatchGradientDescent.h"
#include "Logger.h"
#include "BinaryClassificationPipeline.h"
#include "RegressionPipeline.h"
#include "RegressionOutputWriter.h"
#include "BinaryClassificationWriter.h"

#include <iostream>
#include <exception>
#include <type_traits>
#include <utility>
#include <filesystem>

/**
 * @brief Loads, trains, and evaluates a binary-classification pipeline.
 *
 * @tparam T Floating-point type used for model calculations.
 */
template <typename T>
void runBinaryClassification(Logger &logger) {
	static_assert(std::is_floating_point_v<T>,
		"runBinaryClassification requires floating point data type!");

	const std::filesystem::path datasetPath = "classification.csv";

	ScopedBenchmarkTimer benchmark(logger, "runBinaryClassification");

	const std::filesystem::path dataFile{ datasetPath };
	logger.info() << "Loading dataset from " << dataFile;

	auto samples = CsvDataLoader::load<T>(dataFile);
	logger.info() << "Count of data loaded: " << samples.size();

	auto dataSet = DataSplitter::trainTestSplit(std::move(samples));
	logger.info() << "Count of training set: " << dataSet.trainingData.size();
	logger.info() << "Count of test set: " << dataSet.testData.size();

	using Pipeline = BinaryClassificationPipeline<
		T,
		ZScoreScaler<T>,
		BatchGradientDescent<T>>;

	Pipeline pipeline{
		ZScoreScaler<T>{},
		BatchGradientDescent<T>{},
		GradientDescentOptions<T>{
			.learningRate = T(0.01),
			.epochs = 100
		},
		logger
	};

	pipeline.fit(dataSet.trainingData);

	const std::filesystem::path outFile{ "binaryOutput.csv" };
	BinaryClassificationWriter::writeCsv(outFile, dataSet.testData, pipeline);

	logger.info() << "Predictions are recorded to: " << outFile.string();
}

/**
 * @brief Loads, trains, and evaluates a linear regression model pipeline.
 *
 * @tparam T Floating-point type used for model calculations.
 */
template <typename T>
void runLinearRegression(Logger& logger) {
	static_assert(std::is_floating_point_v<T>,
		"runLinearRegression requires floating point data type!");

	const std::filesystem::path datasetPath = "lineardata.csv";

	ScopedBenchmarkTimer benchmark(logger, "runLinearRegression");

	logger.info() << "Linear regression pipeline started.";

	const std::filesystem::path dataFile{ datasetPath };
	logger.info() << "Loading dataset from " << dataFile;

	auto samples = CsvDataLoader::load<T>(dataFile);
	logger.info() << "Count of data loaded: " << samples.size();

	auto dataSet = DataSplitter::trainTestSplit(std::move(samples));
	logger.info() << "Training samples: " << dataSet.trainingData.size()
				  << ", Test samples: " << dataSet.testData.size();

	using Pipeline = RegressionPipeline<
		T,
		ZScoreScaler<T>,
		BatchGradientDescent<T>>;

	Pipeline pipeline{
		ZScoreScaler<T>{},
		BatchGradientDescent<T>{},
		GradientDescentOptions<T>{
			.learningRate = T(0.01),
			.epochs = 100
		},
		logger
	};

	pipeline.fit(dataSet.trainingData);

	const std::filesystem::path outFile{ "linearOutput.csv" };
	RegressionOutputWriter::writeCsv(outFile, dataSet.testData, pipeline);

	logger.info() << "Predictions are recorded to: " << outFile.string();
}

/**
* @brief Run binary classification and linear regression for sample datasets.
*/
int main() {
	Logger logger(LogLevel::Debug);

	try {
		runBinaryClassification<float>(logger);
	}
	catch (const std::exception& error) {
		logger.critical() << "Binary classification failed!";
		logger.critical() << error.what();

		return 1; // failure
	}

	try {
		runLinearRegression<float>(logger);
	}
	catch (const std::exception& error) {
		logger.critical() << "Linear Regression failed!";
		logger.critical() << error.what();

		return 1; // failure
	}

	return 0;
}