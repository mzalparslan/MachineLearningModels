
#include "CsvDataLoader.h"
#include "DataSplitter.h"
#include "ZScoreScaler.h"
#include "BatchGradientDescent.h"
#include "Logger.h"
#include "BinaryClassificationPipeline.h"
#include "RegressionPipeline.h"
#include "RegressionOutputWriter.h"
#include "BinaryClassificationWriter.h"
#include "DeepNeuralNetwork.h"

#include <iostream>
#include <exception>
#include <type_traits>
#include <utility>
#include <filesystem>

/**
 * @brief Loads, trains, and evaluates a binary-classification pipeline.
 *
 * @tparam T Floating-point mode used for model calculations.
 */
template <typename T>
void runBinaryClassification(Logger &logger) {
	static_assert(std::is_floating_point_v<T>,
		"runBinaryClassification requires floating point data mode!");

	const std::filesystem::path dataFile = "classification.csv";

	ScopedBenchmarkTimer benchmark(logger, "runBinaryClassification");

	logger.info() << "Loading dataset from " << dataFile.string();

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
		ExecutionStrategy<T>{},
		logger
	};

	pipeline.fit(dataSet.trainingData);

	const BinaryClassificationMetrics testMetrics = pipeline.evaluate(dataSet.testData);
	logger.info() << "Test accuracy: " << testMetrics.accuracy
		<< ", precision: " << testMetrics.precision
		<< ", recall: " << testMetrics.recall
		<< ", F1: " << testMetrics.f1Score
		<< ", cross-entropy: " << testMetrics.binaryCrossEntropy;

	const std::filesystem::path outFile{ "binaryOutput.csv" };
	BinaryClassificationWriter::writeCsv(outFile, dataSet.testData, pipeline);

	logger.info() << "Predictions are recorded to: " << outFile.string();
}

/**
 * @brief Loads, trains, and evaluates a linear regression model pipeline.
 *
 * @tparam T Floating-point mode used for model calculations.
 */
template <typename T>
void runLinearRegression(Logger& logger) {
	static_assert(std::is_floating_point_v<T>,
		"runLinearRegression requires floating point data mode!");

	const std::filesystem::path dataFile = "lineardata.csv";

	ScopedBenchmarkTimer benchmark(logger, "runLinearRegression");

	logger.info() << "Linear regression pipeline started.";

	logger.info() << "Loading dataset from " << dataFile.string();

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
		ExecutionStrategy<T>{},
		logger
	};

	pipeline.fit(dataSet.trainingData);

	const RegressionMetrics testMetrics = pipeline.evaluate(dataSet.testData);
	logger.info() << "Test MSE: " << testMetrics.mse
		<< ", RMSE: " << testMetrics.rmse
		<< ", MAE: " << testMetrics.mae
		<< ", R-squared: " << testMetrics.rSquared;

	const std::filesystem::path outFile{ "linearOutput.csv" };
	RegressionOutputWriter::writeCsv(outFile, dataSet.testData, pipeline);

	logger.info() << "Predictions are recorded to: " << outFile.string();
}

/**
 * @brief Trains a small DeepNeuralNetwork on XOR and reports its predictions.
 *
 * XOR is not linearly separable, so it cannot be learned by
 * LinearRegression or LogisticBinaryClassifier -- it exists to
 * demonstrate what a hidden layer buys over a plain linear model.
 *
 * @tparam T Floating-point mode used for model calculations.
 */
template <typename T>
void runNeuralNetwork(Logger& logger) {
	static_assert(std::is_floating_point_v<T>,
		"runNeuralNetwork requires floating point data mode!");

	ScopedBenchmarkTimer benchmark(logger, "runNeuralNetwork");

	logger.info() << "Training DeepNeuralNetwork on XOR.";

	NeuralNetworkOptions<T> options;
	options.learningRate = T(0.5);
	options.epochs = 20000;

	DeepNeuralNetwork<T> model(std::vector<std::size_t>{ 2, 4, 1 }, options);

	const std::vector<DataPoint<T>> trainingSet = {
		DataPoint<T>{ { T(0), T(0) }, T(0) },
		DataPoint<T>{ { T(0), T(1) }, T(1) },
		DataPoint<T>{ { T(1), T(0) }, T(1) },
		DataPoint<T>{ { T(1), T(1) }, T(0) }
	};

	model.fit(trainingSet);

	for (const auto& sample : trainingSet) {
		const T prediction = model.predict(sample.features);
		logger.info() << sample.features[0] << " XOR " << sample.features[1]
			<< " = " << prediction << " (expected " << sample.target << ")";
	}
}

/**
* @brief Run binary classification, linear regression, and a neural
* network demo for sample datasets.
*/
int main() {
	Logger logger(LogLevel::Debug);

	try {
		runBinaryClassification<double>(logger);
	}
	catch (const std::exception& error) {
		logger.critical() << "Binary classification failed!";
		logger.critical() << error.what();

		return 1; // failure
	}

	try {
		runLinearRegression<double>(logger);
	}
	catch (const std::exception& error) {
		logger.critical() << "Linear Regression failed!";
		logger.critical() << error.what();

		return 1; // failure
	}

	try {
		runNeuralNetwork<double>(logger);
	}
	catch (const std::exception& error) {
		logger.critical() << "Neural network demo failed!";
		logger.critical() << error.what();

		return 1; // failure
	}

	return 0;
}