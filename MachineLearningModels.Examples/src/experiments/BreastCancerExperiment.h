#pragma once

#include "BasicNeuralNetwork.h"
#include "BatchGradientDescent.h"
#include "CrossValidation.h"
#include "CsvDataLoader.h"
#include "DeepNeuralNetwork.h"
#include "Logger.h"
#include "LogisticBinaryClassifier.h"
#include "NeuralNetworkOptions.h"
#include "ScopedBenchmarkTimer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

/**
 * @brief Compares a logistic baseline against both neural networks on the
 * Breast Cancer Wisconsin (Diagnostic) dataset (UCI, 569 samples,
 * 30 features, malignant = 1 / benign = 0) using stratified 5-fold
 * cross-validation.
 *
 * Features span several orders of magnitude (area is ~1000, smoothness
 * ~0.1), so each fold scales features with a Z-score scaler fitted on
 * that fold's training part only.
 *
 * Reporting mean +/- standard deviation across folds shows whether a gap
 * between models is larger than the split-to-split variation; a single
 * 20% test split of 569 samples is only ~115 samples, where one sample is
 * a ~0.9 point accuracy swing.
 *
 * @tparam T Floating-point mode used for model calculations.
 */
template <typename T>
void runBreastCancer(Logger& logger) {
	static_assert(std::is_floating_point_v<T>,
		"runBreastCancer requires floating point data mode!");

	constexpr std::size_t folds = 5;
	constexpr std::uint32_t seed = 42;

	const std::filesystem::path dataFile = "breast_cancer.csv";

	ScopedBenchmarkTimer benchmark(logger, "runBreastCancer");

	logger.info() << "Loading dataset from " << dataFile.string();

	const auto samples = CsvDataLoader::load<T>(dataFile);
	logger.info() << "Samples: " << samples.size() << ", folds: " << folds;

	NeuralNetworkOptions<T> options;
	options.learningRate = T(0.05);
	options.epochs = 200;

	{
		ScopedBenchmarkTimer timer(logger, "LogisticBinaryClassifier cross-validation");

		const auto summary = crossValidateBinary<T>(samples, folds, seed, [] {
			return LogisticBinaryClassifier<T, BatchGradientDescent<T>>(
				BatchGradientDescent<T>{},
				GradientDescentOptions<T>{ .learningRate = T(0.1), .epochs = 1000 });
		});

		logCrossValidationSummary(logger, "LogisticBinaryClassifier", summary);
	}

	{
		ScopedBenchmarkTimer timer(logger, "BasicNeuralNetwork cross-validation");

		const auto summary = crossValidateBinary<T>(samples, folds, seed, [&options] {
			return BasicNeuralNetwork<T>(8, options);
		});

		logCrossValidationSummary(logger, "BasicNeuralNetwork(8)", summary);
	}

	{
		ScopedBenchmarkTimer timer(logger, "DeepNeuralNetwork cross-validation");

		const auto summary = crossValidateBinary<T>(samples, folds, seed, [&options] {
			return DeepNeuralNetwork<T>(std::vector<std::size_t>{ 30, 16, 8, 1 }, options);
		});

		logCrossValidationSummary(logger, "DeepNeuralNetwork(30-16-8-1)", summary);
	}
}
