#pragma once

#include "BasicNeuralNetwork.h"
#include "CrossValidation.h"
#include "CsvDataLoader.h"
#include "DeepNeuralNetwork.h"
#include "Logger.h"
#include "LogisticBinaryClassifier.h"
#include "NeuralNetworkOptions.h"
#include "ScopedBenchmarkTimer.h"
#include "StochasticGradientDescent.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <type_traits>
#include <vector>

namespace detail {

	// Finds the (optional, git-ignored) credit card fraud CSV. Searches the
	// working directory, then the repository's resources/external folder
	// relative to both the repository root and the build output folders
	// (build/<config>/bin and bin/<config>/<platform> are three levels
	// below the repository root).
	inline std::optional<std::filesystem::path> findCreditCardCsv() {
		const std::array<std::filesystem::path, 3> candidates = {
			"creditcard.csv",
			"resources/external/creditcard.csv",
			"../../../resources/external/creditcard.csv"
		};

		for (const auto& candidate : candidates) {
			if (std::filesystem::exists(candidate)) {
				return candidate;
			}
		}

		return std::nullopt;
	}

} // namespace detail

/**
 * @brief Evaluates a logistic baseline and both neural networks on the
 * ULB credit card fraud dataset (284,807 transactions, 492 frauds, 0.17%)
 * using stratified 5-fold cross-validation.
 *
 * This is a deliberately *unweighted* baseline: no class weighting, no
 * undersampling, and a fixed 0.5 decision threshold. It shows what the
 * models do with extreme imbalance before any remedy is applied.
 *
 * Accuracy is not reported: always predicting "genuine" already scores
 * ~99.83%, so it cannot distinguish a useful model from a useless one.
 * Precision, recall, F1, and cross-entropy are reported instead.
 *
 * The dataset's `Time` column (seconds since the first transaction) is
 * dropped, leaving 29 features: V1-V28 (PCA components) and Amount.
 *
 * The CSV is ~150 MB and not stored in git (see README). If it cannot be
 * found, the experiment logs a warning and returns without failing.
 *
 * @tparam T Floating-point mode used for model calculations.
 */
template <typename T>
void runCreditCardFraud(Logger& logger) {
	static_assert(std::is_floating_point_v<T>,
		"runCreditCardFraud requires floating point data mode!");

	constexpr std::size_t folds = 5;
	constexpr std::uint32_t seed = 42;

	const auto dataFile = detail::findCreditCardCsv();
	if (false == dataFile.has_value()) {
		logger.warning() << "creditcard.csv not found; skipping fraud experiment. "
			<< "See README for how to download it into resources/external/.";
		return;
	}

	ScopedBenchmarkTimer benchmark(logger, "runCreditCardFraud");

	logger.info() << "Loading dataset from " << dataFile->string();

	auto samples = CsvDataLoader::load<T>(*dataFile);

	// Drop the Time column (first feature).
	std::size_t frauds = 0;
	for (auto& sample : samples) {
		sample.features.erase(sample.features.begin());

		if (sample.target >= T(0.5)) {
			++frauds;
		}
	}

	const double fraudRate =
		static_cast<double>(frauds) / static_cast<double>(samples.size());
	logger.info() << "Samples: " << samples.size() << ", features: "
		<< samples.front().features.size() << ", frauds: " << frauds
		<< " (" << (fraudRate * 100.0) << "%), folds: " << folds;
	logger.info() << "Reference: always predicting 'genuine' scores "
		<< ((1.0 - fraudRate) * 100.0) << "% accuracy with recall 0.";

	// All three models get the same learning rate and number of passes over
	// the training data, and all update per sample, so differences reflect
	// the models rather than their training budgets. (Full-batch gradient
	// descent for the same 20 epochs would be badly undertrained here.)
	constexpr T learningRate = T(0.05);
	constexpr std::size_t epochs = 20;

	{
		ScopedBenchmarkTimer timer(logger, "LogisticBinaryClassifier cross-validation");

		const auto summary = crossValidateBinary<T>(samples, folds, seed, [] {
			return LogisticBinaryClassifier<T, StochasticGradientDescent<T>>(
				StochasticGradientDescent<T>{},
				StochasticGradientDescentOptions<T>{
					.gradientOptions = { .learningRate = learningRate, .epochs = epochs } });
		});

		logCrossValidationSummary(logger, "LogisticBinaryClassifier (SGD)", summary, false);
	}

	NeuralNetworkOptions<T> options;
	options.learningRate = learningRate;
	options.epochs = epochs;

	{
		ScopedBenchmarkTimer timer(logger, "BasicNeuralNetwork cross-validation");

		const auto summary = crossValidateBinary<T>(samples, folds, seed, [&options] {
			return BasicNeuralNetwork<T>(8, options);
		});

		logCrossValidationSummary(logger, "BasicNeuralNetwork(8)", summary, false);
	}

	{
		ScopedBenchmarkTimer timer(logger, "DeepNeuralNetwork cross-validation");

		const auto summary = crossValidateBinary<T>(samples, folds, seed, [&options] {
			return DeepNeuralNetwork<T>(std::vector<std::size_t>{ 29, 16, 8, 1 }, options);
		});

		logCrossValidationSummary(logger, "DeepNeuralNetwork(29-16-8-1)", summary, false);
	}
}
