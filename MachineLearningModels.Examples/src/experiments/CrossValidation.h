#pragma once

#include "BinaryEvaluation.h"
#include "DataPoint.h"
#include "DataSplitter.h"
#include "Logger.h"
#include "metrics.h"
#include "ZScoreScaler.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/**
 * @brief Mean and sample standard deviation of one metric across folds.
 */
struct MetricSummary {
	double mean = 0;
	double stdDev = 0;
};

/**
 * @brief Per-metric summary of a binary classifier across k folds.
 */
struct CrossValidationSummary {
	std::size_t folds = 0;
	MetricSummary accuracy;
	MetricSummary precision;
	MetricSummary recall;
	MetricSummary f1Score;
	MetricSummary binaryCrossEntropy;
};

namespace detail {

	// Mean and sample (n - 1) standard deviation of one metric field.
	inline MetricSummary summarizeMetric(
		const std::vector<BinaryClassificationMetrics>& foldResults,
		double BinaryClassificationMetrics::* metric) {

		double sum = 0.0;
		for (const auto& result : foldResults) {
			sum += result.*metric;
		}

		const double count = static_cast<double>(foldResults.size());
		MetricSummary summary;
		summary.mean = sum / count;

		double squaredDifferences = 0.0;
		for (const auto& result : foldResults) {
			const double difference = (result.*metric) - summary.mean;
			squaredDifferences += difference * difference;
		}

		summary.stdDev = foldResults.size() > 1
			? std::sqrt(squaredDifferences / (count - 1.0))
			: 0.0;

		return summary;
	}

} // namespace detail

/**
 * @brief Runs stratified k-fold cross-validation of a binary classifier.
 *
 * For each fold, a Z-score scaler is fitted on that fold's training part
 * only and applied to both parts, so no test-set statistics leak into
 * training. A fresh model is built per fold by calling makeModel(), so
 * folds never share learned parameters.
 *
 * @tparam T Floating-point mode used for model calculations.
 * @tparam ModelFactory Callable returning a new, unfitted model whose
 * fit() accepts std::vector<DataPoint<T>> and whose predict() returns a
 * positive-class probability.
 *
 * @param samples Full labeled dataset (targets must be 0 or 1).
 * @param folds Number of folds (k), at least 2.
 * @param randomSeed Seed for the stratified fold assignment.
 * @param makeModel Factory building a fresh model for each fold.
 *
 * @return Mean and standard deviation of each metric across the folds.
 */
template <typename T, typename ModelFactory>
[[nodiscard]]
CrossValidationSummary crossValidateBinary(
	const std::vector<DataPoint<T>>& samples,
	std::size_t folds,
	std::uint32_t randomSeed,
	ModelFactory makeModel) {

	auto foldDatasets = DataSplitter::stratifiedKFold(samples, folds, randomSeed);

	std::vector<BinaryClassificationMetrics> foldResults;
	foldResults.reserve(foldDatasets.size());

	for (auto& fold : foldDatasets) {
		ZScoreScaler<T> scaler;
		scaler.fit(fold.trainingData);

		for (auto& sample : fold.trainingData) {
			scaler.transform(sample.features);
		}
		for (auto& sample : fold.testData) {
			scaler.transform(sample.features);
		}

		auto model = makeModel();
		model.fit(fold.trainingData);

		foldResults.push_back(scoreBinaryModel<T>(model, fold.testData));
	}

	CrossValidationSummary summary;
	summary.folds = foldResults.size();
	summary.accuracy = detail::summarizeMetric(foldResults, &BinaryClassificationMetrics::accuracy);
	summary.precision = detail::summarizeMetric(foldResults, &BinaryClassificationMetrics::precision);
	summary.recall = detail::summarizeMetric(foldResults, &BinaryClassificationMetrics::recall);
	summary.f1Score = detail::summarizeMetric(foldResults, &BinaryClassificationMetrics::f1Score);
	summary.binaryCrossEntropy = detail::summarizeMetric(
		foldResults, &BinaryClassificationMetrics::binaryCrossEntropy);

	return summary;
}

/**
 * @brief Logs a cross-validation summary as "mean +/- stddev" per metric.
 *
 * @param includeAccuracy Set to false for heavily imbalanced datasets,
 * where accuracy is dominated by the majority class and misleads.
 */
inline void logCrossValidationSummary(
	Logger& logger,
	std::string_view name,
	const CrossValidationSummary& summary,
	bool includeAccuracy = true) {

	auto entry = logger.info();
	entry << name << " (" << summary.folds << "-fold)";

	if (includeAccuracy) {
		entry << " accuracy: " << summary.accuracy.mean
			<< " +/- " << summary.accuracy.stdDev << ",";
	}

	entry << " precision: " << summary.precision.mean << " +/- " << summary.precision.stdDev
		<< ", recall: " << summary.recall.mean << " +/- " << summary.recall.stdDev
		<< ", F1: " << summary.f1Score.mean << " +/- " << summary.f1Score.stdDev
		<< ", cross-entropy: " << summary.binaryCrossEntropy.mean
		<< " +/- " << summary.binaryCrossEntropy.stdDev;
}
