#pragma once

#include "DataPoint.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * @brief Contains training and test datasets.
 *
 * @tparam T Floating-point data mode for values.
 */
template <typename T>
struct Dataset {
	std::vector<DataPoint<T>> trainingData;
	std::vector<DataPoint<T>> testData;
};

/**
 * @brief Provides operations to split samples into 
 * training and test datasets.
 *
 * Samples are shuffled to prevent ordering bias 
 * before splitting into training and test sets.
 */
class DataSplitter {
public:
	/**
     * @brief Randomly splits samples into training and test datasets.
     *
     * Random seed makes shuffle reproducible.
     *
     * @tparam T Floating-point data mode.
     * @param samples Samples to shuffle and split.
     * @param trainingRatio Proportion assigned to training dataset.
	 * Must be strictly between 0 and 1. Default is 0.8 (80% training, 20% test).
     * @param randomSeed Seed used by random-number generator.
     *
     * @return Dataset containing training and test datasets.
     *
     * @throws std::invalid_argument If samples is empty, trainingRatio is
	 * invalid, or split ratio is invalid.
     */
	template <typename T>
	[[nodiscard]]
	static Dataset<T> trainTestSplit(
		std::vector<DataPoint<T>> samples,
		double trainingRatio = 0.8,
		std::uint32_t randomSeed = 42) {

		if (false == std::isfinite(trainingRatio)) {
			throw std::invalid_argument("DataSplitter: "
				"Training ratio is not finite!");
		}

		if (trainingRatio <= 0.0 || trainingRatio >= 1.0) {
			throw std::invalid_argument("DataSplitter: "
				"Training ratio must be between 0 and 1!");
		}

		if (true == samples.empty()) {
			throw std::invalid_argument(
				"DataSplitter: Samples vector is empty!");
		}

		// Shuffle samples using the provided random seed.
		std::mt19937 randomGenerator(randomSeed);
		std::shuffle(samples.begin(), samples.end(), randomGenerator);

		const double totalSamples = static_cast<double>(samples.size());
		const double trainingSamples = totalSamples * trainingRatio;

		const std::size_t trainingSize = 
			static_cast<std::size_t>(trainingSamples);
		if (trainingSize == 0 || trainingSize >= samples.size()) {
			throw std::invalid_argument("DataSplitter: Invalid training size!");
		}

		const auto splitPosition = samples.begin() + trainingSize;
		
		Dataset<T> dataset;
		// Move samples into training and test datasets.
		dataset.trainingData.assign(
			std::make_move_iterator(samples.begin()), 
			std::make_move_iterator(splitPosition));

		dataset.testData.assign(
			std::make_move_iterator(splitPosition), 
			std::make_move_iterator(samples.end()));

		return dataset;
	}

	/**
	 * @brief Splits samples into training and test datasets while
	 * preserving each target value's proportion in both splits.
	 *
	 * Uniform shuffling before a plain split can hand an imbalanced
	 * classification set a test fold with almost none of minority
	 * class, which makes accuracy look excellent and mean nothing.
	 * This splits within each distinct target value separately (one
	 * stratum per class) and combines results, so both splits keep
	 * approximately same class balance as full dataset.
	 *
	 * Intended for discrete targets such as binary classification
	 * labels; a continuous regression target would put nearly every
	 * sample in its own single-sample stratum.
	 *
	 * @tparam T Floating-point data mode.
	 * @param samples Samples to shuffle and split.
	 * @param trainingRatio Proportion assigned to training dataset.
	 * Must be strictly between 0 and 1. Default is 0.8 (80% training, 20% test).
	 * @param randomSeed Seed used by random-number generator.
	 *
	 * @return Dataset containing training and test datasets.
	 *
	 * @throws std::invalid_argument If samples is empty, trainingRatio is
	 * invalid, or any one target value has too few samples to split at
	 * that ratio.
	 */
	template <typename T>
	[[nodiscard]]
	static Dataset<T> stratifiedSplit(
		std::vector<DataPoint<T>> samples,
		double trainingRatio = 0.8,
		std::uint32_t randomSeed = 42) {

		if (false == std::isfinite(trainingRatio)) {
			throw std::invalid_argument("DataSplitter: "
				"Training ratio is not finite!");
		}

		if (trainingRatio <= 0.0 || trainingRatio >= 1.0) {
			throw std::invalid_argument("DataSplitter: "
				"Training ratio must be between 0 and 1!");
		}

		if (true == samples.empty()) {
			throw std::invalid_argument("DataSplitter: Samples vector is empty!");
		}

		// Group sample indices by target value; one stratum per distinct value.
		std::map<T, std::vector<std::size_t>> strata;
		for (std::size_t i = 0; i < samples.size(); i++) {
			strata[samples[i].target].push_back(i);
		}

		std::mt19937 randomGenerator(randomSeed);

		Dataset<T> dataset;
		for (auto& [target, indices] : strata) {
			std::shuffle(indices.begin(), indices.end(), randomGenerator);

			const double strataSamples = static_cast<double>(indices.size());
			const std::size_t trainingSize =
				static_cast<std::size_t>(strataSamples * trainingRatio);

			if (0 == trainingSize || trainingSize >= indices.size()) {
				throw std::invalid_argument(
					"DataSplitter::stratifiedSplit: "
					"A target value has too few samples to split at this ratio!");
			}

			for (std::size_t i = 0; i < trainingSize; i++) {
				dataset.trainingData.push_back(std::move(samples[indices[i]]));
			}

			for (std::size_t i = trainingSize; i < indices.size(); i++) {
				dataset.testData.push_back(std::move(samples[indices[i]]));
			}
		}

		// Shuffle the combined splits so samples aren't grouped by stratum.
		std::shuffle(dataset.trainingData.begin(), dataset.trainingData.end(), randomGenerator);
		std::shuffle(dataset.testData.begin(), dataset.testData.end(), randomGenerator);

		return dataset;
	}

	/**
	 * @brief Splits samples into k stratified folds for cross-validation.
	 *
	 * Each distinct target value is shuffled and dealt round-robin across
	 * the folds, so every fold keeps approximately the full dataset's
	 * class balance and fold sizes differ by at most one sample per class.
	 *
	 * Element i of the result uses fold i as its test set and the
	 * remaining k - 1 folds as its training set, so every sample is
	 * tested exactly once across the k datasets.
	 *
	 * @tparam T Floating-point data mode.
	 * @param samples Samples to partition. Not modified.
	 * @param folds Number of folds (k). Must be at least 2.
	 * @param randomSeed Seed used by random-number generator.
	 *
	 * @return k datasets, one per fold.
	 *
	 * @throws std::invalid_argument If samples is empty, folds is less
	 * than 2, or any target value has fewer samples than folds.
	 */
	template <typename T>
	[[nodiscard]]
	static std::vector<Dataset<T>> stratifiedKFold(
		const std::vector<DataPoint<T>>& samples,
		std::size_t folds = 5,
		std::uint32_t randomSeed = 42) {

		if (folds < 2) {
			throw std::invalid_argument(
				"DataSplitter::stratifiedKFold: Fold count must be at least 2!");
		}

		if (true == samples.empty()) {
			throw std::invalid_argument(
				"DataSplitter::stratifiedKFold: Samples vector is empty!");
		}

		// Group sample indices by target value; one stratum per distinct value.
		std::map<T, std::vector<std::size_t>> strata;
		for (std::size_t i = 0; i < samples.size(); i++) {
			strata[samples[i].target].push_back(i);
		}

		std::mt19937 randomGenerator(randomSeed);

		// Deal each shuffled stratum round-robin across folds. The fold
		// counter carries over between strata so remainders are spread
		// across different folds rather than always landing in fold 0.
		std::vector<std::size_t> foldOfSample(samples.size(), 0);
		std::size_t nextFold = 0;
		for (auto& [target, indices] : strata) {
			if (indices.size() < folds) {
				throw std::invalid_argument(
					"DataSplitter::stratifiedKFold: "
					"A target value has fewer samples than folds!");
			}

			std::shuffle(indices.begin(), indices.end(), randomGenerator);

			for (const std::size_t index : indices) {
				foldOfSample[index] = nextFold;
				nextFold = (nextFold + 1) % folds;
			}
		}

		std::vector<Dataset<T>> result(folds);
		for (std::size_t fold = 0; fold < folds; fold++) {
			for (std::size_t i = 0; i < samples.size(); i++) {
				if (fold == foldOfSample[i]) {
					result[fold].testData.push_back(samples[i]);
				}
				else {
					result[fold].trainingData.push_back(samples[i]);
				}
			}

			// Shuffle so order-sensitive learners (per-sample SGD) don't
			// depend on the input ordering.
			std::shuffle(result[fold].trainingData.begin(),
				result[fold].trainingData.end(), randomGenerator);
			std::shuffle(result[fold].testData.begin(),
				result[fold].testData.end(), randomGenerator);
		}

		return result;
	}
};
