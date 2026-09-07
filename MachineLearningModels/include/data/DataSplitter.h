#pragma once

#include "DataPoint.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * @brief Contains training and test datasets.
 *
 * @tparam T Floating-point data type for values.
 */
template <typename T>
struct Dataset {
	std::vector<DataPoint<T>> trainingData;
	std::vector<DataPoint<T>> testData;
};

/**
 * @brief Provides operations for splitting samples into dataset partitions.
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
     * @tparam T Floating-point data type.
     * @param samples Samples to shuffle and split.
     * @param trainingRatio Proportion assigned to the training dataset.
	 * Must be strictly between 0 and 1. Default is 0.8 (80% training, 20% test).
     * @param randomSeed Seed used by the random-number generator.
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
			throw std::invalid_argument("DataSplitter: Samples vector is empty!");
		}

		// Shuffle samples using the provided random seed.
		std::mt19937 randomGenerator(randomSeed);
		std::shuffle(samples.begin(), samples.end(), randomGenerator);

		const double totalSamples = static_cast<double>(samples.size());
		const double trainingSamples = totalSamples * trainingRatio;

		const std::size_t trainingSize = static_cast<std::size_t>(trainingSamples);
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
};
