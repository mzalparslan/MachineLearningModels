#pragma once

#include "DataPoint.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Writes Binary Classification predictions to CSV file.
 */
class BinaryClassificationWriter {
public:
    BinaryClassificationWriter() = delete;

    /**
     * @brief Writes expected classes, probabilities, predicted classes,
     * and correctness results to a CSV file.
     *
     * @tparam T Floating-point mode used by the classifier.
     * @tparam Pipeline Binary-classification pipeline mode.
     *
     * @param filePath Destination CSV file path.
     * @param testData Test samples used to generate predictions.
     * @param pipeline Fitted binary-classification pipeline.
     * @param threshold Minimum probability required for the positive class.
     *
     * @throws std::invalid_argument If testData is empty or threshold is invalid.
     * @throws std::runtime_error If the output file cannot be opened or written.
     */
    template <typename T, typename Pipeline>
    static void writeCsv(
        const std::filesystem::path& filePath,
        const std::vector<DataPoint<T>>& testData,
        const Pipeline& pipeline,
        T threshold = T(0.5))
    {
        static_assert(
            std::is_floating_point_v<T>,
            "BinaryClassificationWriter requires a floating-point mode!");

        if (true == testData.empty()) {
            throw std::invalid_argument(
                "BinaryClassificationWriter: Test data is empty.");
        }

        if (false == std::isfinite(threshold) ||
            threshold <= T(0) ||
            threshold >= T(1)) {
            throw std::invalid_argument(
                "BinaryClassificationWriter: "
                "Threshold must be between zero and one!");
        }

        std::ofstream output{ filePath };

        if (false == output.is_open()) {
            throw std::runtime_error(
                "BinaryClassificationWriter: "
                "Unable to open output file: " +
                filePath.string());
        }

        output << std::setprecision(
            std::numeric_limits<T>::max_digits10);

        output << "expected,probability,predicted,correct\n";

        for (const auto& dataPoint : testData) {
            // Calculate the probability once to avoid scaling twice.
            const T probability = pipeline.predict(dataPoint.features);

            // Classify through the pipeline rather than comparing
            // probability >= threshold here: the decision rule belongs to
            // the model, and reimplementing it in the writer would mean
            // the two silently drift if that rule ever changes.
            const bool predictedClass = pipeline.classify(probability, threshold);

            const bool expectedClass = dataPoint.target >= threshold;

            const bool correct = predictedClass == expectedClass;

            output
                << dataPoint.target << ','
                << probability << ','
                << predictedClass << ','
                << correct << '\n';
        }

        output.flush();

        if (false == output.good()) {
            throw std::runtime_error(
                "BinaryClassificationWriter: "
                "Error while writing output file: " +
                filePath.string());
        }
    }
};