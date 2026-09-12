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
 * @brief Writes regression predictions to CSV files.
 */
class RegressionOutputWriter {
public:
    RegressionOutputWriter() = delete;

    /**
     * @brief Writes expected targets, predictions, and residual errors.
     *
     * Error is calculated as:
     *
     *     predicted - expected
     *
     * A positive error means the model overestimated the target. A negative
     * error means it underestimated the target.
     *
     * @tparam T Floating-point mode used for targets and predictions.
     * @tparam Pipeline Regression pipeline providing predict().
     *
     * @param filePath Destination CSV file path.
     * @param testData Test samples used to generate predictions.
     * @param pipeline Fitted regression pipeline.
     *
     * @throws std::invalid_argument If testData is empty.
     * @throws std::runtime_error If the output file cannot be opened or written,
     * or if a calculated result is not finite.
     */
    template <typename T, typename Pipeline>
    static void writeCsv(
        const std::filesystem::path& filePath,
        const std::vector<DataPoint<T>>& testData,
        const Pipeline& pipeline)
    {
        static_assert(
            std::is_floating_point_v<T>,
            "RegressionOutputWriter requires a floating-point mode!");

        if (true == testData.empty()) {
            throw std::invalid_argument(
                "RegressionOutputWriter: Test data is empty.");
        }

        std::ofstream output{ filePath };

        if (false == output.is_open()) {
            throw std::runtime_error(
                "RegressionOutputWriter: Unable to open output file: " +
                filePath.string());
        }

        // Preserve enough digits to reconstruct floating-point values.
        output << std::setprecision(
            std::numeric_limits<T>::max_digits10);

        output << "expected,predicted,error\n";

        for (const auto& dataPoint : testData) {
            const T prediction =
                pipeline.predict(dataPoint.features);

            const T error =
                prediction - dataPoint.target;

            if (false == std::isfinite(prediction) ||
                false == std::isfinite(error)) {
                throw std::runtime_error(
                    "RegressionOutputWriter: "
                    "Prediction or error is not finite.");
            }

            output
                << dataPoint.target << ','
                << prediction << ','
                << error << '\n';
        }

        output.flush();

        if (false == output.good()) {
            throw std::runtime_error(
                "RegressionOutputWriter: Error while writing file: " +
                filePath.string());
        }
    }
};