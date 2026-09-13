#pragma once

#include "DataPoint.h"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @brief Loads supervised learning samples from CSV file.
 *
 * Each row must contain one or more feature values followed
 * by target value in final column.
 * Header line is optional and skipped if present.
 *
 * Loader expects unquoted numeric fields. Quoted values, escaped
 * delimiters, and multiline CSV fields are not supported.
 *
 * All samples are loaded into memory and returned to caller.
 */
class CsvDataLoader {
	// Prevent instantiation of this class.
    CsvDataLoader() = delete;

public:
    /**
     * @brief Loads labeled samples from a CSV file.
     *
     * When hasHeader is true, first line is skipped. 
	 * File is parsed line by line. Each line is split 
     * into tokens using delimiter.
     * Last column in line is target value and 
     * those preceding columns are parsed as features.
     *
     * @tparam T Floating-point mode used for features and targets.
     *
     * @param filePath Full path and filename of CSV file to load.
	 * @param delimiter Character separating columns in CSV file (Default: comma).
     * @param hasHeader Whether first line contains column names.
     *
     * @return Samples parsed from CSV file.
     *
     * @throws std::runtime_error If file cannot be opened or read,
     * contains no samples, empty lines, or contains an invalid data.
     */
    template <typename T>
    [[nodiscard]]
    static std::vector<DataPoint<T>> load(
        const std::filesystem::path& filePath,
        char delimiter = ',',
        bool hasHeader = true)
    {
        static_assert(
            std::is_floating_point_v<T>,
            "CsvDataLoader requires floating-point data mode!");

        std::ifstream file(filePath);
        if (false == file.is_open()) {
            throw std::runtime_error(
                "CsvDataLoader: Unable to open file: " + 
                filePath.string());
        }

        std::vector<DataPoint<T>> samples;
        std::string line;
		// Skip header line if assigned as present. Header line not parsed, 
        // but can be used to validate column count if needed.
        if (true == hasHeader) {
            if (!std::getline(file, line)) {
                throw std::runtime_error(
                    "CsvDataLoader: Unable to read header: " +
                    filePath.string());
            }
        }

        std::size_t lineNumber = hasHeader ? 1 : 0;
        std::size_t expectedColumns = 0;
        while (std::getline(file, line)) {
            lineNumber++;

            // Tolerate CRLF line endings (files authored on Windows,
            // read back on a platform whose streams don't strip '\r').
            if (false == line.empty() && '\r' == line.back()) {
                line.pop_back();
            }

            try {
                DataPoint<T> sample = parseSample<T>(line, delimiter, expectedColumns);

                if (0 == expectedColumns) {
                    expectedColumns = sample.features.size() + 1;
                }

                samples.push_back(std::move(sample));
            }
            catch (const std::exception& error) {
                throw std::runtime_error(
                    "CsvDataLoader: Invalid data at line " +
                    std::to_string(lineNumber) + ": " +
                    error.what());
            }
        }

        if (true == file.bad()) {
            throw std::runtime_error(
                "CsvDataLoader: Error reading file: " +
                filePath.string());
        }

        if (true == samples.empty()) {
            throw std::runtime_error(
                "CsvDataLoader: File contains no samples.");
        }

        return samples;
    }

private:
	// Parses a single token into a floating-point value of mode T.
	// Uses std::from_chars rather than std::stold: it is locale-independent
	// (stold honours the current C locale, where some locales use ','
	// as the decimal separator -- same character as our default
	// delimiter), does not throw, and avoids an intermediate allocation.
    template <typename T>
    static T parseValue(std::string_view token) {
        T value{};

        const auto result = std::from_chars(
            token.data(), token.data() + token.size(), value);
		
        if (result.ec != std::errc{} ||
            result.ptr != token.data() + token.size()) {
            throw std::invalid_argument(
                "CsvDataLoader: Invalid token: " + std::string(token));
        }

        if (false == std::isfinite(value)) {
            throw std::invalid_argument(
                "CsvDataLoader: Value is NaN, Inf, or out of range!");
        }

        return value;
    }

	// Parses a single CSV line into a DataPoint<T> object.
	//
	// @param expectedColumns Column count (features + target) every row
	// must match, or zero to accept this row's count as authoritative.
    template <typename T>
    static DataPoint<T> parseSample(
        std::string_view line, char delimiter, std::size_t expectedColumns) {

		if (true == line.empty()) {
			throw std::invalid_argument("CsvDataLoader: Empty line in CSV file!");
        }

        if (delimiter == line.back()) {
            throw std::invalid_argument("CsvDataLoader: Line ends with delimiter!");
        }

        std::vector<T> rowValues;

        std::size_t tokenStart = 0;
        while (true) {
            const std::size_t delimiterPosition = line.find(delimiter, tokenStart);
			// Extract token from line, either up to next delimiter or to end of line.
            const std::string_view token = (std::string_view::npos == delimiterPosition)
                ? line.substr(tokenStart)
                : line.substr(tokenStart, delimiterPosition - tokenStart);

            if (true == token.empty()) {
                throw std::invalid_argument(
                    "CsvDataLoader: CSV row contains an empty value!");
            }

            rowValues.push_back(parseValue<T>(token));
			// If no more delimiters, break out of loop.
            if (std::string_view::npos == delimiterPosition) {
                break;
            }

			// Skip delimiter and start next token after it.
            tokenStart = delimiterPosition + 1;
        }

        if (rowValues.size() < 2) {
            throw std::invalid_argument(
                "CsvDataLoader: Requires at least 1 feature and 1 target!");
        }

        if (0 != expectedColumns && rowValues.size() != expectedColumns) {
            throw std::invalid_argument(
                "CsvDataLoader: Expected " + std::to_string(expectedColumns) +
                " columns but found " + std::to_string(rowValues.size()) + "!");
        }

        const T target = rowValues.back();
        rowValues.pop_back();

        return DataPoint<T>{std::move(rowValues), target};
    }
};