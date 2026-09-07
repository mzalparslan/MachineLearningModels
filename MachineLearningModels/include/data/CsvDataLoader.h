#pragma once

#include "DataPoint.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @brief Loads supervised-learning samples from numeric CSV files.
 *
 * Each non-header row must contain one or more feature values followed
 * by the target value in the final column.
 *
 * The loader expects unquoted numeric fields. Quoted values, escaped
 * delimiters, and multiline CSV fields are not supported.
 *
 * All samples are loaded into memory and returned to the caller.
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
     * @tparam T Floating-point type used for features and targets.
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
            "CsvDataLoader requires floating-point data type!");

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
        while (std::getline(file, line)) {
            ++lineNumber;
            
            try {
                samples.push_back(parseSample<T>(line, delimiter));
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
	// Parses a single token into a floating-point value of type T.
    template <typename T>
    static T parseValue(const std::string& token) {
        std::size_t parsedCharacters = 0;

        const long double parsedValue =
            std::stold(token, &parsedCharacters);

        if (parsedCharacters != token.size()) {
            throw std::invalid_argument(
                "CsvDataLoader: Invalid token: " + token);
        }

        const T value = static_cast<T>(parsedValue);
        if (false == std::isfinite(value)) {
            throw std::invalid_argument(
                "CsvDataLoader: Value is NaN, Inf, or out of range!");
        }

        return value;
    }

	// Parses a single CSV line into a DataPoint<T> object.
    template <typename T>
    static DataPoint<T> parseSample(const std::string& line, char delimiter) {

		if (true == line.empty()) {
			throw std::invalid_argument("CsvDataLoader: Empty line in CSV file!");
        }

        if (delimiter == line.back()) {
            throw std::invalid_argument("CsvDataLoader: Line ends with delimiter!");
        }

        std::stringstream stream(line);
        std::string token;
        std::vector<T> rowValues;

        while (std::getline(stream, token, delimiter)) {
            if (true == token.empty()) {
                throw std::invalid_argument(
                    "CsvDataLoader: CSV row contains an empty value!");
            }

            rowValues.push_back(parseValue<T>(token));
        }

        if (rowValues.size() < 2) {
            throw std::invalid_argument(
                "CsvDataLoader: Requires at least 1 feature and 1 target!");
        }

        const T target = rowValues.back();
        rowValues.pop_back();

        return DataPoint<T>{std::move(rowValues), target};
    }
};