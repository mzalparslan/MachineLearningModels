#pragma once

#include "BenchmarkTimer.h"
#include "Logger.h"

#include <string>
#include <string_view>

/**
 * @brief Automatically measures and logs the duration of a scope.
 *
 * The timer starts during construction and logs elapsed milliseconds when
 * the object leaves scope, including when the scope exits through an exception.
 *
 * The supplied Logger must outlive the ScopedBenchmarkTimer.
 */
class ScopedBenchmarkTimer {
public:
    /**
     * @brief Starts timing a named scope.
     *
     * @param logger Logger used to report elapsed time.
     * @param scopeName Name of the operation being measured.
     */
    explicit ScopedBenchmarkTimer(
        Logger& logger,
        std::string_view scopeName);

    ScopedBenchmarkTimer(
        const ScopedBenchmarkTimer&) = delete;

    ScopedBenchmarkTimer& operator=(
        const ScopedBenchmarkTimer&) = delete;

    ScopedBenchmarkTimer(
        ScopedBenchmarkTimer&&) = delete;

    ScopedBenchmarkTimer& operator=(
        ScopedBenchmarkTimer&&) = delete;

    /**
     * @brief Stops the timer and logs the elapsed duration.
     */
    ~ScopedBenchmarkTimer() noexcept;

private:
    Logger& logger;
    std::string scopeName;
    BenchmarkTimer timer;
};