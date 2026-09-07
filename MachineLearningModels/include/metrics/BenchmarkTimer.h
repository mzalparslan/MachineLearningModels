#pragma once

#include <chrono>
#include <stdexcept>

/**
 * @brief Measures elapsed execution time for a code section.
 *
 * The timer uses std::chrono::steady_clock so elapsed measurements are
 * unaffected by system-clock adjustments.
 */
class BenchmarkTimer {
private:
    using Clock = std::chrono::steady_clock;
    using Milliseconds =
        std::chrono::duration<double, std::milli>;

public:
    BenchmarkTimer() = default;

    BenchmarkTimer(const BenchmarkTimer&) = delete;
    BenchmarkTimer& operator=(const BenchmarkTimer&) = delete;

    /**
     * @brief Starts the timer.
     *
     * @throws std::logic_error If the timer is already running.
     */
    void start();

    /**
     * @brief Stops the timer and returns the elapsed duration.
     *
     * @return Elapsed time in milliseconds.
     *
     * @throws std::logic_error If the timer has not been started.
     */
    [[nodiscard]]
    double stop();

private:
    Clock::time_point startTime_;
    bool isRunning_ = false;
};