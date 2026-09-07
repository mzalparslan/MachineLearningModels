#include "BenchmarkTimer.h"

void BenchmarkTimer::start() {
    if (true == isRunning_) {
        throw std::logic_error(
            "BenchmarkTimer::start: Timer is already running.");
    }

    startTime_ = Clock::now();
    isRunning_ = true;
}

double BenchmarkTimer::stop() {
    if (false == isRunning_) {
        throw std::logic_error(
            "BenchmarkTimer::stop: Timer has not been started.");
    }

    const Clock::time_point endTime =
        Clock::now();

    isRunning_ = false;

    const Milliseconds elapsed =
        endTime - startTime_;

    return elapsed.count();
}