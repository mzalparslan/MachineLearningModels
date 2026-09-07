#include "ScopedBenchmarkTimer.h"

ScopedBenchmarkTimer::ScopedBenchmarkTimer(
    Logger& logger,
    std::string_view scopeName)
    : logger(logger),
    scopeName(scopeName) {
    timer.start();
}

ScopedBenchmarkTimer::~ScopedBenchmarkTimer() noexcept {
    try {
        const double elapsedMilliseconds =
            timer.stop();

        logger.debug()
            << scopeName
            << " elapsed time: "
            << elapsedMilliseconds
            << " ms.";
    }
    catch (...) {
        // Exceptions must not escape from a destructor.
    }
}