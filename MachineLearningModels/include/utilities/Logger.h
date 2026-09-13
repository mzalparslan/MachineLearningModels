#pragma once

#include <iostream>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>

/**
 * @brief Severity levels supported by Logger.
 */
enum class LogLevel {
    Debug = 0,
    Info,
    Warning,
    Error,
    Critical
};

/**
 * @brief Provides simple non-thread-safe stream-based logging.
 *
 * @warning Not safe to call from multiple threads concurrently, including
 * from within a hypothesis or callback invoked by a model trained with
 * ExecutionMode::Parallel or ParallelVectorized (see ExecutionStrategy.h)
 * once that parallelism actually spans the sample loop rather than a
 * single dot product. Logging from inside such a parallel region is a
 * data race.
 */
class Logger {
private:
    class LogEntry {
    public:
        LogEntry(Logger& logger, LogLevel level);

        LogEntry(const LogEntry&) = delete;
        LogEntry& operator=(const LogEntry&) = delete;

        LogEntry(LogEntry&& other) noexcept;
        LogEntry& operator=(LogEntry&& other) = delete;

        ~LogEntry() noexcept;

        template <typename Value>
        LogEntry& operator<<(const Value& value) {
            if (true == enabled_) {
                stream_ << value;
            }

            return *this;
        }

    private:
        Logger* logger_;
        LogLevel level_;
        bool enabled_;
        std::ostringstream stream_;
    };

public:
    explicit Logger(
        LogLevel minimumLevel = LogLevel::Info,
        std::ostream& output = std::clog);

    /**
     * @brief Shared logger used by callers that don't supply their own.
     *
     * A single, program-wide instance (LogLevel::Info, writing to
     * std::clog) -- not one instance per caller. Callers that only need
     * a default (RegressionPipeline and BinaryClassificationPipeline,
     * for instance) should use this rather than keeping their own
     * function-local static Logger: a static local inside a member
     * function of a class template is per-instantiation, so each
     * distinct template instantiation would otherwise get its own,
     * separately configured default logger instead of sharing one.
     *
     * @return The shared default logger.
     */
    [[nodiscard]]
    static Logger& instance();

    [[nodiscard]]
    LogEntry debug();

    [[nodiscard]]
    LogEntry info();

    [[nodiscard]]
    LogEntry warning();

    [[nodiscard]]
    LogEntry error();

    [[nodiscard]]
    LogEntry critical();

    void log(LogLevel level, std::string_view message);

private:
    [[nodiscard]]
    bool isEnabled(LogLevel level) const noexcept;

    [[nodiscard]]
    static std::string_view levelName(LogLevel level) noexcept;

    LogLevel minimumLevel;
    std::ostream* output;
};