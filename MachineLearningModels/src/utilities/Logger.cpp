#include "Logger.h"

Logger::LogEntry::LogEntry(
    Logger& logger,
    LogLevel level)
    : logger_(&logger),
    level_(level),
    enabled_(logger.isEnabled(level)) {
}

Logger::LogEntry::LogEntry(
    LogEntry&& other) noexcept
    : logger_(other.logger_),
    level_(other.level_),
    enabled_(other.enabled_),
    stream_(std::move(other.stream_)) {
    other.enabled_ = false;
}

Logger::LogEntry::~LogEntry() noexcept {
    if (false == enabled_) {
        return;
    }

    try {
        logger_->log(level_, stream_.str());
    }
    catch (...) {
        // Logging failures must not escape from a destructor.
    }
}

Logger::Logger(
    LogLevel minimumLevel,
    std::ostream& output)
    : minimumLevel(minimumLevel),
    output(&output) {
}

Logger::LogEntry Logger::debug() {
    return LogEntry(*this, LogLevel::Debug);
}

Logger::LogEntry Logger::info() {
    return LogEntry(*this, LogLevel::Info);
}

Logger::LogEntry Logger::warning() {
    return LogEntry(*this, LogLevel::Warning);
}

Logger::LogEntry Logger::error() {
    return LogEntry(*this, LogLevel::Error);
}

Logger::LogEntry Logger::critical() {
    return LogEntry(*this, LogLevel::Critical);
}

void Logger::log(
    LogLevel level,
    std::string_view message) {
    if (false == isEnabled(level)) {
        return;
    }

    (*output)
        << '[' << levelName(level) << "] "
        << message
        << '\n';
}

bool Logger::isEnabled(
    LogLevel level) const noexcept {
    return level >= minimumLevel;
}

std::string_view Logger::levelName(
    LogLevel level) noexcept {
    switch (level) {
    case LogLevel::Debug:
        return "DEBUG";

    case LogLevel::Info:
        return "INFO";

    case LogLevel::Warning:
        return "WARNING";

    case LogLevel::Error:
        return "ERROR";

    case LogLevel::Critical:
        return "CRITICAL";
    }

    return "UNKNOWN";
}