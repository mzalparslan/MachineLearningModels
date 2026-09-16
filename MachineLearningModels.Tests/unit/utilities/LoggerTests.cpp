#include "pch.h"
#include "Logger.h"
#include <sstream>

TEST(LoggerTest, StreamLogging) {
    std::ostringstream ss;
    Logger logger(LogLevel::Debug, ss);

    logger.debug() << "Debug message " << 123;
    std::string output = ss.str();

    EXPECT_NE(output.find("DEBUG"), std::string::npos);
    EXPECT_NE(output.find("Debug message 123"), std::string::npos);
}

TEST(LoggerTest, MinLogLevelFilter) {
    std::ostringstream ss;
    Logger logger(LogLevel::Warning, ss);

    logger.info() << "This info should be ignored";
    EXPECT_TRUE(ss.str().empty());

    logger.warning() << "This warning should be printed";
    EXPECT_FALSE(ss.str().empty());
    EXPECT_NE(ss.str().find("WARNING"), std::string::npos);
}

TEST(LoggerTest, LogMethod) {
    std::ostringstream ss;
    Logger logger(LogLevel::Info, ss);

    logger.log(LogLevel::Error, "Error occurred");
    std::string output = ss.str();

    EXPECT_NE(output.find("ERROR"), std::string::npos);
    EXPECT_NE(output.find("Error occurred"), std::string::npos);
}

TEST(LoggerTest, InstanceReturnsSameSharedLogger) {
    // instance() must hand back one program-wide object, not a fresh
    // Logger (or one scoped to a particular caller) on every call.
    Logger& first = Logger::instance();
    Logger& second = Logger::instance();

    EXPECT_EQ(&first, &second);
}
