#include "pch.h"
#include "ScopedBenchmarkTimer.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

static_assert(
    false == std::is_copy_constructible_v<ScopedBenchmarkTimer>);

static_assert(
    false == std::is_copy_assignable_v<ScopedBenchmarkTimer>);

static_assert(
    false == std::is_move_constructible_v<ScopedBenchmarkTimer>);

static_assert(
    false == std::is_move_assignable_v<ScopedBenchmarkTimer>);


TEST(
    ScopedBenchmarkTimerTests,
    LogsScopeNameWhenScopeEnds)
{
    std::ostringstream output;
    Logger logger{ LogLevel::Debug, output };

    {
        ScopedBenchmarkTimer timer{
            logger,
            "TestOperation"
        };
    }

    const std::string message =
        output.str();

    EXPECT_NE(
        message.find("[DEBUG]"),
        std::string::npos);

    EXPECT_NE(
        message.find("TestOperation"),
        std::string::npos);

    EXPECT_NE(
        message.find("elapsed time:"),
        std::string::npos);

    EXPECT_NE(
        message.find("ms."),
        std::string::npos);
}

TEST(
    ScopedBenchmarkTimerTests,
    DoesNotLogBeforeScopeEnds)
{
    std::ostringstream output;
    Logger logger{ LogLevel::Debug, output };

    {
        ScopedBenchmarkTimer timer{
            logger,
            "TestOperation"
        };

        EXPECT_TRUE(output.str().empty());
    }

    EXPECT_FALSE(output.str().empty());
}

TEST(
    ScopedBenchmarkTimerTests,
    LogsWhenScopeExitsThroughException)
{
    std::ostringstream output;
    Logger logger{ LogLevel::Debug, output };

    try {
        ScopedBenchmarkTimer timer{
            logger,
            "FailingOperation"
        };

        throw std::runtime_error(
            "Simulated failure");
    }
    catch (const std::runtime_error&) {
        // Expected exception.
    }

    const std::string message =
        output.str();

    EXPECT_NE(
        message.find("FailingOperation"),
        std::string::npos);

    EXPECT_NE(
        message.find("elapsed time:"),
        std::string::npos);
}

TEST(
    ScopedBenchmarkTimerTests,
    DebugTimingIsSuppressedByInfoLevel)
{
    std::ostringstream output;
    Logger logger{ LogLevel::Info, output };

    {
        ScopedBenchmarkTimer timer{
            logger,
            "HiddenOperation"
        };
    }

    EXPECT_TRUE(output.str().empty());
}