#include "pch.h"
#include "BenchmarkTimer.h"
#include <thread>
#include <chrono>

TEST(BenchmarkTimerTest, StartAndStopTimer) {
    BenchmarkTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double elapsed = timer.stop();

    EXPECT_GT(elapsed, 0.0);
}

TEST(BenchmarkTimerTest, DoubleStartThrows) {
    BenchmarkTimer timer;
    timer.start();
    EXPECT_THROW(timer.start(), std::logic_error);
}

TEST(BenchmarkTimerTest, StopWithoutStartThrows) {
    BenchmarkTimer timer;
    EXPECT_THROW(timer.stop(), std::logic_error);
}
