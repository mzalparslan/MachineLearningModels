
#include <iostream>
#include "ExecutionStrategyBenchmarks.h"

constexpr std::array featureCounts{
    std::size_t{16'384},
    std::size_t{32'768},
    std::size_t{65'536},
    std::size_t{131'072},
    std::size_t{262'144},
    std::size_t{524'288},
    std::size_t{1'000'000}
};

int main()
{
    std::cout
        << "strategy,feature_count,iterations,"
        << "average_microseconds\n";

    for (const std::size_t featureCount : featureCounts) {
        runBenchmarks<double>(featureCount);
    }

    return 0;
}