
#include <iostream>
#include "ExecutionStrategyBenchmarks.h"

// Real models built with this library typically have somewhere between
// 1 and 50 features (see ExecutionStrategy.h): the small counts below
// show the regime this library actually runs in, where thread-pool
// dispatch overhead dwarfs the dot product it's parallelizing. The large
// counts show where parallel execution eventually wins, a regime this
// library never enters.
constexpr std::array featureCounts{
    std::size_t{1},
    std::size_t{5},
    std::size_t{10},
    std::size_t{25},
    std::size_t{50},
    std::size_t{100},
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