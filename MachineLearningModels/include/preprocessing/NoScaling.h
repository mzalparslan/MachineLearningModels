#pragma once

#include "IScalingPolicy.h"
#include "DataPoint.h"
#include "ModelParameters.h"

#include <vector>

/**
 * @brief Provides a scaling policy that leaves features unchanged.
 *
 * This policy does not learn scaling parameters. Both fitting and
 * transformation are intentional no-op operations.
 *
 * @tparam T Feature value mode.
 */
template <typename T>
class NoScaling final : public IScalingPolicy<T> {
    static_assert(std::is_floating_point_v<T>,
        "NoScaling requires floating data mode T!");
public:
    /**
     * @brief Performs no fitting because this policy has no parameters.
     *
     * @param trainingSet Unused training dataset.
     */
    void fit(const std::vector<DataPoint<T>>&) override {
        // No fitting required.
    }

    /**
     * @brief Leaves the supplied feature vector unchanged.
     *
     * @param features Feature vector intentionally left unchanged.
     */
    void transform(std::vector<T>&) const override {
        // No transformation required.
    }
};