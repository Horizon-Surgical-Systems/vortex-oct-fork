#pragma once

#include <xtensor/containers/xtensor.hpp>

namespace vortex::engine {

    template<typename T>
    struct source_t {
        using element_t = T;

        size_t triggers_per_second = 100'000;
        size_t clock_rising_edges_per_trigger = 1376;
        T duty_cycle = 0.5;
        T imaging_depth_meters = 0.01;

        xt::xtensor<T, 1> clock_rising_edges_seconds;

        bool has_clock() const {
            return clock_rising_edges_seconds.size() > 0;
        }

        auto& clock_edges_seconds() {
            return clock_rising_edges_seconds;
        }
        const auto& clock_edges_seconds() const {
            return clock_rising_edges_seconds;
        }
    };

}