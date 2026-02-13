#pragma once

#include <optional>

#include <xtensor/containers/xtensor.hpp>

#include <vortex/core.hpp>

namespace vortex::motion {

    template<typename T>
    struct limits_t {
        range_t<T> position = { -10, 10 };
        T velocity = 100;
        T acceleration = 10'000;
    };

    template<typename T>
    struct state_t {
        T position;
        T velocity;
    };

    struct options_t {
        bool include_initial = false;
        bool include_final = false;
        bool bypass_limits_check = false;
        std::optional<size_t> fixed_samples;
    };

#if defined(VORTEX_ENABLE_REFLEXXES)

    xt::xtensor<double, 2> plan(size_t dimension, double dt, const state_t<const double*>& start, const state_t<const double*>& end, const limits_t<double>* limits, const options_t& options = {});

    template<size_t N, typename T>
    auto plan(T dt, const state_t<const xt_point<T, N>&>& start, const state_t<const xt_point<T, N>&>& end, const limits_t<T>* limits, const options_t& options = {}) {
        return plan(N, dt, { start.position.data(), start.velocity.data() }, { end.position.data(), end.velocity.data() }, limits, options);
    }

    template<typename T>
    auto plan(T dt, const state_t<xt::xtensor<T, 1>>& start, const state_t<xt::xtensor<T, 1>>& end, const limits_t<T>* limits, const options_t& options = {}) {
        // dimension is inferred from the boundary conditions
        size_t dimension = start.position.shape(0);
        if (start.velocity.shape(0) != dimension || end.position.shape(0) != dimension || end.velocity.shape(0) != dimension) {
            throw std::invalid_argument("dimension mismatch in start/end conditions for motion plan");
        }

        return plan(dimension, dt, { start.position.data(), start.velocity.data() }, { end.position.data(), end.velocity.data() }, limits, options);
    }

#endif

}
