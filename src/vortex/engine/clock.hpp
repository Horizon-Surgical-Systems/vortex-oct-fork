#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xadapt.hpp>

namespace vortex::engine {

    template<typename T = double, typename E>
    xt::xtensor<T, 1> find_rising_edges(const xt::xexpression<E>& signal_, size_t samples_per_second, size_t number_of_edges = 0) {
        auto& signal = signal_.derived_cast();
        std::vector<T> edges;

        // obtain the demeaned clock signal
        auto clock = xt::eval(xt::cast<T>(signal));
        clock -= xt::mean(clock);

        // linear search for rising edges
        for (size_t i = 1; i < clock.shape(0); i++) {
            // check for rising edge
            if (clock(i - 1) < 0 && clock(i) > 0) {
                // find exact zero crossing by linear interpolation
                auto slope = clock(i) - clock(i - 1);
                auto intercept = -clock(i - 1) / slope;
                edges.push_back((i + intercept) / samples_per_second);
            }

            // check if requested number found
            if (number_of_edges > 0 && edges.size() >= number_of_edges) {
                break;
            }
        }

        // ensure the minimum number of rising edges were found
        if (edges.size() < number_of_edges) {
            throw std::runtime_error(fmt::format("failed to recover the expected number of rising edges: {} vs {}", edges.size(), number_of_edges));
        }

        return xt::adapt(edges);
    }

    template<typename source_t, typename T = typename source_t::element_t>
    auto compute_resampling(const source_t& source, size_t samples_per_second, size_t samples_per_ascan, T clock_delay_samples = 0) {
        const auto& ideal_seconds = source.clock_edges_seconds();

        // fraction of sweep completed for a uniformly progressing sweep
        auto ideal_sweep = xt::linspace<T>(0, 1, ideal_seconds.size());
        auto target_sweep = xt::linspace<T>(0, 1, samples_per_ascan);

        // interpolate to find at what time the uniformly-spaced samples occur
        auto target_seconds = xt::interp(target_sweep, ideal_sweep, ideal_seconds);
        auto target_samples = target_seconds * double(samples_per_second) + clock_delay_samples;

        return xt::eval(target_samples);
    }

}

#if defined(VORTEX_ENABLE_ALAZAR)

#include <vortex/acquire/dso/alazar/host.hpp>

#include <spdlog/logger.h>

namespace vortex::engine {

    template<typename acquisition_t, typename source_t>
    auto acquire_alazar_clock(const source_t& source, const typename acquisition_t::config_t& acquire_config, alazar::channel_t clock_channel, std::shared_ptr<spdlog::logger> log = nullptr) {
        // start with the acquisition configurations
        typename acquisition_t::config_t ac = acquire_config;

        // access the board for help choosing samples rates and record lengths
        auto board = alazar::board_t(ac.device.system_index, ac.device.board_index);

        // sample at maximum rate for best clock digitization
        auto samples_per_second = board.info().max_sampling_rate();

        // configure the acquisition to span the whole active portion of the sweep
        auto samples_per_sweep = source.duty_cycle * samples_per_second / double(source.triggers_per_second);
        ac.samples_per_record() = board.info().smallest_aligned_samples_per_record(samples_per_sweep);
        ac.records_per_block() = 1;

        // configure for clock acquisition
        ac.clock = alazar::clock::internal_t{ samples_per_second };
        // NOTE: resize the inputs rather that create a new input to retain any settings (e.g., input range) that were passed in
        ac.inputs.resize(1);
        ac.inputs[0].channel = clock_channel;
        ac.options.clear();

        // setup the acquisition
        acquisition_t a(log);
        a.initialize(ac);
        a.prepare();

        // perform the acquisition
        xt::xarray<typename acquisition_t::output_element_t> buffer;
        buffer.resize(ac.shape());
        a.start();
        auto n = a.next(buffer.shape(0), vortex::view(buffer));
        a.stop();
        if (n != buffer.shape(0)) {
            throw std::runtime_error("failed to acquire clock");
        }

        // return the clock
        auto clock = xt::eval(xt::view(buffer, 0, xt::all(), 0));
        return std::make_tuple(samples_per_second, std::move(clock));
    }

}

#endif
