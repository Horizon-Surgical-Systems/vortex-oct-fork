#pragma once

#include <spdlog/spdlog.h>

#include <vortex/acquire/dso.hpp>

#include <vortex/driver/alazar/board.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/variant.hpp>

namespace vortex::alazar {

    //
    // clock
    //

    namespace clock {
        struct internal_t {
            size_t samples_per_second = 800'000'000;

            template<typename board_t>
            void apply(board_t& board, channel_t channels, std::shared_ptr<spdlog::logger>& log) const {
                if(log) { log->debug("configuring internal clock at {} samples/second", samples_per_second); }
                board.configure_clock_internal(samples_per_second);
            }
        };

        struct external_t {
            float level_ratio = 0.5f;
            coupling_t coupling = coupling_t::AC;
            clock_edge_t edge = clock_edge_t::rising;
            bool dual = false;

            template<typename board_t>
            void apply(board_t& board, channel_t channels, std::shared_ptr<spdlog::logger>& log) const {
                // reduce noise during external clocking for select boards
                if (board.info().features.adc_calibration_sampling_rate) {
                    const auto& sampling_rate = *board.info().features.adc_calibration_sampling_rate;
                    log->debug("calibrating ADC using internal clock at {} samples/second", sampling_rate);
                    board.configure_clock_internal(sampling_rate);
                }

                if (board.info().features.dual_edge_sampling) {
                    // NOTE: configure dual edge sampling before the external clock per SDK manual
                    // ref: https://docs.alazartech.com/ats-sdk-user-guide/latest/programmers-guide.html#dual-edge-sampling

                    // NOTE: dual edge sampling is only valid for channel A
                    auto channel = channel_t::A;

                    // set or clear the dual edge sampling state
                    if (log) { log->debug("configuring dual edge sampling {} for input {}", dual ? "on" : "off", to_string(channel)); }
                    board.set_dual_edge_sampling(cast(channel), dual);

                } else if (dual) {
                    // check if the dual edge sampling request cannot be satisfied
                    throw std::runtime_error(fmt::format("dual edge sampling is requested but is not supported by the {}", board.info().type.model));
                }

                if (log) { log->debug("configuring external clock on {} edge with level {}% and {} coupling", to_string(edge), 100 * level_ratio, to_string(coupling)); }
                board.configure_clock_external(level_ratio, coupling, edge);
            }
        };
    }
    // template<typename... Ts>
    // struct clock_t : std::variant<Ts...> {
    //     using std::variant<Ts...>::variant;

    //     template<typename board_t>
    //     auto apply(board_t& board, channel_t channels, std::shared_ptr<spdlog::logger>& log) const { DISPATCH_CONST(apply, board, channels, log); }

    //     auto samples_per_second() const { ACCESS_CONST(samples_per_second); }
    // };
    using default_clock_t = std::variant<clock::internal_t, clock::external_t>;

    //
    // trigger
    //

    namespace trigger {
        struct single_external_t {
            size_t range_millivolts = 2500;
            float level_ratio = 0.09f;
            size_t delay_samples = 80;
            trigger_slope_t slope = trigger_slope_t::positive;
            coupling_t coupling = coupling_t::DC;

            template<typename board_t>
            void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
                if(log) { log->debug("configuring single external trigger on {} edge with range {} mV (0V is TTL), {} coupling, level {}%, and delay of {} samples", to_string(slope), range_millivolts, to_string(coupling), 100 * level_ratio, delay_samples); }
                board.configure_single_trigger_external(range_millivolts, level_ratio, delay_samples, slope, coupling);
            }
        };

        struct dual_external_t {
            size_t range_millivolts = 2500;
            std::array<float, 2> level_ratios = { 0.09f, 0.09f };
            size_t delay_samples = 80;
            trigger_slope_t initial_slope = trigger_slope_t::positive;
            coupling_t coupling = coupling_t::DC;

            template<typename board_t>
            void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
                if (log) { log->debug("configuring dual external trigger on {} initial edge with range {} mV (0V is TTL), {} coupling, level {}% / {}%, and delay of {} samples", to_string(initial_slope), range_millivolts, to_string(coupling), 100 * level_ratios[0], 100 * level_ratios[1], delay_samples); }
                board.configure_dual_trigger_external(range_millivolts, level_ratios[0], level_ratios[1], delay_samples, initial_slope, coupling);
            }
        };
    }
    // template<typename... Ts>
    // struct trigger_t : std::variant<Ts...> {
    //     using std::variant<Ts...>::variant;

    //     template<typename board_t>
    //     auto apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const { DISPATCH_CONST(apply, board, log); }
    // };
    using default_trigger_t = std::variant<trigger::single_external_t, trigger::dual_external_t>;

    //
    // inputs
    //

    struct input_t {
        channel_t channel = channel_t::B;
        size_t range_millivolts = 400;
        size_t impedance_ohms = 50;
        coupling_t coupling = coupling_t::DC;

        template<typename board_t>
        void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
            if(log) { log->debug("configuring input {} with range {} mV, impedance {} Ohms, and {} coupling", to_string(channel), range_millivolts, impedance_ohms, to_string(coupling)); }
            board.configure_input(channel, coupling, range_millivolts, impedance_ohms);
        }

        // TODO: allow this to be customized
        size_t bytes_per_sample() const {
            return 2;
        }

        bool operator==(const input_t& o) const {
            return channel == o.channel && range_millivolts == o.range_millivolts && impedance_ohms == o.impedance_ohms && coupling == o.coupling;
        }
    };

    //
    // options
    //

    namespace option {
        struct auxio_trigger_out_t {
            template<typename board_t>
            void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
                if(log) { log->debug("configuring auxiliary I/O to pass trigger"); }
                board.configure_auxio_trigger_out();
            }

            bool operator==(const auxio_trigger_out_t& o) const {
                return true;
            }
        };

        struct auxio_clock_out_t {
            template<typename board_t>
            void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
                if(log) { log->debug("configuring auxiliary I/O to pass clock"); }
                board.configure_auxio_clock_out();
            }

            bool operator==(const auxio_clock_out_t& o) const {
                return true;
            }
        };

        struct auxio_pacer_out_t {
            U32 divider = 2;

            template<typename board_t>
            void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
                if(log) { log->debug("configuring auxiliary I/O to pass pacer with divider {}", divider); }
                board.configure_auxio_pacer_out(divider);
            }

            bool operator==(const auxio_pacer_out_t& o) const {
                return divider == o.divider;
            }
        };

        struct oct_ignore_bad_clock_t {
            double good_seconds = 4.95e-6;
            double bad_seconds = 4.95e-6;

            template<typename board_t>
            void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
                if(log) { log->debug("configuring OCT ignore bad clock with good {} s and bad {} s", good_seconds, bad_seconds); }
                board.set_ignore_bad_clock(true, good_seconds, bad_seconds);
            }

            bool operator==(const oct_ignore_bad_clock_t & o) const {
                return good_seconds == o.good_seconds && bad_seconds == o.bad_seconds;
            }
        };
   }
    // template<typename... Ts>
    // struct option_t : std::variant<Ts...> {
    //     using base_t = std::variant<Ts...>;
    //     using base_t::base_t;

    //     template<typename board_t>
    //     auto apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const { DISPATCH_CONST(apply, board, log); }

    //     bool operator==(const option_t& o) const {
    //         return static_cast<const base_t&>(*this) == static_cast<const base_t&>(o);
    //     }
    // };
    using default_option_t = std::variant<option::auxio_trigger_out_t, option::auxio_clock_out_t, option::auxio_pacer_out_t, option::oct_ignore_bad_clock_t>;

}
