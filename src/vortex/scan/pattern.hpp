#pragma once

#include <vortex/util/variant.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xaxis_iterator.hpp>

#include <vortex/scan/segmented.hpp>

namespace vortex::scan {

    namespace detail {

        template<typename It, typename S>
        auto load_waypoints(const It& waypoints_begin, const It& waypoints_end, size_t index, bool reversed, S& segment) {
            // check source index
            size_t segments_per_volume = std::distance(waypoints_begin, waypoints_end);
            if (index >= segments_per_volume) {
                throw std::runtime_error(fmt::format("assignment index is greater than available waypoints: {} >= {}", index, segments_per_volume));
            }
            const auto& wp = *(waypoints_begin + index);

            // check pre-condition for calculating velocity
            auto samples_per_segment = wp.shape(0);
            if (samples_per_segment < 2) {
                throw std::runtime_error(fmt::format("waypoints requires at least two samples per segment: {}", shape_to_string(wp.shape())));
            }

            // load position
            if (reversed) {
                xt::noalias(segment.position) = xt::flip(wp, 0);
            } else {
                xt::noalias(segment.position) = wp;
            }

            // NOTE: allow segment to calculate its own velocity

            return samples_per_segment;
        }


        template<typename T, typename inactive_policy_t, typename waypoints_t, typename pattern_t>
        struct patterned_config_factory_t : segmented_config_t<T, inactive_policy_t>, waypoints_t, pattern_t {

            auto to_segments() const {
                return to_pattern(to_waypoints());
            }

            using waypoints_t::to_waypoints;
            using pattern_t::to_pattern;
        };

        template<typename T, typename marker_t, typename config_t>
        class patterned_scan_factory_t : public segmented_scan_t<T, marker_t, config_t> {
        public:

            using base_t = segmented_scan_t<T, marker_t, config_t>;
            using segment_t = scan::segment_t<T, marker_t>;

            void initialize(const config_t& config) {
                auto segments = _check_and_apply_config(config, false);

                // NOTE: restart after config is applied in case channels_per_sample is not default
                // NOTE: restart before _change_segments to that the scan state is initialized
                restart();
                _change_segments(std::move(segments), false);
            }

            void change(const config_t& new_config, bool restart = false) {
                std::unique_lock<std::mutex> lock(_mutex);

                auto segments = _check_and_apply_config(new_config, true);
                _change_segments(std::move(segments), restart, {});
            }
            void change(const config_t& new_config, bool restart, marker::event::eid_t event_id) {
                std::unique_lock<std::mutex> lock(_mutex);

                auto segments = _check_and_apply_config(new_config, true);
                _change_segments(std::move(segments), restart, { event_id });
            }

            using base_t::restart;

        protected:

            auto _check_and_apply_config(const config_t& new_config, bool change) {
                if (change) {
                    // check that channel count is the same
                    if (_config.channels_per_sample != new_config.channels_per_sample) {
                        throw std::runtime_error(fmt::format("changing the number of channels per sample is not permitted: {} != {}", _config.channels_per_sample, new_config.channels_per_sample));
                    }
                }

                // perform configuration-level validation
                new_config.validate();

                // generate segments
                auto segments = new_config.to_segments();

                // perform scan-level validation against new configuration
                _validate_segments(new_config, segments);

                // accept configuration
                _config = std::move(new_config);

                return segments;
            }

            using base_t::_config, base_t::_mutex, base_t::_change_segments, base_t::_validate_segments;
            using base_t::_buffer_sample_base, base_t::_state, base_t::_active_sample;

        };

    }

    template<typename T, typename marker_t>
    struct manual_pattern_t {

        using segment_t = scan::segment_t<T, marker_t>;
        std::vector<segment_t> pattern;

        template<typename... Args>
        const auto& to_pattern(const Args&...) const {
            return pattern;
        }

    };

    template<typename marker_t, typename flags_t>
    struct sequential_pattern_t {

        // alternate segment directions
        bool bidirectional_segments = false;

        // execute the volume forwards and then backwards
        bool bidirectional_volumes = false;

        // segment mask
        flags_t flags = flags_t::all();

        template<typename T>
        auto to_pattern(const xt::xtensor<T, 3>& waypoints) const {
            std::vector<decltype(xt::view(waypoints, size_t(0), xt::all(), xt::all()))> list;
            for (size_t i = 0; i < waypoints.shape(0); i++) {
                list.push_back(xt::view(waypoints, i, xt::all(), xt::all()));
            }

            return to_pattern<T>(list.begin(), list.end());
        }

        template<typename T>
        auto to_pattern(const std::vector<xt::xtensor<T, 2>>& waypoints) const {
            return to_pattern<T>(waypoints.begin(), waypoints.end());
        }

        template<typename T, typename It>
        auto to_pattern(const It& waypoints_begin, const It& waypoints_end) const {
            std::vector<segment_t<T, marker_t>> segments;
            auto segments_per_volume = std::distance(waypoints_begin, waypoints_end);

            size_t n = segments_per_volume;
            if (bidirectional_volumes) {
                n *= 2;
            }
            segments.resize(n);

            bool reversed = false;
            for (size_t dst = 0; dst < segments.size(); dst++) {
                // map counter into correct segment
                size_t src;
                if (dst < segments_per_volume) {
                    src = dst;
                } else {
                    src = 2 * segments_per_volume - dst - 1;
                }

                // load segment waypoints
                auto samples_per_segment = detail::load_waypoints(waypoints_begin, waypoints_end, src, reversed, segments[dst]);

                // set up volume markers
                if (dst == 0) {
                    segments[dst].markers.push_back(marker::volume_boundary(0, 0, 0, false, segments_per_volume, flags));
                } else if (dst == segments_per_volume) {
                    segments[dst].markers.push_back(marker::volume_boundary(0, 1, 1, true, segments_per_volume, flags));
                }
                // each volume starts from segment 0
                segments[dst].markers.push_back(marker::segment_boundary(0, dst % segments_per_volume, src, reversed, samples_per_segment, flags));

                // alternate segment directions
                if (bidirectional_segments) {
                    reversed = !reversed;
                }
            }

            return segments;
        }

    };

    template<typename marker_t, typename flags_t>
    struct repeated_pattern_t {

        // store repeats in order of acquisition
        // scan is performed ABCABCABC and is stored ABCABCABC
        struct repeat_order {
            flags_t flags = flags_t::all();
        };

        // store repeats sequentially in the volume
        // scan is performed ABCABCABC but is stored AAABBBCCC
        struct repeat_pack {
            flags_t flags = flags_t::all();
        };

        // each repeat uses different flags from the provided list
        struct repeat_flags {
            // flags to assign to each repetition
            // NOTE: length of flags must match repeat count
            std::vector<flags_t> flags;
        };
        using repeat_strategy_t = std::variant<repeat_order, repeat_pack, repeat_flags>;

        // number of segments to execute in order before repeating
        size_t repeat_period = 2;

        // number of times to repeat each segment
        size_t repeat_count = 3;

        // strategy for organizing repeated segments
        repeat_strategy_t repeat_strategy{ repeat_pack{} };

        // alternate segment directions
        // NOTE: repeat period should be even if each repetition should use the same direction
        bool bidirectional_segments = false;

        template<typename T>
        auto to_pattern(const xt::xtensor<T, 3>& waypoints) const {
            std::vector<decltype(xt::view(waypoints, size_t(0), xt::all(), xt::all()))> list;
            for (size_t i = 0; i < waypoints.shape(0); i++) {
                list.push_back(xt::view(waypoints, i, xt::all(), xt::all()));
            }

            return to_pattern<T>(list.begin(), list.end());
        }

        template<typename T>
        auto to_pattern(const std::vector<xt::xtensor<T, 2>>& waypoints) const {
            return to_pattern<T>(waypoints.begin(), waypoints.end());
        }

        template<typename T, typename It>
        auto to_pattern(const It& waypoints_begin, const It& waypoints_end) const {
            std::vector<segment_t<T, marker_t>> segments;
            auto segments_per_volume = std::distance(waypoints_begin, waypoints_end);

            // calculate number of repetition sets (groups of segments executed in sequence)
            size_t repetition_sets = segments_per_volume / repeat_period;
            if (repetition_sets * repeat_period != segments_per_volume) {
                throw std::runtime_error(fmt::format("repeat period must evenly divide number of segments per volume: {} % {} == {}", segments_per_volume, repeat_period, segments_per_volume % repeat_period));
            }

            // validate repeat strategy
            std::visit(overloaded{
                [&](const repeat_flags& s) {
                    if (repeat_count != s.flags.size()) {
                        throw std::runtime_error(fmt::format("repeat count and number of flags must match for flags repeat strategy: {} != {}", repeat_count, s.flags.size()));
                    }
                },
                [](const auto&) {}
            }, repeat_strategy);

            segments.resize(segments_per_volume * repeat_count);

            size_t dst = 0;
            bool reversed = false;

            // loop over groups of segments executed in sequence
            for (size_t set = 0; set < repetition_sets; set++) {
                // loop over each repetition
                for (size_t rep = 0; rep < repeat_count; rep++) {

                    // look up flags for this repetition
                    auto flags = std::visit(overloaded{
                        [&](const repeat_order& s) { return s.flags; },
                        [&](const repeat_pack& s) { return s.flags; },
                        [&](const repeat_flags& s) { return s.flags[rep]; }
                    }, repeat_strategy);

                    // loop sequentially through segments within a repetition
                    for (size_t offset = 0; offset < repeat_period; offset++) {

                        // segment to execute
                        size_t src = (set * repeat_period) + offset;
                        // segment to assign in output volume
                        auto logical_dst = std::visit(overloaded{
                            [&](const repeat_order& s) {
                                // store repeats in order of execution
                                return dst;
                            },
                            [&](const repeat_pack& s) {
                                // store repeats in adjacent segments in output volume
                                return (src * repeat_count) + rep;
                            },
                            [&](const repeat_flags& s) {
                                // no grouping, repeats are differentiated by flags
                                return src;
                            }
                        }, repeat_strategy);

                        // load segment waypoints
                        auto samples_per_segment = detail::load_waypoints(waypoints_begin, waypoints_end, src, reversed, segments[dst]);

                        // set up markers
                        if (logical_dst == 0) {
                            segments[dst].markers.push_back(marker::volume_boundary(0, 0, 0, false, segments_per_volume, flags));
                        }
                        segments[dst].markers.push_back(marker::segment_boundary(0, dst, logical_dst, reversed, samples_per_segment, flags));

                        // alternate segment directions
                        if (bidirectional_segments) {
                            reversed = !reversed;
                        }

                        // next output segment
                        dst++;
                    }
                }
            }

            return segments;
        }

    };

}
