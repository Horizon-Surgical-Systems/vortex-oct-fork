#pragma once

#include <variant>
#include <optional>
#include <vector>

#include <xtensor/misc/xsort.hpp>

#include <vortex/scan/base.hpp>

#include <vortex/marker/marker.hpp>

#include <vortex/driver/motion.hpp>

namespace vortex::scan {

    namespace inactive_policy {

#if defined(VORTEX_ENABLE_REFLEXXES)

        struct minimum_dynamic_limited_t {

            template<typename T>
            bool compatible(const motion::state_t<xt::xtensor<T, 1>>& a, const  motion::state_t<xt::xtensor<T, 1>>& b) const {
                // require position and velocity to match
                return xt::allclose(a.position, b.position) && xt::allclose(a.velocity, b.velocity);
            }

            template<typename T>
            xt::xtensor<T, 2> generate(bool inter_segment, T dt, const motion::state_t<xt::xtensor<T, 1>>& start, const  motion::state_t<xt::xtensor<T, 1>>& end, const std::vector<motion::limits_t<T>>& limits, const motion::options_t& options) const {
                // generate smooth motion plan
                return motion::plan(dt, start, end, limits.data(), options);
            }

        };

        struct fixed_dynamic_limited_t {
            size_t inter_segment_samples = 100;
            size_t inter_volume_samples = 100;

            void set_samples(size_t n) { inter_segment_samples = inter_volume_samples = n; }

            template<typename T>
            bool compatible(const motion::state_t<xt::xtensor<T, 1>>& a, const  motion::state_t<xt::xtensor<T, 1>>& b) const {
                // require position and velocity to match
                return xt::allclose(a.position, b.position) && xt::allclose(a.velocity, b.velocity);
            }

            template<typename T>
            xt::xtensor<T, 2> generate(bool inter_segment, T dt, const motion::state_t<xt::xtensor<T, 1>>& start, const  motion::state_t<xt::xtensor<T, 1>>& end, const std::vector<motion::limits_t<T>>& limits, motion::options_t options) const {
                // set the fixed duration
                options.fixed_samples = { inter_segment ? inter_segment_samples : inter_volume_samples };
                // generate smooth motion plan
                return motion::plan(dt, start, end, limits.data(), options);
            }
        };

#endif

        struct fixed_linear_t {
            size_t inter_segment_samples = 100;
            size_t inter_volume_samples = 100;

            void set_samples(size_t n) { inter_segment_samples = inter_volume_samples = n; }

            template<typename T>
            bool compatible(const motion::state_t<xt::xtensor<T, 1>>& a, const  motion::state_t<xt::xtensor<T, 1>>& b) const {
                // require only position to match
                return xt::allclose(a.position, b.position);
            }

            template<typename T>
            xt::xtensor<T, 2> generate(bool inter_segment, T dt, const motion::state_t<xt::xtensor<T, 1>>& start, const  motion::state_t<xt::xtensor<T, 1>>& end, const std::vector<motion::limits_t<T>>& limits, const motion::options_t& options) const {
                // NOTE: add two to requested number of samples to account for start and end position
                auto samples = (inter_segment ? inter_segment_samples : inter_volume_samples) + 2;
                auto r = xt::view(xt::linspace<T>(0, 1, samples), xt::all(), xt::newaxis());

                // linearly interpolate between endpoints
                auto path =
                    (1 - r) * xt::view(start.position, xt::newaxis(), xt::all()) + \
                       r    * xt::view(end.position,   xt::newaxis(), xt::all());

                // handle options to path endpoints
                size_t start_index = 0;
                if (!options.include_initial) {
                    start_index++;
                }
                size_t end_index = samples;
                if (!options.include_final) {
                    // NOTE: guaranteed to be non-negative because samples is a minimum 2
                    end_index--;
                }

                return xt::view(path, xt::range(start_index, end_index), xt::all());
            }
        };

    }

    template<typename T, typename marker_t>
    struct segment_t {

        // 2D array -- shape unknown for now
        //
        //     <-- N -->
        //   ^ . . . . .
        //   | . . . . .
        //   | . . . . .
        //   t . . . . .
        //   | . . . . .
        //   | . . . . .
        //   v . . . . .
        using array_t = xt::xtensor<T, 2>;

        // 1D array of N numbers -- so a N-dimensional vector
        //
        // ^ .
        // | .
        // N .
        // | .
        // v .
        using point_t = xt::xtensor<T, 1>;

        array_t position;
        std::optional<point_t> entry_delta, exit_delta;

        std::vector<marker_t> markers;

        point_t entry_position() const {
            if (position.shape(0) >= 1) {
                return xt_row(position, 0);
            } else {
                throw std::runtime_error("cannot calculate entry position with no samples");
            }
        }
        point_t exit_position() const {
            if (position.shape(0) >= 1) {
                return xt_row(position, -1);
            } else {
                throw std::runtime_error("cannot calculate exit position with no samples");
            }
        }

        point_t entry_velocity(size_t samples_per_second) const {
            if (entry_delta) {
                return *entry_delta * samples_per_second;
            } else if (position.shape(0) >= 2) {
                return (xt_row(position, 1) - xt_row(position, 0)) * samples_per_second;
            } else {
                throw std::runtime_error("missing entry delta and cannot calculate entry velocity with fewer than two samples");
            }
        }
        point_t exit_velocity(size_t samples_per_second) const {
            if (exit_delta) {
                return *exit_delta * samples_per_second;
            } else if (position.shape(0) >= 2) {
                return (xt_row(position, -1) - xt_row(position, -2)) * samples_per_second;
            } else {
                throw std::runtime_error("missing exit delta and cannot calculate entry velocity with fewer than two samples");
            }
        }
    };

}

namespace vortex::scan::detail {

    template<typename T, typename inactive_policy_t>
    struct segmented_config_t : scan_config_t<T> {

        std::vector<motion::limits_t<T>> limits;
        bool bypass_limits_check = false;

        // NOTE: defaults to first variant alternative: minimum_dynamic_limited_t if Reflexxes is available, otherwise fixed_linear_t
        inactive_policy_t inactive_policy;

        segmented_config_t() {
            limits.resize(channels_per_sample);
            for (auto& l : limits) {
                l = { {-12.5, 12.5}, 8e3, 5e6 };  // ThorLabs, assuming square wave at 100 Hz
            }
        }

        void validate() const override {
            base_t::validate();

            // check number of limits
            if (limits.size() != channels_per_sample) {
                throw std::runtime_error(fmt::format("mismatch between number of limits and channels per sample: {} != {}", limits.size(), channels_per_sample));
            }
        }

        using base_t = scan_config_t<T>;
        using base_t::channels_per_sample;

    };

    template<typename T, typename marker_t_, typename config_t>
    class segmented_scan_t : public scan_t<T, marker_t_, config_t> {
    public:

        using base_t = scan_t<T, marker_t_, config_t>;

        using marker_t = marker_t_;
        using segment_t = scan::segment_t<T, marker_t>;

        using array_t = typename segment_t::array_t;
        using point_t = typename segment_t::point_t;

        void restart() {
            auto zero = xt::zeros<T>({ _config.channels_per_sample });
            restart(0, zero, zero, true);
        }
        void restart(counter_t sample, point_t position, point_t velocity, bool include_start) {
            // check dimensions
            if (position.shape(0) != _config.channels_per_sample || velocity.shape(0) != _config.channels_per_sample) {
                throw std::invalid_argument(fmt::format("dimension mismatch for restart position or velocity: {} and {} vs {}", position.shape(0), velocity.shape(0), _config.channels_per_sample));
            }

            // save the buffering status because changing _state and _scan_interrupted modifies it
            auto is_fully_buffered = _fully_buffered();

            // change to pre-scan state but then return to current state since the scan buffers have not been invalidated
            _state = pre_scan{ position, velocity, 0, include_start, false /* is_fully_buffered */ };
            // indicate scan interrupt so that stream buffers are used for the pre-scan path
            _scan_interrupted = true;

            // invalidate buffers
            //if (!is_fully_buffered) {
                // XXX: workaround to fix bug in restarting buffered scans (#92)
                _discard_scan_buffer();
            //}
            _discard_stream_buffer();

            // reset sample base
            base_t::restart(sample);
        }

        const std::vector<segment_t>& scan_segments() const {
            std::unique_lock<std::mutex> lock(_mutex);

            return _segments;
        }

    protected:

        struct pre_scan { point_t position, velocity; counter_t index; bool include_start = false; bool fully_buffered = false; };
        struct pre_segment { counter_t index; bool inactive = true; };
        struct post_segment { counter_t index; };
        struct post_scan {};

        using state_t = std::variant<pre_scan, pre_segment, post_segment, post_scan>;
        state_t _state = post_scan{};

        void _prepare_next_chunk() override {
            // choose which buffers/markers to use based on interruption status
            auto* markers = &_scan_markers;
            auto* buffers = &_scan_buffers;
            if (_scan_interrupted) {
                markers = &_stream_markers;
                buffers = &_stream_buffers;
            }

            _state = std::visit(overloaded{
                [&](const pre_scan& s) -> state_t {
                    // check if current position and velocity match the initial segment position and velocity
                    auto compatible = CALL_CONST(_config.inactive_policy, template compatible<T>,
                        { s.position, s.velocity },
                        { _segments[s.index].entry_position(), _segments[s.index].entry_velocity(_config.samples_per_second) }
                    );

                    if (!compatible) {
                        // plan segment to the first segment start
                        auto r = CALL_CONST(_config.inactive_policy, generate, false,
                            _config.sampling_interval(),
                            { s.position, s.velocity },
                            { _segments[s.index].entry_position(), _segments[s.index].entry_velocity(_config.samples_per_second) },
                            _config.limits, { s.include_start, false, _config.bypass_limits_check }
                        );

                        if (r.size() > 0) {
                            // scan starts inactive
                            _stream_markers.push_back(marker::inactive_lines(_scan_buffer_initial_sample()));
                            _stream_buffers.emplace_back(std::move(r));

                            // add change event to the end
                            // NOTE: it is critical that the change event occur at a sample after inactive lines to avoid bad memcpy calls
                            // TODO: figure out why this is the case
                            if (_change_event_id) {
                                _stream_markers.push_back(marker::event(_scan_buffer_initial_sample() - 1, *_change_event_id));
                                _change_event_id.reset();
                            }

                            // adjust the existing scan markers
                            _shift_scan_markers(_stream_buffer_size());
                        }
                    }

                    // do not clear interrupt status until change event is emitted to avoid buffering and replaying it
                    if (s.index == 0 && !_change_event_id) {
                        // now at start of scan
                        _scan_interrupted = false;
                    }

                    if (s.fully_buffered) {
                        // no work left
                        return post_scan{};
                    } else {
                        // work on initial segment
                        return pre_segment{ s.index, true };
                    }
                },
                [&](const pre_segment& s) -> state_t {
                    // add change event marker if not already present
                    if (_change_event_id) {
                        markers->push_back(marker::event(_next_unbuffered_sample(), *_change_event_id));
                        _change_event_id.reset();
                    }

                    // add regular markers
                    _copy_and_update_markers(_segments[s.index].markers, *markers, _next_unbuffered_sample());
                    if (s.inactive) {
                        markers->push_back(marker::active_lines(_next_unbuffered_sample()));
                    }

                    // add the next segment
                    buffers->push_back(_segments[s.index].position);

                    return post_segment{ s.index };
                },
                [&](const post_segment& s) -> state_t {
                    bool inactive = false;

                    // next segment index or empty if done with scan
                    std::optional<size_t> next;

                    // check if the scan is done
                    if (s.index + 1 >= _segments.size()) {
                        if (_config.loop) {
                            // move to beginning
                            next = 0;
                        }
                    } else {
                        // move to next active segment
                        next = s.index + 1;
                    }

                    // plan to start of next segment if needed
                    if (next) {
                        if (_post_segments[s.index].size() == 0) {
                            // plan segment to the next segment start
                            auto r = CALL_CONST(_config.inactive_policy, generate, true,
                                _config.sampling_interval(),
                                { _segments[s.index].exit_position(), _segments[s.index].exit_velocity(_config.samples_per_second) },
                                { _segments[*next].entry_position(), _segments[*next].entry_velocity(_config.samples_per_second) },
                                _config.limits, { false, false, _config.bypass_limits_check }
                            );
                            _post_segments[s.index] = std::move(r);
                        }

                        if (_post_segments[s.index].size() > 0) {
                            // update outputs
                            markers->push_back(marker::inactive_lines{ _next_unbuffered_sample() });
                            buffers->emplace_back(_post_segments[s.index]);
                            inactive = true;
                        }
                    }

                    // attempt to update samples per scan
                    if (!next || *next == 0) {
                        if (!_scan_interrupted) {
                            // update total scan sample count
                            _samples_per_scan = _scan_buffer_size();
                        }
                    }

                    if (next) {
                        if (*next == 0) {
                            if (!_scan_interrupted) {
                                // scan is fully buffered
                                return post_scan{};
                            }

                            // now at start of scan
                            _scan_interrupted = false;
                        }

                        // advance to next segment
                        return pre_segment{ *next, inactive };
                    }

                    // finish
                    return post_scan{};
                },
                [&](const post_scan& s) -> state_t {
                    // nothing to do
                    return post_scan{};
                }
            }, _state);
        }

        bool _fully_buffered() override {
            // XXX: workaround an issue where an interrupted non-looping scan enters an infinite loop in prepare() (#92)
            return (!_scan_interrupted || !_config.loop) && std::holds_alternative<post_scan>(_state);
        }

        virtual void _change_segments(std::vector<segment_t> new_segments, bool restart, std::optional<marker::event::eid_t> change_event_id = {}) {
            // check if need to find out where in the cycle the scan is
            if (!std::holds_alternative<post_scan>(_state)) {
                // still building the scan buffer so just move everything to prebuffer
                _shift_scan_to_stream();
            } else {

                // can revise buffers but not prebuffers
                counter_t sample = _scan_buffer_initial_sample();
                if (sample > _active_sample) {
                    // still in prebuffers which means the scan buffer is at the scan start
                    _state = pre_segment{ 0 };

                    // no split required
                } else {
                    // find out which segment is executing
                    for (size_t i = 0; i < _segments.size(); i++) {

                        // check if currently executing this segment
                        sample += _segments[i].position.shape(0);
                        if (sample >= _active_sample) {

                            // state begins with the post segment
                            _state = post_segment{ i };
                            break;
                        }

                        sample += _post_segments[i].shape(0);
                        // check if currently executing this post-segment
                        if (sample > _active_sample) {
                            // state begins with the next pre-segment
                            if (i + 1 == _segments.size()) {
                                if (_config.loop) {
                                    // first segment
                                    _state = pre_segment{ 0 };
                                } else {
                                    // scan is too far done
                                    throw std::runtime_error("scan is complete and cannot be changed");
                                }
                            } else {
                                // the next segment
                                _state = pre_segment{ i + 1 };
                            }
                            break;
                        }

                    }

                    // cut scan buffer at target point
                    if (sample < _next_unbuffered_sample()) {
                        _shift_scan_to_stream(sample);
                    }
                }

                // clear out the existing scan
                _discard_scan_buffer();

                if (std::holds_alternative<post_scan>(_state)) {
                    throw std::runtime_error("failed to determine current state");
                }
            }

            // map current state onto new state
            _state = std::visit(overloaded{
                [&](const pre_scan& s) -> state_t {
                    // no change
                    return s;
                },
                [&](const pre_segment& s) -> state_t {
                    size_t idx;
                    // check if restarting or this segment no longer exists
                    if (restart || s.index >= new_segments.size()) {
                        // move to the start of first segment
                        idx = 0;
                    } else {
                        // move to start of the new next segment, which might be at a different position
                        idx = s.index;
                    }

                    // will require planning or a check in all cases, starting from the beginning of this segment
                    // NOTE: need to include the start point because the presegment does not include it already
                    return pre_scan{ _segments[s.index].entry_position(), _segments[s.index].entry_velocity(_config.samples_per_second), idx, true };
                },
                [&](const post_segment& s) -> state_t {
                    size_t idx;
                    // check if restarting or this segment no longer exists
                    if (restart || s.index + 1 >= new_segments.size()) {
                        // move to the start of first segment
                        idx = 0;
                    } else {
                        // move to start of the new next segment, which might be at a different position
                        idx = s.index + 1;
                    }

                    // will require planning in all cases, starting from the end of this segment
                    return pre_scan{ _segments[s.index].exit_position(), _segments[s.index].exit_velocity(_config.samples_per_second), idx, false };
                },
                [&](const post_scan& s) -> state_t {
                    throw std::runtime_error("should not reach this state");
                }
            }, _state);

            // invalidate results from existing segments

            _segments = std::move(new_segments);
            _change_event_id = std::move(change_event_id);

            _post_segments.clear();
            _post_segments.resize(_segments.size());

            _samples_per_scan.reset();
            _scan_interrupted = true;
        }

        virtual void _validate_segments(const config_t& cfg, const std::vector<segment_t>& segments) {
            // check number of segments
            if (segments.empty()) {
                throw std::runtime_error("scan must have at least one segment");
            }

            // check segments
            for (size_t i = 0; i < segments.size(); i++) {
                auto& s = segments[i];

                // check shapes
                if (s.position.shape(0) == 0) {
                    throw std::runtime_error(fmt::format("segment {} must have more than zero samples", i));
                }
                if (s.position.shape(1) != cfg.channels_per_sample) {
                    throw std::runtime_error(fmt::format("mismatch between position channels and channels per sample for segment {}: {} != {}", i, s.position.shape(1), cfg.channels_per_sample));
                }
                if (s.entry_velocity(0).shape(0) != cfg.channels_per_sample) {
                    throw std::runtime_error(fmt::format("mismatch between entry velocity channels and channels per sample for segment {}: {} != {}", i, s.entry_velocity(0).shape(1), cfg.channels_per_sample));
                }
                if (s.exit_velocity(0).shape(0) != cfg.channels_per_sample) {
                    throw std::runtime_error(fmt::format("mismatch between exit velocity channels and channels per sample for segment {}: {} != {}", i, s.exit_velocity(0).shape(1), cfg.channels_per_sample));
                }

                if (!cfg.bypass_limits_check) {

                    // check position limits
                    auto max_position = xt::amax(s.position, 0, xt::evaluation_strategy::immediate);
                    auto min_position = xt::amin(s.position, 0, xt::evaluation_strategy::immediate);
                    for (size_t d = 0; d < cfg.channels_per_sample; d++) {
                        if (min_position(d) < cfg.limits[d].position.min()) {
                            throw std::runtime_error(fmt::format("axis {} violated lower position limit during segment {}: {} < {}", d, i, min_position(d), cfg.limits[d].position.min()));
                        }
                        if (max_position(d) > cfg.limits[d].position.max()) {
                            throw std::runtime_error(fmt::format("axis {} violated upper position limit during segment {}: {} > {}", d, i, max_position(d), cfg.limits[d].position.max()));
                        }
                    }

                    if (s.position.shape(0) >= 2) {
                        // check velocity limits
                        auto velocity = xt::diff(s.position, 1, 0) / cfg.sampling_interval();
                        auto index = xt::argmax(xt::abs(velocity), 0);
                        for (size_t d = 0; d < cfg.channels_per_sample; d++) {
                            auto max = velocity(index(d), d);
                            if (max > cfg.limits[d].velocity) {
                                throw std::runtime_error(fmt::format("axis {} violated velocity limit during segment {} at sample {}: {} > {}", d, i, index(d), max, cfg.limits[d].velocity));
                            }
                        }
                    }

                    if (s.position.shape(0) >= 3) {
                        // check acceleration limits
                        auto acceleration = xt::diff(s.position, 2, 0) / std::pow(cfg.sampling_interval(), 2);
                        auto index = xt::argmax(xt::abs(acceleration), 0);
                        for (size_t d = 0; d < cfg.channels_per_sample; d++) {
                            auto max = acceleration(index(d), d);
                            if (max > cfg.limits[d].acceleration) {
                                throw std::runtime_error(fmt::format("axis {} violated acceleration limit during segment {} at sample {}: {} > {}", d, i, index(d), max, cfg.limits[d].acceleration));
                            }
                        }
                    }

                }
            }
        }

        std::optional<size_t> _samples_per_scan;

    private:

        void _copy_and_update_markers(const std::vector<marker_t>& src, std::vector<marker_t>& dst, counter_t sample) {
            for (auto& marker : src) {
                dst.push_back(std::visit(vortex::overloaded{ [&](const auto& m) -> marker_t {
                    // copy and update sample
                    auto m2 = m;
                    m2.sample += sample;
                    return m2;
                }}, marker));
            }
        }

        std::optional<marker::event::eid_t> _change_event_id;
        bool _scan_interrupted = true;

        std::vector<array_t> _post_segments;
        std::vector<segment_t> _segments;

    protected:

        using base_t::_config, base_t::_mutex;
        using base_t::_active_sample, base_t::_buffer_sample_base, base_t::_next_unbuffered_sample, base_t::_scan_buffer_initial_sample, base_t::_stream_buffer_size;
        using base_t::_scan_buffer_size;
        using base_t::_shift_scan_to_stream, base_t::_discard_scan_buffer, base_t::_discard_stream_buffer, base_t::_shift_scan_markers;
        using base_t::_scan_markers, base_t::_scan_buffers, base_t::_stream_markers, base_t::_stream_buffers, base_t::_last_samples;

    };

}
