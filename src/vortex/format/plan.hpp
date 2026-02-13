/** \rst

    decoding of markers to extract segments

    The format planner interprets the marker stream and generates
    a sequence of instructions for extracting partial or whole
    segments for blocks.  The planner assumes a rectangular volume,
    although they actual formatting implementation is determined
    by the executor used to execute the planner output.  The
    planner is capable of handling bidiretional volumes and segments
    and will generate appropriate intructions for reordering them
    on the fly.

 \endrst */

 #pragma once

#include <compare>
#include <optional>

#include <spdlog/spdlog.h>

#include <vortex/marker/marker.hpp>

#include <vortex/util/variant.hpp>
#include <vortex/util/cast.hpp>

namespace vortex::format {

    struct format_planner_config_t {
        using flags_t = marker::segment_boundary::flags_t;

        std::array<size_t, 2> shape = { std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max() };
        auto& segments_per_volume() { return shape[0]; }
        const auto& segments_per_volume() const { return shape[0]; }
        auto& records_per_segment() { return shape[1]; }
        const auto& records_per_segment() const { return shape[1]; }

        flags_t mask = flags_t::all();

        // load reversed segments in reverse order to flip them
        bool flip_reversed = true;

        // skip over inactive lines
        bool strip_inactive = true;

        bool adapt_shape = false;

    };

    namespace action {
        struct copy {
            size_t count, block_offset;
            size_t buffer_segment, buffer_record;
            bool reverse;
	        auto operator<=>(const copy&) const = default;
        };

        struct resize {
            std::array<size_t, 2> shape;
            auto& segments_per_volume() { return shape[0]; }
            const auto& segments_per_volume() const { return shape[0]; }
            auto& records_per_segment() { return shape[1]; }
            const auto& records_per_segment() const { return shape[1]; }
	        auto operator==(const resize& rhs) const{ return shape == rhs.shape; };
        };

        struct finish_segment {
            counter_t sample;
            size_t scan_index, volume_index, segment_index_buffer;
	        auto operator<=>(const finish_segment&) const = default;
        };

        struct finish_volume {
            counter_t sample;
            size_t scan_index, volume_index;
	        auto operator<=>(const finish_volume&) const = default;
        };

        struct finish_scan {
            counter_t sample;
            size_t scan_index;
	        auto operator<=>(const finish_scan&) const = default;
        };

        struct event {
            counter_t sample;
            counter_t id;
            auto operator<=>(const event&) const = default;
        };
    }
    using format_action_t = std::variant<action::copy, action::resize, action::finish_segment, action::finish_volume, action::finish_scan, action::event>;
    using format_plan_t = std::vector<format_action_t>;

    template<typename config_t_>
    class format_planner_t {
    public:

        using config_t = config_t_;

        format_planner_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            std::swap(_config, config);
            _shape = _config.shape;

            reset();
        }

        template<typename markers_t>
        auto next(size_t block_sample, size_t block_length, const markers_t& markers) {
            format_plan_t plan;
            next(plan, block_sample, block_length, markers);
            return plan;
        }

        template<typename markers_t>
        void next(format_plan_t& plan, size_t block_sample, size_t block_length, const markers_t& markers) {
            _active_sample = block_sample;

            for (auto& marker : markers) {
                // check that marker is not after this block
                std::visit([&](auto& m) {
                    if (m.sample >= block_sample + block_length) {
                        raise(_log, "marker at sample {} is after end of block: {} > {}", m.sample, m.sample, block_sample + block_length);
                    }
                }, marker);

                std::visit(overloaded{
                    [&](const marker::active_lines& m) {
                        // update state
                        _lines_active = true;
                        _active_sample = m.sample;
                    },
                    [&](const marker::inactive_lines& m) {
                        // check if need to copy prior active lines
                        if (_lines_active && _config.strip_inactive) {
                            // advance position by number of loaded records
                            _active_sample = _setup_load(plan, block_sample, _active_sample, m.sample);
                        }

                        // update state
                        _lines_active = false;
                    },
                    [&](const marker::scan_boundary& m) {
                        // check if need to finish copying prior
                        _active_sample = _finish_prior(plan, block_sample, m, _active_sample);

                        // check the mask
                        bool scan_accepted = m.flags.matches(_config.mask);

                        // emit notification in reverse order of size
                        // NOTE: a scan boundary ends all active segments and volumes regardless of mask
                        if (_last_segment_index) {
                            _handle_segment(plan, *_last_segment_index);
                            _last_segment_index.reset();
                        }
                        if (_last_volume_index) {
                            _handle_volume(plan, *_last_volume_index);
                            _last_volume_index.reset();
                        }
                        if (scan_accepted) {
                            if (_last_scan_index) {
                                _handle_scan(plan, *_last_scan_index);
                            }
                            _last_scan_index = m.sequence;
                        }

                    },
                    [&](const marker::volume_boundary& m) {
                        // check if need to finish copying prior
                        _active_sample = _finish_prior(plan, block_sample, m, _active_sample);

                        // check the mask
                        bool volume_accepted = m.flags.matches(_config.mask);

                        // emit notification in reverse order of size
                        // NOTE: a volume boundary ends all active segments regardless of mask
                        if (_last_segment_index) {
                            _handle_segment(plan, *_last_segment_index);
                            _last_segment_index.reset();
                        }
                        if (volume_accepted) {
                            if (_last_volume_index) {
                                _handle_volume(plan, *_last_volume_index);
                            }
                            _last_volume_index = m.index_in_scan;
                        }
                    },
                    [&](const marker::segment_boundary& m) {
                        // check if need to finish copying prior
                        _active_sample = _finish_prior(plan, block_sample, m, _active_sample);

                        // emit notification
                        if (_last_segment_index) {
                            _handle_segment(plan, *_last_segment_index);
                            _last_segment_index.reset();
                        }

                        // check the mask
                        _segment_accepted = m.flags.matches(_config.mask);
                        if (!_segment_accepted) {
                            return;
                        }

                        // advance position to start of marked segment
                        _segment_reversed = _config.flip_reversed && m.reversed;
                        // set record load position
                        if (_segment_reversed) {
                             // at the end of the segment buffer
                            // NOTE: the write count is subtracted from _buffer_record_position so records_per_segment() is correct
                            if (m.record_count_hint > 0) {
                                _buffer_record_position = m.record_count_hint;
                            } else {
                                _buffer_record_position = records_per_segment();
                            }
                        } else {
                            // at the start of the segment buffer
                            _buffer_record_position = 0;
                        }
                        // NOTE: _buffer_segment_position always is a valid segment index in the buffer
                        _buffer_segment_position = m.index_in_volume;

                        size_t needed_volume_size = segments_per_volume();
                        if (_buffer_segment_position >= segments_per_volume()) {
                            if (_config.adapt_shape) {
                                // enlarge buffer
                                needed_volume_size = _buffer_segment_position + 1;
                            }
                        }

                        // check that buffer is appropriately sized
                        if (needed_volume_size > segments_per_volume()) {
                            // enlarge buffer
                            if (_log) { _log->debug("enlarging buffer from {} segments to {} segments", segments_per_volume(), needed_volume_size); }
                            segments_per_volume() = needed_volume_size;
                            plan.push_back({ action::resize{ _shape} });
                        }

                        // update index for notification
                        if (_buffer_segment_position < segments_per_volume()) {
                            _last_segment_index = _buffer_segment_position;
                        }
                    },
                    [&](const marker::event& m) {
                        // pass matching events through
                        if (m.flags.matches(_config.mask)) {
                            plan.push_back({ action::event{ m.sample, m.id } });
                        }
                    },
                    [&](const auto&) {} // ignore
                }, marker);
            }

            // check for remaining records in this block
            if (block_length > 0 && (_lines_active || !_config.strip_inactive)) {
                // load any remaining records in this block
                _setup_load(plan, block_sample, _active_sample, block_sample + block_length);
            }
        }

        auto finish() {
            format_plan_t plan;
            finish(plan);
            return plan;
        }
        void finish(format_plan_t& plan) {
            // emit final notifications
            if (_last_segment_index) {
                _handle_segment(plan, *_last_segment_index);
            }
            if (_last_volume_index) {
                _handle_volume(plan, *_last_volume_index);
            }
            if (_last_scan_index) {
                _handle_scan(plan, *_last_scan_index);
            }
        }

        void reset() {
            _active_sample = 0;
            _buffer_segment_position = 0;
            _buffer_record_position = 0;

            _segment_reversed = false;
            _segment_accepted = true;
            _lines_active = false;

            _last_scan_index.reset();
            _last_volume_index.reset();
            _last_segment_index.reset();
        }

        const auto& segments_per_volume() const { return _shape[0]; }
        const auto& records_per_segment() const { return _shape[1]; }

    protected:

        auto& segments_per_volume() { return _shape[0]; }
        auto& records_per_segment() { return _shape[1]; }

        template<typename marker_t>
        auto _finish_prior(format_plan_t& plan, size_t block_sample, const marker_t& m, size_t active_sample) {
            // check if need any copying is left over from prior blocks
            if (_lines_active || !_config.strip_inactive) {
                // load any remaining records in this segment
                active_sample = _setup_load(plan, block_sample, active_sample, m.sample);
            }
            return active_sample;
        }

        auto _setup_load(format_plan_t& plan, size_t block_sample, size_t start_sample, size_t end_sample) {
            // find how many records to load
            auto record_count = end_sample - start_sample;

            // skip copy if no records or if the segment is invalid
            if (_segment_accepted && record_count > 0 && _buffer_segment_position < segments_per_volume()) {

                size_t needed_segment_size = records_per_segment();
                if (_segment_reversed) {
                    // check that will not underflow segment
                    if (_buffer_record_position - record_count < 0) {
                        if (_config.adapt_shape) {
                            // enlarge buffer
                            needed_segment_size += record_count - _buffer_record_position;
                            // reposition such that copy will succeed
                            _buffer_record_position = record_count;
                        } else {
                            // truncate segment
                            record_count = _buffer_record_position;
                        }
                    }
                } else {
                    // check that will not overflow segment
                    if (_buffer_record_position + record_count > records_per_segment()) {
                        if (_config.adapt_shape) {
                            // enlarge buffer
                            needed_segment_size = _buffer_record_position + record_count;
                        } else {
                            // truncate segment
                            record_count = records_per_segment() - _buffer_record_position;
                        }
                    }
                }

                // check that buffer is appropriately sized
                if (records_per_segment() < needed_segment_size) {
                    if (_log) { _log->debug("enlarging buffer from {} records to {} records", records_per_segment(), needed_segment_size); }
                    records_per_segment() = needed_segment_size;
                    plan.push_back({ action::resize{_shape} });
                }

                // perform load
                if (record_count > 0) {
                    auto load_position = downcast<size_t>(_buffer_record_position);
                    if (_segment_reversed) {
                        load_position -= record_count;
                    }
                    if (_log) { _log->trace("loading {} records at segment {}, record {}{}", record_count, _buffer_segment_position, load_position, _segment_reversed ? " reversed" : ""); }
                    plan.push_back({ action::copy{ record_count, start_sample - block_sample, _buffer_segment_position, load_position, _segment_reversed } });
                }
            }

            if (_segment_reversed) {
                _buffer_record_position -= record_count;
            } else {
                _buffer_record_position += record_count;
            }
            return end_sample;
        }

        void _handle_scan(format_plan_t& plan, size_t index) {
            if (_log) { _log->trace("formatted scan {}", index); }
            plan.push_back({ action::finish_scan{ _active_sample, index } });
        };
        void _handle_volume(format_plan_t& plan, size_t index) {
            if (_log) { _log->trace("formatted volume {}", index); }
            plan.push_back({ action::finish_volume{ _active_sample, _last_scan_index.value_or(-1), index } });
        };
        void _handle_segment(format_plan_t& plan, size_t index) {
            if (_log) { _log->trace("formatted segment {}", index); }
            plan.push_back({ action::finish_segment{ _active_sample, _last_scan_index.value_or(-1), _last_volume_index.value_or(-1), index } });
        };

        std::shared_ptr<spdlog::logger> _log;

        counter_t _active_sample;
        size_t _buffer_segment_position;
        ptrdiff_t _buffer_record_position;
        bool _segment_reversed;
        bool _segment_accepted;
        bool _lines_active;

        std::optional<size_t> _last_scan_index, _last_volume_index, _last_segment_index;

        std::array<size_t, 2> _shape;

        config_t _config;

    };

}
