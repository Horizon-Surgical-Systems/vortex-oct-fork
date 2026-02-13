/** \rst

    base class and configuration for scan generation

    The base class is abstract and requires no knowledge of the actual scan
    pattern except to know its dimension.  It further handles buffers and
    efficient looping of scan patterns, requiring the complex scan pattern to
    be generated only once.  At minimum, derived classes must implement
    two pure virtual functions to indicate when the scan pattern is fully
    buffered and prepare the next chunk of the scan.  The chunk can be any
    division of the scan pattern that is convenient for the derived
    implementation.

    The scan pattern is stored in 2D array where time increases along the rows
    and channel (e.g., X or Y galvo signal) varies with column.  The data is
    stored in row-major format.

    The scan pattern is stored in "prebuffers" and "buffers".  The content
    of the prebuffers is ephemeral; it is discarded once it is released.
    The content of the buffers is persistent (until cleared); it is used to
    loop the scan pattern once completed, if requested.  The prebuffers are
    used for temporary scan segments required to match segment initial conditions
    when the scan is first started or when a change occurs.

 \endrst */

 #pragma once

#include <algorithm>
#include <optional>
#include <mutex>

#include <spdlog/spdlog.h>

#include <xtensor/containers/xtensor.hpp>

#include <vortex/core.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/variant.hpp>

namespace vortex::scan::detail {

    template<typename T>
    struct scan_config_t {

        size_t channels_per_sample = 2;
        size_t samples_per_second = 100'000;

        auto sampling_interval() const { return T(1) / samples_per_second; }

        bool loop = true;
        bool consolidate = false;

        virtual void validate() const {};

    };

    template<typename T, typename marker_t_, typename config_t_>
    class scan_t {
    public:

        using element_t = T;

        using config_t = config_t_;
        using marker_t = marker_t_;

        using point_t = xt::xtensor<T, 1>;
        using array_t = xt::xtensor<T, 2>;

        scan_t() {}
        scan_t(std::shared_ptr<spdlog::logger> log)
            : _log(std::move(log)) {

        }

        virtual ~scan_t() {}

        const config_t& config() const {
            return _config;
        }

        virtual void restart(counter_t sample) {
            // shift time base
            _shift_scan_markers(sample - _buffer_sample_base);
            _buffer_sample_base = _active_sample = sample;

            // clear history
            _last_samples.first.reset();
            _last_samples.second.reset();
        }

        void prepare() {
            std::unique_lock<std::mutex> lock(_mutex);

            _prepare();
        }
        void prepare(size_t count) {
            std::unique_lock<std::mutex> lock(_mutex);

            _prepare(count);
        }

        template<typename markers_t>
        size_t next(markers_t& markers, const fixed_cpu_view_t<T, 2>& buffer) {
            std::unique_lock<std::mutex> lock(_mutex);

            // verify stream shape
            size_t needed = buffer.shape(0);
            if (buffer.shape(1) != _config.channels_per_sample) {
                raise(_log, "invalid buffer shape: [{}] vs [{} x {}]", shape_to_string(buffer.shape()), needed, _config.channels_per_sample);
            }

            // generate samples if needed or possible
            _prepare(needed);

            // number of samples copied into block this call
            size_t released = 0;

            // release the prebuffers first
            if (!_stream_buffers.empty()) {
                _release(markers, buffer, 0, _stream_buffers, _stream_markers, released, needed, true);
            }

            // release the regular buffers next
            while (needed > 0) {
                _release(markers, buffer, 0, _scan_buffers, _scan_markers, released, needed, false);

                // check for buffer exhaustion
                if (needed > 0) {
                    if (_config.loop) {
                        // restart the volume
                        _scans_completed++;
                        _buffer_sample_base += _scan_buffer_size();

                        // shift the markers
                        _shift_scan_markers(_scan_buffer_size());
                    } else {
                        // no more samples
                        break;
                    }
                }
            }

            // advance the released sample
            _active_sample += released;

            // record last samples
            if (released >= 1) {
                if (released >= 2) {
                    // lookup two samples before end
                    _last_samples.second = xt::view(buffer.to_xt(), released - 2, xt::all());
                } else {
                    // shift prior sample back
                    std::swap(_last_samples.first, _last_samples.second);
                }

                // lookup one sample before end
                _last_samples.first = xt::view(buffer.to_xt(), released - 1, xt::all());
            }

            return released;
        }

        // return last position and velocity as a tuple, if they are available
        auto last() const {
            std::optional<point_t> position, velocity;
            if (_last_samples.second) {
                velocity = (*_last_samples.first - *_last_samples.second) * _config.samples_per_second;
            }
            if (_last_samples.first) {
                position = _last_samples.first;
            }
            return std::make_tuple(std::move(position), std::move(velocity));
        }

        virtual void consolidate() {
            std::unique_lock<std::mutex> lock(_mutex);

            _consolidate(_scan_buffers);
        }

        virtual const array_t& scan_buffer() {
            std::unique_lock<std::mutex> lock(_mutex);

            _prepare();

            if (_scan_buffers.empty()) {
                throw std::runtime_error("empty scan pattern");
            }

            return _scan_buffers[0];
        }
        virtual std::vector<marker_t> scan_markers() {
            std::unique_lock<std::mutex> lock(_mutex);

            _prepare();

            // copy and rebase marker samples
            auto markers = _scan_markers;
            for (auto& marker : markers) {
                std::visit(overloaded{ [&](auto& m) { m.sample -= _scan_buffer_initial_sample(); } }, marker);
            }

            return markers;
        }

    protected:

        //
        // child interface
        //

        // returns true if all futures samples can be generated by repeating the buffer
        virtual bool _fully_buffered() = 0;

        // generate the next chunk of samples and add to buffer
        virtual void _prepare_next_chunk() = 0;

        //
        // buffer position helpers
        //

        counter_t _next_unbuffered_sample() {
            return _buffer_sample_base + _stream_buffer_size() + _scan_buffer_size();
        }
        counter_t _scan_buffer_initial_sample() {
            return _buffer_sample_base + _stream_buffer_size();
        }
        counter_t _stream_buffer_initial_sample() {
            return _buffer_sample_base;
        }

        size_t _stream_buffer_size() {
            return _length(_stream_buffers);
        }
        size_t _scan_buffer_size() {
            return _length(_scan_buffers);
        }

        //
        // buffer manipulations
        //

        void _consolidate(std::vector<array_t>& buffers) {
            // nothing to do if only a single buffer
            if (buffers.size() < 2) {
                return;
            }

            // determine length of buffers
            size_t n = _length(buffers);

            // check if actually any data to process
            if (n == 0) {
                buffers.clear();
                return;
            }

            // allocate
            array_t total;
            total.resize({ n, _config.channels_per_sample });

            // copy in the data
            size_t offset = 0;
            for (auto& b : buffers) {
                xt::view(total, xt::range(offset, offset + b.shape(0)), xt::all()) = b;
                offset += b.shape(0);
            }

            // replace the buffers with the total
            buffers.clear();
            buffers.emplace_back(std::move(total));
        }

        void _discard_scan_buffer() {
            _scan_buffers.clear();
            _scan_markers.clear();
        }
        void _discard_stream_buffer() {
            _stream_buffers.clear();
            _stream_markers.clear();
        }

        void _shift_scan_markers(ptrdiff_t delta_samples) {
            for (auto& m : _scan_markers) {
                std::visit(overloaded{ [&](auto& m2) { m2.sample += delta_samples; } }, m);
            }
        }

        // ensures that the requested sample starts on a chunk
        void _split_scan_buffer(counter_t requested_sample) {
            // sample of the start of the buffer in the loop
            auto buffer_sample = _scan_buffer_initial_sample();
            if (requested_sample < buffer_sample) {
                throw std::runtime_error("requested sample is before scan buffer");
            }
            if (requested_sample > _next_unbuffered_sample()) {
                throw std::runtime_error("requested sample is after scan buffer");
            }

            for (size_t i = 0; i < _scan_buffers.size(); i++) {
                auto& buffer = _scan_buffers[i];

                // check if buffer end is after the first requested sample
                if (buffer_sample + buffer.shape(0) > requested_sample) {
                    // map the requested sample into the buffer
                    size_t index = requested_sample - buffer_sample;

                    if (index == 0) {
                        // no split required
                        break;
                    }

                    // separate into two chunks
                    array_t before = xt::view(buffer, xt::range(0, index), xt::all());
                    array_t after = xt::view(buffer, xt::range(index, buffer.shape(0)), xt::all());

                    // update the buffers
                    _scan_buffers[i] = std::move(before);
                    _scan_buffers.insert(_scan_buffers.begin() + i + 1, std::move(after));

                    break;
                }

                // advance buffer sample
                buffer_sample += buffer.shape(0);
            }
        }

        void _shift_scan_to_stream() {
            // perform the move
            _stream_buffers.insert(_stream_buffers.end(), std::make_move_iterator(_scan_buffers.begin()), std::make_move_iterator(_scan_buffers.end()));
            _stream_markers.insert(_stream_markers.end(), std::make_move_iterator(_scan_markers.begin()), std::make_move_iterator(_scan_markers.end()));

            // reset the counts
            _scan_buffers.clear();
            _scan_markers.clear();
        }

        // move scan buffers to streams buffers such that the scan buffer starts at the requested sample
        void _shift_scan_to_stream(counter_t requested_sample) {
            _split_scan_buffer(requested_sample);

            size_t consumed;

            auto buffer_sample = _scan_buffer_initial_sample();

            consumed = 0;
            for (auto& buffer : _scan_buffers) {
                // advance buffer sample
                buffer_sample += buffer.shape(0);

                // check if buffer end is after the requested sample
                if (buffer_sample > requested_sample) {
                    break;
                }

                // increment number of buffers to move
                consumed++;
            }

            if (consumed > 0) {
                // perform the move
                _stream_buffers.insert(_stream_buffers.end(), std::make_move_iterator(_scan_buffers.begin()), std::make_move_iterator(_scan_buffers.begin() + consumed));
                _scan_buffers.erase(_scan_buffers.begin(), _scan_buffers.begin() + consumed);
            }

            // NOTE: the list of markers must be sorted
            consumed = 0;
            for (auto& marker : _scan_markers) {
                auto c = std::visit(overloaded{ [&](auto& m) {
                    if (m.sample >= requested_sample) {
                        // done with search
                        return false;
                    } else {
                        // increment number of markers to erase
                        consumed++;

                        return true;
                    }
                } }, marker);
                if (!c) {
                    break;
                }
            }
            if (consumed > 0) {
                // perform the move
                _stream_markers.insert(_stream_markers.end(), std::make_move_iterator(_scan_markers.begin()), std::make_move_iterator(_scan_markers.begin() + consumed));
                _scan_markers.erase(_scan_markers.begin(), _scan_markers.begin() + consumed);
            }
        }

        //
        // scan generation and release
        //

        virtual void _prepare() {
            // buffer everything possible
            while (!_fully_buffered()) {
                _prepare_next_chunk();
            }

            // go ahead and consolidate
            _consolidate(_scan_buffers);
        }
        virtual void _prepare(size_t count) {
            while (true) {
                if (_fully_buffered()) {
                    break;
                }

                // find out how many samples are available
                size_t available = _next_unbuffered_sample() - _active_sample;
                if (available >= count) {
                    break;
                }

                // make more samples
                _prepare_next_chunk();
            }

            if (_config.consolidate && _fully_buffered()) {
                // no more samples to prepare so consolidate
                _consolidate(_scan_buffers);
            }
        }

        template<typename markers_t>
        void _release(markers_t& output_markers, fixed_cpu_view_t<T, 2> output_buffer, size_t offset, std::vector<array_t>& buffers, std::vector<marker_t>& markers, size_t& released, size_t& needed, bool ephemeral) {
            size_t consumed;

            // sample of the start of the buffer in the loop
            auto buffer_sample = _buffer_sample_base;

            consumed = 0;
            for (auto& buffer : buffers) {
                // sample of the first requested sample this iteration
                auto requested_sample = _active_sample + released;

                // check if buffer end is after the first requested sample
                if (buffer_sample + buffer.shape(0) > requested_sample) {
                    // map the requested sample into the buffer
                    size_t index = requested_sample - buffer_sample;
                    // released as many samples as the buffer contains or are needed
                    size_t count = std::min(buffer.shape(0) - index, needed);

                    // perform the copy
                    auto dst = xt::view(output_buffer.to_xt(), xt::range(offset + released, offset + released + count), xt::all());
                    auto src = xt::view(buffer, xt::range(index, index + count), xt::all());
                    dst = src;

                    // update state
                    needed -= count;
                    released += count;
                }

                // advance buffer sample
                buffer_sample += buffer.shape(0);

                // check if request satisfied
                if (needed == 0) {
                    break;
                }

                // increment number of buffers to erase
                consumed++;
            }
            if (ephemeral) {
                // update the buffer sample base for consumed buffers only
                for (size_t i = 0; i < consumed; i++) {
                    _buffer_sample_base += buffers[i].shape(0);
                }

                // erase consumed buffers
                buffers.erase(buffers.begin(), buffers.begin() + consumed);
            }

            // release corresponding markers
            // NOTE: the list of markers must be sorted
            consumed = 0;
            for (auto& marker : markers) {
                auto c = std::visit([&](auto& m) {
                    if (m.sample >= _active_sample + released) {
                        // done with search
                        return false;
                    }
                    if (m.sample >= _active_sample) {
                        // copy the marker
                        output_markers.push_back(m);
                    }

                    // increment number of markers to erase
                    consumed++;

                    return true;
                }, marker);
                if (!c) {
                    break;
                }
            }
            if (ephemeral) {
                // erase consumed markers
                markers.erase(markers.begin(), markers.begin() + consumed);

                // check for inconsistencies
                if (buffers.empty() && !markers.empty()) {
                    throw std::runtime_error(fmt::format("stranded markers: buffers are empty but there are still {} markers", markers.size()));
                }
            }
        }

        std::shared_ptr<spdlog::logger> _log;

        std::vector<marker_t> _stream_markers, _scan_markers;
        std::vector<array_t> _stream_buffers, _scan_buffers;

        // sample that corresponds to first available buffer
        counter_t _buffer_sample_base, _active_sample;

        counter_t _scans_completed = 0;

        std::pair<std::optional<point_t>, std::optional<point_t>> _last_samples;

        mutable std::mutex _mutex;

        config_t _config;

    private:

        size_t _length(const std::vector<array_t>& buffer) {
            return std::accumulate(buffer.cbegin(), buffer.cend(), size_t(0), [](size_t n, const auto& b) { return n + b.shape(0); });
        }

    };

}
