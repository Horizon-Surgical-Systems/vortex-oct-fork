#pragma once

#include <queue>
#include <functional>
#include <optional>
#include <mutex>

#include <vortex/core.hpp>

#include <vortex/marker/marker.hpp>
#include <vortex/util/exception.hpp>

namespace vortex::engine {

    template<typename T, typename marker_t>
    class scan_queue_t {
    public:

        using element_t = T;
        using point_t = xt::xtensor<T, 1>;
        using array_t = xt::xtensor<T, 2>;

        enum class event_t {
            start,
            finish,
            interrupt,
            abort
        };
        using scan_callback_t = std::function<void(size_t, event_t)>;

        struct online_scan_queue_t {
            online_scan_queue_t(scan_queue_t& sq)
                : _sq(sq) {}

            template<typename S>
            void append(std::shared_ptr<S>& scan, scan_callback_t&& callback = {}, marker::scan_boundary marker = {}) {
                if (!scan) {
                    throw std::invalid_argument("non-null scan required");
                }

                // queue the scan
                _sq._enqueue(scan, std::forward<scan_callback_t>(callback), std::move(marker));

                // NOTE: generate() will take care of restarting the scan
            }

        protected:

            scan_queue_t& _sq;

        };
        using empty_callback_t = std::function<void(online_scan_queue_t&)>;

    protected:

        struct scan_t {
            std::function<size_t(std::vector<marker_t>&, const fixed_cpu_view_t<T, 2>&)> generate;
            std::function<std::tuple<std::optional<point_t>, std::optional<point_t>>()> last;
            std::function<void(counter_t, point_t, point_t, bool)> restart;
            std::function<size_t()> channels_per_sample;

            scan_callback_t callback;
            marker::scan_boundary marker;
        };

    public:

        scan_queue_t() {
            reset();
        }
        scan_queue_t(counter_t sample, const point_t& position, const point_t& velocity) {
            reset(sample, position, velocity);
        }

        void reset() {
            std::unique_lock<std::mutex> lock(_mutex);

            _last_position.reset();
            _last_velocity.reset();

            _reset(0);
        }
        void reset(counter_t sample, const point_t& position, const point_t& velocity) {
            std::unique_lock<std::mutex> lock(_mutex);

            _last_position = position;
            _last_velocity = velocity;

            _reset(sample);
        }

        void rebase(counter_t sample) {
            std::unique_lock<std::mutex> lock(_mutex);

            _rebase(sample);
        }

        template<typename S>
        void append(std::shared_ptr<S>& scan, scan_callback_t&& callback = {}, marker::scan_boundary marker = {}) {
            std::unique_lock<std::mutex> lock(_mutex);

            if (!scan) {
                throw std::invalid_argument("non-null scan required");
            }

            // queue the scan
            _enqueue(scan, std::forward<scan_callback_t>(callback), std::move(marker));
        }

        template<typename S>
        void interrupt(std::shared_ptr<S>& scan, scan_callback_t&& callback = {}, marker::scan_boundary marker = {}) {
            std::unique_lock<std::mutex> lock(_mutex);

            if (!scan) {
                throw std::invalid_argument("non-null scan required");
            }

            // notify of interruption
            _emit(event_t::interrupt);
            _clear();

            // queue the scan
            _enqueue(scan, std::forward<scan_callback_t>(callback), std::move(marker));
        }

        auto generate(std::vector<marker_t>& markers, const fixed_cpu_view_t<T, 2>& buffer, bool zero_order_hold = true) {
            std::unique_lock<std::mutex> lock(_mutex);
            auto start_sample = _last_sample;

            size_t released = 0;
            size_t count = buffer.shape(0);
            while (released < count) {
                bool scan_boundary = false;

                // check if scan is active
                if (!_active_scan) {

                    // handle empty queue without current scan
                    if (_queue.empty()) {
                        // see if another scan is to be appended
                        if (_empty_callback) {
                            // invoke callback with online interface
                            online_scan_queue_t osq(*this);
                            _empty_callback(osq);
                        }
                    }

                    // handle non-empty queue without current scan
                    if (!_queue.empty()) {
                        // load next scan
                        _active_scan = _queue.front();
                        _queue.pop();
                        scan_boundary = true;

                        // initialize scan
                        _restart();
                        _emit(event_t::start);
                    }

                }

                // stop generation when no more scans available
                if (!_active_scan) {
                    break;
                }

                auto marker_count = markers.size();

                // generate samples
                auto added = _active_scan->generate(markers, buffer.range(released, count));
                if (added > 0) {
                    // start position is no longer required
                    _include_start = false;
                }
                released += added;

                // add the scan boundary marker if needed
                if (scan_boundary && added > 0) {
                    // move marker to current sample
                    auto marker = _active_scan->marker;
                    marker.sample = _last_sample;

                    // NOTE: insert before the newly added markers to the list remains sorted
                    markers.insert(markers.begin() + marker_count, std::move(marker));
                }

                // update last state
                _last_sample = start_sample + released;
                std::tie(_last_position, _last_velocity) = _active_scan->last();

                // check if scan is done
                if (released < count) {
                    _emit(event_t::finish);
                    _active_scan.reset();
                }
            }
            size_t scan_samples = released;

            // scan queue has finished
            if (released < count) {
                if (zero_order_hold) {
                    // hold the last known position to finish out the block, if possible
                    {
                        auto dst = buffer.range(released, count).to_xt();
                        if (_last_position) {
                            dst = xt::broadcast(*_last_position, dst.shape());
                        } else {
                            dst.fill(0);
                        }
                    }
                    released = count;

                    // mark the held samples as inactive
                    if (!_marked_inactive) {
                        markers.push_back(marker::inactive_lines(_last_sample));
                        _marked_inactive = true;
                    }

                    // update state
                    // NOTE: position is unchanged by zero-order hold
                    _last_sample = start_sample + released;
                    if (_last_velocity) {
                        _last_velocity->fill(0);
                    }
                }
            }

            return std::make_tuple(scan_samples, released);
        }

        void clear() {
            std::unique_lock<std::mutex> lock(_mutex);

            _clear();
        }

        void set_empty_callback(empty_callback_t&& callback) {
            std::unique_lock<std::mutex> lock(_mutex);

            _empty_callback = std::forward<empty_callback_t>(callback);
        }
        const auto& empty_callback() const { return _empty_callback; }

        auto state() const {
            std::unique_lock<std::mutex> lock(_mutex);
            return std::make_tuple(_last_sample, _last_position, _last_velocity);
        }

    protected:

        template<typename S>
        void _enqueue(std::shared_ptr<S>& scan, scan_callback_t&& callback, marker::scan_boundary marker) {
            // push the wrapper onto the queue
            _queue.push({
                [scan](std::vector<marker_t>& markers, const fixed_cpu_view_t<T, 2>& buffer) {
                    return scan->next(markers, buffer);
                },
                [scan]() {
                    return scan->last();
                },
                [scan](counter_t sample, point_t position, point_t velocity, bool include_start) {
                    scan->restart(sample, position, velocity, include_start);
                },
                [scan]() {
                    return scan->config().channels_per_sample;
                },
                std::forward<scan_callback_t>(callback),
                std::move(marker)
            });

            // allow marking after scan finishes
            _marked_inactive = false;
        }

        void _restart() {
            auto zero = xt::zeros<T>({ _active_scan->channels_per_sample() });
            _active_scan->restart(_last_sample, _last_position.value_or(zero), _last_velocity.value_or(zero), _include_start);
        }

        void _reset(counter_t sample) {
            _clear();
            _rebase(sample);
        }

        void _rebase(counter_t sample) {
            if (_active_scan) {
                throw traced<std::runtime_error>("cannot rebase scan queue with active scan");
            }

            _last_sample = sample;
        }

        void _clear() {
            do {
                _emit(event_t::abort);
                _active_scan.reset();

                if (!_queue.empty()) {
                    _active_scan = _queue.front();
                    _queue.pop();
                }
            } while (_active_scan);
        }

        void _emit(event_t e) {
            if (_active_scan && _active_scan->callback) {
                std::invoke(_active_scan->callback, _last_sample, e);
            }
        }

        std::optional<scan_t> _active_scan;
        std::queue<scan_t> _queue;

        bool _marked_inactive = false;

        bool _include_start = true;

        counter_t _last_sample;
        std::optional<point_t> _last_position, _last_velocity;

        empty_callback_t _empty_callback;

        mutable std::mutex _mutex;

    };

}
