#pragma once

#include <optional>
#include <shared_mutex>

#include <vortex/driver/cuda/runtime.hpp>

#include <vortex/format.hpp>

#include <vortex/util/tuple.hpp>
#include <vortex/util/exception.hpp>

namespace vortex::endpoint {

    namespace detail {

        struct notify {

            // callback at end of block
            using update_callback_t = std::function<void()>;
            update_callback_t update_callback;

            // callback with segments grouped within a volume
            using aggregate_segment_callback_t = std::function<void(std::vector<size_t>)>;
            aggregate_segment_callback_t aggregate_segment_callback;

            // callback with all segments grouped with a block
            using block_segment_callback_t = std::function<void(std::vector<size_t>)>;
            block_segment_callback_t block_segment_callback;

            using segment_callback_t = std::function<void(counter_t, size_t, size_t, size_t)>;
            segment_callback_t segment_callback;

            using volume_callback_t = std::function<void(counter_t, size_t, size_t)>;
            volume_callback_t volume_callback;

            using scan_callback_t = std::function<void(counter_t, size_t)>;
            scan_callback_t scan_callback;

            using event_callback_t = std::function<void(counter_t, size_t)>;
            event_callback_t event_callback;

        protected:

            void _default(std::shared_ptr<spdlog::logger> log, std::vector<size_t>& block_segments, std::vector<size_t>& volume_segments, const format::format_action_t& action) {
                // default actions
                std::visit(overloaded{
                    [&](const format::action::finish_segment& a) {
                        // update aggregate segments
                        volume_segments.push_back(a.segment_index_buffer);
                        block_segments.push_back(a.segment_index_buffer);
                        _notify(log, segment_callback, a.sample, a.scan_index, a.volume_index, a.segment_index_buffer);
                    },
                    [&](const format::action::finish_volume& a) {
                        // notify of aggregate segments before volume
                        if (!volume_segments.empty()) {
                            _notify(log, aggregate_segment_callback, std::move(volume_segments));
                            volume_segments.clear();
                        }
                        _notify(log, volume_callback, a.sample, a.scan_index, a.volume_index);
                    },
                    [&](const format::action::finish_scan& a) {
                        _notify(log, scan_callback, a.sample, a.scan_index);
                    },
                    [&](const format::action::event& a) {
                        _notify(log, event_callback, a.sample, a.id);
                    },
                    [](const auto&) {}
                }, action);
            }

            void _finish(std::shared_ptr<spdlog::logger> log, std::vector<size_t>& block_segments, std::vector<size_t>& volume_segments) {
                // handle any remaining segments
                if (!volume_segments.empty()) {
                    _notify(log, aggregate_segment_callback, std::move(volume_segments));
                }
                
                // handle block segments
                if (!block_segments.empty()) {
                    _notify(log, block_segment_callback, std::move(block_segments));
                }

                // notify of update completion
                _notify(log, update_callback);
            }

            template<typename Callback, typename... Args>
            void _notify(std::shared_ptr<spdlog::logger> log, const Callback& callback, Args&&... args) {
                if (callback) {
#if defined(VORTEX_EXCEPTION_GUARDS)
                    try {
#endif
                        std::invoke(callback, std::forward<Args>(args)...);
#if defined(VORTEX_EXCEPTION_GUARDS)
                    } catch (const std::exception& e) {
                        if (log) { log->critical("unhandled callback exception: {}\n{}", to_string(e), check_trace(e)); }
                    }
#endif
                }
            }

        };

        template<typename T, typename = typename std::enable_if_t<std::is_integral_v<T>>>
        auto combine(const std::vector<T>& segments) {
            std::vector<vortex::range_t<T>> chunks;

            chunks.push_back({ segments.front(), segments.front() + 1 });
            for (size_t i = 1; i < segments.size(); i++) {
                auto& idx = segments[i];

                if (idx == chunks.back().max()) {
                    chunks.back().max()++;
                } else {
                    chunks.push_back({ idx, idx + 1 });
                }
            }

            return chunks;
        }

        template<size_t index>
        struct select_streams_t {
            template<typename... T, typename spectra_stream_t, typename ascan_stream_t>
            static const auto& select(const std::tuple<T...>& streams, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                return std::get<index>(streams);
            }
            template<typename streams_t, typename spectra_stream_t, typename ascan_stream_t, typename = std::enable_if_t<!is_tuple<std::decay_t<streams_t>>::value>>
            static const auto& select(const streams_t& streams, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                return streams;
            }
        };
        struct select_spectra_t {
            template<typename streams_t, typename spectra_stream_t, typename ascan_stream_t>
            static const auto& select(const streams_t& streams, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                return spectra;
            }
        };
        struct select_ascans_t {
            template<typename streams_t, typename spectra_stream_t, typename ascan_stream_t>
            static const auto& select(const streams_t& streams, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
                return ascans;
            }
        };

    }

    struct null : detail::notify {
        using base_t = detail::notify;

        null(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) {}

        void allocate(const std::optional<cuda::device_t> spectra, const std::optional<cuda::device_t>& ascans) {}

        template<typename block_t, typename spectra_stream_t, typename ascan_stream_t>
        void handle(const format::format_plan_t& plan, const block_t& block, const spectra_stream_t& spectra, const ascan_stream_t& ascans) {
            std::vector<size_t> block_segments, volume_segments;

            // process actions
            for (auto& action : plan) {
                std::visit([&](const auto& a) { _default(_log, block_segments, volume_segments, a); }, action);
            }

            _finish(_log, block_segments, volume_segments);
        }

    protected:

        std::shared_ptr<spdlog::logger> _log;

        using base_t::_notify, base_t::_default, base_t::_finish;

    };

}
