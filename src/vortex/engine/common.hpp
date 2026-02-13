#pragma once

#include <functional>

#include <vortex/core.hpp>

#include <vortex/marker.hpp>

namespace vortex::engine {

#define VORTEX_PROFILER_ENVVAR  "VORTEX_PROFILER_LOG"
#define VORTEX_PROFILER_META    size_t(-1)
#define VORTEX_PROFILER_VERSION 0

    enum class profiler_job_mark_t : uint8_t {
        create = 0,
        clearance,
        generate_scan,
        generate_strobe,
        acquire_dispatch_begin,
        acquire_dispatch_end,
        acquire_join,
        process_dispatch_begin,
        process_dispatch_end,
        format_join,
        recycle,
    };
    enum class profiler_task_mark_t : uint8_t {
        acquire_complete = 0,
        process_complete,
        format_begin,
        format_plan,
        format_end,
    };

    struct profiler_entry_t {
        size_t code;
        size_t index;
        size_t job;
        vortex::time_t timestamp;
    };


    struct timing_t {
        vortex::time_t create, service, scan, acquire, process, format, recycle;
    };

    enum class event_t : uint8_t {
        launch = 0, // session is starting up
        start,      // all acquisitions have started
        run,        // session has spawned all child threads
        stop,       // session is waiting for all inflight tasks to finish
        complete,   // scan has completed or all blocks have been dispatched
        shutdown,   // scan has ended early (i.e., not completed)
        exit,       // session has exited all child threads
        error,      // session has encountered an error
        abort       // session has encountered an error and cannot continue
    };
    using event_callback_t = std::function<void(event_t, std::exception_ptr)>;

    struct session_status_t {
        size_t allocated, inflight, dispatched, limit;

        auto available() const {
            return allocated - inflight;
        }

        auto progress() const {
            if (limit > 0) {
                return dispatched / double(limit);
            } else {
                return 0.0;
            }
        }

        auto utilization() const {
            return inflight / double(allocated);
        }
    };

    using job_callback_t = std::function<void(size_t, session_status_t, timing_t)>;

    namespace strobe {
        using flags_t = vortex::default_marker_flags_t;

        enum class polarity_t {
            low = 0,
            high,
        };

        namespace detail {
            struct physical {
                size_t line;
                polarity_t polarity;
                size_t duration;

                physical(size_t line_ = 0, polarity_t polarity_ = polarity_t::high, size_t duration_ = 1)
                    : line(line_), polarity(polarity_), duration(duration_) {}

                auto operator<=>(const physical&) const = default;
            };
            struct flagged {
                flags_t flags;
                size_t delay;

                flagged(size_t delay_ = 0, flags_t flags_ = flags_t::all())
                    : delay(delay_), flags(flags_) {}
                auto operator<=>(const flagged&) const = default;
            };
        }

        struct sample : detail::physical {
            size_t divisor;
            size_t phase;

            sample(size_t line_ = 0, size_t divisor_ = 2, polarity_t polarity_ = polarity_t::high, size_t duration_ = 1, size_t phase_ = 0)
                : detail::physical(line_, polarity_, duration_), divisor(divisor_), phase(phase_) {}

            auto operator<=>(const sample&) const = default;
        };
        struct segment : detail::physical, detail::flagged {
            segment(size_t line_ = 0, polarity_t polarity_ = polarity_t::high, size_t duration_ = 10, size_t delay_ = 0, flags_t flags_ = flags_t::all())
                : physical(line_, polarity_, duration_), flagged(delay_, flags_) {}
            auto operator<=>(const segment&) const = default;
        };
        struct volume : detail::physical, detail::flagged {
            volume(size_t line_ = 0, polarity_t polarity_ = polarity_t::high, size_t duration_ = 10, size_t delay_ = 0, flags_t flags_ = flags_t::all())
                : physical(line_, polarity_, duration_), flagged(delay_, flags_) {}
            auto operator<=>(const volume&) const = default;
        };
        struct scan : detail::physical, detail::flagged {
            scan(size_t line_ = 0, polarity_t polarity_ = polarity_t::high, size_t duration_ = 10, size_t delay_ = 0, flags_t flags_ = flags_t::all())
                : physical(line_, polarity_, duration_), flagged(delay_, flags_) {}
            auto operator<=>(const scan&) const = default;
        };
        struct event : detail::physical, detail::flagged {
            event(size_t line_ = 0, polarity_t polarity_ = polarity_t::high, size_t duration_ = 10, size_t delay_ = 0, flags_t flags_ = flags_t::all())
                : physical(line_, polarity_, duration_), flagged(delay_, flags_) {}
            auto operator<=>(const event&) const = default;
        };
    }
    using strobe_t = std::variant<strobe::sample, strobe::segment, strobe::volume, strobe::scan, strobe::event>;

}
