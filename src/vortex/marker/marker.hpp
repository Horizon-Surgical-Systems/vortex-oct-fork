/** \rst

    definition of markers

    Markers indicate an important event at a particular instant in time.
    Examples include start of a B-scan or start of inactive lines. Some
    markers are associated with flags which are intended to support
    bitwise masking in filtering applications.  See `vortex::format::planner_t`
    for an example of such filtering.

 \endrst */

#pragma once

#include <compare>
#include <variant>

#include <vortex/core.hpp>

#include <vortex/marker/flags.hpp>

namespace vortex::marker {

    namespace detail {
        struct base {
            counter_t sample;

            using flags_t = marker::flags_t<>;

            base() : base(0) {}
            base(counter_t sample_) { sample = sample_; }

            auto operator <=>(const base& o) const = default;
        };
    }

    struct scan_boundary : detail::base {
        counter_t sequence;
        size_t volume_count_hint;
        flags_t flags;

        scan_boundary()
            : scan_boundary(0, 0) {}
        scan_boundary(counter_t sample, counter_t sequence, size_t volume_count_hint = 0)
            : scan_boundary(sample, sequence, volume_count_hint, flags_t::all()) {
        }
        scan_boundary(counter_t sample_, counter_t sequence_, size_t volume_count_hint_, flags_t flags_)
            : base(sample_) {
            sequence = sequence_; volume_count_hint = volume_count_hint_; flags = flags_;
        }

        auto operator <=>(const scan_boundary&) const = default;

    };

    struct volume_boundary : detail::base {
        counter_t sequence, index_in_scan;
        bool reversed;
        size_t segment_count_hint;
        flags_t flags;

        volume_boundary()
            : volume_boundary(0, 0, false) {}
        volume_boundary(counter_t sample, counter_t sequence, bool reversed)
            : volume_boundary(sample, sequence, sequence, reversed) {
        }
        volume_boundary(counter_t sample, counter_t sequence, counter_t index_in_scan, bool reversed, size_t segment_count_hint = 0)
            : volume_boundary(sample, sequence, index_in_scan, reversed, segment_count_hint, flags_t::all()) {
        }
        volume_boundary(counter_t sample_, counter_t sequence_, counter_t index_in_scan_, bool reversed_, size_t segment_count_hint_, flags_t flags_)
            : base(sample_) {
            sequence = sequence_; index_in_scan = index_in_scan_; reversed = reversed_; segment_count_hint = segment_count_hint_; flags = flags_;
        }

        auto operator <=>(const volume_boundary&) const = default;

    };

    struct segment_boundary : detail::base {
        counter_t sequence, index_in_volume;
        bool reversed;
        size_t record_count_hint;
        flags_t flags;

        segment_boundary()
            : segment_boundary(0, 0, false) {}
        segment_boundary(counter_t sample, counter_t sequence, bool reversed)
            : segment_boundary(sample, sequence, sequence, reversed) {
        }
        segment_boundary(counter_t sample, counter_t sequence, counter_t index_in_volume, bool reversed, size_t record_count_hint = 0)
            : segment_boundary(sample, sequence, index_in_volume, reversed, record_count_hint, flags_t::all()) {
        }
        segment_boundary(counter_t sample_, counter_t sequence_, counter_t index_in_volume_, bool reversed_, size_t record_count_hint_, flags_t flags_)
            : base(sample_) {
            sequence = sequence_; index_in_volume = index_in_volume_; reversed = reversed_; record_count_hint = record_count_hint_; flags = flags_;
        }
        auto operator <=>(const segment_boundary&) const = default;
    };

    struct active_lines : detail::base {
        using detail::base::base;
        auto operator<=>(const active_lines&) const = default;
    };
    struct inactive_lines : detail::base {
        using detail::base::base;
        auto operator<=>(const inactive_lines&) const = default;
    };

    struct event : detail::base {
        using detail::base::base;

        using eid_t = size_t;
        eid_t id;
        flags_t flags;

        event()
            : event(0, 0, flags_t::all()) {}
        event(counter_t sample)
            : event(sample, 0) {}
        event(counter_t sample, eid_t id)
            : event(sample, id, flags_t::all()) {}
        event(counter_t sample_, eid_t id_, flags_t flags_)
            : base(sample_) {
            id = id_; flags = flags_;
        }

        auto operator<=>(const event&) const = default;
    };

    // template<typename... Ts>
    // struct marker_t : std::variant<Ts...> {
    //     using std::variant<Ts...>::variant;
    //     using flags_t = marker::detail::base::flags_t;
    // };

}
