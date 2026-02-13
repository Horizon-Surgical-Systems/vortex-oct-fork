#pragma once

#include <variant>

#include <vortex/marker/marker.hpp>
#include <vortex/marker/flags.hpp>

namespace vortex {

    using default_marker_t = std::variant<
        marker::scan_boundary,
        marker::volume_boundary,
        marker::segment_boundary,
        marker::active_lines,
        marker::inactive_lines,
        marker::event
    >;
    using default_marker_flags_t = vortex::marker::detail::base::flags_t;

}