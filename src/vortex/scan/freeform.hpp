#pragma once

#include <vortex/scan/waypoints.hpp>
#include <vortex/scan/pattern.hpp>

namespace vortex::scan {

    template<typename T, typename inactive_policy_t, typename marker_t>
    using freeform_config_t = detail::patterned_config_factory_t<
        T,
        inactive_policy_t,
        null_waypoints_t<T>,
        manual_pattern_t<T, marker_t>
    >;

    template<typename T, typename marker_t, typename config_t>
    using freeform_scan_t = detail::patterned_scan_factory_t<T, marker_t, config_t>;

}
