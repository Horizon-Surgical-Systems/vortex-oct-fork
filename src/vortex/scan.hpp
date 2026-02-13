#pragma once

#include <vortex/marker.hpp>

#include <vortex/scan/freeform.hpp>
#include <vortex/scan/raster.hpp>
#include <vortex/scan/radial.hpp>
#include <vortex/scan/spiral.hpp>

namespace vortex {

    using default_warp_t = std::variant<
        scan::warp::none_t,
        scan::warp::angular_t,
        scan::warp::telecentric_t
    >;
    using default_inactive_policy_t = std::variant<
#if defined(VORTEX_ENABLE_REFLEXXES)
        scan::inactive_policy::minimum_dynamic_limited_t,
        scan::inactive_policy::fixed_dynamic_limited_t,
#endif
        scan::inactive_policy::fixed_linear_t
    >;

    using freeform_scan_config_t = scan::freeform_config_t<double, default_inactive_policy_t, default_marker_t>;
    using freeform_scan_t = scan::freeform_scan_t<double, default_marker_t, freeform_scan_config_t>;

    using raster_scan_config_t = scan::raster_config_t<double, default_inactive_policy_t, default_warp_t, default_marker_t, default_marker_flags_t>;
    using raster_scan_t = scan::raster_scan_t<double, default_marker_t, raster_scan_config_t>;

    using repeated_raster_scan_config_t = scan::repeated_raster_config_t<double, default_inactive_policy_t, default_warp_t, default_marker_t, default_marker_flags_t>;
    using repeated_raster_scan_t = scan::raster_scan_t<double, default_marker_t, repeated_raster_scan_config_t>;

    using radial_scan_config_t = scan::radial_config_t<double, default_inactive_policy_t, default_warp_t, default_marker_t, default_marker_flags_t>;
    using radial_scan_t = scan::radial_scan_t<double, default_marker_t, radial_scan_config_t>;

    using repeated_radial_scan_config_t = scan::repeated_radial_config_t<double, default_inactive_policy_t, default_warp_t, default_marker_t, default_marker_flags_t>;
    using repeated_radial_scan_t = scan::radial_scan_t<double, default_marker_t, repeated_radial_scan_config_t>;

    using spiral_scan_config_t = scan::spiral_config_t<double, default_inactive_policy_t, default_warp_t, default_marker_t, default_marker_flags_t>;
    using spiral_scan_t = scan::spiral_scan_t<double, default_marker_t, spiral_scan_config_t>;

}
