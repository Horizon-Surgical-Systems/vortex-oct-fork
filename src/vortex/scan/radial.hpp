#pragma once

#include <vortex/scan/waypoints.hpp>
#include <vortex/scan/pattern.hpp>

namespace vortex::scan {

    template<typename T, typename warp_t>
    struct radial_waypoints_t : detail::xy_waypoints_t<T, warp_t> {

        radial_waypoints_t() {
            extents = { { {0, pi}, {-2, 2} } };

            set_half_evenly_spaced(10);
        }

        void set_half_evenly_spaced(size_t n) {
            bscans_per_volume() = n;
            volume_extent().min() = 0;
            volume_extent().max() = pi * (n - 1) / n;
        }

        void set_evenly_spaced(size_t n) {
            bscans_per_volume() = n;
            volume_extent().min() = 0;
            volume_extent().max() = 2 * pi * (n - 1) / n;
        }

        void set_aiming() {
            set_half_evenly_spaced(2);
        }

        auto to_waypoints() const {
            // generate scan polar coordinates
            auto [ti, ri] = xt::meshgrid(
                xt::linspace(extents[0].min(), extents[0].max(), shape[0]),
                xt::linspace(extents[1].min(), extents[1].max(), shape[1])
            );

            // prepare output views
            xt::xtensor<double, 3> wp({ shape[0], shape[1], 2 });
            auto xo = xt::view(wp, xt::all(), xt::all(), 0);
            auto yo = xt::view(wp, xt::all(), xt::all(), 1);

            // convert to Cartesian and apply rotation and offset
            xo = ri * xt::cos(ti + angle) + offset(0);
            yo = ri * xt::sin(ti + angle) + offset(1);

            // apply warp
            CALL_CONST(warp, inverse, wp);

            return wp;
        }

        using base_t = detail::xy_waypoints_t<T, warp_t>;
        using base_t::extents, base_t::shape, base_t::angle, base_t::offset, base_t::warp;
        using base_t::volume_extent, base_t::bscans_per_volume;

    };

    template<typename T, typename inactive_policy_t, typename warp_t, typename marker_t, typename flags_t>
    using radial_config_t = detail::patterned_config_factory_t<
        T,
        inactive_policy_t,
        radial_waypoints_t<T, warp_t>,
        sequential_pattern_t<marker_t, flags_t>
    >;

    template<typename T, typename inactive_policy_t, typename warp_t, typename marker_t, typename flags_t>
    using repeated_radial_config_t = detail::patterned_config_factory_t<
        T,
        inactive_policy_t,
        radial_waypoints_t<T, warp_t>,
        repeated_pattern_t<marker_t, flags_t>
    >;

    template<typename T, typename marker_t, typename config_t>
    using radial_scan_t = detail::patterned_scan_factory_t<T, marker_t, config_t>;

}
