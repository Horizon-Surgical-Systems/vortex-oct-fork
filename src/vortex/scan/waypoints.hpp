#pragma once

#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/views/xview.hpp>

#include <vortex/scan/warp.hpp>

namespace vortex::scan {

    template<typename T>
    struct null_waypoints_t {

        xt::xtensor<T, 3> to_waypoints() const {
            return {};
        }

    };

    namespace detail {

        template<typename T, size_t N, typename warp_t>
        struct grid_waypoints_t {

            xt_point<T, N> offset;
            // scan rotation angle in radians, about offset
            T angle = 0;

            // extent of the scan, before rotation
            std::array<range_t<T>, N> extents;

            // resolution of the scan
            std::array<size_t, N> shape;

            // scan warp
            // NOTE: defaults to warp::none_t (the first variant alternative)
            warp_t warp;

            grid_waypoints_t() {
                offset.fill(0);
                extents.fill(range_t<T>::symmetric(1));
                shape.fill(100);
            }

        };

        template<typename T, typename warp_t>
        struct xy_waypoints_t : grid_waypoints_t<T, 2, warp_t> {

            range_t<T>& bscan_extent() { return extents[1]; }
            const range_t<T>& bscan_extent() const { return extents[1]; }
            range_t<T>& volume_extent() { return extents[0]; }
            const range_t<T>& volume_extent() const { return extents[0]; }

            range_t<T>& segment_extent() { return extents[1]; }
            const range_t<T>& segment_extent() const { return extents[1]; }

            size_t& ascans_per_bscan() { return shape[1]; }
            const size_t& ascans_per_bscan() const { return shape[1]; }
            size_t& bscans_per_volume() { return shape[0]; }
            const size_t& bscans_per_volume() const { return shape[0]; }

            size_t& samples_per_segment() { return shape[1]; }
            const size_t& samples_per_segment() const { return shape[1]; }
            size_t& segments_per_volume() { return shape[0]; }
            const size_t& segments_per_volume() const { return shape[0]; }

            xy_waypoints_t() {
                extents = { { {-1, 1}, {-1, 1} } };
                shape = { 10, 500 };
            }

            using base_t = grid_waypoints_t<T, 2, warp_t>;
            using base_t::extents, base_t::shape;

        };

    }

}
