#pragma once

#include <vortex/scan/waypoints.hpp>
#include <vortex/scan/pattern.hpp>

// Hybrid spiral scan generation
//
// This file handles generation of hybrid spiral scanning patterns,
// which are an Archimedes spiral wit a particular time
// parameterization that gives constant angular velocity (CAV)
// behavior near the middle and constant linear velocity (CLV)
// behavior towards the outside. Using a CAV approach in the middle
// region respects the bandwidth limits of the scanner, while CLV in
// the outer regions respects the slew rate limit.
//
// For more information about spiral scanning, see [1] (Izatt lab
// publication about CLV scanning) and [2] (paper from AFM describing
// hybrid scanning).
//
// [1] https://www.osapublishing.org/boe/abstract.cfm?URI=boe-9-10-5052
// [2] https://ieeexplore.ieee.org/document/7589006

namespace vortex::scan {

    template<typename T, typename warp_t>
    struct spiral_waypoints_t : detail::xy_waypoints_t<T, warp_t> {

        T angular_velocity = 10;
        T linear_velocity = 0;
        T acceleration_limit = 0;

        auto& inner_radius() { return volume_extent().min(); }
        const auto& inner_radius() const { return volume_extent().min(); }
        auto& outer_radius() { return volume_extent().max(); }
        const auto& outer_radius() const { return volume_extent().max(); }

        size_t rings_per_spiral = 10;

        auto radial_pitch() const { return volume_extent().length() / rings_per_spiral; }

        spiral_waypoints_t() {
            extents = { { {0, 2}, {0, 0} } };

            set_isotropic(100'000);
        }

        void set_isotropic(size_t samples_per_second) {
            angular_velocity = 0;
            linear_velocity = radial_pitch() * samples_per_second;

            auto t_start = _clv_duration(inner_radius());
            auto t_end = _clv_duration(outer_radius());
            if(acceleration_limit != 0) {
                t_start = _clv_inverse_time_scale(t_start);
                t_end = _clv_inverse_time_scale(t_end);
            }
            auto duration = t_end - t_start;

            segments_per_volume() = 1; // one giant b-scan
            samples_per_segment() = std::round(duration * samples_per_second);
        }

        void set_hybrid(size_t samples_per_second) {
            // compute total spiral duration
            auto r_swap = linear_velocity / angular_velocity;
            auto duration = _cav_duration(r_swap) - _cav_duration(inner_radius()) + _clv_duration(outer_radius()) - _clv_duration(r_swap);

            segments_per_volume() = 1; // one giant b-scan
            samples_per_segment() = std::round(duration * samples_per_second);
        }

        auto to_waypoints() const {
            xt::xtensor<T, 1> r, theta;

            if (linear_velocity == 0) {

                // constant angular velocity spiral
                auto t = xt::linspace(_cav_duration(inner_radius()), _cav_duration(outer_radius()), samples_per_segment());
                std::tie(r, theta) = _cav_spiral(t);

            } else if (angular_velocity == 0) {

                // constant linear velocity sprial

                if (acceleration_limit == 0) {

                    // standard spiral
                    auto t = xt::linspace(_clv_duration(inner_radius()), _clv_duration(outer_radius()), samples_per_segment());
                    std::tie(r, theta) = _clv_spiral(t);

                } else {

                    // acceleration-limited spiral

                    // map time endpoints in spiral to time endpoints in scaled time
                    auto t_start = _clv_inverse_time_scale(_clv_duration(inner_radius()));
                    auto t_end = _clv_inverse_time_scale(_clv_duration(outer_radius()));

                    // sample uniformly in scaled time
                    auto t = xt::linspace(t_start, t_end, samples_per_segment());
                    std::tie(r, theta) = _clv_spiral(_clv_forward_time_scale(t));

                }

            } else {

                // hybrid CAV->CLV spiral

                // transition when they reach the same radius
                auto r_swap = linear_velocity / angular_velocity;
                if (r_swap < inner_radius() || r_swap > outer_radius()) {
                    throw std::runtime_error(fmt::format("swap radius of {} is not within radius range [{}, {}]", r_swap, inner_radius(), outer_radius()));
                }

                // compute spiral durations in their individual time bases
                auto t_start_cav = _cav_duration(inner_radius());
                auto t_end_cav = _cav_duration(r_swap);

                auto t_start_clv = _clv_duration(r_swap);
                auto t_end_clv = _clv_duration(outer_radius());

                // sample number at which transition occurs
                size_t n = std::floor((t_end_cav - t_start_cav) / (t_end_cav - t_start_cav + t_end_clv - t_start_clv) * samples_per_segment());

                // generate each spiral
                auto t_cav = xt::linspace(t_start_cav, t_end_cav, n - 1, false);
                auto [r_cav, theta_cav] = _cav_spiral(t_cav);

                auto t_clv = xt::linspace(t_start_clv, t_end_clv, samples_per_segment() - n + 1);
                auto [r_clv, theta_clv] = _clv_spiral(t_clv);

                // concatenate into single waveform
                r = xt::concatenate(xtuple(r_cav, r_clv));
                theta = xt::concatenate(xtuple(theta_cav, theta_clv));

            }

            // prepare output views
            xt::xtensor<T, 3> wp({ shape[0], shape[1], 2 });
            auto xo = xt::view(wp, xt::all(), xt::all(), 0);
            auto yo = xt::view(wp, xt::all(), xt::all(), 1);

            // convert to Cartesian and apply rotation and offset
            xo = r * xt::cos(theta + angle) + offset(0);
            yo = r * xt::sin(theta + angle) + offset(1);

            // apply warp
            CALL_CONST(warp, inverse, wp);

            return wp;
        }

        using base_t = detail::xy_waypoints_t<T, warp_t>;
        using base_t::extents, base_t::shape, base_t::angle, base_t::offset, base_t::warp;
        using base_t::volume_extent, base_t::samples_per_segment, base_t::segments_per_volume;

    protected:

        auto _cav_duration(T radius) const {
            return 2 * pi * radius / (radial_pitch() * angular_velocity);
        }
        auto _cav_spiral(const auto& t) const {
            auto r = radial_pitch() * angular_velocity * t / (2 * pi);
            auto theta = t * angular_velocity;

            return std::make_tuple(r, theta);
        }

        auto _clv_duration(T radius) const {
            return pi * radius * radius / (radial_pitch() * linear_velocity);
        }
        auto _clv_spiral(const auto& t) const {
            auto r = xt::sqrt((radial_pitch() * linear_velocity / pi) * t);
            auto theta = xt::sqrt((4 * pi * linear_velocity / radial_pitch()) * t);

            return std::make_tuple(r, theta);
        }

        auto _clv_time_scale_setup() const {
            // Mathematica: D[ A Sqrt[t] Cos[B Sqrt[t]], {t, 2}] to find envelope
            // yields 3 terms and this one is the dominant one
            auto A = std::sqrt(linear_velocity * radial_pitch() / pi);
            auto B = std::sqrt(4 * pi * linear_velocity / radial_pitch());
            //auto envelope  = A * B * B / (4 * xt::sqrt(t));

            auto C = acceleration_limit;

            // time at which CLV changes from violating to meeting limits
            // from solving when enevelope intersects acceleration limit
            auto s_switch = std::pow((A * B * B) / (4 * C), 2);

            return std::make_tuple(A, B, C, s_switch);
        }

        auto _clv_forward_time_scale(T t) const {
            double A, B, C, s_switch;
            std::tie(A, B, C, s_switch) = _clv_time_scale_setup();

            // Mathematica: Int[ Sqrt[ (A B^2)/(4 C Sqrt[t]) ], t] and then invert
            auto s_ = [&](auto& t_) { return std::pow(3 / (2 * B) * std::sqrt(C / A) * t_, T(4) / 3); };
            auto t_switch = _clv_inverse_time_scale(s_switch);

            return t < t_switch ? s_(t) : t - t_switch + s_switch;
        }
        template<typename E>
        auto _clv_forward_time_scale(xt::xexpression<E>& t_in) const {
            auto& t = t_in.derived_cast();
            double A, B, C, s_switch;
            std::tie(A, B, C, s_switch) = _clv_time_scale_setup();

            // Mathematica: Int[ Sqrt[ (A B^2)/(4 C Sqrt[t]) ], t] and then invert
            auto s_ = [&](auto& t_) { return xt::pow(3 / (2 * B) * std::sqrt(C / A) * t_, T(4) / 3); };
            auto t_switch = _clv_inverse_time_scale(s_switch);

            return xt::eval(xt::where(t < t_switch, s_(t), t - t_switch + s_switch));
        }

        auto _clv_inverse_time_scale(T s) const {
            double A, B, C, s_switch;
            std::tie(A, B, C, s_switch) = _clv_time_scale_setup();

            // Mathematica: Int[ Sqrt[ (A B^2)/(4 C Sqrt[t]) ], t]
            auto t_ = [&](auto& s_) { return (T(2) / 3) * B * std::sqrt(A / C) * std::pow(s_, T(3) / 4); };

            // time at which CLV changes from violating to satisfying limits
            // from solving when enevelope intersects acceleration limit
            auto t_switch = t_(s_switch);

            return s < s_switch ? t_(s) : s - s_switch + t_switch;
        }
        template<typename E>
        auto _clv_inverse_time_scale(xt::xexpression<E>& s_in) const {
            auto& s = s_in.derived_cast();
            double A, B, C, s_switch;
            std::tie(A, B, C, s_switch) = _clv_time_scale_setup();

            // Mathematica: Int[ Sqrt[ (A B^2)/(4 C Sqrt[t]) ], t]
            auto t_ = [&](auto& s_) { return (T(2) / 3) * B * std::sqrt(A / C) * xt::pow(s_, T(3) / 4); };

            // time at which CLV changes from violating to meeting limits
            // from solving when enevelope intersects acceleration limit
            auto t_switch = _clv_inverse_time_scale(s_switch);

            return xt::eval(xt::where(s < s_switch, t_(s), s - s_switch + t_switch));
        }

    };

    template<typename T, typename inactive_policy_t, typename warp_t, typename marker_t, typename flags_t>
    using spiral_config_t = detail::patterned_config_factory_t<
        T,
        inactive_policy_t,
        spiral_waypoints_t<T, warp_t>,
        sequential_pattern_t<marker_t, flags_t>
    >;

    template<typename T, typename marker_t, typename config_t>
    using spiral_scan_t = detail::patterned_scan_factory_t<T, marker_t, config_t>;

}
