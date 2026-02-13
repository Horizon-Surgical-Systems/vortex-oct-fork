#pragma once

#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xnoalias.hpp>

#include <vortex/util/variant.hpp>

namespace vortex::scan {

    namespace warp {
        struct none_t {
            // TODO: figure out if xt::noalias(...) is actully the correct solution here
            template<typename E1, typename E2>
            void forward(xt::xexpression<E1>& in, xt::xexpression<E2>& out) const {
                xt::noalias(out.derived_cast()) = in.derived_cast();
            }

            template<typename E1, typename E2>
            void inverse(xt::xexpression<E1>& in, xt::xexpression<E2>& out) const {
                xt::noalias(out.derived_cast()) = in.derived_cast();
            }

            template<typename E>
            void forward(xt::xexpression<E>& inout) const {}

            template<typename E>
            void inverse(xt::xexpression<E>& inout) const {}
        };

        // from galvo scan angle to sample scan angle
        struct angular_t {
            double factor = 2;

            angular_t() {}
            angular_t(double factor_) : angular_t() { factor = factor_; }

            template<typename E1, typename E2>
            auto forward(xt::xexpression<E1>& in, xt::xexpression<E2>& out) const {
                xt::noalias(out.derived_cast()) = in.derived_cast() * factor;
            }

            template<typename E1, typename E2>
            auto inverse(xt::xexpression<E1>& in, xt::xexpression<E2>& out) const {
                xt::noalias(out.derived_cast()) = in.derived_cast() / factor;
            }

            template<typename E>
            void forward(xt::xexpression<E>& inout) const {
                inout.derived_cast() *= factor;
            }

            template<typename E>
            void inverse(xt::xexpression<E>& inout) const {
                inout.derived_cast() /= factor;
            }
        };

        // from galvo scan angle to telecentric sample position
        struct telecentric_t {
            double galvo_lens_spacing = 100;
            double scale = 2 * pi / 180;

            telecentric_t() {}
            telecentric_t(double galvo_lens_spacing_) : telecentric_t() { galvo_lens_spacing = galvo_lens_spacing_; }
            telecentric_t(double galvo_lens_spacing_, double scale_) : telecentric_t(galvo_lens_spacing_) { scale = scale_; }

            template<typename E1, typename E2>
            auto forward(xt::xexpression<E1>& in, xt::xexpression<E2>& out) const {
                xt::noalias(out.derived_cast()) = galvo_lens_spacing * xt::tan(in.derived_cast() * scale);
            }

            template<typename E1, typename E2>
            auto inverse(xt::xexpression<E1>& in, xt::xexpression<E2>& out) const {
                xt::noalias(out.derived_cast()) = xt::atan2(in.derived_cast(), galvo_lens_spacing) / scale;
            }

            template<typename E>
            void forward(xt::xexpression<E>& inout) const {
                inout.derived_cast() = galvo_lens_spacing * xt::tan(inout.derived_cast() * scale);
            }

            template<typename E>
            void inverse(xt::xexpression<E>& inout) const {
                inout.derived_cast() = xt::atan2(inout.derived_cast(), galvo_lens_spacing) / scale;
            }
        };

    }

    // template<typename... Ts>
    // struct warp_t : std::variant<Ts...> {
    //     using std::variant<Ts...>::variant;

    //     template<typename T1, typename T2>
    //     auto forward(T1& in, T2& out) const { DISPATCH_CONST(forward, in, out); }
    //     template<typename T1>
    //     auto forward(T1& inout) const { DISPATCH_CONST(forward, inout); }

    //     template<typename T1, typename T2>
    //     auto inverse(T1& in, T2& out) const { DISPATCH_CONST(inverse, in, out); }
    //     template<typename T1>
    //     auto inverse(T1& inout) const { DISPATCH_CONST(inverse, inout); }
    // };

}
