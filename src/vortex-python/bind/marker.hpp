#pragma once
#include <vortex-python/bind/common.hpp>

#include <vortex/marker.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<vortex::default_marker_t>);
PYBIND11_MAKE_OPAQUE(std::vector<vortex::default_marker_flags_t>);

// namespace pybind11 {
//     namespace detail {
//         template <typename... Ts>
//         struct type_caster<vortex::marker::marker_t<Ts...>> : variant_caster<vortex::marker::marker_t<Ts...>> {};
//     }
// }
