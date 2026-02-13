#pragma once

#include <vortex/util/tuple.hpp>

namespace vortex {

    namespace detail {

        template<typename element_t>
        struct stream_element_is_same {

            template<typename stream_t>
            struct evaluate {
                using T = element_t;
                using U = typename std::decay_t<stream_t>::derived_t::element_t;
                static auto constexpr value = 
                    std::is_same_v<T, U> ||
                    (std::is_integral_v<T> && std::is_integral_v<U> && std::is_signed_v<T> == std::is_signed_v<U> && sizeof(T) == sizeof(U));
            };
        };

    }

    template<typename T, typename... stream_ts, typename function_t>
    inline void select(const std::tuple<stream_ts...>& streams, size_t index, function_t&& function) {
        runtime_get_condition<detail::stream_element_is_same<T>>(streams, index, std::forward<function_t>(function));
    }

    template<typename T, typename... stream_ts, typename function_t>
    inline void select(const std::tuple<const stream_ts...>& streams, size_t index, function_t&& function) {
        runtime_get_condition<detail::stream_element_is_same<T>>(streams, index, std::forward<function_t>(function));
    }

    template<typename... stream_ts, typename function_t>
    inline void select_any(const std::tuple<stream_ts...>& streams, size_t idx, function_t&& function) {
        runtime_get(streams, idx, function);
    }
    template<typename... stream_ts, typename function_t>
    inline void select_any(const std::tuple<const stream_ts...>& streams, size_t idx, function_t&& function) {
        runtime_get(streams, idx, function);
    }

}
