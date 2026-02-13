#pragma once

#include <fmt/format.h>

#include <vortex/util/exception.hpp>

namespace vortex {

    namespace detail {
        template<typename To, typename Convert, typename From>
        void check_upper(From o) {
            auto ulimit = std::numeric_limits<To>::max();
            if (Convert(o) > Convert(ulimit)) {
                throw traced<std::out_of_range>(fmt::format("downcast upper limit exceeded for {}: {} > {}", typeid(To).name(), o, ulimit));
            }
        }

        template<typename To, typename Convert, typename From>
        void check_lower(From o) {
            auto llimit = std::numeric_limits<To>::lowest();
            if (Convert(o) < Convert(llimit)) {
                throw traced<std::out_of_range>(fmt::format("downcast lower limit exceeded for {}: {} < {}", typeid(To).name(), o, llimit));
            }
        }
    }

    template<typename T, typename U>
    T downcast(U o) {
        if constexpr (std::is_floating_point<U>::value) {
            detail::check_upper<T, double>(o);
            detail::check_lower<T, double>(o);
        } else if constexpr (std::is_unsigned<T>::value) {
            if constexpr (std::is_signed<U>::value) {
                // unsigned signed
                detail::check_upper<T, uintmax_t>(o);
                detail::check_lower<T, intmax_t>(o);
            } else {
                // unsigned unsigned
                detail::check_upper<T, uintmax_t>(o);
            }
        } else {
            if constexpr (std::is_signed<U>::value) {
                // signed signed
                detail::check_upper<T, intmax_t>(o);
                detail::check_lower<T, intmax_t>(o);
            } else {
                // signed unsigned
                detail::check_upper<T, uintmax_t>(o);
            }
        }
        return static_cast<T>(o);
    }

    template<typename T, typename = std::enable_if_t<std::is_enum<T>::value>>
    constexpr std::underlying_type_t<T> cast(const T& v) {
        return static_cast<std::underlying_type_t<T>>(v);
    }

}
