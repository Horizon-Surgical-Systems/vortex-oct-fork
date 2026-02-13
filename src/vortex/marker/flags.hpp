#pragma once

#include <type_traits>
#include <limits>

namespace vortex::marker {

    template<typename T = uintmax_t, typename index_t = uint8_t>
    struct flags_t {
        static_assert(std::is_integral_v<T>, "only integral types supported");
        static_assert(std::is_unsigned_v<T>, "only unsigned types supported");

        flags_t() :
            flags_t(0) {

        }
        flags_t(T flags_) {
            value = flags_;
        }

        void set(index_t idx) {
            value |= (1uLL << idx);
        }
        void set() {
            value = -1;
        }

        void clear(index_t idx) {
            value &= ~(1uLL << idx);
        }
        void clear() {
            value = 0;
        }

        bool matches(const flags_t& o) const {
            return (value & o.value) != 0;
        }
        static bool matches(const flags_t& a, const flags_t& b) {
            return a.matches(b);
        }

        static flags_t all() {
            return flags_t(-1);
        }
        static flags_t none() {
            return flags_t(0);
        }

        static constexpr auto max_unique() {
            return std::numeric_limits<T>::digits;
        }
        auto operator <=>(const flags_t&) const = default;

        T value;
    };

}