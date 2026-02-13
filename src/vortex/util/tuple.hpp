#pragma once

#include <tuple>

#include <vortex/util/exception.hpp>

namespace vortex {

    // ref: https://stackoverflow.com/questions/13101061/detect-if-a-type-is-a-stdtuple
    template <typename> struct is_tuple : std::false_type {};
    template <typename... T> struct is_tuple<std::tuple<T...>> : std::true_type {};

    template<typename Tuple, typename Function>
    inline void for_each_tuple(Tuple&& tup, Function&& func) {
        // ref: https://stackoverflow.com/questions/16387354/template-tuple-calling-a-function-on-each-element
        std::apply([&](auto&& ...x) { (static_cast<void>(func(std::forward<decltype(x)>(x))), ...); }, tup);
    }

    // ref: https://stackoverflow.com/questions/28997271/c11-way-to-index-tuple-at-runtime-without-using-switch
    template<class Func, class Tuple, size_t N = 0>
    inline void runtime_get(Tuple& tup, size_t idx, const Func& func) {
        if (N == idx) {
            std::invoke(func, std::get<N>(tup));
            return;
        }

        if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
            return runtime_get<Func, Tuple, N + 1>(tup, idx, func);
        }

        throw traced<std::invalid_argument>("tuple index out of bounds");
    }
    template<class Func, class Tuple, size_t N = 0>
    inline void runtime_get(const Tuple& tup, size_t idx, const Func& func) {
        if (N == idx) {
            std::invoke(func, std::get<N>(tup));
            return;
        }

        if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
            return runtime_get<Func, Tuple, N + 1>(tup, idx, func);
        }

        throw traced<std::invalid_argument>("tuple index out of bounds");
    }

    // ref: https://stackoverflow.com/questions/28997271/c11-way-to-index-tuple-at-runtime-without-using-switch
    template<typename Condition, class Func, class Tuple, size_t N = 0>
    inline void runtime_get_condition(Tuple& tup, size_t idx, const Func& func) {
        if (N == idx) {
            if constexpr (Condition::template evaluate<std::tuple_element_t<N, Tuple>>::value) {
                // NOTE: put inside constexpr so compiler does not generate this code path for invalid accesses, which suppresses warnings
                std::invoke(func, std::get<N>(tup));
                return;
            }

            throw traced<std::runtime_error>("tuple condition failed to match at target index");
        }

        if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
            return runtime_get_condition<Condition, Func, Tuple, N + 1>(tup, idx, func);
        }

        throw traced<std::invalid_argument>("tuple index out of bounds");
    }
    template<typename Condition, class Func, class Tuple, size_t N = 0>
    inline void runtime_get_condition(const Tuple& tup, size_t idx, const Func& func) {
        if (N == idx) {
            if constexpr (Condition::template evaluate<std::tuple_element_t<N, Tuple>>::value) {
                // NOTE: put inside constexpr so compiler does not generate this code path for invalid accesses, which suppresses warnings
                std::invoke(func, std::get<N>(tup));
                return;
            }

            throw traced<std::runtime_error>("tuple condition failed to match at target index");
        }

        if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
            return runtime_get_condition<Condition, Func, Tuple, N + 1>(tup, idx, func);
        }

        throw traced<std::invalid_argument>("tuple index out of bounds");
    }

    namespace detail::tuple {

        // ref: https://stackoverflow.com/questions/55941964/how-to-filter-duplicate-types-from-tuple-c
        template <typename T, typename... Ts>
        struct unique : std::type_identity<T> {};

        template <typename... Ts, typename U, typename... Us>
        struct unique<std::tuple<Ts...>, U, Us...>
            : std::conditional_t<(std::is_same_v<U, Ts> || ...)
            , unique<std::tuple<Ts...>, Us...>
            , unique<std::tuple<Ts..., U>, Us...>> {};

    }

    template <typename... Ts>
    using unique_tuple = typename detail::tuple::unique<std::tuple<>, Ts...>::type;

    namespace detail::tuple {

        // ref: https://stackoverflow.com/questions/15880756/how-to-get-the-position-of-a-tuple-element
        template<size_t I, size_t N, typename T, typename... Args>
        struct index_by_value {
            static ptrdiff_t call(const std::tuple<Args...>& t, T&& val) {
                return (std::get<I>(t) == val) ? I : index_by_value<I + 1, N, T, Args...>::call(t, std::forward<T>(val));
            }
        };

        template<size_t N, typename T, typename... Args>
        struct index_by_value<N, N, T, Args...>{
            static ptrdiff_t call(const std::tuple<Args...>& t, T&& val) {
                return -1;
            }
        };

        template<size_t I, size_t N, typename T, typename... Args>
        struct index_by_pointer {
            static ptrdiff_t call(const std::tuple<Args...>& t, const T& val) {
                return ((void*)&std::get<I>(t) == (void*)&val) ? I : index_by_pointer<I + 1, N, T, Args...>::call(t, val);
            }
        };

        template<size_t N, typename T, typename... Args>
        struct index_by_pointer<N, N, T, Args...> {
            static ptrdiff_t call(const std::tuple<Args...>& t, const T& val) {
                return -1;
            }
        };

    }

    template<typename T, typename... Args>
    ptrdiff_t index_by_value(const std::tuple<Args...>& t, T&& val) {
        return detail::tuple::index_by_value<0, sizeof...(Args), T, Args...>::call(t, std::forward<T>(val));
    }

    template<typename T, typename... Args>
    ptrdiff_t index_by_pointer(const std::tuple<Args...>& t, const T& val) {
        return detail::tuple::index_by_pointer<0, sizeof...(Args), T, Args...>::call(t, val);
    }

}
