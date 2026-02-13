#pragma once

#include <type_traits>
#include <variant>

namespace vortex {

    template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
    template<class... Ts> overloaded(Ts...)->overloaded<Ts...>; // not needed as of C++20

#define CALL(obj, function, ...) std::visit([&](auto& c) { return c.function(__VA_ARGS__); }, obj)
#define CALL_CONST(obj, function, ...) std::visit([&](const auto& c) { return c.function(__VA_ARGS__); }, obj)
#define DISPATCH(function, ...) return std::visit([&](auto& c) { return c.function(__VA_ARGS__); }, *this)
#define DISPATCH_CONST(function, ...) return std::visit([&](const auto& c) { return c.function(__VA_ARGS__); }, *this)
#define ACCESS(attribute) return std::visit([](auto& c) { return c.attribute; }, *this)
#define ACCESS_CONST(attribute) return std::visit([](const auto& c) { return c.attribute; }, *this)

    namespace variant::detail {

        // ref: https://stackoverflow.com/questions/55941964/how-to-filter-duplicate-types-from-tuple-c
        template <typename T, typename... Ts>
        struct unique : std::type_identity<T> {};

        template <typename... Ts, typename U, typename... Us>
        struct unique<std::variant<Ts...>, U, Us...>
            : std::conditional_t<(std::is_same_v<U, Ts> || ...)
            , unique<std::variant<Ts...>, Us...>
            , unique<std::variant<Ts..., U>, Us...>> {};

    }

    template <typename... Ts>
    using unique_variant = typename variant::detail::unique<std::variant<>, Ts...>::type;

    template<typename T, typename F>
    void for_each(T& list, F&& function) {
        for (auto& obj : list) {
            std::visit(function, obj);
        }
    }
    template<typename T, typename F>
    void for_each(const T& list, F&& function) {
        for (const auto& obj : list) {
            std::visit(function, obj);
        }
    }

    template<typename T, typename F>
    void for_each_key(T& map, F&& function) {
        for (auto& [key, _] : map) {
            std::visit(function, key);
        }
    }
    template<typename T, typename F>
    void for_each_key(const T& map, F&& function) {
        for (const auto& [key, _] : map) {
            std::visit(function, key);
        }
    }

    template<typename T, typename F>
    void for_each_key_value(T& map, F&& function) {
        for (auto& [key, val_] : map) {
            auto& val = val_; // to make clang happy
            std::visit([&](auto& inner) { std::invoke(function, inner, val); }, key);
        }
    }
    template<typename T, typename F>
    void for_each_key_value(const T& map, F&& function) {
        for (const auto& [key, val_] : map) {
            auto& val = val_; // to make clang happy
            std::visit([&function, &val](const auto& inner) { std::invoke(function, inner, val); }, key);
        }
    }

}
