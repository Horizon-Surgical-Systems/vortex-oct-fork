#pragma once

#include <string_view>

#include <fmt/format.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

const char* doc(const std::string_view& key);

// shortcut for common case
inline const char* doc(const std::string_view& key1, const std::string_view& key2) {
    return doc(fmt::format("{}.{}", key1, key2));
}
template<typename... Keys>
const char* doc(const std::string_view& key1, const std::string_view& key2, const std::string_view& key3, Keys&&... keys) {
    return doc(fmt::format("{}.{}.{}", key1, key2, key3), std::forward<Keys...>(keys)...);
}

template<typename... Keys>
const char* doc(const py::module& m, Keys&&... keys) {
    auto mod = m.attr("__name__").template cast<std::string>();
    return doc(mod, std::forward<Keys...>(keys)...);
}
template<typename... Args, typename... Keys>
const char* doc(const py::class_<Args...>& o, Keys&&... keys) {
    auto mod = o.attr("__module__").template cast<std::string>();
    auto cls = o.attr("__qualname__").template cast<std::string>();
    return doc(mod, cls, std::forward<Keys...>(keys)...);
}
