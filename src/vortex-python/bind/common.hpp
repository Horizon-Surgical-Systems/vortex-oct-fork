#pragma once

#include <sstream>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/chrono.h>

#include <xtensor/containers/xtensor.hpp>

#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyarray.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <spdlog/logger.h>

#include <vortex-python/doc/docstring.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

template<typename T>
struct xtensor_to_pytensor {
   using type = xt::pytensor<typename T::value_type, std::tuple_size_v<typename T::shape_type>>;
};
template<typename T>
using xtensor_to_pytensor_t = typename xtensor_to_pytensor<T>::type;

template<typename T>
struct xtensor_fixed_to_pytensor {
    using type = xt::pytensor<typename T::value_type, T::N>;
};
template<typename T>
using xtensor_fixed_to_pytensor_t = typename xtensor_to_pytensor<T>::type;

template<typename T>
struct xtensor_to_pyarray {
    using type = xt::pyarray<typename T::value_type>;
};
template<typename T>
using xtensor_to_pyarray_t = typename xtensor_to_pyarray<T>::type;

#define CLS_VAL(name, ...) auto c = py::class_<C>(m, #name, doc(m, #name));
#define CLS_PTR(name, ...) auto c = py::class_<C, std::shared_ptr<C>>(m, #name, doc(m, #name));
#define CLS_BASE_PTR(name, base, ...) auto c = py::class_<C, base, std::shared_ptr<C>>(m, #name, doc(m, #name));

#define SRO_VAR(var, ...) c.def_readonly_static(#var, &C::var, doc(c, #var), ##__VA_ARGS__)
#define RW_VAR(var, ...) c.def_readwrite(#var, &C::var, doc(c, #var), ##__VA_ARGS__);
#define RW_VAR_XT(var, ...) c.def_property(#var, [](C& o) -> std::decay_t<decltype(std::declval<C>().var)>& { return o.var; }, [](C& o, xtensor_to_pytensor_t<std::decay_t<decltype(std::declval<C>().var)>>& val) { o.var = val; }, doc(c, #var), ##__VA_ARGS__);

// TODO: avoid returning a copy
#define RW_VAR_XT_FIXED(var, ...) c.def_property(#var, [](C& o) -> xtensor_fixed_to_pytensor_t<std::decay_t<decltype(std::declval<C>().var)>> { return o.var; }, [](C& o, xtensor_fixed_to_pytensor_t<std::decay_t<decltype(std::declval<C>().var)>>& val) { o.var = val; }, doc(c, #var), ##__VA_ARGS__);

#define RW_MUT(var, ...) c.def_property(#var, [](C& o) -> std::decay_t<decltype(std::declval<C>().var())> { return o.var(); }, [](C& o, std::decay_t<decltype(std::declval<C>().var())>& val) { o.set_##var(std::forward<std::decay_t<decltype(std::declval<C>().var())>>(val)); }, doc(c, #var), ##__VA_ARGS__);
#define RW_MUT_GIL(var, ...) c.def_property(#var, [](C& o) -> std::decay_t<decltype(std::declval<C>().var())> { return o.var(); }, [](C& o, std::decay_t<decltype(std::declval<C>().var())>& val) { o.set_##var(std::forward<std::decay_t<decltype(std::declval<C>().var())>>(val)); }, py::call_guard<py::gil_scoped_release>(), doc(c, #var), ##__VA_ARGS__);
#define RW_ACC(var, ...) c.def_property(#var, [](C& o) -> std::decay_t<decltype(std::declval<C>().var())>& { return o.var(); }, [](C& o, std::decay_t<decltype(std::declval<C>().var())>& val) { o.var() = val; }, doc(c, #var), ##__VA_ARGS__);
#define RW_ACC_GIL(var, ...) c.def_property(#var, [](C& o) -> std::decay_t<decltype(std::declval<C>().var())>& { return o.var(); }, [](C& o, std::decay_t<decltype(std::declval<C>().var())>& val) { o.var() = val; }, py::call_guard<py::gil_scoped_release>(), doc(c, #var), ##__VA_ARGS__);
#define RW_ACC_XT(var, ...) c.def_property(#var, [](C& o) -> std::decay_t<decltype(std::declval<C>().var())>& { return o.var(); }, [](C& o, xtensor_to_pytensor_t<std::decay_t<decltype(std::declval<C>().var())>>& val) { o.var() = val; }, doc(c, #var), ##__VA_ARGS__);
#define RO_VAR(var, ...) c.def_readonly(#var, &C::var, doc(c, #var), ##__VA_ARGS__);
#define RO_ACC(var, ...) c.def_property_readonly(#var, [](const C& o) { return o.var(); }, doc(c, #var), ##__VA_ARGS__);
#define RO_ACC_GIL(var, ...) c.def_property_readonly(#var, [](const C& o) { return o.var(); }, py::call_guard<py::gil_scoped_release>(), doc(c, #var), ##__VA_ARGS__);
#define FXN(var, ...) c.def(#var, &C::var, doc(c, #var), ##__VA_ARGS__);
#define FXN_GIL(var, ...) c.def(#var, &C::var, py::call_guard<py::gil_scoped_release>(), doc(c, #var), ##__VA_ARGS__);
#define SFXN(var, ...) c.def_static(#var, &C::var, doc(c, #var), ##__VA_ARGS__);
#define PY_REPR(var) py::cast<std::string>(py::repr(py::cast(var)))

#define SHALLOW_COPY() c.def("copy", [](const C& o) { return o; }, py::return_value_policy::copy, doc(c, "copy"));

namespace detail {
    template<typename T> struct dtype {};

    template<> struct dtype<int8_t>  { constexpr static const char name[] = "int8";  constexpr static const char display_name[] = "Int8"; };
    template<> struct dtype<int16_t> { constexpr static const char name[] = "int16"; constexpr static const char display_name[] = "Int16"; };
    template<> struct dtype<int32_t> { constexpr static const char name[] = "int32"; constexpr static const char display_name[] = "Int32"; };
    template<> struct dtype<int64_t> { constexpr static const char name[] = "int64"; constexpr static const char display_name[] = "Int64"; };
    template<> struct dtype<uint8_t>  { constexpr static const char name[] = "uint8"; constexpr static const char display_name[] = "UInt8"; };
    template<> struct dtype<uint16_t> { constexpr static const char name[] = "uint16"; constexpr static const char display_name[] = "UInt16"; };
    template<> struct dtype<uint32_t> { constexpr static const char name[] = "uint32"; constexpr static const char display_name[] = "UInt32"; };
    template<> struct dtype<uint64_t> { constexpr static const char name[] = "uint64"; constexpr static const char display_name[] = "UInt64"; };
    template<> struct dtype<float> { constexpr static const char name[] = "float32"; constexpr static const char display_name[] = "Float32"; };
    template<> struct dtype<double> { constexpr static const char name[] = "float64"; constexpr static const char display_name[] = "Float64"; };
    template<> struct dtype<std::complex<float>> { constexpr static const char name[] = "complex64"; constexpr static const char display_name[] = "Complex64"; };
    template<> struct dtype<std::complex<double>> { constexpr static const char name[] = "complex128"; constexpr static const char display_name[] = "Complex128"; };
}
template<typename T>
using dtype = detail::dtype<T>;

template<typename Container>
std::string list_repr(const Container& cpp_list, const std::string& prefix = "") {
    auto py_list = py::cast(cpp_list);
    
    std::ostringstream out;
    out << prefix << "[";

    for (auto it = py_list.begin(); it != py_list.end(); it++) {
        out << py::repr(*it);
        if (it != py_list.begin()) {
            out << ", ";
        }
    }

    out << "]";
    return out.str();
}
