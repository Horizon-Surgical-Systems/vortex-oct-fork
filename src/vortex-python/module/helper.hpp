#pragma once

#if defined(VORTEX_ENABLE_MODULAR_BUILD)
// each compilation unit needs this defined
#  define FORCE_IMPORT_ARRAY
#endif

#include <vortex-python/bind/common.hpp>
#include <vortex-python/module/loader.hpp>

#if defined(VORTEX_ENABLE_MODULAR_BUILD)

// define an entry function that connects up docstring lookup correctly
#  define VORTEX_MODULE(name) \
    static module_info_t::doc_ptr_t _doc = nullptr; \
    const char* doc(const std::string_view& key) { \
        if (!_doc) { \
            throw std::runtime_error("docstring handler is missing"); \
        } \
        return _doc(key); \
    } \
    \
    static void _bind_root(py::module&); \
    extern "C" void bind_##name(const module_info_t& info) { \
        _doc = info.doc; \
        xt::import_numpy(); \
        _bind_root(info.root); \
    } \
    \
    static void _bind_root(py::module& root)

#else

// define the function for direct linking
#  define VORTEX_MODULE(name) \
    void bind_##name(py::module& root)

#endif

#define VORTEX_BIND(name) \
    auto name = root.attr(#name).cast<py::module>(); \
    _bind_##name(name);
