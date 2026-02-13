#pragma once

#define VORTEX_LOADER_ENVVAR  "VORTEX_LOADER_LOG"

#include <filesystem>

#include <vortex/util/platform.hpp>

#include <vortex-python/doc/docstring.hpp>

#if defined(VORTEX_PLATFORM_WINDOWS)
#  include <Windows.h>
using module_t = HMODULE;
#elif defined(VORTEX_PLATFORM_LINUX)
#  include <dlfcn.h>
using module_t = void*;
#endif

struct module_info_t {
    py::module& root;

    using doc_ptr_t = const char* (*) (const std::string_view&);
    doc_ptr_t doc;
};

using init_ptr_t = void(*)(const module_info_t&);

std::filesystem::path current_module_path();

module_t load_library(const char* path);
void free_library(module_t handle);

void* lookup_function(module_t handle, const char* name);
template<typename T>
T lookup_function(module_t handle, const char* name) {
    return reinterpret_cast<T>(lookup_function(handle, name));
}

bool load_and_bind_module(py::module& root, const std::string_view& name, const std::string_view& file_name);
