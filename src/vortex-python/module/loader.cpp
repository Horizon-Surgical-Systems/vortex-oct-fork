#include <vortex-python/module/loader.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <vortex/core.hpp>
#include <vortex/util/platform.hpp>

static std::mutex _logger_mutex;
static std::shared_ptr<spdlog::logger> _logger;

#if defined(VORTEX_PLATFORM_WINDOWS)

std::filesystem::path current_module_path() {
    HMODULE handle;
    if (!::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCTSTR)&current_module_path, &handle)) {
        throw std::runtime_error(fmt::format("failed to access current module: {}", vortex::error_message_with_number()));
    }
    constexpr auto increment = 1024;

    std::vector<char> buffer;
    buffer.resize(increment);

    DWORD count = 0;
    while(true) {
        count = ::GetModuleFileName(handle, buffer.data(), buffer.size());
        if (count == 0) {
            throw std::runtime_error(fmt::format("failed to retrieve module path: {}", vortex::error_message_with_number()));
        } else if(count == buffer.size()) {
            buffer.resize(buffer.size() + increment);
        } else {
            break;
        }
    }

    return std::filesystem::path(buffer.begin(), buffer.begin() + count);
}

HMODULE load_library(const char* path) {
    // suppress error dialogs
    DWORD old_mode;
    ::SetThreadErrorMode(0, &old_mode);

    // load the library
    // NOTE: pass the same flags that Python uses with its new restricted DLL search path (https://github.com/python/cpython/pull/12302)
    auto handle = ::LoadLibraryEx(path, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
    auto error = ::GetLastError();

    // restore error mode
    ::SetThreadErrorMode(old_mode, NULL);

    // check that load succeeded
    if(handle == NULL) {
        throw std::runtime_error(fmt::format("failed to load library \"{}\": {}", path, vortex::error_message_with_number(error)));
    }

    return handle;
}

void free_library(HMODULE handle) {
    auto success = ::FreeLibrary(handle);
    if (!success) {
        throw std::runtime_error(fmt::format("failed to free library: {}", vortex::error_message_with_number()));
    }
}

void* lookup_function(HMODULE handle, const char* name) {
    auto ptr = ::GetProcAddress(handle, name);
    if(ptr == NULL) {
        throw std::runtime_error(fmt::format("failed to lookup function \"{}\": {}", name, vortex::error_message_with_number()));
    }

    return reinterpret_cast<void*>(ptr);
}

#elif defined(VORTEX_PLATFORM_LINUX)

std::filesystem::path current_module_path() {
    Dl_info info;
    auto result = dladdr((void*)&current_module_path, &info);
    if(result == 0) {
        throw std::runtime_error("failed to retrieve current module path");
    }

    return std::filesystem::path(info.dli_fname);
}

void* load_library(const char* path) {
    auto handle = ::dlopen(path, RTLD_NOW);
    if(handle == NULL) {
        throw std::runtime_error(fmt::format("failed to load library \"{}\": {}", path, ::dlerror()));
    }

    return handle;
}

void free_library(void* handle) {
    auto success = ::dlclose(handle);
    if (!success) {
        throw std::runtime_error(fmt::format("failed to free library: {}", ::dlerror()));
    }
}

void* lookup_function(void* handle, const char* name) {
    auto ptr = ::dlsym(handle, name);
    if(ptr == NULL) {
        throw std::runtime_error(fmt::format("failed to lookup function \"{}\": {}", name, ::dlerror()));
    }

    return reinterpret_cast<void*>(ptr);
}
#endif

template<typename... Args>
void _log(spdlog::level::level_enum level, const char* msg, Args&&... args) {
    std::unique_lock<std::mutex> lock(_logger_mutex);

    // ensure logger is set up
    if (!_logger) {
        _logger = spdlog::create<spdlog::sinks::stderr_color_sink_st>("loader");

        _logger->set_pattern("[%d-%b-%Y %H:%M:%S.%f] %-10n %^(%L) %v%$");
        _logger->set_level(spdlog::level::critical);

        // determine logging level
        auto var = vortex::envvar(VORTEX_LOADER_ENVVAR);
        try {
            if (var && std::stoi(*var) != 0) {
                _logger->set_level(spdlog::level::debug);
            }
        } catch (const std::exception&) {
            // no change
        }
    }

    // generate message
#if FMT_VERSION >= 80000
    _logger->log(level, fmt::runtime(msg), std::forward<Args>(args)...);
#else
    _logger->log(level, msg, std::forward<Args>(args)...);
#endif
}

bool load_and_bind_module(py::module& root, const std::string_view& name, const std::string_view& file_name) {
    _log(spdlog::level::info, "request to activate module \"{}\"", name);

    module_t handle = nullptr;
    std::filesystem::path load_path;
    bool success;

    try {

        // determine module path
        load_path = current_module_path().parent_path() / file_name;
        _log(spdlog::level::info, "loading \"{}\" from \"{}\"", name, load_path.string());

        // warn if does not exist to help with debugging
        if (!std::filesystem::exists(load_path)) {
            _log(spdlog::level::warn, "module path \"{}\" does not exist", load_path.string());
        }

        // attempt to load module
        handle = load_library(load_path.string().c_str());

        // attempt to initialize module
        auto init_name = fmt::format("bind_{}", name);
        auto init_ptr = lookup_function<init_ptr_t>(handle, init_name.c_str());
        _log(spdlog::level::info, "invoking initialization function \"{}\" at {}", init_name, (const void*)init_ptr);
        (*init_ptr)({ root, doc });

        // record outcome
        _log(spdlog::level::info, "module \"{}\" activated successfully", name);
        success = true;

    } catch (const std::exception& e) {
        if (handle) {
            _log(spdlog::level::critical, "module \"{}\" loaded but failed to initialize (likely bug): {}", name, e.what());
        } else {
            _log(spdlog::level::err, "module \"{}\" failed to load: {}", name, e.what());
        }
        
        // record outcome
        success = false;
    }

    if (handle && !success) {
        // clean up
        // NOTE: free the library outside the exception handler because the exception likely has allocated memory inside the library
        _log(spdlog::level::info, "unloading module \"{}\" from \"{}\"", name, load_path.string());
        free_library(handle);
    }

    return success;
}
