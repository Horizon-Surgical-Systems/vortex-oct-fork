#include <vortex-python/bind/common.hpp>

#include <mutex>

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <vortex/util/platform.hpp>

#include <vortex-python/bind/log.hpp>
#include <vortex-python/module/loader.hpp>

#if !defined(NDEBUG)
#  define _ASYNC_LOGGING
#endif

template<typename T>
static auto _to_string_view(T& o) {
    return std::string_view(o.data(), o.size());
}

// duplicate most of spdlog::sink::basic_file_sink here since it is marked final
template <typename Mutex>
class file_sink : public spdlog::sinks::base_sink<Mutex> {
public:
    explicit file_sink(const spdlog::filename_t& filename, bool truncate = false) {
        file_helper_.open(filename, truncate);
    }
    const spdlog::filename_t& filename() const {
        return file_helper_.filename();
    }
    void change(const spdlog::filename_t& filename, bool truncate = false) {
        std::lock_guard<Mutex> lock(spdlog::sinks::base_sink<Mutex>::mutex_);

        file_helper_.close();
        file_helper_.open(filename, truncate);
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        file_helper_.write(formatted);
    }
    void flush_() override {
        file_helper_.flush();
    }

private:
    spdlog::details::file_helper file_helper_;
};

using file_sink_mt = file_sink<std::mutex>;
using file_sink_st = file_sink<spdlog::details::null_mutex>;

template<typename Mutex>
class python_sink : public spdlog::sinks::base_sink<Mutex> {
public:
    python_sink() {
        _module_name = current_module_path().filename().string();

        py::gil_scoped_acquire gil;

        auto logging = py::module::import("logging");
        _getLogger = logging.attr("getLogger");
        _LogRecord = logging.attr("LogRecord");
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        // try to avoid copies
        auto name = _to_string_view(msg.logger_name);
        auto payload = _to_string_view(msg.payload);

        // check the cache for the appropriate Python logger
        auto& log = _loggers[name];

        py::gil_scoped_acquire gil;
        if (!log) {
            // create and cache the needed Python logger
            log = _getLogger(name).attr("handle");
        }

        // forward on to Python, mapping the level appropriately and rewriting the creation time
        auto record = _LogRecord(
            name,
            msg.level * 10,
            msg.source.empty() ? _module_name.c_str() : msg.source.filename,
            msg.source.line, payload,
            py::tuple(), // args
            py::none(), // exc_info
            msg.source.funcname
        );
        record.attr("created") = std::chrono::duration<double>(msg.time.time_since_epoch()).count();
        log(record);
    }

    void flush_() override {
        // no flush capability for a Python logger, only handlers
    }

    std::unordered_map<std::string_view, py::object> _loggers;

    py::object _getLogger, _LogRecord;
    std::string _module_name;
};
using python_sink_mt = python_sink<std::mutex>;
using python_sink_st = python_sink<spdlog::details::null_mutex>;

#if defined(_ASYNC_LOGGING)
    // thread-safe logger hands off messages to thread pool (single thread by default) that writes to non-thread-safe sinks in the background
#   define logger_t spdlog::async_logger
#   define stdout_color_sink_t spdlog::sinks::stdout_color_sink_st
#   define stderr_color_sink_t spdlog::sinks::stderr_color_sink_st
#   define file_sink_t file_sink_st
#   define python_sink_t python_sink_st
#else
    // thread-safe logger writes messages directly to thread-safe sinks to ensure messages captured immediately
#   define logger_t spdlog::logger
#   define stdout_color_sink_t spdlog::sinks::stdout_color_sink_mt
#   define stderr_color_sink_t spdlog::sinks::stderr_color_sink_mt
#   define file_sink_t file_sink_mt
#   define python_sink_t python_sink_mt
#endif

#define DEFAULT_PATTERN "d-%b-%Y %H:%M:%S.%f] %-10n %^(%L) %v%$"

static auto _make_logger_multi(std::string name, std::vector<spdlog::sink_ptr> sinks) {
    // remove duplicates
    std::sort(sinks.begin(), sinks.end());
    sinks.erase(std::unique(sinks.begin(), sinks.end()), sinks.end());

#if defined(_ASYNC_LOGGING)
    auto logger = std::make_shared<spdlog::async_logger>(std::move(name), sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::overrun_oldest);
#else
    auto logger = std::make_shared<spdlog::logger>(std::move(name), sinks.begin(), sinks.end());
#endif

    logger->set_pattern(DEFAULT_PATTERN);
    return logger;
}
static auto _make_logger_single(std::string name, spdlog::sink_ptr sink) {
    return _make_logger_multi(std::move(name), std::vector<spdlog::sink_ptr>{ sink });
}
static auto _make_logger_empty(std::string name) {
    return _make_logger_multi(std::move(name), std::vector<spdlog::sink_ptr>{});
}

static auto get_console_logger(const std::string& name, int level) {
    auto logger = _make_logger_single(name, std::make_shared<stderr_color_sink_t>());

    logger->set_level(static_cast<spdlog::level::level_enum>(level));

    return logger;
}

static auto get_file_logger(const std::string& name, const std::string& filename, int level) {
    auto logger = _make_logger_single(name, std::make_shared<file_sink_t>(filename));

    logger->set_level(static_cast<spdlog::level::level_enum>(level));

    return logger;
}

static auto get_python_logger(std::string name, const std::string& scope, int level) {
    name = scope + "." + name;

    auto logger = _make_logger_single(name, std::make_shared<python_sink_t>());
    logger->set_level(static_cast<spdlog::level::level_enum>(level));

    return logger;
}

void bind_log(py::module& root) {

    root.def("get_console_logger", &get_console_logger, "name"_a, "level"_a = static_cast<int>(spdlog::level::info));
    root.def("get_file_logger", &get_file_logger, "name"_a, "filename"_a, "level"_a = static_cast<int>(spdlog::level::info));
    root.def("get_python_logger", &get_python_logger, "name"_a, "scope"_a = "vortex", "level"_a = static_cast<int>(spdlog::level::debug));

    auto m = root.def_submodule("log");

    py::bind_vector<std::vector<spdlog::sink_ptr>>(m, "SinkList");

#if defined(_ASYNC_LOGGING)
    spdlog::init_thread_pool(8192, 1, []() { vortex::set_thread_name("spdlog Worker"); });
#endif

    {
        using C = spdlog::sinks::sink;
        CLS_PTR(Sink);
    }

    {
        using C = python_sink_t;
        CLS_BASE_PTR(PythonSink, spdlog::sinks::sink);

        c.def(py::init(), doc(c, "__init__"));

        setup_sink(c);

        c.def("__repr__", [](const C& o) { return "<PythonSink>"; });
    }

    {
        using C = stdout_color_sink_t;
        CLS_BASE_PTR(StdOutSink, spdlog::sinks::sink);

        c.def(py::init(), doc(c, "__init__"));

        setup_sink(c);

        // FXN(set_color);

        c.def("__repr__", [](const C& o) { return "<StdOutSink>"; });
    }

    {
        using C = stderr_color_sink_t;
        CLS_BASE_PTR(StdErrSink, spdlog::sinks::sink);

        c.def(py::init(), doc(c, "__init__"));

        setup_sink(c);

        // FXN(set_color);

        c.def("__repr__", [](const C& o) { return "<StdErrSink>"; });
    }
    m.attr("ConsoleSink") = m.attr("StdErrSink");

    {
        using C = file_sink_t;
        CLS_BASE_PTR(FileSink, spdlog::sinks::sink);

        c.def(py::init<const spdlog::filename_t&, bool>(), "filename"_a, "truncate"_a = false, doc(c, "__init__"));

        setup_sink(c);

        RO_ACC(filename);
        FXN_GIL(change, "filename"_a, "truncate"_a = false);

        c.def("__repr__", [](const C& o) { return fmt::format("<FileSink: {}>", o.filename()); });
    }

    {
        using C = logger_t;
        CLS_PTR(Logger);

        c.def(py::init(&_make_logger_empty), "name"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "__init__"));
        c.def(py::init(&_make_logger_single), "name"_a, "sink"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "__init__"));
        c.def(py::init(&_make_logger_multi), "name"_a, "sinks"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "__init__"));

        RO_ACC(name);
        c.def_property("level", [](C& o) { return static_cast<int>(o.level()); }, [](C& o, int v) { o.set_level(static_cast<spdlog::level::level_enum>(v)); }, doc(c, "level"));
        RO_ACC(sinks);

        FXN_GIL(set_pattern);
        c.def("add_sink", [](C& o, spdlog::sink_ptr sink) {
            auto& sinks = o.sinks();
            if (std::find(sinks.begin(), sinks.end(), sink) == sinks.end()) {
                sinks.emplace_back(std::move(sink));
            }
        }, py::call_guard<py::gil_scoped_release>(), doc(c, "add_sink"));
        c.def("remove_sink", [](C& o, const spdlog::sink_ptr& sink) {
            auto& sinks = o.sinks();
            sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
        }, py::call_guard<py::gil_scoped_release>(), doc(c, "remove_sink"));

        // NOTE: release the GIL because the Python sink will try to acquire it, causing deadlock in debug builds
        c.def("log", [](C& o, int level, const std::string& msg) { o.log(spdlog::level::level_enum(level), msg); }, "level"_a, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "log"));
        c.def("trace", [](C& o, const std::string& msg) { o.trace(msg); }, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "trace"));
        c.def("debug", [](C& o, const std::string& msg) { o.debug(msg); }, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "debug"));
        c.def("info", [](C& o, const std::string& msg) { o.info(msg); }, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "info"));
        c.def("warn", [](C& o, const std::string& msg) { o.warn(msg); }, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "warn"));
        c.def("error", [](C& o, const std::string& msg) { o.error(msg); }, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "error"));
        c.def("critical", [](C& o, const std::string& msg) { o.critical(msg); }, "msg"_a, py::call_guard<py::gil_scoped_release>(), doc(c, "critical"));

        FXN_GIL(flush);
        FXN_GIL(clone, "name"_a);

        c.def("__repr__", [](const C& o) { return fmt::format("<Logger: {} -> {}>", o.name(), list_repr(o.sinks())); });
    }
}
